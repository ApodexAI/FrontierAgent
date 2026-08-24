"""ContextSizeGuard — pre-empt LLM context-window overflow."""
from __future__ import annotations

import logging
from typing import Any

from frontier_agent.core.loop_types import (
    BaseObserver,
    Intervention,
    LoopConfig,
    TurnContext,
)
from frontier_agent.core.runtime.loop.compact import (
    COMPACTION_SEQ_KEY,
    FORCE_COMPACTION_KEY,
)

logger = logging.getLogger(__name__)


class ContextSizeGuard(BaseObserver):
    """Stop at ``max_input_tokens``, optionally after one forced compaction.

    Critical observer — its ``stop_reason`` must reach the loop driver.
    """

    critical: bool = True

    def __init__(
        self,
        max_input_tokens: int,
        *,
        force_compaction_first: bool = False,
    ) -> None:
        if max_input_tokens <= 0:
            raise ValueError(
                f"max_input_tokens must be positive, got {max_input_tokens}",
            )
        self._limit = max_input_tokens
        self._tripped = False
        self._force_compaction_first = force_compaction_first
        self._requested_at_seq: int | None = None
        self._rearms_left = 0

    async def on_loop_start(self, config: LoopConfig) -> None:
        # Defensive: reset state if the same instance is ever reused.
        self._tripped = False
        self._requested_at_seq = None
        self._rearms_left = 0

    async def on_llm_response(self, ctx: TurnContext) -> Intervention | None:
        if self._tripped or ctx.usage is None:
            return None
        # Normalised usage (infra/usage.py) exposes ``prompt_tokens``; raw
        # Anthropic payloads use ``input_tokens``. Read prompt_tokens first —
        # reading only ``input_tokens`` meant this guard saw 0 and never tripped.
        used = int(ctx.usage.get("prompt_tokens") or ctx.usage.get("input_tokens") or 0)
        if used <= self._limit:
            self._requested_at_seq = None
            return None
        if self._force_compaction_first:
            seq = int(ctx.metadata.get(COMPACTION_SEQ_KEY, 0) or 0)
            if self._requested_at_seq is None:
                self._requested_at_seq = seq
                self._rearms_left = 1
                ctx.metadata[FORCE_COMPACTION_KEY] = True
                logger.info(
                    "ContextSizeGuard: turn=%d input_tokens=%d > limit=%d — "
                    "forcing one compaction pass before stopping",
                    ctx.turn, used, self._limit,
                )
                return None
            # Only turn end advances COMPACTION_SEQ_KEY, and a turn can skip it
            # entirely: a ``no_tool`` nudge and an observer asking for
            # ``continue_to_next_turn`` both ``continue`` before it. Re-arming
            # on an unadvanced seq without a bound therefore never stops — the
            # loop keeps issuing over-limit requests until the attempt budget
            # runs out, where one over-limit request used to end it. Allow a
            # single retry for a skipped turn, then stop.
            if seq <= self._requested_at_seq and self._rearms_left > 0:
                self._rearms_left -= 1
                ctx.metadata[FORCE_COMPACTION_KEY] = True
                logger.info(
                    "ContextSizeGuard: turn=%d compaction has not run yet "
                    "(seq=%d) — re-arming once before stopping",
                    ctx.turn, seq,
                )
                return None
        self._tripped = True
        logger.warning(
            "ContextSizeGuard: turn=%d input_tokens=%d > limit=%d — "
            "stopping early to force a clean final answer",
            ctx.turn, used, self._limit,
        )
        return Intervention(
            stop_reason="budget_exhausted",
            inject_messages=[
                f"Context approaching the model's limit "
                f"({used:,} > {self._limit:,} tokens). Stopping now to "
                f"deliver the answer based on information gathered so far."
            ],
        )

    async def on_loop_end(self, result: Any) -> None:
        return None
