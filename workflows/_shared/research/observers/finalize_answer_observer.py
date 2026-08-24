"""Stops the agent loop on a terminal "I am done" tool call."""
from __future__ import annotations

import logging
from collections.abc import Iterable

from frontier_agent.core.loop_types import (
    BaseObserver,
    Intervention,
    ToolResult,
    TurnContext,
)

logger = logging.getLogger(__name__)

FINALIZE_TOOL_NAME = "finalize_answer"


class FinalizeAnswerObserver(BaseObserver):
    critical = True

    def __init__(
        self,
        tool_names: Iterable[str] = (FINALIZE_TOOL_NAME,),
        content_arg_name: str = "content",
    ) -> None:
        self._tool_names = frozenset(tool_names)
        self._content_arg_name = content_arg_name
        self._triggered = False

    async def on_tool_result(self, ctx: TurnContext, result: ToolResult) -> None:
        if result.name not in self._tool_names:
            return
        if result.is_error:
            # Let the model see the error and try again; don't latch.
            return

        args = result.args if isinstance(result.args, dict) else {}
        content = (args.get(self._content_arg_name) or "").strip()
        if not content:
            return

        try:
            confidence = max(0.0, min(1.0, float(args.get("confidence", 0.7))))
        except (TypeError, ValueError):
            confidence = 0.7

        self._triggered = True
        if isinstance(ctx.metadata, dict):
            ctx.metadata["final_answer"] = content
            ctx.metadata["final_answer_confidence"] = confidence

        logger.info(
            "FinalizeAnswerObserver latched via %s: turn=%d, len=%d, conf=%.2f",
            result.name, ctx.turn, len(content), confidence,
        )

    async def on_turn_end(self, ctx: TurnContext) -> Intervention | None:
        if not self._triggered:
            return None
        # One-shot: subsequent on_turn_end calls shouldn't re-emit.
        self._triggered = False
        return Intervention(stop_reason="final_answer")
