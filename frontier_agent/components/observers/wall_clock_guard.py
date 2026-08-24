"""Stop sub-agent loops cleanly before their hard wall-time cancellation.

The guard reserves one worst-case turn and stamps the deadline so retries are
also bounded, preserving partial work for normal finalization.
"""
from __future__ import annotations

import logging

from frontier_agent.components.observers.wall_clock_observer import (
    WallClockDeadlineObserver,
)
from frontier_agent.core.loop_types import LoopConfig

logger = logging.getLogger(__name__)

_DEFAULT_RESERVE_S = 600.0


class WallClockGuard(WallClockDeadlineObserver):
    """Stop the loop ``reserve`` seconds before ``budget`` is exhausted.

    Critical observer (inherited) — its ``stop_reason`` reaches the loop
    driver.

    Args:
        budget_s: Total wall-time the loop is allowed, i.e. the same value
            the hard ``asyncio.wait_for`` uses. Pass ``SpawnGuard.timeout_s``
            so the two can never drift apart.
        reserve_s: Headroom subtracted from ``budget_s``. Raised at
            ``on_loop_start`` to at least ``llm_timeout + tool_timeout + 60``
            so a turn beginning just under the soft deadline still finishes
            inside the hard one.
    """

    # A sub-agent's way to wrap up is to submit its report — it has no
    # sub-agents of its own to stop spawning, which is what the parent's
    # coordinator-facing default tells it to do.
    STOP_MESSAGE = (
        "Wall-clock budget nearly exhausted ({elapsed}s of {budget}s used). "
        "Stop gathering and deliver your report now, based on the information "
        "you already have."
    )
    WARN_MESSAGE = (
        "Warning: ~{remaining}s of usable time left on your wall-clock "
        "budget. Start consolidating what you have into your report rather "
        "than opening new lines of investigation."
    )

    def __init__(
        self, budget_s: float, reserve_s: float = _DEFAULT_RESERVE_S,
    ) -> None:
        if budget_s <= 0:
            raise ValueError(f"budget_s must be positive, got {budget_s}")
        super().__init__(budget_s, reserve_s=reserve_s)
        # The reserve the CALLER asked for. ``on_loop_start`` raises it to the
        # config-derived floor, which is not knowable at construction time —
        # AgentBus has the SpawnGuard budget but not the loop's timeouts.
        self._requested_reserve_s = float(reserve_s)

    async def on_loop_start(self, config: LoopConfig) -> None:
        # One worst-case turn = a full LLM timeout plus a full tool timeout,
        # because the stop check only runs between turns. Note this covers a
        # single LLM ATTEMPT; retries are bounded instead by the scope stamp
        # the parent publishes here (see the module docstring).
        floor = float(
            getattr(config, "llm_timeout", 0) or 0,
        ) + float(
            getattr(config, "tool_timeout", 0) or 0,
        ) + 60.0
        self.reserve_s = max(self._requested_reserve_s, floor)
        # Recompute before delegating: the parent stamps the ABSOLUTE soft
        # deadline into the execution scope from ``self.soft_deadline_s``, so
        # it has to be final by the time we call up. The parent's own
        # half-budget clamp (``max(deadline - reserve, deadline * 0.5)``)
        # applies, keeping the loop runnable when the floor exceeds the
        # budget.
        self.soft_deadline_s = max(
            self.deadline_s - self.reserve_s, self.deadline_s * 0.5,
        )
        await super().on_loop_start(config)
        logger.debug(
            "WallClockGuard armed: budget=%.0fs reserve=%.0fs "
            "soft_deadline=%.0fs",
            self.deadline_s, self.reserve_s, self.soft_deadline_s,
        )
