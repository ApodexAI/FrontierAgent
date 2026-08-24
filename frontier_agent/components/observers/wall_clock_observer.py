"""WallClockDeadlineObserver — stop the loop gracefully before a hard."""
from __future__ import annotations

import time

from frontier_agent.core.execution_context import get_current_execution_scope
from frontier_agent.core.loop_types import (
    WALL_DEADLINE_MONOTONIC_KEY,
    Intervention,
    LoopConfig,
    TurnContext,
)
from frontier_agent.infra.wall_time_lease import (
    WALL_TIME_LEASE_SCOPE_KEY,
    RenewableWallTimeDeadline,
    RenewableWallTimeLease,
)

# ``WALL_DEADLINE_MONOTONIC_KEY`` is re-exported here for the historical
# import path — the key itself moved to ``frontier_agent.core.loop_types``
# so framework-level readers (``llm_client.call_llm``) don't depend on
# the observer layer.
__all__ = ["WALL_DEADLINE_MONOTONIC_KEY", "WallClockDeadlineObserver"]


class WallClockDeadlineObserver:
    """Force a graceful loop exit shortly before a hard wall-clock cap.

    critical = True  → awaited; Intervention return values are collected.

    Args:
        deadline_s: Total wall-clock budget for the loop, in seconds —
            mirror the outer ``run_timeout_s`` cap.
        reserve_s: Seconds to reserve before ``deadline_s`` for any post-loop
            work that is meant to remain inside the budget. Dedicated reporter
            phases can set this to zero when they are budgeted separately. The
            soft deadline is ``deadline_s - reserve_s``.
        warn_ratio: Inject a one-shot "wrap up soon" nudge once the
            elapsed fraction of the soft budget crosses this ratio.
    """

    critical = True

    # Nudge templates, as class attributes so a subclass can re-word them for
    # a different audience without duplicating the deadline logic.
    # ``STOP_MESSAGE`` is formatted with ``elapsed`` / ``budget`` (ints, in
    # seconds); ``WARN_MESSAGE`` with ``remaining``. The defaults address a
    # fan-out COORDINATOR, whose way to wrap up is to stop spawning and
    # finalize; see ``WallClockGuard`` for the sub-agent wording.
    STOP_MESSAGE = (
        "Time budget nearly exhausted ({elapsed}s / {budget}s wall-clock). "
        "Stop spawning, searching, or waiting on sub-agents now. Preserve any "
        "existing artifacts and finalize a best-effort answer immediately "
        "from the work already completed."
    )
    WARN_MESSAGE = (
        "Warning: ~{remaining}s of usable time left before the wall-clock "
        "deadline. Enter finalization now: stop new exploration, finish and "
        "publish the best available deliverables while tools are still "
        "available, run only essential checks, and prepare the final answer."
    )

    def __init__(
        self,
        deadline_s: float,
        *,
        reserve_s: float = 150.0,
        warn_ratio: float = 0.8,
    ) -> None:
        self.deadline_s = float(deadline_s)
        self.reserve_s = float(reserve_s)
        self.warn_ratio = float(warn_ratio)
        # Soft deadline never goes below half the budget — a tiny budget
        # with a large reserve must not stop the loop before turn one.
        self.soft_deadline_s = max(
            self.deadline_s - self.reserve_s, self.deadline_s * 0.5,
        )
        self._start: float | None = None
        self._warned: bool = False
        self._lease: RenewableWallTimeLease | None = None
        self._renewable_deadline: RenewableWallTimeDeadline | None = None
        self._lease_sequence = 0

    async def on_loop_start(self, config: LoopConfig) -> None:
        self._start = time.monotonic()
        self._warned = False
        self._lease = None
        self._renewable_deadline = None
        # Publish the absolute soft deadline into the execution scope so
        # long-blocking tools (collect_reports) can clamp their wait and
        # return before the hard cap cancels the loop mid-call. Best-effort
        # — a missing scope just means tools fall back to their own timeout.
        try:
            scope = get_current_execution_scope()
            if scope is not None:
                lease = scope.metadata.get(WALL_TIME_LEASE_SCOPE_KEY)
                if isinstance(lease, RenewableWallTimeLease):
                    renewable_deadline = lease.bind_duration(self.soft_deadline_s)
                    self._lease = lease
                    self._renewable_deadline = renewable_deadline
                    self._lease_sequence = lease.sequence
                    scope.metadata[WALL_DEADLINE_MONOTONIC_KEY] = renewable_deadline
                else:
                    scope.metadata[WALL_DEADLINE_MONOTONIC_KEY] = (
                        self._start + self.soft_deadline_s
                    )
        except Exception:
            pass

    async def on_llm_response(self, ctx: TurnContext) -> None:
        pass

    async def on_tool_result(self, ctx: TurnContext, result: object) -> None:
        pass

    async def on_turn_end(self, ctx: TurnContext) -> Intervention | None:
        if self._start is None:
            return None
        if self._lease is not None:
            sequence = self._lease.sequence
            if sequence != self._lease_sequence:
                self._lease_sequence = sequence
                self._warned = False
            renewable_deadline = self._renewable_deadline
            if renewable_deadline is None:
                return None
            elapsed = renewable_deadline.elapsed_s()
            ctx.metadata["walltime_reset_seq"] = sequence
        else:
            elapsed = time.monotonic() - self._start
        ctx.metadata["wall_elapsed_s"] = int(elapsed)
        ctx.metadata["wall_soft_deadline_s"] = int(self.soft_deadline_s)

        if elapsed >= self.soft_deadline_s:
            return Intervention(
                stop_reason="wall_deadline",
                inject_messages=[
                    self.STOP_MESSAGE.format(
                        elapsed=int(elapsed), budget=int(self.deadline_s),
                    )
                ],
            )

        if (
            self.soft_deadline_s > 0
            and elapsed >= self.soft_deadline_s * self.warn_ratio
            and not self._warned
        ):
            self._warned = True
            return Intervention(
                inject_messages=[
                    self.WARN_MESSAGE.format(
                        remaining=int(self.soft_deadline_s - elapsed),
                    )
                ],
            )

        return None

    async def on_loop_end(self, result: object) -> None:
        pass
