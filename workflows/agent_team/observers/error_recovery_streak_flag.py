"""ErrorRecoveryStreakFlag — logging-only observer that flags when an
agent hits several consecutive tool-call errors in a row.

Real, added 2026-09-08. This is a real, deliberately SIMPLIFIED version of
a much larger "RecoveryScaffoldingObserver" design proposed this session
(tracking recovery-scaffolding byte volume vs. task content, cumulative
ratios, oscillation correlation). That fuller design was not built:
"recovery scaffolding bytes" has no clean, direct signal in the real
tool-call/tool-result data this pipeline actually exposes -- it would
need real, additional design work to operationalize honestly, not a
straightforward port of the pattern used for the other observers built
tonight.

This simplified version uses a real, concrete, directly-observable proxy
instead: consecutive tool-call errors (ToolResult.is_error), which
genuinely represents an agent stuck trying to recover from repeated
failures -- the same underlying concern ("is this agent burning turns on
self-repair instead of real progress?") without needing to measure
scaffolding volume at all.

This observer does NOT change behavior -- matching the established,
deliberate design of the other real observers built this session: it
only logs a flagged anomaly.
"""

from __future__ import annotations

import logging

from frontier_agent.core.loop_types import BaseObserver, ToolResult, TurnContext

logger = logging.getLogger(__name__)

__all__ = ["ErrorRecoveryStreakFlag"]


class ErrorRecoveryStreakFlag(BaseObserver):
    """Real, logging-only: tracks consecutive real tool-call errors per
    agent; flags when the streak reaches ``streak_threshold``."""

    def __init__(self, *, streak_threshold: int = 3) -> None:
        self.streak_threshold = max(2, int(streak_threshold))
        self._streak: dict[str, int] = {}
        self._flagged: set[str] = set()

    async def on_tool_result(
        self, ctx: TurnContext, result: ToolResult,
    ) -> ToolResult | None:
        if not result.is_error:
            self._streak[ctx.task_id] = 0
            self._flagged.discard(ctx.task_id)
            return None

        streak = self._streak.get(ctx.task_id, 0) + 1
        self._streak[ctx.task_id] = streak

        if streak < self.streak_threshold or ctx.task_id in self._flagged:
            return None
        self._flagged.add(ctx.task_id)

        # Real, print()-based (not logger.warning()), matching the
        # established, confirmed-necessary pattern from earlier this same
        # session -- logger output has nowhere real to go here.
        print(
            f"[ERROR_RECOVERY_STREAK_FLAG] task={ctx.task_id} "
            f"role={ctx.role_id} turn={ctx.turn} -- {streak} consecutive "
            f"real tool-call errors (most recent: {result.name!r}). "
            f"Possible real recovery-loop behavior. This is a real, "
            f"plausible risk signal, not a confirmed fabrication cause.",
            flush=True,
        )
        return None


__all__ = ["ErrorRecoveryStreakFlag"]
