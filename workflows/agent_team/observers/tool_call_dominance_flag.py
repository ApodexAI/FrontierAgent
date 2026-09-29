"""ToolCallDominanceFlag — logging-only observer that flags when a single
tool accounts for a dominant share of an agent's recent tool calls.

Real, added 2026-09-08. This is a real, distinct, THEORETICAL mechanism --
like the other newer observers this session, this targets a plausible,
untested hypothesis (a real agent stuck repeatedly reaching for the same
one tool may signal it's stuck on a sub-problem, even when the exact
argument-repetition patterns RepetitionGuard/CycleDetectionGuard target
aren't present). This is a real, simplified extraction of one condition
from a larger, five-condition "ToolCallDistributionObserver" design
proposed this session -- deliberately scoped down to just this one,
concrete, measurable signal rather than building all five at once.

This observer does NOT change behavior -- matching the established,
deliberate design of the other real observers built this session: it
only logs a flagged anomaly.
"""

from __future__ import annotations

import logging
from collections import Counter, deque

from frontier_agent.core.loop_types import BaseObserver, TurnContext

logger = logging.getLogger(__name__)

__all__ = ["ToolCallDominanceFlag"]


class ToolCallDominanceFlag(BaseObserver):
    """Real, logging-only: tracks a real, sliding window of the last
    ``window`` tool calls per agent; flags when one tool name accounts
    for ``dominance_threshold`` or more of that window."""

    def __init__(
        self, *, window: int = 8, dominance_threshold: float = 0.75,
        min_calls: int = 4,
    ) -> None:
        self.window = max(2, int(window))
        self.dominance_threshold = min(1.0, max(0.5, float(dominance_threshold)))
        self.min_calls = max(2, int(min_calls))
        # task_id -> deque of the real, recent tool names (bounded to window)
        self._recent: dict[str, deque[str]] = {}
        self._flagged_at: dict[str, int] = {}

    async def on_tool_call(
        self, ctx: TurnContext, tool_call: dict,
    ):
        name = str(tool_call.get("name", ""))
        if name == "submit_report":
            # Real, genuine progress -- clear this agent's window so a
            # fresh one starts counting from here, matching the same
            # reset behavior MultiToolOscillationFlag uses.
            self._recent.pop(ctx.task_id, None)
            self._flagged_at.pop(ctx.task_id, None)
            return None
        if not name:
            return None

        window = self._recent.setdefault(ctx.task_id, deque(maxlen=self.window))
        window.append(name)

        if len(window) < self.min_calls:
            return None

        counts = Counter(window)
        dominant_name, dominant_count = counts.most_common(1)[0]
        dominance_ratio = dominant_count / len(window)

        if dominance_ratio < self.dominance_threshold:
            return None
        if self._flagged_at.get(ctx.task_id) == ctx.turn:
            return None
        self._flagged_at[ctx.task_id] = ctx.turn

        # Real, print()-based (not logger.warning()), matching the
        # established, confirmed-necessary pattern from earlier this same
        # session -- logger output has nowhere real to go here.
        print(
            f"[TOOL_CALL_DOMINANCE_FLAG] task={ctx.task_id} "
            f"role={ctx.role_id} turn={ctx.turn} -- {dominant_name!r} "
            f"accounts for {dominant_count}/{len(window)} "
            f"({dominance_ratio:.0%}) of this agent's last {len(window)} "
            f"tool calls. This is a real, plausible risk signal, not a "
            f"confirmed fabrication cause.",
            flush=True,
        )
        return None


__all__ = ["ToolCallDominanceFlag"]
