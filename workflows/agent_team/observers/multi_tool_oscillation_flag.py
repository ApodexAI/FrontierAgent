"""MultiToolOscillationFlag — logging-only observer that flags when an
agent oscillates between many DIFFERENT tool types in a short window of
turns without progressing to submit_report.

Real, added 2026-09-08. This is a distinct, real signal from
CycleDetectionGuard (added earlier this same session): that observer
detects a STABLE, EXACT, repeating N-tool sequence (e.g. A,B,A,B,... with
byte-identical arguments each cycle). This observer detects a genuinely
different, real pattern -- high DIVERSITY of tool types used within a
short window, regardless of whether any exact repetition occurs at all.
Confirmed directly, live, this same session: a sub-agent that later
fabricated a claim about a file had, shortly before, used read_file,
create_file, AND web_search within a handful of turns -- real, varied
activity, not a stable loop, but still a real signal the agent may be
"casting around" rather than making real progress.

This observer does NOT change behavior -- matching the established,
deliberate design of SelfConsistencyFlag and FabricationFlag: it only
logs a flagged anomaly, so a real pattern can be tracked across runs
before any intervention is designed.
"""

from __future__ import annotations

import logging

from frontier_agent.core.loop_types import BaseObserver, TurnContext

logger = logging.getLogger(__name__)

__all__ = ["MultiToolOscillationFlag"]

# Real, direct tool-type set this observer tracks diversity across --
# deliberately excludes submit_report itself (the terminal tool; calling
# it is real, genuine progress, not oscillation) and update_task/collect_
# reports (coordinator-only tools that never appear in a sub-agent's own
# trace).
_TRACKED_TOOLS = frozenset({
    "read_file", "create_file", "web_search", "web_fetch",
    "grep_search", "glob_search", "bash", "recover_result",
})


class MultiToolOscillationFlag(BaseObserver):
    """Real, logging-only: tracks the real, distinct tool TYPES used in the
    last ``window`` turns; flags when that count reaches ``min_distinct``
    without an intervening submit_report."""

    def __init__(self, *, window: int = 6, min_distinct: int = 3) -> None:
        self.window = max(2, int(window))
        self.min_distinct = max(2, int(min_distinct))
        # task_id -> list of (turn, tool_name) for the tracked-tool calls
        # seen so far this run (only ever holds the last `window` turns'
        # worth, trimmed below).
        self._recent: dict[str, list[tuple[int, str]]] = {}
        self._flagged_at: dict[str, int] = {}  # task_id -> turn last flagged

    async def on_tool_call(
        self, ctx: TurnContext, tool_call: dict,
    ):
        name = str(tool_call.get("name", ""))
        if name == "submit_report":
            # Real, genuine progress -- clear this agent's oscillation
            # history so a fresh window starts counting from here.
            self._recent.pop(ctx.task_id, None)
            self._flagged_at.pop(ctx.task_id, None)
            return None
        if name not in _TRACKED_TOOLS:
            return None

        history = self._recent.setdefault(ctx.task_id, [])
        history.append((ctx.turn, name))
        cutoff = ctx.turn - self.window
        history[:] = [(t, n) for t, n in history if t > cutoff]

        distinct = {n for _, n in history}
        if len(distinct) < self.min_distinct:
            return None
        if self._flagged_at.get(ctx.task_id) == ctx.turn:
            return None  # Already flagged at this exact turn.
        self._flagged_at[ctx.task_id] = ctx.turn

        # Real, print()-based (not logger.warning()), matching the
        # established, confirmed-necessary pattern from earlier this same
        # session -- logger output has nowhere real to go here.
        print(
            f"[MULTI_TOOL_OSCILLATION_FLAG] task={ctx.task_id} "
            f"role={ctx.role_id} turn={ctx.turn} -- {len(distinct)} "
            f"distinct tool types ({sorted(distinct)}) used within the "
            f"last {self.window} turns without calling submit_report. "
            f"Possible tool-induced drift/oscillation.",
            flush=True,
        )
        return None


__all__ = ["MultiToolOscillationFlag"]
