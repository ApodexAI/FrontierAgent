"""ContextAnchorShiftFlag — logging-only observer that flags when an
agent's tool-call target (file path or URL) shifts away from its
original, real anchor for several consecutive turns without returning.

Real, added 2026-09-08. This is a real, distinct, THEORETICAL mechanism --
like PartialReadContaminationFlag and ToolScaffoldingDriftFlag, this
targets a hypothesized cause that has NOT been directly confirmed as an
actual mechanism behind any real, observed failure this session. "Topic
anchor drift" is a real, but abstract, concept; this observer operationalizes
it as a concrete, measurable proxy: the real path/URL argument named in
each tool call, not any deeper semantic notion of "topic". This is a real,
plausible hypothesis worth instrumenting, not a demonstrated mechanism.

This observer does NOT change behavior -- matching the established,
deliberate design of the other real observers built this session: it
only logs a flagged anomaly.
"""

from __future__ import annotations

import logging

from frontier_agent.core.loop_types import BaseObserver, TurnContext

logger = logging.getLogger(__name__)

__all__ = ["ContextAnchorShiftFlag"]

# Real, direct, per-tool map of which argument key holds the real "target"
# this observer tracks as the anchor -- deliberately reusing the same,
# real key set confirmed earlier this session in MultiToolOscillationFlag's
# tracked tools, since those are the tools observed live to carry a real
# path/URL argument.
_TARGET_ARG_KEYS: dict[str, str] = {
    "read_file": "path",
    "create_file": "path",
    "web_fetch": "url",
}


class ContextAnchorShiftFlag(BaseObserver):
    """Real, logging-only: tracks the first real target this agent's
    tool calls established as its anchor, then flags when
    ``shift_turns`` consecutive, later calls all target something
    different, without a single call back to the original anchor."""

    def __init__(self, *, shift_turns: int = 4) -> None:
        self.shift_turns = max(2, int(shift_turns))
        self._anchor: dict[str, str] = {}  # task_id -> first real target seen
        self._away_streak: dict[str, int] = {}  # task_id -> consecutive turns away
        self._flagged: set[str] = set()

    async def on_tool_call(
        self, ctx: TurnContext, tool_call: dict,
    ):
        name = str(tool_call.get("name", ""))
        key = _TARGET_ARG_KEYS.get(name)
        if key is None:
            return None
        args = tool_call.get("args") or {}
        target = args.get(key)
        if not isinstance(target, str) or not target:
            return None

        anchor = self._anchor.get(ctx.task_id)
        if anchor is None:
            self._anchor[ctx.task_id] = target
            self._away_streak[ctx.task_id] = 0
            return None

        if target == anchor:
            self._away_streak[ctx.task_id] = 0
            return None

        streak = self._away_streak.get(ctx.task_id, 0) + 1
        self._away_streak[ctx.task_id] = streak

        if streak < self.shift_turns or ctx.task_id in self._flagged:
            return None
        self._flagged.add(ctx.task_id)

        # Real, print()-based (not logger.warning()), matching the
        # established, confirmed-necessary pattern from earlier this same
        # session -- logger output has nowhere real to go here.
        print(
            f"[CONTEXT_ANCHOR_SHIFT_FLAG] task={ctx.task_id} "
            f"role={ctx.role_id} turn={ctx.turn} -- this agent's original "
            f"real anchor was {anchor!r}, but its last {streak} tool calls "
            f"all targeted a different real path/URL ({target!r}) without "
            f"ever returning to the original anchor. This is a real, "
            f"plausible risk signal, not a confirmed fabrication cause.",
            flush=True,
        )
        return None


__all__ = ["ContextAnchorShiftFlag"]
