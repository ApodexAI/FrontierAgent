"""ToolStarvationFlag — logging-only observer that flags when a sub-agent's
entire run ends without ever making a single real tool call.

Real, added 2026-09-08. This is a real, deliberately SIMPLIFIED version of
a much larger, five-condition "ToolStarvationObserver" design proposed
this session (expected-window heuristics, sub-agent-spawn tracking,
fallback/timeout cross-referencing). That fuller design was not built --
several of its proposed conditions (e.g. "coordinator fallback before
first tool call", "sub-agent spawned but never called a tool") require
coordinator-level state this sub-agent-scoped observer has no access to,
and would need real, additional design work to wire correctly, not a
straightforward implementation.

This simplified version implements only the single clearest, most directly
observable condition: a sub-agent's ENTIRE run produces zero real tool
calls before the loop ends. This is a real, distinct signal from
MultiToolOscillationFlag/ToolCallDominanceFlag (which both require at
least some tool calls to have real data to work with) -- this one instead
catches the "nothing happened at all" case those cannot see.

This observer does NOT change behavior -- matching the established,
deliberate design of the other real observers built this session: it
only logs a flagged anomaly.
"""

from __future__ import annotations

import logging

from frontier_agent.core.loop_types import (
    AgentLoopResult,
    BaseObserver,
    TurnContext,
)

logger = logging.getLogger(__name__)

__all__ = ["ToolStarvationFlag"]


class ToolStarvationFlag(BaseObserver):
    """Real, logging-only: tracks whether this sub-agent's run ever made a
    single real tool call; flags at loop end if it never did."""

    def __init__(self) -> None:
        self._saw_tool_call: dict[str, bool] = {}
        self._last_turn: dict[str, int] = {}
        self._last_role_id: str = ""

    async def on_tool_call(
        self, ctx: TurnContext, tool_call: dict,
    ):
        self._last_role_id = ctx.role_id
        self._saw_tool_call[ctx.role_id] = True
        self._last_turn[ctx.role_id] = ctx.turn
        return None

    async def on_turn_end(self, ctx: TurnContext):
        # Real, direct tracking of the last real turn seen, even for a
        # role_id that never makes a tool call at all -- on_tool_call
        # alone would never fire for a genuinely tool-starved agent, so
        # this hook is the only reliable place to learn the real role_id
        # and final turn count for such a run.
        self._last_role_id = ctx.role_id
        self._last_turn[ctx.role_id] = ctx.turn
        self._saw_tool_call.setdefault(ctx.role_id, False)
        return None

    async def on_loop_end(self, result: AgentLoopResult) -> None:
        role_id = self._last_role_id
        saw_tool_call = self._saw_tool_call.pop(role_id, None)
        last_turn = self._last_turn.pop(role_id, 0)
        if saw_tool_call is None or saw_tool_call:
            return  # Either never tracked, or genuinely made a real call.
        # Real, print()-based (not logger.warning()), matching the
        # established, confirmed-necessary pattern from earlier this same
        # session -- logger output has nowhere real to go here.
        print(
            f"[TOOL_STARVATION_FLAG] role={role_id or '?'} "
            f"stopped_by={result.stopped_by} last_turn={last_turn} -- "
            f"this agent's entire run ended without a single real tool "
            f"call. This is a real, plausible risk signal, not a "
            f"confirmed root cause.",
            flush=True,
        )


__all__ = ["ToolStarvationFlag"]
