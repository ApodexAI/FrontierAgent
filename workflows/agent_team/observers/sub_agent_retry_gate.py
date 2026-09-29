"""SubAgentRetryGate — bounds coordinator-level re-assignment to a sub-agent
that keeps failing to call its own required tool (submit_report).

Real, added 2026-09-06, closing a genuine gap found during this same
session's own Planning Gate verification run: the per-run no_tool retry cap
(frontier_agent.core.loop_types.LoopConfig.no_tool_max_retries, default 2)
already exists and works correctly -- a sub-agent that replies with plain
text twice in a row genuinely stops with status="incomplete",
reason="no_tool", confirmed directly in a live log. But nothing tracked that
failure ACROSS separate assign_task calls, so the coordinator kept
re-assigning the same task to the same agent indefinitely, hitting the exact
same wall every time -- confirmed directly, live: doc2_summarizer failed
identically on 3 consecutive retries before this session stopped the run
manually (it never reached a natural end on its own).

This observer tracks, per sub-agent, how many CONSECUTIVE times its most
recent report came back as `status="incomplete" reason="no_tool"`, and
blocks a further assign_task to that same agent once a real, configurable
limit is reached -- forcing the coordinator to mark the task `blocked` and
move on, rather than repeating a call already proven not to work.
"""

from __future__ import annotations

import logging
import re

from frontier_agent.core.loop_types import (
    BaseObserver,
    ToolCallIntervention,
    ToolResult,
    TurnContext,
)
from plugins.tools._coerce import coerce_json_list

logger = logging.getLogger(__name__)

# Matches <report agent="NAME" status="incomplete" reason="no_tool"> (and
# tolerates the reverse attribute order / extra attributes some renderers
# use) -- real, direct parsing of collect_reports' own, documented wire
# format, not a guess at its shape.
_REPORT_TAG_RE = re.compile(
    r'<report\s+agent="(?P<agent>[^"]+)"[^>]*\bstatus="(?P<status>[^"]+)"'
    r'(?:[^>]*\breason="(?P<reason>[^"]+)")?[^>]*>',
)


class SubAgentRetryGate(BaseObserver):
    def __init__(self, max_consecutive_no_tool: int = 2) -> None:
        self._limit = max(1, int(max_consecutive_no_tool))
        # agent_name -> consecutive no_tool count. Any OTHER, real status
        # (complete, or incomplete for a different reason) resets this to 0
        # for that agent -- this tracks a STREAK, not a lifetime total, so an
        # agent that fails once, then genuinely succeeds, then fails again
        # later is not unfairly blocked from that single, earlier failure.
        self._streaks: dict[str, int] = {}
        self._blocked: set[str] = set()

    async def on_tool_result(
        self, ctx: TurnContext, result: ToolResult,
    ) -> ToolResult | None:
        if result.name != "collect_reports" or result.is_error:
            return None
        for match in _REPORT_TAG_RE.finditer(result.result):
            agent = match.group("agent")
            status = match.group("status")
            reason = match.group("reason") or ""
            if status == "incomplete" and reason == "no_tool":
                self._streaks[agent] = self._streaks.get(agent, 0) + 1
                if self._streaks[agent] >= self._limit:
                    self._blocked.add(agent)
                    logger.info(
                        "SubAgentRetryGate: %s hit %d consecutive no_tool "
                        "reports (task=%s) -- blocking further assign_task "
                        "to this agent",
                        agent, self._streaks[agent], ctx.task_id,
                    )
            else:
                # Real, genuine success or a different failure reason --
                # this agent is no longer on a no_tool streak.
                self._streaks.pop(agent, None)
                self._blocked.discard(agent)
        return None

    async def on_tool_call(
        self, ctx: TurnContext, tool_call: dict,
    ) -> ToolCallIntervention | None:
        if tool_call.get("name") != "assign_task" or not self._blocked:
            return None
        args = tool_call.get("args") or {}
        tasks = coerce_json_list(args.get("tasks")) or []
        hit = [
            str(t.get("agent", "")) for t in tasks
            if isinstance(t, dict) and str(t.get("agent", "")) in self._blocked
        ]
        if not hit:
            return None
        names = ", ".join(sorted(set(hit)))
        return ToolCallIntervention(skip_with_result=(
            f"Blocked: {names} failed to call its required tool "
            f"(submit_report) {self._limit} times in a row on this same "
            "task -- re-assigning it again will not help. Mark the "
            "corresponding task_board item `blocked` via update_task "
            "(with a note explaining why), and either try a DIFFERENT "
            "agent for this sub-question or proceed without it."
        ))


__all__ = ["SubAgentRetryGate"]
