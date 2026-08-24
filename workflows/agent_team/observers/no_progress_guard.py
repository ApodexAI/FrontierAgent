"""NoProgressGuard — break the main coordinator's create/assign spin."""

from __future__ import annotations

import logging
import re

from frontier_agent.components.agent_bus.fan_in import ORCHESTRATOR_AGENT_NAME
from frontier_agent.core.loop_types import (
    BaseObserver,
    Intervention,
    LoopConfig,
    ToolResult,
    TurnContext,
)
from plugins.tools._coerce import coerce_json_list

logger = logging.getLogger(__name__)

# Extracts the ``agent="…"`` name from every ``<report agent="…">`` block
# in a collect_reports result. A name other than the orchestrator means a
# sub-agent actually delivered a report (real progress); an orchestrator-only
# result is a status-only notice (no_work_queued / all_collected).
_REPORT_AGENT_RE = re.compile(r'<report\s+agent="([^"]*)"')

# A task prompt shorter than this (after stripping trailing punctuation) is
# too terse to be a real research instruction — one of the trivial wind-down
# markers a spin assigns to throwaway agents ("End.", "OK", "Done", …).
_MIN_MEANINGFUL_PROMPT_LEN = 8

# Session-churn tools. A turn calling ONLY these does no first-hand work —
# it just manages the sub-agent roster. Board tools (add_task / update_task),
# data-gathering tools, and finalize_answer are deliberately excluded: they
# represent progress and reset the streak.
_MGMT_TOOLS = frozenset({
    "create_subagent",
    "assign_task",
    "collect_reports",
    "stop_subagent",
})

# Trivial task prompts a wind-down spin assigns to throwaway agents. Used
# only to ACCELERATE the hard stop (strong signal the agent is not doing
# real work), never as the sole trigger.
_DEGENERATE_PROMPTS = frozenset({
    "end", "done", "complete", "completed", "final", "finalize", "finish",
    "finished", "x", "ok", "okay", "stop", "confirm", "nothing", "na", "n/a",
})


class NoProgressGuard(BaseObserver):
    critical: bool = True

    def __init__(
        self,
        *,
        soft_streak: int = 6,
        hard_streak: int = 12,
        min_repeated_creates: int = 3,
        cooldown_turns: int = 3,
    ) -> None:
        self._soft_streak = max(1, int(soft_streak))
        self._hard_streak = max(self._soft_streak + 1, int(hard_streak))
        self._min_repeated_creates = max(2, int(min_repeated_creates))
        self._cooldown_turns = max(1, int(cooldown_turns))
        self._reset()
        self._last_soft_turn = -(10**9)
        self._report_agents_this_turn: set[str] = set()
        # Agent names whose most recently observed assignment was a trivial
        # wind-down prompt. A report from one of these agents is completion
        # of the spin itself, not evidence that substantive research happened.
        self._degenerate_assignment_agents: set[str] = set()

    def _reset(self) -> None:
        self._mgmt_streak = 0
        self._creates_in_streak = 0
        self._degenerate_in_streak = 0

    async def on_loop_start(self, config: LoopConfig) -> None:
        self._reset()
        self._last_soft_turn = -(10**9)
        self._report_agents_this_turn.clear()
        self._degenerate_assignment_agents.clear()

    async def on_tool_result(
        self, ctx: TurnContext, result: ToolResult,
    ) -> ToolResult | None:
        """Flag genuine progress when collect_reports drains a real sub-agent report.

        Fires before ``on_turn_end`` in the same turn, so a wave that both
        spawns and collects real research counts as progress there.
        """
        if (
            result is not None
            and not result.is_error
            and str(result.name) == "collect_reports"
        ):
            self._report_agents_this_turn.update(
                self._subagent_report_agents(result.result)
            )
        return None

    async def on_turn_end(self, ctx: TurnContext) -> Intervention | None:
        # Remember which agents were assigned substantive vs. trivial work
        # before classifying reports. This also handles a turn containing both
        # assign_task and collect_reports calls.
        self._record_assignment_quality(ctx.tool_calls)

        # A wave that returned fresh *substantive* sub-agent research is real
        # progress. Reports produced by "End." / "Done." assignments are part
        # of the wind-down spin and must not disable the guard.
        report_agents = self._report_agents_this_turn
        self._report_agents_this_turn = set()
        fresh_report = any(
            name not in self._degenerate_assignment_agents
            for name in report_agents
        )
        self._degenerate_assignment_agents.difference_update(report_agents)
        if fresh_report:
            self._reset()
            return None

        names = {str(tc.get("name", "")) for tc in (ctx.tool_calls or [])}
        # Empty turn, or any non-management tool → real work / a different
        # mode. Reset and let the natural finalize path run.
        if not names or not names.issubset(_MGMT_TOOLS):
            self._reset()
            return None

        self._mgmt_streak += 1
        if "create_subagent" in names:
            self._creates_in_streak += 1
        if "assign_task" in names and self._assign_is_degenerate(ctx.tool_calls):
            self._degenerate_in_streak += 1

        # Not the pathology yet: without REPEATED new-agent creation this is
        # just normal fan-out / waiting on running sub-agents.
        if self._creates_in_streak < self._min_repeated_creates:
            return None

        degenerate = self._degenerate_in_streak >= 2

        # HARD: guarantee termination. force_final_answer (post-loop) turns
        # the accumulated reports into a plain-text answer.
        if self._mgmt_streak >= self._hard_streak or (
            degenerate and self._mgmt_streak >= self._soft_streak + 2
        ):
            logger.warning(
                "NoProgressGuard: forcing stop at turn %d "
                "(mgmt_streak=%d, creates=%d, degenerate=%d)",
                ctx.turn, self._mgmt_streak, self._creates_in_streak,
                self._degenerate_in_streak,
            )
            return Intervention(stop_reason="thrash_no_progress")

        # SOFT: nudge, cooldown-limited.
        if (
            self._mgmt_streak >= self._soft_streak
            and ctx.turn - self._last_soft_turn >= self._cooldown_turns
        ):
            self._last_soft_turn = ctx.turn
            logger.info(
                "NoProgressGuard: soft nudge at turn %d "
                "(mgmt_streak=%d, creates=%d)",
                ctx.turn, self._mgmt_streak, self._creates_in_streak,
            )
            return Intervention(inject_messages=[
                "You have repeatedly created and assigned sub-agents without "
                "gathering any new information — your sub-agents have already "
                "returned complete reports covering the question. STOP "
                "delegating: do not call create_subagent or assign_task "
                "again. Synthesize everything you already have and deliver "
                "your COMPLETE final answer as plain text now (no tool call)."
            ])
        return None

    def _assign_is_degenerate(self, tool_calls: list[dict]) -> bool:
        """True when an assign_task call's prompts are all trivial wind-down text."""
        try:
            for tc in tool_calls or []:
                if str(tc.get("name", "")) != "assign_task":
                    continue
                args = tc.get("args") or {}
                tasks = coerce_json_list(args.get("tasks") or []) or []
                prompts = [
                    str(t.get("prompt") or t.get("task") or "").strip()
                    for t in tasks
                    if isinstance(t, dict)
                ]
                if prompts and all(self._is_degenerate(p) for p in prompts):
                    return True
        except (AttributeError, TypeError, ValueError):
            return False
        return False

    def _record_assignment_quality(self, tool_calls: list[dict]) -> None:
        """Track whether each agent's latest assignment is substantive."""
        try:
            for tc in tool_calls or []:
                if str(tc.get("name", "")) != "assign_task":
                    continue
                args = tc.get("args") or {}
                tasks = coerce_json_list(args.get("tasks") or []) or []
                for task in tasks:
                    if not isinstance(task, dict):
                        continue
                    agent = str(task.get("agent") or "").strip()
                    if not agent:
                        continue
                    prompt = str(
                        task.get("prompt") or task.get("task") or ""
                    ).strip()
                    if self._is_degenerate(prompt):
                        self._degenerate_assignment_agents.add(agent)
                    else:
                        self._degenerate_assignment_agents.discard(agent)
        except (AttributeError, TypeError, ValueError):
            return

    @staticmethod
    def _is_degenerate(prompt: str) -> bool:
        s = prompt.strip().lower().strip(".!?。！ ")
        return len(s) < _MIN_MEANINGFUL_PROMPT_LEN or s in _DEGENERATE_PROMPTS

    @staticmethod
    def _subagent_report_agents(result: str) -> set[str]:
        """Return agent names for real reports in a collect_reports result.

        Status-only notices (no_work_queued / all_collected) are wrapped in a
        ``<report agent="orchestrator" …>`` envelope and are excluded.
        """
        if not result:
            return set()
        try:
            return {
                name
                for name in _REPORT_AGENT_RE.findall(result)
                if name != ORCHESTRATOR_AGENT_NAME
            }
        except (AttributeError, TypeError):
            return set()


__all__ = ["NoProgressGuard"]
