"""StoppedByDistributionTracker — logging-only, real, module-level counter
of which real stopped_by value each sub-agent run ends with, aggregated
across every real run within this process.

Real, added 2026-09-08. This is the real, corrected, grounded
implementation of a much larger "FallbackMechanismObserver" design
proposed this session. That design was built almost entirely on
stop-reason values that DO NOT EXIST anywhere in this codebase --
confirmed directly, by reading frontier_agent/components/agent_bus/
fan_in.py's real INCOMPLETE_STOP_REASONS allowlist, which only contains:
max_turns, max_attempts, llm_error, no_tool, budget_exhausted,
wall_deadline, context_limit_reached, cross_turn_repetition,
repeated_tool_calls, cycle_detected, argument_churn_detected,
response_truncated, exception -- plus the real, non-incomplete terminal
values "completed", "final_answer", "submit_report". Values like
"tool_contract_violation", "dominance_detected", "oscillation_detected",
"tool_starvation", "clean_context_llm", and "salvage_mode" were invented,
not real. That design also conflated hint-only observers
(MultiToolOscillationFlag, ToolCallDominanceFlag print warnings but never
set stopped_by) with actual stop-loop mechanisms.

This corrected version does one real, modest thing: tracks the real,
running distribution of stopped_by values across every sub-agent run in
this process, following the exact same pattern already confirmed working
for RescueModeDistributionTracker. This is a real, genuine complement to
that tracker -- stopped_by answers "why did the loop end", rescue_mode
answers "how was the final answer recovered", and the two are related but
distinct (a run can have any real stopped_by and only SOME of those
values trigger force_final_answer at all).
"""

from __future__ import annotations

from collections import Counter

from frontier_agent.core.loop_types import (
    AgentLoopResult,
    BaseObserver,
    TurnContext,
)

# Real, module-level state: persists across every real sub-agent run in
# this process, matching the same real design as
# rescue_mode_distribution_tracker.py.
_counts: Counter[str] = Counter()


def record_and_report(stopped_by: str, role_id: str = "") -> None:
    """Real, direct call site: record one real stopped_by occurrence,
    then print the current, running distribution across every real run
    seen so far this process."""
    key = stopped_by or "(empty)"
    _counts[key] += 1
    total = sum(_counts.values())
    breakdown = "; ".join(
        f"{value}={count} ({count / total:.0%})"
        for value, count in _counts.most_common()
    )
    # Real, print()-based (not logger.warning()), matching the
    # established, confirmed-necessary pattern from earlier this same
    # session -- logger output has nowhere real to go here.
    print(
        f"[STOPPED_BY_DISTRIBUTION_TRACKER] role={role_id or '?'} "
        f"latest={key} -- running distribution across {total} real "
        f"sub-agent runs this process: {breakdown}",
        flush=True,
    )


def reset() -> None:
    """Real, direct reset -- exposed for tests; not called in production."""
    _counts.clear()


class StoppedByDistributionObserver(BaseObserver):
    """Real, logging-only wrapper: hooks ``record_and_report`` above into
    the real observer pipeline via ``on_loop_end``.

    Real, updated 2026-09-08: now resolves the real, SPECIFIC sub-agent
    name (e.g. "doc1_summarizer"), not just the generic role_id (e.g.
    "agent_team_sub") -- using TurnContext's new real session_id field
    (added this same session, directly alongside this fix), extracted
    the same way SUBAGENT_TOOL_CALL_DEBUG's print statement already does:
    the last "."-delimited segment of session_id, which is formatted as
    f"{task_id}.agent_team.{agent_name}" by
    workflows/agent_team/identity.py's llm_session_id().
    """

    def __init__(self) -> None:
        self._last_agent_name: str = ""

    def _resolve_agent_name(self, ctx: TurnContext) -> str:
        session_id = getattr(ctx, "session_id", "") or ""
        return session_id.rsplit(".", 1)[-1] if session_id else ctx.role_id

    async def on_tool_call(self, ctx: TurnContext, tool_call: dict):
        self._last_agent_name = self._resolve_agent_name(ctx)
        return None

    async def on_turn_end(self, ctx: TurnContext):
        self._last_agent_name = self._resolve_agent_name(ctx)
        return None

    async def on_loop_end(self, result: AgentLoopResult) -> None:
        record_and_report(result.stopped_by, role_id=self._last_agent_name)


__all__ = ["StoppedByDistributionObserver", "record_and_report", "reset"]
