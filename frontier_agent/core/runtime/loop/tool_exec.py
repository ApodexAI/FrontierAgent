"""FrontierAgent policies for AgentCore's shared tool execution engine.

AgentCore owns dispatch (parallelism, interrupts, error envelopes); this module
supplies the product policy through :class:`ToolExecutionHooks`: per-tool
timeout floors, the bash budget contextvar, usage metering, the configurable
result cap with spill, and the per-turn aggregate budget.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Iterator
from contextlib import contextmanager
from dataclasses import replace
from typing import Any

from agent_core.runtime.loop.tool_exec import (
    DefaultToolResultPostProcessor,
    ToolExecutionHooks,
    ToolLike,
    ToolResultPostProcessor,
)
from agent_core.runtime.loop.tool_exec import execute_tools as _execute_tools

from frontier_agent.core.loop_types import ToolResult

__all__ = [
    "PROTECTED_FANIN_TOOLS",
    "TOOL_EXECUTION_HOOKS",
    "TOOL_RESULT_MAX_CHARS",
    "DefaultToolResultPostProcessor",
    "ToolResultPostProcessor",
    "execute_tools",
    "max_tool_wall_time_s",
]

TOOL_RESULT_MAX_CHARS = 150_000


def _result_cap() -> int:
    """Effective global result cap — :data:`TOOL_RESULT_MAX_CHARS` unless
    ``TOOL_RESULT_MAX_CHARS`` is set in the environment.

    Read per call, not at import: a stress run lowers it to force the truncation
    path on the tools that opt out of a per-tool cap, and a misconfigured value
    must fall back to the default rather than uncapping the context.
    """
    from frontier_agent.infra.config import get_config

    try:
        configured = int(get_config().tool_result_max_chars or 0)
    except Exception:
        return TOOL_RESULT_MAX_CHARS
    return configured if configured > 0 else TOOL_RESULT_MAX_CHARS
_AGGREGATION_TOOLS = frozenset({"collect_reports", "collect_results"})

# Tools that enforce their own wall-clock deadline internally and return a
# STRUCTURED error when it fires. The outer wait must never be shorter than
# that deadline, or the loop cancels the tool mid-flight and the caller gets
# a bare "timed out after Ns" instead of the tool's own diagnosis (partial
# file cleaned up, disk-limit hit, upstream 404, …).
#
# Unlike ``_AGGREGATION_TOOLS`` the deadline is a module constant rather than
# a call argument, so it is declared here as a floor. Keep each entry in sync
# with the owning tool module.
#
# Why this matters now: the agent_team profiles moved ``tool_timeout_s``
# 1800 -> 360 (it doubles as the WallClockGuard reserve floor). That number
# was measured over bash / web_search / web_fetch only — ``download_file``
# was not in the sample and legitimately runs to 660s, so without this floor
# a large download would be cancelled at 360s.
_SELF_TIMING_TOOL_FLOORS: dict[str, int] = {
    # plugins/tools/download_file.py: _TIMEOUT = _TOTAL_TIMEOUT + 60 = 660.
    "download_file": 660,
}

# Tools that read their deadline from the loop instead of owning a constant:
# they call :func:`get_current_tool_budget` and stop at that budget, so the
# outer wait must sit a little ABOVE it. Without the headroom the two
# deadlines are equal and it is a coin flip whether the tool's own error
# ("timed out after 900 seconds" plus a recovery hint) or the loop's bare
# cancel reaches the model.
#
# ``bash`` is the case this exists for: it used to enforce a hardcoded 300s
# regardless of ``tool_timeout_s``, so on the Frontier Challenge every ORCA /
# GROMACS / LAMMPS run died at 5 minutes inside an 1800s budget and the model
# burned turns on chunk-and-restart workarounds. See issue #53.
_BUDGET_AWARE_TOOLS = frozenset({"bash"})

# Headroom granted to a budget-aware tool between its own deadline and the
# outer wait. Must exceed the sandbox's post-exit output drain
# (``_sandbox._POST_EXIT_DRAIN_S`` = 5s) plus the killed command's teardown,
# output masking, and overflow spill — 30s matches the margin
# ``run_python_code`` already leaves over its inner ``timeout -s KILL``.
_BUDGET_GRACE_S = 30

# Fan-in / orchestration tools whose results carry a sub-agent's actual
# findings. Single source of truth for the set that tiered compaction PROTECTS
# from Tier1 age-based blanking in agent_team (imported by nodes/main_agent.py).
PROTECTED_FANIN_TOOLS = _AGGREGATION_TOOLS | frozenset(
    {"assign_task", "submit_report", "create_subagent"}
)




def _effective_tool_timeout(name: str, args: dict, default_timeout: int) -> int:
    """Return the outer wait timeout for one tool call.

    Aggregation tools already accept their own ``timeout`` argument and
    use it to bound the internal wait. The outer loop timeout must not
    be shorter than that value, otherwise a profile-level 60s
    ``tool_timeout`` cancels ``collect_reports(timeout=1800)`` before
    the tool's own wait semantics can run.

    Tools in :data:`_SELF_TIMING_TOOL_FLOORS` own a fixed internal deadline
    instead of taking one as an argument; the same rule applies, with the
    floor read from the table.

    Tools in :data:`_BUDGET_AWARE_TOOLS` invert the relationship — they take
    their deadline FROM the loop via ``get_current_tool_budget`` — so the outer
    wait is the configured timeout plus :data:`_BUDGET_GRACE_S`, leaving the
    tool room to report its own timeout first. Their budget comes back from
    :func:`_tool_budget`.
    """
    floor = _SELF_TIMING_TOOL_FLOORS.get(name)
    if floor is not None:
        return max(int(default_timeout), floor)
    if name in _BUDGET_AWARE_TOOLS:
        return int(default_timeout) + _BUDGET_GRACE_S
    if name not in _AGGREGATION_TOOLS:
        return int(default_timeout)
    requested = 0
    if isinstance(args, dict):
        try:
            requested = int(float(args.get("timeout", 0) or 0))
        except (TypeError, ValueError):
            requested = 0
    if requested <= 0:
        return int(default_timeout)
    return max(int(default_timeout), requested + 5)


def _truncate_with_recovery(name: str, result: str) -> str:
    """Cut a result to :data:`TOOL_RESULT_MAX_CHARS`, spilling the middle first.

    This cap is the ONLY one that applies to the tools whose ``ToolMeta`` sets
    ``max_result_chars=0`` — web_fetch / web_search / read_file, where content
    density is judged to reward full bodies. For those, ``maybe_overflow`` is a
    documented no-op, so before this the discarded remainder was simply gone: a
    300K paper lost its second half with nothing on disk and no path to recover
    it, which is the opposite of what every other truncation in the system does
    (``maybe_overflow`` and ``spill_compacted_body`` both persist before cutting).

    Spill is best-effort by design — a backend with no agent-readable filesystem
    returns no path — so the marker degrades to the plain char count rather than
    failing the tool call.
    """
    from plugins.tools._overflow import budgeted_preview, spill_compacted_body

    try:
        ref = spill_compacted_body(name, result)
    except Exception:  # pragma: no cover - diagnostics must not fail a tool call
        ref = None
    return budgeted_preview(result, cap=_result_cap(), ref=ref or "", tool_name=name)


def _apply_aggregate_budget(results: list[ToolResult]) -> list[ToolResult]:
    """Hold one turn's tool results to the per-turn total.

    Every cap above this one is per result, so N parallel calls each landing just
    under their own cap still add up without bound — and the uncapped tools cap
    at 150K EACH, so two fetches in one turn can outweigh the whole rest of the
    context. The budget existed but had no call site; this is it.
    """
    from plugins.tools._overflow import (
        MAX_AGGREGATE_RESULT_CHARS,
        check_aggregate_budget,
    )

    if sum(len(r.result) for r in results) <= MAX_AGGREGATE_RESULT_CHARS:
        return results
    bodies = check_aggregate_budget(
        [r.result for r in results], [r.name for r in results],
    )
    return [
        r if body == r.result else replace(r, result=body)
        for r, body in zip(results, bodies, strict=True)
    ]


def _tool_budget(name: str, effective_timeout: int) -> float | None:
    """Return the deadline to publish to the tool, or ``None`` to leave it alone.

    Only :data:`_BUDGET_AWARE_TOOLS` get a budget; every other tool either has
    no internal deadline or owns a fixed one that
    :func:`_effective_tool_timeout` already floors, and handing those a number
    they would ignore only invites a future reader to wire it up backwards.

    The budget is the outer wait minus :data:`_BUDGET_GRACE_S` — i.e. exactly
    the configured ``tool_timeout`` — so the tool's own deadline fires first.
    """
    if name not in _BUDGET_AWARE_TOOLS:
        return None
    return float(max(effective_timeout - _BUDGET_GRACE_S, 1))


def max_tool_wall_time_s(tool_timeout: float) -> float:
    """Longest a single tool call can occupy the loop, for this configured
    ``tool_timeout``.

    Wall-clock reserves must budget against THIS rather than the configured
    value. A budget-aware tool's outer wait is deliberately ``tool_timeout +
    _BUDGET_GRACE_S`` so the tool reports its own timeout instead of being
    cancelled mid-flight — which means a tool started just before the research
    deadline overruns it by the grace as well, eating into the reserve that
    finalization was promised.

    Note this does NOT fold in :data:`_SELF_TIMING_TOOL_FLOORS`. ``download_file``
    can already outlast a small ``tool_timeout`` (its floor is 660s) and the
    reserves never accounted for that either; correcting it would silently move
    agent_team's research window by several minutes, so it is left as its own
    decision rather than smuggled in here.
    """
    return max(float(tool_timeout), 0.0) + _BUDGET_GRACE_S


@contextmanager
def _tool_call_scope(call: dict[str, Any], timeout: float) -> Iterator[None]:
    """Expose the tool-call id and (for budget-aware tools) its deadline.

    Dispatcher tools (assign_task / create_subagent) stamp the id on the
    sub-agent they spawn; ``bash`` reads the budget so its own timeout fires
    before the loop's outer wait.
    """
    from frontier_agent.core.execution_context import (
        reset_current_tool_budget,
        reset_current_tool_call_id,
        set_current_tool_budget,
        set_current_tool_call_id,
    )

    name = str(call.get("name", "") or "")
    tc_token = set_current_tool_call_id(str(call.get("id", "") or ""))
    budget_token = set_current_tool_budget(_tool_budget(name, int(timeout)))
    try:
        yield
    finally:
        reset_current_tool_budget(budget_token)
        reset_current_tool_call_id(tc_token)


def _record_tool_call(name: str) -> None:
    from frontier_agent.infra.usage_meter import record_tool_call

    record_tool_call(name)


def _transform_result(name: str, result: str) -> str:
    return _truncate_with_recovery(name, result) if len(result) > _result_cap() else result


def _timeout_result(name: str, _elapsed_s: float, effective_timeout: float) -> str:
    return f"Error: tool '{name}' timed out after {int(effective_timeout)}s"


TOOL_EXECUTION_HOOKS = ToolExecutionHooks(
    resolve_timeout=lambda name, args, timeout: float(_effective_tool_timeout(name, args, timeout)),
    call_scope=_tool_call_scope,
    on_call=_record_tool_call,
    transform_result=_transform_result,
    transform_batch=_apply_aggregate_budget,
    timeout_result=_timeout_result,
)


async def execute_tools(
    tool_calls: list[dict],
    tool_map: dict[str, ToolLike],
    timeout: int,
    turn: int,
    count_offset: int,
    interrupt_waiter: Callable[[dict], Awaitable[bool]] | None = None,
) -> list[ToolResult]:
    """Execute ``tool_calls`` in parallel with the product policies applied."""
    return await _execute_tools(
        tool_calls,
        tool_map,
        timeout=timeout,
        turn=turn,
        count_offset=count_offset,
        interrupt_waiter=interrupt_waiter,
        hooks=TOOL_EXECUTION_HOOKS,
    )
