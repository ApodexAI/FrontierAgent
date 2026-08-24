"""Drain completed sub-agent reports and classify session state.

The tool short-circuits when no work remains and emits heartbeats while waiting
so long-running sub-agents do not leave streaming connections idle.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
from typing import Any

from frontier_agent.components.agent_bus import (
    AgentBus,
    SessionWaitOutcome,
    SubAgentResult,
)
from frontier_agent.components.agent_bus.fan_in import (
    format_status_line,
    format_status_report_block,
    process_collected,
    session_state,
)
from frontier_agent.core.execution_context import get_current_execution_scope
from frontier_agent.core.loop_types import wall_deadline_remaining_s
from frontier_agent.core.runtime.registries import services as registry
from frontier_agent.core.tool import tool
from plugins.tools._bus_scope import resolve_bus_task_id

logger = logging.getLogger(__name__)

# Snapshot cadence while ``bus.wait_any_session`` blocks. This is a *UI*
# clock, not a transport keep-alive: the heartbeat publishes sub-agent
# progress to observers that opt in via ``on_subagent_status`` (today the
# TUI), and it is not started at all when none do. One second is what keeps
# elapsed counters and the spinner moving; the work per tick is one in-memory
# snapshot per session, so the cost is negligible.
#
# A deployment that streams over chunked HTTP and needs a real keep-alive
# should register its own observer here rather than widening this interval —
# the two concerns want different cadences and different failure handling.
_HEARTBEAT_INTERVAL_S = 1.0

# Upper bound on how long teardown waits for the heartbeat's final
# ``done=True`` publish. Bounded so a wedged observer cannot stall the tool's
# return, but awaited at all so the "clear the card" message is queued before
# a subsequent ``collect_reports`` starts drawing a new one.
_HEARTBEAT_STOP_TIMEOUT_S = 2.0

# When the remaining wall-clock budget is at or below this, treat the run as
# out of time: skip the blocking wait and tell the agent to finalize rather
# than re-poll (a re-poll would be cancelled by the hard cap anyway).
_WALL_FINALIZE_BUFFER_S = 10.0

# Consecutive same-state status-only short-circuits tolerated before the
# return escalates from guidance to a hard stop-polling directive. The
# planner is told "do NOT call collect_reports again" on the very first
# notice, but on trivial queries fanned out at full width (e.g. a bare
# greeting forced to tier 3) smaller planner models have been observed
# re-polling 5+ times anyway, flooding the protocol stream with status
# churn. Counter lives in ``scope.metadata`` so it lasts exactly as long
# as the run; reset whenever a real report (or a different state) comes
# back.
_STATUS_CHURN_KEY = "collect_reports_status_churn"
_STATUS_CHURN_LIMIT = 2


def _bump_status_churn(scope: Any, state: str) -> int:
    """Increment and return the consecutive count for ``state``."""
    md = getattr(scope, "metadata", None)
    if md is None:
        return 1
    prev_state, count = md.get(_STATUS_CHURN_KEY) or ("", 0)
    count = count + 1 if prev_state == state else 1
    md[_STATUS_CHURN_KEY] = (state, count)
    return count


def _reset_status_churn(scope: Any) -> None:
    md = getattr(scope, "metadata", None)
    if md:
        md.pop(_STATUS_CHURN_KEY, None)


def _status_notice(
    scope: Any,
    state: str,
    body: str,
    *,
    extra_status: str | None = None,
    track_churn: bool = True,
) -> str:
    """Render a status-only return: ``<report>`` block + ``[status]`` line.

    Every short-circuit return goes through here so SDK consumers
    parsing ``tool_finished.result_preview`` always find a well-formed
    ``<report>`` block (see
    :func:`workflows.agent_team._report_format.format_status_report_block`)
    — bare free-form status text used to be pasted back to end users
    verbatim with a parse-failure marker.

    With ``track_churn`` (idle states only), the same state returned
    more than :data:`_STATUS_CHURN_LIMIT` times in a row swaps the body
    for a hard stop-polling directive and tags
    ``reason="<state>_stop_polling"`` so trace consumers can spot the
    churn at a glance.
    """
    reason = state
    if track_churn:
        count = _bump_status_churn(scope, state)
        if count > _STATUS_CHURN_LIMIT:
            reason = f"{state}_stop_polling"
            body = (
                f"This is consecutive '{state}' notice #{count}. STOP "
                "calling collect_reports — repeating it cannot return "
                "anything new. Call finalize_answer NOW with your best "
                "synthesis of the reports already collected, or "
                "assign_task with a concrete follow-up question if (and "
                "only if) one is genuinely needed."
            )
    else:
        _reset_status_churn(scope)
    out = f"{format_status_report_block(reason, body)}\n\n[status] {state}"
    if extra_status:
        out += f"\n{extra_status}"
    return out


@tool
async def collect_reports(timeout: int = 1800) -> str:
    """Collect reports from any sub-agents that have finished their tasks.

    Reports also drain automatically between turns when agents finish
    *during* a turn, but agents still running at the start of a new turn
    won't appear unless you call this. Block-wait here is the right move
    whenever you need a running agent's output to proceed.

    Behaviour:
      1. Non-blocking drain: any sub-agent that already finished returns
         its ``<report>`` immediately, regardless of ``timeout``.
      2. If the drain found nothing, classify the session state:
         * no agents at all → ``"create_subagent first"``.
         * agents created but never assigned → ``[status] no_work_queued``,
           call ``assign_task`` (do NOT poll again).
         * all submitted work already collected → ``[status] all_collected``,
           synthesize or assign a follow-up (do NOT poll again).
         * at least one session running → wait up to ``timeout`` seconds
           for the FIRST completion, then drain any others that
           completed during the wait.

    Reports come back as
    ``<report agent="NAME" status="complete|incomplete|failed|paused"
    reason="...">…</report>`` — treat ``incomplete`` / ``failed`` as
    partial evidence, not a final answer.

    Args:
        timeout: Max seconds to wait for the first completion when
                 something is genuinely running. Default 1800 (30 min).
                 Sub-agents on hard BrowseComp-style questions routinely
                 take 3-10 minutes — short timeouts are a poll, not a
                 wait. Pass a smaller value only when you specifically
                 want a cheap poll.

    Returns:
        One or more ``<report>`` blocks plus a status line. Status-only
        outcomes (nothing to wait for / timeout) come back as a single
        ``<report agent="orchestrator" …>`` block carrying the
        directive, followed by the ``[status]`` line.
    """
    scope = get_current_execution_scope()
    if scope is None:
        return (
            "Error: collect_reports can only be called inside an "
            "active ReAct execution."
        )

    bus = registry.get(AgentBus)
    task_id = resolve_bus_task_id(scope)

    # Clamp the blocking wait to the run's remaining wall-clock budget so a
    # late ``collect_reports(timeout=1800)`` can't block past the per-run cap
    # and get hard-cancelled mid-call (which would skip ``force_final_answer``
    # and leave a dead ``partial`` run). The soft deadline is published by
    # ``WallClockDeadlineObserver`` at loop start; absent it (direct loop use,
    # tests) the caller's ``timeout`` is used unchanged.
    wall_budget_exhausted = False
    remaining = wall_deadline_remaining_s()
    if remaining is not None:
        wall_budget_exhausted = remaining <= _WALL_FINALIZE_BUFFER_S
        # ``max(0, …)`` turns an already-passed deadline into a final
        # non-blocking drain so the loop regains control immediately and the
        # observer's between-turn check stops it (leaving the reserve for
        # ``force_final_answer``).
        clamped = max(0, int(remaining))
        if clamped < int(timeout):
            logger.info(
                "collect_reports: clamped timeout %ds → %ds "
                "(%.0fs left to wall deadline)",
                int(timeout), clamped, remaining,
            )
            timeout = clamped

    collected: list[tuple[str, SubAgentResult]] = []
    # Seeded rather than left unbound: every path that reaches the empty-handed
    # reporting below goes through the blocking wait, but the default states
    # the honest fallback — no wait happened, so no time elapsed.
    wait_outcome = SessionWaitOutcome(None, "no_pending", 0.0)

    # 1) Drain anything already ready, non-blocking.
    while True:
        ready = await bus.wait_any_session(task_id, timeout=0.0)
        if ready is None:
            break
        collected.append(ready)

    # 2) Nothing was ready → before blocking, classify the fan-out state
    #    and short-circuit when there is genuinely nothing to wait for.
    #    Each branch returns a distinct, actionable line so the main
    #    agent prompt can map state → next action without polling.
    if not collected:
        sessions = bus.list_sessions_for_task(task_id)
        state = session_state(sessions)

        if state == "no_subagents":
            return _status_notice(
                scope, "no_subagents",
                "No sub-agents have been created yet. "
                "Call create_subagent first.",
            )
        if state == "no_work_queued":
            return _status_notice(
                scope, "no_work_queued",
                "Sub-agents exist, but none has been assigned work yet. "
                "Call assign_task to give them tasks; do NOT call "
                "collect_reports again.",
            )
        if state == "all_collected":
            return _status_notice(
                scope, "all_collected",
                "All submitted sub-agent work has already been collected. "
                "Do NOT call collect_reports again. Synthesize now, or "
                "call assign_task with a specific follow-up question.",
            )

        # state in {"running", "ready_to_collect"} → there IS outstanding
        # work. ready_to_collect should be impossible here (the drain
        # above would have caught it), but treat it the same as running
        # for safety. Wrap the blocking wait with a periodic heartbeat
        # so the SDK stdout / SSE stream doesn't go silent for the
        # entire ``timeout`` (sub-agents on hard browsecomp questions
        # routinely take 2-5+ minutes; without a heartbeat any reverse
        # proxy will close the chunked stream long before the wait
        # returns).
        hb_task = _start_heartbeat(
            scope_metadata=scope.metadata,
            bus=bus,
            task_id=task_id,
            timeout_s=int(timeout),
        )
        try:
            wait_outcome = await bus.wait_any_session_detailed(
                task_id, timeout=float(timeout),
            )
        finally:
            await _stop_heartbeat(hb_task)
        if wait_outcome.result is not None:
            collected.append(wait_outcome.result)
            # Drain any others that completed during the wait.
            while True:
                extra = await bus.wait_any_session(task_id, timeout=0.0)
                if extra is None:
                    break
                collected.append(extra)

    if not collected:
        status = format_status_line(bus, task_id)
        if wall_budget_exhausted:
            # Out of wall-clock budget — re-polling would just be cancelled.
            # Steer the agent to finalize with the evidence already gathered.
            return _status_notice(
                scope, "wall_deadline",
                "The time budget is nearly exhausted — do NOT call "
                "collect_reports again. Finalize your answer now using the "
                "evidence already collected.",
                extra_status=status,
                track_churn=False,
            )
        elapsed = wait_outcome.elapsed_s
        if wait_outcome.reason == "no_pending":
            return _status_notice(
                scope, "no_waitable_tasks",
                "No runnable sub-agent task remained when the wait began. "
                "The session state changed or a terminal task was reconciled; "
                "do not count the requested timeout as elapsed time.",
                extra_status=status,
                track_churn=False,
            )
        if wait_outcome.reason == "unpublished":
            return _status_notice(
                scope, "no_report_published",
                f"A sub-agent task ended during an actual {elapsed:.1f}s wait "
                "but published no report. Its session has been reconciled, so "
                "call collect_reports once more to pick up the recorded "
                "failure, or reassign the work to another sub-agent.",
                extra_status=status,
                track_churn=False,
            )
        return _status_notice(
            scope, "wait_timeout",
            f"No completion arrived during an actual {elapsed:.1f}s wait "
            f"(requested maximum {timeout}s). If sub-agents are unresponsive, "
            "consider calling stop_subagent to cancel them and switch to "
            "fallback data or a simplified delivery, rather than waiting indefinitely.",
            extra_status=status,
            track_churn=False,
        )

    _reset_status_churn(scope)
    batch = process_collected(bus, task_id, collected)
    status = format_status_line(
        bus, task_id,
        paused_names=batch.paused_names,
        incomplete_count=batch.incomplete_count,
    )
    if batch.evidence_count or batch.assertion_count:
        status += (
            f" harvested={batch.evidence_count} evidence "
            f"+ {batch.assertion_count} assertions"
        )
    return "\n\n".join([*batch.blocks, status])


def _start_heartbeat(
    *,
    scope_metadata: dict[str, Any] | None,
    bus: AgentBus,
    task_id: str,
    timeout_s: int,
) -> asyncio.Task[None] | None:
    """Publish live sub-agent snapshots while the fan-in wait is blocked."""
    observers = list((scope_metadata or {}).get("sdk_extra_observers") or [])
    callbacks = [
        callback
        for observer in observers
        if (callback := getattr(observer, "on_subagent_status", None)) is not None
    ]
    if not callbacks:
        return None

    async def _publish(*, done: bool = False) -> None:
        # Everything but cancellation is swallowed: this is decoration around
        # the caller's real work, and a snapshot or renderer fault must not
        # kill the heartbeat (which would freeze the UI for the rest of the
        # wait) nor surface as an unretrieved task exception.
        try:
            snapshots = bus.describe_sessions_for_task(task_id)
        except Exception:
            logger.debug("sub-agent snapshot failed", exc_info=True)
            return
        for callback in callbacks:
            try:
                result = callback(snapshots, done=done, timeout_s=timeout_s)
                if inspect.isawaitable(result):
                    await result
            except Exception:
                logger.debug("sub-agent heartbeat callback failed", exc_info=True)

    async def _run() -> None:
        try:
            while True:
                await _publish()
                await asyncio.sleep(_HEARTBEAT_INTERVAL_S)
        finally:
            # Runs under cancellation. The callbacks are synchronous in
            # practice, so this completes without a real suspension; an async
            # observer that does suspend would see the pending cancellation
            # here, which ``_publish`` reports and the caller's bounded wait
            # absorbs.
            await _publish(done=True)

    return asyncio.create_task(_run(), name=f"subagent-heartbeat:{task_id}")


async def _stop_heartbeat(hb_task: asyncio.Task[None] | None) -> None:
    """Cancel the progress heartbeat and let its final publish land.

    Teardown is awaited (with a bound) rather than fire-and-forget so the
    ``done=True`` "clear the card" callback is delivered before this tool
    returns. Otherwise a second ``collect_reports`` can mount a fresh status
    card and have the previous call's late teardown remove it.
    """
    if hb_task is None:
        return
    hb_task.cancel()
    await asyncio.wait({hb_task}, timeout=_HEARTBEAT_STOP_TIMEOUT_S)


__all__ = ["collect_reports"]
