"""Shared report-formatting helpers for sub-agent fan-in paths."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from frontier_agent.components.agent_bus.models import SubAgentResult

if TYPE_CHECKING:
    from frontier_agent.components.agent_bus.bus import AgentBus


# Stop reasons that mean "agent finished cleanly via a terminal tool".
# Anything outside this set + INCOMPLETE_STOP_REASONS + ``"paused"``
# falls through to a generic ``incomplete`` label with the raw
# stopped_by value (sanitized) as the ``reason`` attribute, which keeps
# unknown observer-driven stops from being silently mislabelled as
# complete.
COMPLETE_STOP_REASONS: frozenset[str] = frozenset({
    "",
    "completed",
    "final_answer",
    "submit_report",
})

# Stop reasons that mean "agent was truncated mid-task" — the report
# may be partial. ``force_final_answer()`` may rescue a best-effort
# plain-text conclusion, but the structural status is still
# incomplete and the main agent should treat it as such.
#
# This set is the curated label table consumed by
# :func:`classify_completion` (status="incomplete" + a specific NOTE), AND
# the trigger list for ``force_final_answer()``: both rescue gates in
# ``subagent_runtime`` read ``stopped_by not in INCOMPLETE_STOP_REASONS``
# and return early, so it is an allowlist. A new observer-driven stop
# reason that is not added here still classifies as ``incomplete`` (via the
# catch-all at the bottom of :func:`classify_completion`) but silently
# loses its forced-final answer, which is the whole point of stopping a
# looping agent early. Add the reason here when you add the stop.
INCOMPLETE_STOP_REASONS: frozenset[str] = frozenset({
    "max_turns",
    "max_attempts",
    "llm_error",
    "no_tool",
    "budget_exhausted",
    # Ran out of wall clock rather than tokens — emitted by WallClockGuard
    # (a sub-agent's SpawnGuard slot), WallClockDeadlineObserver (the whole
    # run's deadline), and agent_loop's mid-turn wall refusal.
    "wall_deadline",
    "context_limit_reached",
    "cross_turn_repetition",
    "repeated_tool_calls",
    # The output cap cut every continuation off mid-sentence. Distinct from
    # ``no_tool``: the agent never chose to stop, so its report is unfinished
    # rather than merely answer-less.
    "response_truncated",
    "exception",
})

_INCOMPLETE_NOTES: dict[str, str] = {
    "max_turns": "agent reached the max_turns budget; report is partial",
    "max_attempts": (
        "agent exceeded the per-turn LLM retry budget; report is partial"
    ),
    "llm_error": "sub-agent's LLM call failed; report is partial",
    "no_tool": "agent stopped without producing a final answer",
    "response_truncated": (
        "agent's replies kept hitting the output token limit; report is partial"
    ),
    "budget_exhausted": "agent exhausted its token budget; report is partial",
    "wall_deadline": (
        "agent ran out of wall-clock time; report is partial"
    ),
    "context_limit_reached": (
        "agent ran out of model context window mid-task; report is partial"
    ),
    "cross_turn_repetition": (
        "agent stopped after repeating itself across turns; "
        "report is a best-effort partial"
    ),
    "repeated_tool_calls": (
        "agent stopped after re-issuing the same tool call with identical "
        "arguments; report is a best-effort partial"
    ),
    "exception": (
        "agent terminated with an unhandled exception; report is partial"
    ),
}


CompletionStatus = Literal["complete", "incomplete", "failed", "paused"]
SessionState = Literal[
    "no_subagents",
    "ready_to_collect",
    "running",
    "no_work_queued",
    "all_collected",
]


# Attribute values must be stable short tokens: XML-attribute-safe,
# log-grep-safe, prompt-template-safe. Raw error strings (with quotes,
# newlines, angle brackets, …) belong in the NOTE / body, never in the
# ``reason`` attribute.
_ATTR_TOKEN_RE = re.compile(r"[^a-z0-9_.-]+")


@dataclass(frozen=True)
class CompletionPolicy:
    """Workflow-specific stop-reason labels for the shared fan-in mechanics."""

    complete_reasons: frozenset[str]
    incomplete_notes: Mapping[str, str]
    include_error_class: bool = True


DEFAULT_COMPLETION_POLICY = CompletionPolicy(
    complete_reasons=COMPLETE_STOP_REASONS,
    incomplete_notes=_INCOMPLETE_NOTES,
)


def _safe_reason(value: str) -> str:
    """Sanitize an arbitrary string into an XML-attribute-safe token."""
    token = (value or "unknown").strip().lower()
    token = _ATTR_TOKEN_RE.sub("_", token).strip("_")
    return token[:80] or "unknown"


@dataclass(frozen=True)
class CompletionInfo:
    """Translation of sub-agent stop reason into a UI-facing label.

    ``reason`` is ``""`` for clean completion; otherwise a sanitized
    short tag (``"max_turns"``, ``"timeout"``, …). ``note`` is rendered
    as ``[NOTE: <note>.]`` at the top of the report body when non-empty.
    """

    status: CompletionStatus
    reason: str
    note: str


def classify_completion(
    result: SubAgentResult,
    *,
    policy: CompletionPolicy = DEFAULT_COMPLETION_POLICY,
) -> CompletionInfo:
    """Inspect ``result`` and return how the report should be labelled.

    Decision tree:

    1. ``success=False``                                 → ``failed``
    2. ``stopped_by="paused"``                           → ``paused``
    3. ``stopped_by`` in :data:`INCOMPLETE_STOP_REASONS` → ``incomplete``
    4. ``stopped_by`` in :data:`COMPLETE_STOP_REASONS`   → ``complete``
    5. **Anything else** → ``incomplete`` with the sanitized stop
       reason. Catch-all for observer-driven stops we don't have a
       curated note for; silently labelling them ``complete`` would be
       exactly the F2 failure mode this helper exists to prevent.
    """
    md = result.metadata or {}
    stopped_by = (md.get("stopped_by") or "").strip()

    if not result.success:
        err = (result.error or "unknown error").strip()
        cls = (result.error_class or "").strip() if policy.include_error_class else ""
        display = f"{cls}: {err}" if cls else err
        return CompletionInfo(
            status="failed",
            reason=_safe_reason(err),
            note=f"agent failed mid-task ({display}); output is partial",
        )

    if stopped_by == "paused":
        return CompletionInfo(
            status="paused",
            reason="paused",
            note="agent paused (resumable from checkpoint)",
        )

    if stopped_by in policy.incomplete_notes:
        return CompletionInfo(
            status="incomplete",
            reason=stopped_by,
            note=policy.incomplete_notes[stopped_by],
        )

    if stopped_by in policy.complete_reasons:
        return CompletionInfo(status="complete", reason="", note="")

    return CompletionInfo(
        status="incomplete",
        reason=_safe_reason(stopped_by),
        note=(
            f"agent stopped by observer reason `{stopped_by}`; "
            "report may be partial"
        ),
    )


def format_report_block(
    name: str,
    result: SubAgentResult,
    info: CompletionInfo | None = None,
    *,
    policy: CompletionPolicy = DEFAULT_COMPLETION_POLICY,
) -> str:
    """Render one ``<report agent="..." status="..." reason="...">`` block.

    Pass a pre-computed ``info`` to skip a redundant
    :func:`classify_completion` call when the caller already inspected
    the status (the fan-in path needs both the rendered block and the
    status bucket).
    """
    if info is None:
        info = classify_completion(result, policy=policy)
    body = (result.final_content or "(empty report)").strip()
    if info.note:
        body = f"[NOTE: {info.note}.]\n{body}"
    if info.reason:
        return (
            f'<report agent="{name}" status="{info.status}" '
            f'reason="{info.reason}">\n{body}\n</report>'
        )
    return f'<report agent="{name}" status="{info.status}">\n{body}\n</report>'


# Synthetic agent name for orchestrator-level status notices. The
# status short-circuits in ``collect_reports`` (all_collected /
# no_work_queued / …) used to return bare ``[status] …`` text; SDK
# consumers that parse ``tool_finished.result_preview`` for
# ``<report>`` blocks choked on those payloads and pasted them back to
# end users verbatim with a parse-failure marker. Wrapping the notice
# in the same envelope keeps every collect_reports return parseable by
# a single code path.
ORCHESTRATOR_AGENT_NAME = "orchestrator"


def format_status_report_block(reason: str, body: str) -> str:
    """Render an orchestrator status notice as a ``<report>`` block.

    Same envelope as :func:`format_report_block` so both consumers —
    the main agent reading the tool result and SDK callers parsing
    ``tool_finished.result_preview`` — handle status-only returns with
    the report-block code path instead of free-form text.
    ``status="complete"`` keeps the attribute within the documented
    vocabulary (``complete|incomplete|failed|paused``); the actual
    state token travels in ``reason``.
    """
    return (
        f'<report agent="{ORCHESTRATOR_AGENT_NAME}" status="complete" '
        f'reason="{_safe_reason(reason)}">\n{body.strip()}\n</report>'
    )


@dataclass
class FanInBatch:
    """Aggregated outcome of draining a batch of sub-agent results.

    ``blocks`` are rendered ``<report>`` strings ready to join with
    ``"\\n\\n"``. The other fields feed
    :func:`format_status_line` (paused bucket, partial-output count) and
    callers that want the cumulative evidence/assertion totals harvested
    from this batch.
    """

    blocks: list[str] = field(default_factory=list)
    paused_names: set[str] = field(default_factory=set)
    incomplete_count: int = 0
    evidence_count: int = 0
    assertion_count: int = 0


def process_collected(
    bus: AgentBus,
    task_id: str,
    collected: list[tuple[str, SubAgentResult]],
    *,
    policy: CompletionPolicy = DEFAULT_COMPLETION_POLICY,
) -> FanInBatch:
    """Single-pass fan-in: classify, render, harvest evidence/assertions.

    ``classify_completion`` runs exactly once per result; the same
    :class:`CompletionInfo` is then threaded into
    :func:`format_report_block` to avoid a redundant second pass.

    Evidence / assertions are harvested EVEN on failure — a sub-agent
    that ran 20 web_searches then crashed in the result adapter still
    contributed real evidence we don't want to silently drop.
    """
    batch = FanInBatch()
    for session_id, result in collected:
        name = session_id.split("::", 1)[-1]
        info = classify_completion(result, policy=policy)
        if info.status == "paused":
            batch.paused_names.add(name)
        if info.status in ("incomplete", "failed"):
            batch.incomplete_count += 1
        batch.blocks.append(format_report_block(name, result, info))

        ev = list(result.metadata.get("evidence_cards", []))
        asserts = list(result.metadata.get("assertions", []))
        if ev or asserts:
            bus.accumulate_task_metadata(
                task_id,
                evidence_cards=ev,
                assertions=asserts,
            )
            batch.evidence_count += len(ev)
            batch.assertion_count += len(asserts)

    return batch


def session_state(sessions: list[Any]) -> SessionState:
    """Classify the swarm's overall session state for ``collect_reports``.

    Used to give the main agent an actionable next-step hint instead of
    a generic "wait more" line.
    """
    if not sessions:
        return "no_subagents"
    if any(getattr(s, "pending_results", None) for s in sessions):
        return "ready_to_collect"
    # A queued task on any session counts as "running": the next
    # ``wait_any_session`` call will eventually surface its report,
    # so the main agent should keep waiting rather than concluding
    # all_collected.
    if any(
        getattr(s, "current_job_id", None) is not None
        or getattr(s, "pending_tasks", None)
        for s in sessions
    ):
        return "running"
    all_unassigned = all(
        getattr(s, "total_task_count", 0) == 0
        and not getattr(s, "last_report", "")
        for s in sessions
    )
    if all_unassigned:
        return "no_work_queued"
    return "all_collected"


def format_status_line(
    bus: AgentBus,
    task_id: str,
    *,
    paused_names: set[str] | None = None,
    incomplete_count: int = 0,
) -> str:
    """One-line summary of sub-agent state for the main agent.

    Deliberately does NOT name idle sessions — idle here means
    "already fanned in" or "freshly created, awaiting assignment",
    neither of which is actionable. Showing them tends to mislead the
    main agent into re-issuing tasks or polling unnecessarily.

    When no session is running, ready_to_collect, or paused, the line
    collapses to ``no_work_queued`` (next step: ``assign_task``) or
    ``all_collected`` (next step: synthesize / follow-up).
    """
    paused_names = paused_names or set()
    sessions = bus.list_sessions_for_task(task_id)
    if not sessions:
        return "[status] no sub-agents"

    running: list[str] = []
    ready: list[str] = []
    paused: list[str] = []
    total_task_count = 0
    has_last_report = False

    for s in sessions:
        total_task_count += getattr(s, "total_task_count", 0)
        if getattr(s, "last_report", ""):
            has_last_report = True
        if s.name in paused_names:
            paused.append(s.name)
        elif (
            getattr(s, "current_job_id", None) is not None
            or getattr(s, "pending_tasks", None)
        ):
            queued = len(getattr(s, "pending_tasks", []) or [])
            label = f"{s.name}+{queued}q" if queued else s.name
            running.append(label)
        elif getattr(s, "pending_results", None):
            ready.append(s.name)

    bits: list[str] = []
    if running:
        bits.append(f"running={running}")
    if ready:
        bits.append(f"ready_to_collect={ready}")
    if paused:
        bits.append(f"paused={paused}")

    if not bits:
        if total_task_count == 0 and not has_last_report:
            bits.append("no_work_queued")
        else:
            bits.append("all_collected")

    if incomplete_count:
        bits.append(f"incomplete_this_batch={incomplete_count}")

    return "[status] " + " ".join(bits)


__all__ = [
    "COMPLETE_STOP_REASONS",
    "DEFAULT_COMPLETION_POLICY",
    "INCOMPLETE_STOP_REASONS",
    "ORCHESTRATOR_AGENT_NAME",
    "CompletionInfo",
    "CompletionPolicy",
    "CompletionStatus",
    "FanInBatch",
    "SessionState",
    "classify_completion",
    "format_report_block",
    "format_status_line",
    "format_status_report_block",
    "process_collected",
    "session_state",
]
