"""Data models for AgentBus.

Keeping these models separate lets ``agent_bus.py`` focus on lifecycle logic
while preserving the same import surface for callers via re-export.
"""

from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from frontier_agent.core.messages import Message
from frontier_agent.core.runtime.loop.message_trimmer import MessageTrimmer, TaskBoundary
from frontier_agent.core.tool import Tool


class DepthLimitExceeded(RuntimeError):
    """Raised when a sub-agent dispatch would exceed the configured depth."""


@dataclass(slots=True)
class SubTask:
    """Single sub-agent task."""

    question: str
    role_id: str = "researcher"
    system_prompt: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SubAgentResult:
    """Normalized result returned from a sub-agent execution.

    ``metadata`` is the workflow-agnostic artifact bag — runtime specs
    and result adapters stash domain-specific payloads (research
    evidence cards, assertions, tool trails, …) there. The kernel data
    model itself carries no domain vocabulary; the legacy dedicated
    ``evidence_cards`` / ``assertions`` fields were removed as part of
    the thin-kernel migration.
    """

    question: str
    role_id: str
    final_content: str
    success: bool
    error: str | None = None
    error_class: str | None = None
    job_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class JobEntry:
    """Tracks a single async sub-agent job.

    ``task`` is ``None`` while the job is still queued — sessions execute
    tasks strictly serially, so submitted tasks can sit behind a running
    one without an asyncio.Task spawned yet. The ``"queued"`` status
    marks that state; transitions to ``"submitted"`` (then ``"running"``)
    when the bus dequeues and dispatches it.
    """

    job_id: str
    parent_task_id: str
    item: SubTask
    task: asyncio.Task[SubAgentResult] | None = None
    status: Literal[
        "queued", "submitted", "running", "completed", "failed", "aborted"
    ] = "submitted"
    result: SubAgentResult | None = None
    submitted_at: float = 0.0
    completed_at: float | None = None


@dataclass(slots=True)
class PendingSessionTask:
    """A session task waiting in the queue behind the running one.

    Holds every input ``submit_task_to_session`` was called with so the
    bus can dispatch the queued task identically once the previous one
    finishes — same boundary semantics, same runtime spec, same guard
    bookkeeping. Created at submit time, drained FIFO when
    ``session.current_job_id`` clears.
    """

    job_id: str
    task_prompt: str
    max_turns: int | None = None
    observers: list[Any] | None = None
    estimated_tokens: int = 0
    runtime_spec: SubAgentRuntimeSpec | None = None
    # ``spawn_context`` carries the parent's run_id, the verbatim delegation
    # prompt, allowed tools snapshot, depth, and budget. Bus stamps it onto
    # the sub-agent's trace observer at dispatch time so a downstream
    # training-data pipeline can reconstruct the delegation graph
    # (parent → child + the exact instruction).
    spawn_context: dict[str, Any] | None = None
    task_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CollectResult:
    """Aggregated result from collect()."""

    completed: list[SubAgentResult] = field(default_factory=list)
    pending: list[str] = field(default_factory=list)
    failed: list[SubAgentResult] = field(default_factory=list)


@dataclass(slots=True)
class SessionWaitOutcome:
    """Detailed outcome of waiting for one reusable sub-agent session.

    ``wait_any_session`` keeps its historical ``result | None`` API, while
    orchestration tools use this richer form so ``None`` is not automatically
    (and incorrectly) described as a full timeout.

    ``reason`` distinguishes the three empty-handed outcomes, which need
    different advice for the coordinator:

    ``no_pending``
        Nothing was waitable when the wait began — no time passed, so the
        requested timeout must not be reported as elapsed.
    ``timeout``
        A live task was waited on and did not finish within the budget.
    ``unpublished``
        A task *did* finish during the wait but produced no collectable
        result. Real time elapsed here, so this is not a ``no_pending``.
    """

    result: tuple[str, SubAgentResult] | None
    reason: Literal["ready", "timeout", "no_pending", "unpublished"]
    elapsed_s: float


@dataclass(slots=True)
class SubAgentRuntimeSpec:
    """Optional runtime injection for generic sub-agent execution."""

    config_builder: Callable[[str, SubTask, int], Any] | None = None
    # ``task_index`` is the 1-based ordinal of this task within the
    # session (== ``session.total_task_count`` at submit time). Lets
    # observers tell first-run from reuse without consulting the bus.
    observers_builder: Callable[
        [str, SubTask, int], list[Any]
    ] | None = None
    # Sync or async — bus awaits when the return is a coroutine.
    result_adapter: Callable[
        [Any, str, SubTask],
        SubAgentResult | Awaitable[SubAgentResult],
    ] | None = None
    model_profile: Any = None
    history_policy: Any = None
    # Invoked AFTER ``run_agent_loop`` returns and BEFORE the bus
    # absorbs ``raw_result.messages`` back into ``session.messages``.
    # Sync or async. Mutates ``raw_result`` in place (or returns a
    # replacement). Anything appended to ``result.messages`` flows
    # naturally into ``session.messages`` and the closing boundary;
    # rewriting ``result.final_content`` propagates to
    # ``session.last_report``. Use this for "no-tool recovery"
    # logic (e.g. swarm's ``force_final_answer``) — putting it here
    # rather than in ``result_adapter`` ensures session bookkeeping
    # sees the rescue.
    force_finalizer: Callable[
        [Any, SubTask],
        Any | Awaitable[Any],
    ] | None = None
    # Sync context manager ``setup(job_id, item)`` wrapping the whole
    # sub-agent ``run_agent_loop`` coroutine. Runs in the sub-agent's own
    # asyncio task, so any contextvar it sets (e.g. agent_team's per-sub
    # bwrap sandbox binding ``/workspace`` + ``/inputs`` (ro) + ``/outputs``
    # (rw)) is scoped to that sub-agent and reset on exit. ``None`` → no
    # wrapping (agent_team leases its sandbox through its own mechanism).
    context_setup: Callable[[str, SubTask], Any] | None = None


@dataclass
class SubAgentSession:
    """A durable sub-agent session that accumulates history across tasks.

    Boundary invariant
    ------------------
    Each *closed* boundary ``(start, end)`` in ``task_boundaries`` is
    guaranteed to contain at least one ``AIMessage`` with no
    ``tool_calls`` between ``start + 1`` and ``end`` inclusive. The
    bus enforces this on boundary closure: when the loop exits without
    one, the bus appends a synthetic ``AIMessage`` populated from
    ``raw_result.final_content`` (or a deterministic stub naming
    ``stopped_by``) and points ``end`` at that synthetic message.

    This lets ``TaskBoundaryTrimmer`` always surface a "final report"
    for completed tasks; without it, max_turns / llm_error / aborted
    sub-agents would be silently dropped from the trimmed history on
    reuse.
    """

    session_id: str
    task_id: str
    name: str
    role_id: str
    system_prompt: str
    tools: list[Tool]
    llm: Any
    trimmer: MessageTrimmer
    max_turns: int = 100
    tool_result_max_chars: int | None = None
    # Per-LLM-call timeout (seconds) used by the agent loop's LLM client.
    # ``None`` means defer to ``LoopConfig.llm_timeout`` (default 180s).
    # Set this when the session's model has high response latency variance
    # (e.g. slow auditor / strong reasoning model used for review).
    llm_timeout: int | None = None
    messages: list[Message] = field(default_factory=list)
    task_boundaries: list[TaskBoundary] = field(default_factory=list)
    total_task_count: int = 0
    current_job_id: str | None = None
    # Wall-clock seconds the most recently finished task took. Kept so a
    # progress snapshot can report a *stable* duration for an idle session:
    # once ``current_job_id`` clears there is no job entry left to measure,
    # and recomputing from "now" makes finished workers appear to keep
    # running.
    last_task_elapsed_s: float = 0.0
    last_report: str = ""
    runtime_spec: SubAgentRuntimeSpec | None = None
    pending_results: list[SubAgentResult] = field(default_factory=list)
    # Tasks queued behind the currently-running one. Sessions execute
    # serially (the messages list / boundary list cannot be safely
    # mutated by two concurrent runs), so a second ``submit_task_to_session``
    # call enqueues here instead of running in parallel. The queue
    # drains FIFO whenever ``current_job_id`` clears.
    pending_tasks: deque[PendingSessionTask] = field(default_factory=deque)
    # Bounded live event trail consumed by local UIs while a task runs.
    # Full trajectories remain owned by workflow observers; this is only the
    # recent thinking/tool activity needed for an expandable team overview.
    activity_events: deque[dict[str, Any]] = field(
        default_factory=lambda: deque(maxlen=40),
    )
    # Path written by _eager_trim_and_offload for the most-recently dropped
    # message batch. Debug only — None when offload is disabled or skipped.
    offloaded_history_path: Path | None = None
