"""Generic runtime helpers for AgentBus."""

from __future__ import annotations

import logging
from typing import Any

from frontier_agent.components.agent_bus.models import SubAgentResult, SubAgentSession
from frontier_agent.core.events import EventType
from frontier_agent.core.messages import assistant_msg
from frontier_agent.core.protocols import EventSink
from frontier_agent.core.runtime.loop.message_trimmer import find_final_assistant
from frontier_agent.core.runtime.registries import services as registry

logger = logging.getLogger(__name__)


def build_default_subagent_loop_config(
    job_id: str,
    item: Any,
    max_turns: int,
) -> Any:
    """Build the generic default LoopConfig for async sub-agent jobs."""
    from frontier_agent.core.loop_types import LoopConfig

    return LoopConfig(
        max_turns=max_turns,
        task_id=job_id,
        role_id=item.role_id,
    )


def build_default_subagent_observers(
    *,
    event_store: Any | None,
    job_id: str,
) -> list[Any]:
    """Build the generic default observer stack for async sub-agent jobs."""
    del event_store, job_id
    return []


def adapt_default_subagent_result(
    agent_result: Any,
    job_id: str,
    item: Any,
) -> Any:
    """Adapt an AgentLoopResult into a generic SubAgentResult.

    The kernel default forwards the full loop metadata bag untouched —
    any domain-specific fields (evidence_cards, assertions, …) live
    inside ``metadata`` and only appear when workflow observers
    populated them.
    """
    metadata = dict(getattr(agent_result, "metadata", {}) or {})
    return SubAgentResult(
        question=item.question,
        role_id=item.role_id,
        final_content=getattr(agent_result, "final_content", "") or "",
        success=True,
        job_id=job_id,
        metadata=metadata,
    )


def adapt_default_session_result(
    agent_result: Any,
    job_id: str,
    session: Any,
    task_prompt: str,
) -> Any:
    """Adapt a session loop result into a generic SubAgentResult."""
    metadata = dict(getattr(agent_result, "metadata", {}) or {})
    return SubAgentResult(
        question=task_prompt,
        role_id=session.role_id,
        final_content=getattr(agent_result, "final_content", "") or "",
        success=True,
        job_id=job_id,
        metadata=metadata,
    )


def build_session_loop_config(
    session: Any,
    job_id: str,
    max_turns: int,
) -> Any:
    """Build LoopConfig for a session task."""
    from frontier_agent.core.loop_types import LoopConfig

    kwargs: dict[str, Any] = {
        "max_turns": max_turns,
        "task_id": session.task_id,
        "role_id": session.role_id,
        "tool_result_max_chars": session.tool_result_max_chars,
    }
    if getattr(session, "llm_timeout", None) is not None:
        kwargs["llm_timeout"] = session.llm_timeout
    return LoopConfig(**kwargs)


def resolve_session_observers(
    observers: list[Any] | None,
    session: Any | None = None,
) -> list[Any]:
    """Resolve observers for a session task.

    The kernel default is intentionally empty. Workflow-specific observer
    stacks should be injected through ``runtime_spec`` or explicit callers.
    """
    if observers is not None:
        return observers
    del session
    return []


def close_session_boundary_aborted(session: Any) -> None:
    """Close the current session task boundary when cancelled/errored.

    Maintains the SubAgentSession boundary invariant: when no clean
    AIMessage exists in the in-flight slice, append an abort stub so
    the trimmer can still surface "task #N happened, here's what we
    have" when the agent is reused.
    """
    if not session.task_boundaries:
        return
    start, end = session.task_boundaries[-1]
    if end is not None:
        return

    if find_final_assistant(
        session.messages, start + 1, len(session.messages) - 1,
    ) is None:
        session.messages.append(
            assistant_msg("[task aborted before producing a final answer]"),
        )

    tail = max(start, len(session.messages) - 1)
    session.task_boundaries[-1] = (start, tail)


async def emit_session_task_submitted(
    session: SubAgentSession,
    job_id: str,
    task_prompt: str,
    *,
    event_sink: Any = None,
) -> None:
    """Record the submit side of a session-task lifecycle to EventStore.

    ``event_sink`` is an optional ``core.protocols.EventSink`` injected
    by ``AgentBus``. Empty falls back to the
    global registry lookup so existing direct callers keep working.
    """
    event_store = event_sink if event_sink is not None else registry.get_optional(EventSink)
    if event_store is None:
        return
    await event_store.append(
        task_id=session.task_id,
        event_type=EventType.AGENT_ACTION,
        payload={
            "trace_type": "session_task_submitted",
            "session_id": session.session_id,
            "job_id": job_id,
            "task_count": session.total_task_count,
            "is_reuse": session.total_task_count > 1,
            "role_id": session.role_id,
            "agent": session.name,
            "action": "assign_task",
            "detail": (
                f"{'Reusing' if session.total_task_count > 1 else 'Starting'} "
                f"sub-agent '{session.name}' "
                f"(task #{session.total_task_count}): "
                f"{task_prompt[:140]}"
            ),
        },
        agent_role="system",
    )


def safe_metadata(raw_result: Any) -> dict[str, Any]:
    """Best-effort extraction of ``metadata`` from a partial AgentLoopResult.

    Called from exception handlers where ``raw_result`` may be ``None``
    (agent loop never completed) or a completed result whose downstream
    adapter raised. Never raises — returns ``{}`` on any oddity.

    The evidence harvested inside an agent loop is the most expensive
    side-effect of a research run (it cost real web_search/web_fetch
    API calls). Losing it because a post-loop bookkeeping step
    crashed is the worst possible UX: the user still paid for the
    calls but sees nothing in the evidence graph.
    """
    if raw_result is None:
        return {}
    try:
        meta = getattr(raw_result, "metadata", None)
        if isinstance(meta, dict):
            return dict(meta)
    except Exception:
        pass
    return {}


def safe_final_content(raw_result: Any) -> str:
    """Safely pull ``final_content`` off a possibly-None result."""
    if raw_result is None:
        return ""
    try:
        value = getattr(raw_result, "final_content", "") or ""
        return str(value)
    except Exception:
        return ""


async def emit_session_task_completed(
    session: SubAgentSession,
    job_id: str,
    result: SubAgentResult,
    *,
    event_sink: Any = None,
) -> None:
    """Record the completion side of a session-task lifecycle.

    Swallows failures — telemetry must not break the session lifecycle.
    List-valued metadata entries are summarised as ``{key}_count`` to
    keep the event payload compact regardless of workflow-specific
    metadata shape.

    ``event_sink`` accepts a ``core.protocols.EventSink`` injected by
    ``AgentBus``; falls back to the global
    registry when omitted.
    """
    ev_store = event_sink if event_sink is not None else registry.get_optional(EventSink)
    if ev_store is None:
        return
    try:
        metadata_counts = {
            f"{k}_count": len(v)
            for k, v in result.metadata.items()
            if isinstance(v, list)
        }
        detail_parts = [
            f"{len(v)} {k}"
            for k, v in result.metadata.items()
            if isinstance(v, list) and v
        ]
        detail = f"'{session.name}' returned"
        if detail_parts:
            detail += " " + ", ".join(detail_parts)
        if not result.success and result.error:
            # Make the failure legible in both the event stream and
            # any downstream log summarizer — the previous shape left
            # "success: false" with no reason and every sub-agent
            # failure looked identical.
            cls = (result.error_class or "").strip()
            err_short = str(result.error)[:200]
            detail += f" (failed: {cls}: {err_short})" if cls else f" (failed: {err_short})"
        payload: dict[str, Any] = {
            "trace_type": "session_task_completed",
            "session_id": session.session_id,
            "job_id": job_id,
            "success": result.success,
            "agent": session.name,
            "action": "report_returned",
            "detail": detail,
            **metadata_counts,
        }
        if result.error:
            payload["error"] = str(result.error)[:500]
        if result.error_class:
            payload["error_class"] = result.error_class
        await ev_store.append(
            task_id=session.task_id,
            event_type=EventType.AGENT_ACTION,
            payload=payload,
            agent_role="system",
        )
    except Exception:
        pass
