"""stop_subagent — cooperatively ask a running sub-agent to stop."""

from __future__ import annotations

import logging
from typing import Any

from frontier_agent.components.agent_bus import AgentBus
from frontier_agent.components.agent_bus.stop_signal import get_stop_registry
from frontier_agent.core.execution_context import get_current_execution_scope
from frontier_agent.core.runtime import registry
from frontier_agent.core.tool import tool
from plugins.tools._bus_scope import resolve_bus_task_id

logger = logging.getLogger(__name__)


def _current_metadata(bus: AgentBus, session_id: str) -> dict[str, Any]:
    """Metadata of the session's running job, or ``{}`` if it cannot be read.

    The existence check is ``hasattr`` and the call is made on ``bus`` directly.
    Binding the attribute first — ``reader = getattr(bus, ..., None)`` then
    ``callable(reader)`` — narrows it to a bare ``(...) -> object``, which makes
    the ``dict()`` around the result untypeable; going through ``bus`` keeps the
    declared signature. ``hasattr`` also beats catching ``AttributeError``, which
    would swallow one raised *inside* a real implementation.

    A bus-shaped stand-in that predates the accessor is the case being tolerated
    (the tests use one). Any other failure to read metadata degrades to "not a
    publish job" rather than turning a stop request into an exception, and says
    so in the log.
    """
    if not hasattr(bus, "current_job_metadata"):
        return {}
    try:
        return dict(bus.current_job_metadata(session_id) or {})
    except Exception:
        logger.warning(
            "stop_subagent: could not read running-job metadata for %s",
            session_id, exc_info=True,
        )
        return {}


@tool
async def stop_subagent(agent_name: str, force: bool = False) -> str:
    """Ask a running sub-agent to stop exploring as soon as possible.

    Use this when a sub-agent is clearly going off-track, looping on the same
    searches, no longer needed, or burning budget. The sub-agent is told to
    stop and MAY submit a brief report if it already has valuable findings —
    otherwise it just stops. The sub-agent is NOT destroyed: you can assign it
    new work later. Pick up whatever it returns with ``collect_reports``.

    A sub-agent running a PUBLISH task is refused unless ``force`` is set: that
    task writes the run's deliverable, and stopping it throws the deliverable
    away.

    Args:
        agent_name: Name of the sub-agent to stop (the name you created /
            assigned it with).
        force: Stop even a publish task. Only when the publisher is genuinely
            wedged and you accept losing what it was writing.

    Returns:
        A short status line describing what happened.
    """
    scope = get_current_execution_scope()
    if scope is None:
        return (
            "Error: stop_subagent can only be called inside an active "
            "ReAct execution."
        )
    name = (agent_name or "").strip()
    if not name:
        return "Error: `agent_name` was empty."

    bus = registry.get(AgentBus)
    task_id = resolve_bus_task_id(scope)
    session_id = f"{task_id}::{name}"
    session = bus.get_session(session_id)
    if session is None:
        return f"Error: sub-agent {name!r} not found (was it created?)."
    if session.current_job_id is None:
        return (
            f"{name} has no task running right now — nothing to stop. "
            "(It may have already finished; call collect_reports.)"
        )

    # A publish task is the one job whose whole value is the file it writes, and
    # ``StopSignalObserver`` stops by POPPING the LLM turn it just produced —
    # so a stop landing between "decided to write" and "wrote" discards the
    # deliverable and leaves nothing behind. Seen for real: a coordinator
    # dispatched a publish task and stopped it ~5 s later, four times running,
    # burning its publisher's whole task budget while its finished answer never
    # reached /outputs. Publish tasks are short by construction — waiting is
    # almost always right, so make the override explicit.
    if force is not True and _current_metadata(bus, session_id).get("can_publish") is True:
        return (
            f"Refusing to stop {name}: it is running the PUBLISH task, which "
            "writes the final deliverable, and stopping it now would throw "
            "that write away. Publish tasks are short — call collect_reports "
            "to wait for it. Pass force=true only if the publisher is truly "
            "wedged and you accept losing what it was writing."
        )

    # Non-blocking: just register the signal and return. We do NOT wait for
    # the sub-agent to actually stop — it reacts on its own next LLM response.
    # Tag the request with the running job so a stop that outlives its job
    # can't roll back a later task on this (reusable) session.
    get_stop_registry().request_stop(session_id, session.current_job_id or "")
    logger.info("stop_subagent: stop signal queued for %s", session_id)
    return (
        f"Stop signal sent to {name}; it will stop exploring shortly and "
        "may submit a brief report if it has valuable findings."
    )


__all__ = ["stop_subagent"]
