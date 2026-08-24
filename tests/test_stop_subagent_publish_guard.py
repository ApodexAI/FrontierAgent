"""``stop_subagent`` must not throw the deliverable away.

``StopSignalObserver`` stops a sub-agent by POPPING the LLM turn it just
produced, so its pending tool calls are never executed. That is right for
exploration and wrong for the publish task, whose whole value is the file it
writes.

Seen on an APEX trial: the coordinator dispatched a publish task and called
``stop_subagent`` on it ~5 s later, four times in a row. Each publisher reported
"No tool calls were executed — I was stopped before writing", the session burned
its whole task budget, and the finished answer never reached ``/outputs``.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest

from frontier_agent.components.agent_bus import AgentBus
from frontier_agent.components.agent_bus.stop_signal import get_stop_registry
from frontier_agent.core.execution_context import (
    ExecutionScope,
    reset_current_execution_scope,
    set_current_execution_scope,
)
from frontier_agent.core.runtime import registry
from plugins.tools.stop_subagent import stop_subagent

TASK_ID = "current-run"
SESSION_ID = f"{TASK_ID}::data_extractor"


class _FakeBus:
    def __init__(self, job_id: str | None, metadata: dict[str, Any]) -> None:
        self._session = SimpleNamespace(
            session_id=SESSION_ID, current_job_id=job_id,
        )
        self._metadata = metadata

    def get_session(self, session_id: str) -> Any:
        return self._session if session_id == SESSION_ID else None

    def current_job_metadata(self, session_id: str) -> dict[str, Any]:
        return dict(self._metadata) if session_id == SESSION_ID else {}


def _stop(bus: Any, **kwargs: Any) -> str:
    scope = ExecutionScope(task_id=TASK_ID, metadata={})
    saved = registry.snapshot()
    registry.register(AgentBus, bus)  # type: ignore[arg-type]
    token = set_current_execution_scope(scope)
    try:
        return asyncio.run(stop_subagent.func(agent_name="data_extractor", **kwargs))
    finally:
        reset_current_execution_scope(token)
        registry.restore(saved)


@pytest.fixture(autouse=True)
def _clear_registry():
    """The stop registry is a process-wide singleton."""
    yield
    get_stop_registry().consume(SESSION_ID)


def test_a_publish_job_is_not_stopped_and_no_signal_is_queued() -> None:
    result = _stop(_FakeBus("job-1", {"can_publish": True}))

    assert "Refusing to stop" in result
    assert "PUBLISH task" in result
    assert "collect_reports" in result
    # The refusal has to be complete: a queued signal would fire on the
    # publisher's next turn anyway and lose the write regardless.
    assert get_stop_registry().is_requested(SESSION_ID) is False


def test_force_still_stops_a_wedged_publisher() -> None:
    """The capability is kept, just made explicit — a genuinely wedged publisher
    would otherwise hold the run to its turn budget."""
    result = _stop(_FakeBus("job-1", {"can_publish": True}), force=True)

    assert "Stop signal sent to data_extractor" in result
    assert get_stop_registry().is_requested(SESSION_ID) is True


@pytest.mark.parametrize("malformed_force", ["false", "true", 1, [], {}])
def test_only_literal_true_can_force_stop_a_publisher(
    malformed_force: Any,
) -> None:
    """Tool invocation does not enforce annotations at runtime."""
    result = _stop(
        _FakeBus("job-1", {"can_publish": True}),
        force=malformed_force,
    )

    assert "Refusing to stop" in result
    assert get_stop_registry().is_requested(SESSION_ID) is False


def test_an_exploration_job_stops_exactly_as_before() -> None:
    result = _stop(_FakeBus("job-1", {"can_publish": False}))

    assert "Stop signal sent to data_extractor" in result
    assert get_stop_registry().is_requested(SESSION_ID) is True


def test_an_idle_session_is_reported_before_the_publish_check() -> None:
    """Nothing is running, so there is no job to classify — and telling the
    coordinator to wait for a publish task that already finished would send it
    into ``collect_reports`` for a report it can just take."""
    result = _stop(_FakeBus(None, {"can_publish": True}))

    assert "no task running" in result
    assert get_stop_registry().is_requested(SESSION_ID) is False


def test_a_bus_without_the_accessor_does_not_break_a_stop_request() -> None:
    """The guard reads metadata through ``current_job_metadata``. A bus-shaped
    object that predates it must degrade to the old behaviour rather than
    failing the stop outright."""
    bus = SimpleNamespace(
        get_session=lambda session_id: (
            SimpleNamespace(session_id=SESSION_ID, current_job_id="job-1")
            if session_id == SESSION_ID else None
        ),
    )

    result = _stop(bus)

    assert "Stop signal sent to data_extractor" in result
