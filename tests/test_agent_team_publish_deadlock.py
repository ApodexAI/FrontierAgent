"""The publish deadlock: two guards that each told the coordinator to do the
opposite of the other.

Observed on an APEX trial that ran to completion with a finished 8,745-byte
answer and shipped no deliverable at all. Its last six turns:

    assign_task data_extractor   → "5-task limit … create a fresh sub-agent"
    assign_task answer_publisher → "publisher already assigned to
                                    'data_extractor'; reuse that agent"
    assign_task final_publisher  → same
    assign_task filesystem_scanner → same

The task cap says *use a new agent*; the publisher lock said *reuse the capped
one*. Mutually exclusive, so nothing in the run could ever write ``/outputs``
again — and the agent_team main agent has no ``create_file`` or ``bash`` of its
own, so it could not publish instead.

The lock still exists: its job is to stop two sub-agents racing on the same
deliverable. It just is not permanent any more.
"""

from __future__ import annotations

import asyncio
from collections import deque
from types import SimpleNamespace
from typing import Any

import pytest

from frontier_agent.components.agent_bus import AgentBus
from frontier_agent.core.execution_context import (
    ExecutionScope,
    reset_current_execution_scope,
    set_current_execution_scope,
)
from frontier_agent.core.runtime import registry
from plugins.tools._bus_scope import SWARM_SCOPE_KEY
from plugins.tools.assign_task import (
    MAX_TASKS_PER_SESSION,
    _session_at_task_cap,
    _session_has_publish_work,
    agent_team_assign_task,
)

TASK_ID = "current-run"
MANIFEST = ["/outputs/answer.md"]


def _session(name: str, *, dispatched: int = 0, queued: int = 0) -> Any:
    return SimpleNamespace(
        session_id=f"{TASK_ID}::{name}",
        total_task_count=dispatched,
        pending_tasks=deque(range(queued)),
        current_job_id=None,
        tools=[],
        max_turns=30,
    )


class _FakeBus:
    """Just the surface ``assign_task`` touches."""

    def __init__(self, sessions: dict[str, Any]) -> None:
        self._sessions = sessions
        self.submitted: list[tuple[str, dict[str, Any]]] = []

    def get_session(self, session_id: str) -> Any:
        return self._sessions.get(session_id)

    def current_job_metadata(self, session_id: str) -> dict[str, Any]:
        session = self._sessions.get(session_id)
        return dict(getattr(session, "current_metadata", {}) or {})

    def get_last_report(self, *_args: Any, **_kwargs: Any) -> str:
        return ""

    async def submit_task_to_session(
        self,
        session_id: str,
        _prompt: str,
        *,
        spawn_context: dict[str, Any] | None = None,
        task_metadata: dict[str, Any] | None = None,
    ) -> str:
        self.submitted.append((session_id, dict(task_metadata or {})))
        return f"job-{len(self.submitted)}"


@pytest.fixture
def dispatch(monkeypatch: pytest.MonkeyPatch):
    """Call the real ``assign_task`` against a fake bus and publication state."""
    def _run(
        sessions: dict[str, Any],
        tasks: list[dict[str, Any]],
        publication_state: dict[str, Any] | None = None,
    ) -> tuple[str, _FakeBus, dict[str, Any]]:
        bus = _FakeBus(sessions)
        state = publication_state if publication_state is not None else {}
        runtime = SimpleNamespace(
            publication_state=state,
            publication_lock=asyncio.Lock(),
            original_question="what is the net consumer intent?",
        )
        scope = ExecutionScope(
            task_id=TASK_ID,
            metadata={SWARM_SCOPE_KEY: runtime, "run_id": "run-1"},
        )
        saved = registry.snapshot()
        registry.register(AgentBus, bus)  # type: ignore[arg-type]
        token = set_current_execution_scope(scope)
        try:
            result = asyncio.run(agent_team_assign_task.func(tasks=tasks))
        finally:
            reset_current_execution_scope(token)
            registry.restore(saved)
        return result, bus, state

    return _run


# ── the capacity predicate both guards now share ─────────────────────────────


def test_capacity_counts_queued_tasks_and_treats_a_missing_session_as_capped(
) -> None:
    """Queued tasks count: ``total_task_count`` only increments at dispatch, so
    counting dispatches alone let a burst of assignments slip past the cap."""
    assert _session_at_task_cap(None) is True
    assert _session_at_task_cap(_session("a", dispatched=0)) is False
    assert _session_at_task_cap(
        _session("a", dispatched=MAX_TASKS_PER_SESSION - 1)
    ) is False
    assert _session_at_task_cap(
        _session("a", dispatched=MAX_TASKS_PER_SESSION)
    ) is True
    assert _session_at_task_cap(
        _session("a", dispatched=MAX_TASKS_PER_SESSION - 1, queued=1)
    ) is True


def test_publish_work_detection_covers_running_and_queued_jobs() -> None:
    running = _session("running", dispatched=MAX_TASKS_PER_SESSION)
    running.current_job_id = "job-running"
    running.current_metadata = {"can_publish": True}
    queued = _session("queued", dispatched=MAX_TASKS_PER_SESSION - 1)
    queued.pending_tasks.append(SimpleNamespace(
        task_metadata={"can_publish": True},
    ))
    bus = _FakeBus({running.session_id: running, queued.session_id: queued})

    assert _session_has_publish_work(bus, running) is True
    assert _session_has_publish_work(bus, queued) is True


@pytest.mark.parametrize("state", ["running", "queued"])
def test_the_publisher_role_does_not_move_while_publish_work_exists(
    dispatch,
    state: str,
) -> None:
    incumbent = _session("writer", dispatched=MAX_TASKS_PER_SESSION)
    if state == "running":
        incumbent.current_job_id = "job-running"
        incumbent.current_metadata = {"can_publish": True}
    else:
        incumbent.total_task_count = MAX_TASKS_PER_SESSION - 1
        incumbent.pending_tasks.append(SimpleNamespace(
            task_metadata={"can_publish": True},
        ))
    replacement = _session("replacement")

    result, bus, publication = dispatch(
        {
            incumbent.session_id: incumbent,
            replacement.session_id: replacement,
        },
        [{"agent": "replacement", "prompt": "publish", "publish": True,
          "output_paths": MANIFEST}],
        {
            "publisher_agent_name": "writer",
            "deliverable_manifest": tuple(MANIFEST),
        },
    )

    assert "still has a publishing task running or queued" in result
    assert bus.submitted == []
    assert publication["publisher_agent_name"] == "writer"


# ── the lock ─────────────────────────────────────────────────────────────────


def test_a_second_publisher_is_refused_while_the_first_can_still_work(
    dispatch,
) -> None:
    """The lock's actual purpose: one publisher per run, so two sub-agents cannot
    race on the same file."""
    sessions = {
        f"{TASK_ID}::writer": _session("writer", dispatched=1),
        f"{TASK_ID}::other": _session("other"),
    }

    result, bus, state = dispatch(
        sessions,
        [{"agent": "other", "prompt": "publish", "publish": True,
          "output_paths": MANIFEST}],
        {"publisher_agent_name": "writer", "deliverable_manifest": tuple(MANIFEST)},
    )

    assert "publisher already assigned to 'writer'" in result
    assert "reuse that agent" in result
    assert bus.submitted == []
    # The incumbent keeps the role and the manifest.
    assert state["publisher_agent_name"] == "writer"


def test_the_publisher_role_moves_on_once_the_incumbent_is_capped(
    dispatch,
) -> None:
    """The deadlock, gone. A capped incumbent cannot be assigned anything, so
    holding the role would mean the deliverable can never be written."""
    sessions = {
        f"{TASK_ID}::data_extractor": _session(
            "data_extractor", dispatched=MAX_TASKS_PER_SESSION,
        ),
        f"{TASK_ID}::answer_publisher": _session("answer_publisher"),
    }

    result, bus, state = dispatch(
        sessions,
        [{"agent": "answer_publisher", "prompt": "publish the final answer",
          "publish": True, "output_paths": MANIFEST}],
        {
            "publisher_agent_name": "data_extractor",
            "deliverable_manifest": tuple(MANIFEST),
        },
    )

    assert "Submitted 1 task" in result
    assert "answer_publisher" in result
    assert state["publisher_agent_name"] == "answer_publisher"
    assert state["deliverable_manifest"] == tuple(MANIFEST)
    session_id, metadata = bus.submitted[0]
    assert session_id == f"{TASK_ID}::answer_publisher"
    # The new publisher must actually be able to write, or the transfer is
    # cosmetic: this metadata is what scopes the /outputs write policy.
    assert metadata["can_publish"] is True
    assert metadata["output_paths"] == MANIFEST


def test_the_role_moves_on_when_the_incumbent_session_is_gone(
    dispatch,
) -> None:
    """A name the bus no longer knows is as undispatchable as a capped one."""
    sessions = {f"{TASK_ID}::rescue_publisher": _session("rescue_publisher")}

    result, _bus, state = dispatch(
        sessions,
        [{"agent": "rescue_publisher", "prompt": "publish", "publish": True,
          "output_paths": MANIFEST}],
        {
            "publisher_agent_name": "vanished",
            "deliverable_manifest": tuple(MANIFEST),
        },
    )

    assert "Submitted 1 task" in result
    assert state["publisher_agent_name"] == "rescue_publisher"


def test_a_transfer_still_cannot_quietly_change_the_manifest(
    dispatch,
) -> None:
    """Transferring the role is not a licence to change the deliverable: a new
    publisher inherits the fixed manifest and must ask for a replacement."""
    sessions = {
        f"{TASK_ID}::old": _session("old", dispatched=MAX_TASKS_PER_SESSION),
        f"{TASK_ID}::new": _session("new"),
    }

    result, bus, state = dispatch(
        sessions,
        [{"agent": "new", "prompt": "publish", "publish": True,
          "output_paths": ["/outputs/report.docx"]}],
        {"publisher_agent_name": "old", "deliverable_manifest": tuple(MANIFEST)},
    )

    assert "output manifest is already fixed" in result
    assert "replace_manifest=true" in result
    assert bus.submitted == []
    assert state["deliverable_manifest"] == tuple(MANIFEST)


def test_the_capped_incumbent_is_still_refused_its_own_retry(
    dispatch,
) -> None:
    """The cap is unchanged — reusing the saturated session is still refused, and
    the message still points at a fresh sub-agent. That advice only became
    followable because of the transfer above."""
    sessions = {
        f"{TASK_ID}::data_extractor": _session(
            "data_extractor", dispatched=MAX_TASKS_PER_SESSION,
        ),
    }

    result, bus, _state = dispatch(
        sessions,
        [{"agent": "data_extractor", "prompt": "publish", "publish": True,
          "output_paths": MANIFEST}],
        {
            "publisher_agent_name": "data_extractor",
            "deliverable_manifest": tuple(MANIFEST),
        },
    )

    assert f"reached the {MAX_TASKS_PER_SESSION}-task limit" in result
    assert "create a fresh sub-agent" in result
    assert bus.submitted == []
