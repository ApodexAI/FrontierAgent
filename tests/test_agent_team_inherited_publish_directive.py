"""The deliverable directive every sub-agent inherits, and nobody owned.

APEX task text names the deliverable itself — "Write your reply to the user as
``/outputs/answer.md``" — and ``assign_task`` prepends the original question
verbatim to every dispatched prompt. A workspace-only sub-agent therefore read
the user's own instruction to write ``/outputs/answer.md`` and then, further
down, a contract block telling it not to write ``/outputs`` at all. Nothing
connected the two, so the agent had to guess which one governed.

Measured over the 267-task APEX agent_team run (``apex267team``):

* 267/267 task texts carried such a directive;
* 234/267 trials hit the non-publisher write denial;
* 1352/1767 denials (76.5%) targeted exactly ``/outputs/answer.md``.

The agents were obeying the user. The denial then went only to the sub-agent,
which cannot grant itself permission: it wrote the deliverable somewhere else
(usually ``/outputs/scratch``), reported success, and the coordinator never
learned that nothing had landed.

So the contract now names the inherited directive and says who owns it, and a
directive the coordinator wrote itself is reported back on the assign_task
return — the one moment the coordinator can still fix the assignment.
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
from plugins.tools._deliverable_policy import output_write_directives
from plugins.tools.assign_task import agent_team_assign_task

TASK_ID = "current-run"

# The instruction APEX appends to all 480 task texts, verbatim.
APEX_QUESTION = (
    "# Task\nProvide the mean implied equity value for CNS from the "
    "Comparables and Precedents using 2025E EBITDA.\n\n"
    "Write your reply to the user as /outputs/answer.md."
)


def _session(name: str) -> Any:
    return SimpleNamespace(
        session_id=f"{TASK_ID}::{name}",
        total_task_count=0,
        pending_tasks=deque(),
        current_job_id=None,
        tools=[],
        max_turns=30,
    )


class _FakeBus:
    """Just the surface ``assign_task`` touches — but it keeps the prompt."""

    def __init__(self, sessions: dict[str, Any]) -> None:
        self._sessions = sessions
        self.submitted: list[tuple[str, str, dict[str, Any]]] = []

    def get_session(self, session_id: str) -> Any:
        return self._sessions.get(session_id)

    def current_job_metadata(self, session_id: str) -> dict[str, Any]:
        return {}

    def get_last_report(self, *_args: Any, **_kwargs: Any) -> str:
        return ""

    async def submit_task_to_session(
        self,
        session_id: str,
        prompt: str,
        *,
        spawn_context: dict[str, Any] | None = None,
        task_metadata: dict[str, Any] | None = None,
    ) -> str:
        self.submitted.append((session_id, prompt, dict(task_metadata or {})))
        return f"job-{len(self.submitted)}"


@pytest.fixture
def dispatch():
    """Run the real ``agent_team_assign_task`` and keep the prompts it sent."""
    def _run(
        tasks: list[dict[str, Any]],
        *,
        question: str = APEX_QUESTION,
        publication_state: dict[str, Any] | None = None,
    ) -> tuple[str, _FakeBus]:
        names = [str(task.get("agent", "")) for task in tasks]
        sessions = {
            f"{TASK_ID}::{name}": _session(name) for name in names if name
        }
        bus = _FakeBus(sessions)
        runtime = SimpleNamespace(
            publication_state=dict(publication_state or {}),
            publication_lock=asyncio.Lock(),
            original_question=question,
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
        return result, bus

    return _run


# ── the detector ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        # The APEX sentence itself, which is the whole reason this exists.
        ("Write your reply to the user as /outputs/answer.md.",
         ("/outputs/answer.md",)),
        ("Save the edited workbook to /outputs/Model_v2.xlsx",
         ("/outputs/Model_v2.xlsx",)),
        ("cp /workspace/draft.md /outputs/answer.md",
         ("/outputs/answer.md",)),
        ("python3 build.py > /outputs/report.csv",
         ("/outputs/report.csv",)),
        # Reading /outputs is allowed for everyone, so a read must not match.
        ("Read /outputs/answer.md and audit every figure in it.", ()),
        # A past participle describes a file that already exists.
        ("Audit the draft answer against the published file "
         "/outputs/answer.md.", ()),
        # The most common coordinator shape of all: telling it to stay out.
        ("Do not write anything to /outputs/answer.md.", ()),
        ("Never create /outputs/answer.md yourself.", ()),
        # Scratch is writable by every assignment, publisher or not.
        ("Put intermediate tables in /outputs/scratch/tables.csv", ()),
        # A bare mention is not a directive — it is usually "don't touch it".
        ("Do not create any other file in /outputs.", ()),
        ("Run `ls -la /outputs` and report what you see.", ()),
        ("No /outputs anywhere in this prompt.", ()),
    ],
)
def test_the_detector_matches_directives_and_not_mentions(
    text: str, expected: tuple[str, ...],
) -> None:
    """Precision matters and recall does not.

    A miss costs nothing: the write is still blocked at execution time and the
    contract still carries its generic override. A false positive would put a
    sentence in the prompt that contradicts the assignment.
    """
    assert output_write_directives(text) == expected


# ── the inherited directive ──────────────────────────────────────────────────


def test_a_workspace_only_agent_is_told_who_owns_the_inherited_path(
    dispatch,
) -> None:
    result, bus = dispatch([
        {"agent": "researcher", "prompt": "Extract the comparables table.",
         "publish": False},
    ])

    assert "Submitted 1 task(s)" in result
    (_session_id, prompt, metadata) = bus.submitted[0]
    assert metadata == {"can_publish": False, "output_paths": []}
    # The user's instruction really is in this prompt — that is the conflict.
    assert "Write your reply to the user as /outputs/answer.md" in prompt
    # ...and now something in the prompt resolves it.
    assert "The task text above tells you to write `/outputs/answer.md`" in prompt
    assert "addressed to the team, not to you" in prompt
    # It also has to say what to do instead, or the agent has no next move.
    assert "/workspace" in prompt.split("addressed to the team")[1]


def test_a_question_without_a_directive_still_gets_the_generic_override(
    dispatch,
) -> None:
    """Not every workflow's question names a file, and the override must not
    invent one — but a directive can also reach the agent by a route this
    dispatch never sees (a teammate's attached report, the system prompt), so
    the generic sentence stays."""
    _result, bus = dispatch(
        [{"agent": "researcher", "prompt": "Summarise the filing.",
          "publish": False}],
        question="What is the net consumer intent?",
    )

    (_session_id, prompt, _metadata) = bus.submitted[0]
    assert "The task text above tells you to write" not in prompt
    assert "If any instruction you were given names a file under" in prompt
    assert "addressed to the team, not to you" in prompt


def test_the_publisher_keeps_its_publishing_contract_untouched(
    dispatch,
) -> None:
    """The publisher is the agent the inherited directive was addressed to.
    Telling it a different agent owns the path would be exactly wrong."""
    _result, bus = dispatch([
        {"agent": "publisher", "prompt": "Integrate and publish the answer.",
         "publish": True, "output_paths": ["/outputs/answer.md"]},
    ])

    (_session_id, prompt, metadata) = bus.submitted[0]
    assert metadata["can_publish"] is True
    assert "You are the sole publisher for this assignment" in prompt
    assert "addressed to the team, not to you" not in prompt
    assert "Workspace-only contract" not in prompt


# ── the directive the coordinator wrote itself ───────────────────────────────


def test_a_self_contradictory_assignment_dispatches_and_is_reported_back(
    dispatch,
) -> None:
    """``{"prompt": "write /outputs/answer.md", "publish": false}``.

    Rejecting the dispatch was considered and measured: the detector flags only
    39% of the trials that actually hit the denial, so a hard gate would buy
    little recall while costing a coordinator turn on every false positive.
    Dispatching with the conflict named in the prompt AND reported on the
    return is strictly more informative than either half alone.
    """
    result, bus = dispatch([
        {"agent": "writer",
         "prompt": "Write the final memo to /outputs/answer.md.",
         "publish": False},
    ])

    # It still ran — the work is not lost to a validation error.
    assert "Submitted 1 task(s)" in result
    (_session_id, prompt, metadata) = bus.submitted[0]
    assert metadata["can_publish"] is False
    assert "The task text above tells you to write `/outputs/answer.md`" in prompt

    # And the coordinator is told, in the same turn it can still act on.
    assert "Publishing notices:" in result
    assert "writer: dispatched workspace-only" in result
    assert "output_paths=['/outputs/answer.md']" in result


def test_a_plain_workspace_assignment_produces_no_notice(dispatch) -> None:
    """The notice has to stay rare or it becomes noise the coordinator skips."""
    result, _bus = dispatch(
        [{"agent": "researcher",
          "prompt": "Extract every comparable multiple into /workspace/comps.csv.",
          "publish": False}],
        question="Which comparables support the valuation?",
    )

    assert "Submitted 1 task(s)" in result
    assert "Publishing notices:" not in result


# ── who owns the path, when nobody does ──────────────────────────────────────


def test_the_contract_does_not_promise_a_publisher_that_does_not_exist(
    dispatch,
) -> None:
    """"A different agent will write it" is a claim about the run, and in a
    research-only round it is false. An agent that reads it, looks around, and
    concludes nobody is coming is back to guessing — which is the behaviour the
    whole block exists to remove."""
    _result, bus = dispatch([
        {"agent": "researcher", "prompt": "Extract the comparables table.",
         "publish": False},
    ])

    (_session_id, prompt, _metadata) = bus.submitted[0]
    assert "no agent in this run holds the publishing role yet" in prompt
    assert "a different agent holds the publishing role" not in prompt
    # Still says what to do instead.
    assert "/workspace" in prompt.split("addressed to the team")[1]


def test_a_publisher_in_the_same_dispatch_covers_the_path(dispatch) -> None:
    result, bus = dispatch([
        {"agent": "researcher", "prompt": "Extract the comparables table.",
         "publish": False},
        {"agent": "publisher", "prompt": "Integrate and publish.",
         "publish": True, "output_paths": ["/outputs/answer.md"]},
    ])

    prompts = {name: prompt for name, prompt, _meta in
               ((sid.split("::")[1], pr, md) for sid, pr, md in bus.submitted)}
    assert "publisher candidate" in prompts["researcher"]
    assert "will write it" not in prompts["researcher"]
    assert "no agent in this run holds" not in prompts["researcher"]
    assert "No agent in this run can write" not in result


def test_a_publisher_recorded_in_an_earlier_round_still_owns_the_path(
    dispatch,
) -> None:
    """Rounds are dispatched separately. A research round that follows the
    publisher's assignment does have a publisher; saying otherwise would invite
    the write it already told the agent not to make."""
    result, bus = dispatch(
        [{"agent": "researcher", "prompt": "Check the tax basis figure.",
          "publish": False}],
        publication_state={
            "publisher_agent_name": "publisher",
            "deliverable_manifest": ("/outputs/answer.md",),
        },
    )

    (_session_id, prompt, _metadata) = bus.submitted[0]
    assert "publisher candidate" in prompt
    assert "will write it" not in prompt
    assert "Publishing notices:" not in result


def test_a_round_with_no_publisher_tells_the_coordinator_so(dispatch) -> None:
    """Population A of the APEX run: the deliverable is named, every agent is
    workspace-only, and nothing said so until the writes were already denied."""
    result, _bus = dispatch([
        {"agent": "researcher", "prompt": "Extract the comparables table.",
         "publish": False},
        {"agent": "verifier", "prompt": "Check the arithmetic.",
         "publish": False},
    ])

    assert "Submitted 2 task(s)" in result
    assert "Publishing notices:" in result
    assert "No agent in this run can write /outputs/answer.md" in result
    assert "output_paths=['/outputs/answer.md']" in result
    # Not an error: a research round is a legitimate reason to be here.
    assert "Error" not in result
    assert "expected for a research or verification round" in result


def test_a_rejected_publisher_does_not_suppress_the_notice(dispatch) -> None:
    """A spec carrying a manifest is only a request. Runtime checks can reject
    it, and
    then the successfully dispatched research task must not be told that the
    rejected publisher will write the deliverable."""
    result, bus = dispatch([
        {"agent": "researcher", "prompt": "Extract the comparables table.",
         "publish": False},
        {"agent": "final_verifier", "prompt": "Publish the checked answer.",
         "publish": True, "output_paths": ["/outputs/answer.md"]},
    ])

    assert [session_id for session_id, _prompt, _metadata in bus.submitted] == [
        f"{TASK_ID}::researcher",
    ]
    (_session_id, prompt, _metadata) = bus.submitted[0]
    assert "publisher candidate" in prompt
    assert "will write it" not in prompt
    assert "verifier tasks cannot publish files" in result
    assert "No agent in this run can write /outputs/answer.md" in result


def test_a_publisher_for_another_manifest_does_not_own_this_path(dispatch) -> None:
    """Publisher identity is not authority for every output path. The manifest
    has to cover the concrete directive before the coordinator can be told that
    somebody can write it."""
    result, _bus = dispatch([
        {"agent": "researcher", "prompt": "Extract the comparables table.",
         "publish": False},
        {"agent": "publisher", "prompt": "Publish the supporting report.",
         "publish": True, "output_paths": ["/outputs/report.pdf"]},
    ])

    assert "Submitted 2 task(s)" in result
    assert "No agent in this run can write /outputs/answer.md" in result
    assert "replace_manifest=true" in result
    assert (
        "output_paths=['/outputs/report.pdf', '/outputs/answer.md']" in result
    )


# ── the manifest is the grant ────────────────────────────────────────────────


def test_a_manifest_alone_dispatches_an_authorized_publisher(dispatch) -> None:
    """No ``publish`` key, and the sub-agent still gets write authority.

    This is the case the old contract lost. ``publish`` defaulted to false, so
    an omitted boolean collided with the manifest and the whole dispatch was
    rejected with "output_paths requires publish=true" -- while the coordinator
    omitted that boolean on roughly a quarter of its assignments across the
    30-task replay (60/236 control, 63/225 fix).
    """
    result, bus = dispatch([
        {"agent": "writer", "prompt": "Write the final answer.",
         "output_paths": ["/outputs/answer.md"]},
    ])

    assert "Error" not in result
    (_session_id, prompt, metadata) = bus.submitted[0]
    assert metadata["can_publish"] is True
    assert metadata["output_paths"] == ["/outputs/answer.md"]
    # Authority reached the sub-agent's own contract block, not just the metadata.
    assert "/outputs/answer.md" in prompt
    # And the deliverable now has an owner, so the no-owner notice stays quiet.
    assert "No agent in this run can write" not in result


def test_a_manifest_alone_still_bounds_the_run_to_one_publisher(dispatch) -> None:
    """The single-publisher invariant has to key on the manifest as well."""
    result, _bus = dispatch([
        {"agent": "writer", "prompt": "Write the answer.",
         "output_paths": ["/outputs/answer.md"]},
        {"agent": "other", "prompt": "Write it differently.",
         "output_paths": ["/outputs/answer.md"]},
    ])

    assert "only one publishing assignment is allowed per dispatch" in result


def test_a_manifest_alone_still_cannot_go_to_a_verifier(dispatch) -> None:
    result, bus = dispatch([
        {"agent": "final_verifier", "prompt": "Check and publish.",
         "output_paths": ["/outputs/answer.md"]},
    ])

    assert "verifier tasks cannot publish files" in result
    assert bus.submitted == []
