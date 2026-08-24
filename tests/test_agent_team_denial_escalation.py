"""A blocked /outputs write, reported to the one party that can fix it.

The refusal has always gone to the sub-agent that attempted the write — and a
sub-agent cannot authorize itself. Over the 267-task APEX agent_team run
(``apex267team``) 234/267 trials hit that refusal, and the usual next move was
to route the deliverable somewhere else (mostly ``/outputs/scratch``) and report
success. The coordinator, which could have fixed it by dispatching a publish
task, was the only party never told: it read a report that said the work was
done and finished the run with nothing in ``/outputs``.

So the sub-agent's result now carries the refusal to the coordinator.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from frontier_agent.components.agent_bus.models import SubTask
from plugins.tools._deliverable_policy import (
    bash_output_write_error,
    new_output_write_denial_log,
    output_write_error,
    render_denial_escalation,
    reset_deliverable_write_paths,
    reset_output_write_denial_log,
    set_deliverable_write_paths,
)
from workflows.agent_team.subagent_runtime import (
    SwarmSubagentRuntime,
    build_swarm_session_runtime_spec,
)


@pytest.fixture
def recording():
    """A non-publisher policy with an open denial log."""
    policy_token = set_deliverable_write_paths([])
    denial_token, log = new_output_write_denial_log()
    try:
        yield log
    finally:
        reset_output_write_denial_log(denial_token)
        reset_deliverable_write_paths(policy_token)


# ── recording ────────────────────────────────────────────────────────────────


def test_both_write_surfaces_record_the_same_refusal_once(recording) -> None:
    """``bash`` and the file tools are separate enforcement paths, and an agent
    that is refused usually tries the other one next."""
    assert output_write_error("/outputs/answer.md") is not None
    assert bash_output_write_error("cp /workspace/a.md /outputs/answer.md") is not None

    assert recording == [("not_publisher", "/outputs/answer.md")]


def test_an_unverifiable_read_is_not_recast_as_a_failed_write(recording) -> None:
    """Fail-closed bash policy blocks opaque commands even when the caller says
    they only read. The coordinator must hear that uncertainty, not a fabricated
    write attempt or a claim that publication failed."""
    assert (
        bash_output_write_error(
            "python3 unknown_reader.py /outputs/answer.md",
        )
        is not None
    )

    assert recording == [("unverifiable", "/outputs/answer.md")]
    note = render_denial_escalation(recording)
    assert note.startswith("[BLOCKED OUTPUTS ACCESS:")
    assert "could not verify as read-only" in note
    assert "does not establish that the agent attempted a write" in note
    assert "Nothing it produced" not in note


def test_the_shared_scratch_area_is_not_a_refused_deliverable(recording) -> None:
    """Every assignment may write ``/outputs/scratch``, and the quota errors it
    can raise are self-correctable — neither is a lost deliverable."""
    assert output_write_error("/outputs/scratch/notes.csv") is None
    assert bash_output_write_error("cp a /outputs/scratch/notes.csv") is None

    assert recording == []


def test_a_publisher_records_only_what_falls_outside_its_manifest() -> None:
    policy_token = set_deliverable_write_paths(["/outputs/answer.md"])
    denial_token, log = new_output_write_denial_log()
    try:
        assert output_write_error("/outputs/answer.md") is None
        assert output_write_error("/outputs/extra.md") is not None
    finally:
        reset_output_write_denial_log(denial_token)
        reset_deliverable_write_paths(policy_token)

    assert log == [("undeclared", "/outputs/extra.md")]


def test_nothing_is_recorded_outside_a_task() -> None:
    """Workflows that never opted in must not pay for this, and a missing log
    has to be a no-op rather than an error."""
    policy_token = set_deliverable_write_paths([])
    try:
        assert output_write_error("/outputs/answer.md") is not None
    finally:
        reset_deliverable_write_paths(policy_token)


# ── the escalation text ──────────────────────────────────────────────────────


def test_the_non_publisher_note_contradicts_a_false_success_claim() -> None:
    note = render_denial_escalation([("not_publisher", "/outputs/answer.md")])

    assert "/outputs/answer.md" in note
    assert "is not the publisher" in note
    # The failure this exists to stop: the report claims the file was published.
    assert "as false" in note
    assert "/workspace" in note
    assert "assign a publish task with output_paths" in note


def test_the_publisher_note_does_not_claim_the_manifest_was_lost() -> None:
    """A publisher refused on one extra path may still have written every
    declared path successfully."""
    note = render_denial_escalation([("undeclared", "/outputs/extra.md")])

    assert "manifest does not cover" in note
    assert "Nothing it produced" not in note


def test_no_denials_means_no_note() -> None:
    assert render_denial_escalation([]) == ""


# ── the wiring ───────────────────────────────────────────────────────────────


def _spec(**overrides):
    runtime = SwarmSubagentRuntime(
        sandbox_mode="native",
        shared_workspace_dir=None,
        worktree_root=None,
        **overrides,
    )
    return build_swarm_session_runtime_spec(
        runtime, session_name="researcher", task_id="run-1",
    )


def _item() -> SubTask:
    return SubTask(question="extract the comparables", role_id="researcher")


def _result(final_content: str):
    return SimpleNamespace(
        final_content=final_content,
        messages=[],
        metadata={},
        stopped_by="final_answer",
    )


def test_a_refusal_inside_the_task_reaches_the_coordinators_report() -> None:
    """End to end through the real spec: the bus opens ``context_setup`` around
    the loop and calls ``result_adapter`` AFTER it exits, so the log cannot be
    read back off the contextvar and has to be handed over explicitly."""
    spec = _spec()
    item = _item()

    with spec.context_setup("job-1", item):
        # What the sub-agent does mid-task, obeying the user's own instruction.
        assert output_write_error("/outputs/answer.md") is not None

    adapted = asyncio.run(
        spec.result_adapter(_result("Done — wrote /outputs/answer.md."), "job-1", item)
    )

    assert adapted.final_content.startswith("[BLOCKED WRITE:")
    assert "is not the publisher" in adapted.final_content
    # The claim it contradicts is still there for the coordinator to compare.
    assert "Done — wrote /outputs/answer.md." in adapted.final_content
    assert adapted.metadata["blocked_output_writes"] == [
        ["not_publisher", "/outputs/answer.md"]
    ]


def test_an_unverifiable_access_is_not_metadata_labeled_as_a_write() -> None:
    spec = _spec()
    item = _item()

    with spec.context_setup("job-1", item):
        assert (
            bash_output_write_error(
                "python3 unknown_reader.py /outputs/answer.md",
            )
            is not None
        )

    adapted = asyncio.run(
        spec.result_adapter(_result("Could not inspect the answer."), "job-1", item)
    )

    assert adapted.final_content.startswith("[BLOCKED OUTPUTS ACCESS:")
    assert "blocked_output_writes" not in adapted.metadata
    assert adapted.metadata["unverifiable_output_accesses"] == [
        ["unverifiable", "/outputs/answer.md"]
    ]


def test_a_clean_task_report_is_passed_through_unchanged() -> None:
    spec = _spec()
    item = _item()

    with spec.context_setup("job-1", item):
        pass

    adapted = asyncio.run(
        spec.result_adapter(_result("Comparables in /workspace/comps.csv."), "job-1", item)
    )

    assert adapted.final_content == "Comparables in /workspace/comps.csv."
    assert "blocked_output_writes" not in adapted.metadata


def test_one_tasks_refusal_does_not_follow_the_next_one() -> None:
    """Sessions are reusable, so a per-job log that leaked would attach a stale
    refusal to a later task's report and send the coordinator after a
    deliverable that is already published."""
    spec = _spec()
    item = _item()

    with spec.context_setup("job-1", item):
        assert output_write_error("/outputs/answer.md") is not None
    first = asyncio.run(spec.result_adapter(_result("first"), "job-1", item))

    with spec.context_setup("job-2", item):
        pass
    second = asyncio.run(spec.result_adapter(_result("second"), "job-2", item))

    assert first.final_content.startswith("[BLOCKED WRITE:")
    assert second.final_content == "second"


def test_an_abandoned_job_log_cannot_grow_without_bound() -> None:
    """``result_adapter`` never runs for a cancelled or timed-out job, so its
    log is dropped by age instead of on drain."""
    from workflows.agent_team.subagent_runtime import _MAX_TRACKED_DENIAL_JOBS

    spec = _spec()
    item = _item()

    with spec.context_setup("job-abandoned", item):
        assert output_write_error("/outputs/answer.md") is not None
    for index in range(_MAX_TRACKED_DENIAL_JOBS * 2):
        with spec.context_setup(f"job-{index}", item):
            pass  # no adapter call either: every one of these died mid-flight

    # Its log was evicted, so a very late adapter call gets no note rather than
    # a note from a log the store was never allowed to release.
    stale = asyncio.run(
        spec.result_adapter(_result("late"), "job-abandoned", item)
    )
    assert stale.final_content == "late"

    # The newest job is unaffected.
    with spec.context_setup("job-fresh", item):
        assert output_write_error("/outputs/answer.md") is not None
    fresh = asyncio.run(spec.result_adapter(_result("now"), "job-fresh", item))
    assert fresh.final_content.startswith("[BLOCKED WRITE:")
