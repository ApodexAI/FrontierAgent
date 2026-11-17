"""Crash, race, authorization and accounting contracts for durable tasks."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import pytest

from frontier_agent.superagent import Budget, Store
from frontier_agent.superagent.models import Conflict, Fenced


@pytest.fixture
def clock():
    return [1000.0]


@pytest.fixture
def store(tmp_path, clock):
    with Store(tmp_path, clock=lambda: clock[0]) as value:
        yield value


def submit(store, **kwargs):
    return store.submit("Investigate and deliver", request_key="request-1", **kwargs)


def test_request_is_durable_and_conflicting_reuse_rejected(tmp_path):
    with Store(tmp_path) as store:
        task = submit(store)
        store.add_input(task.id, "Preserve compatibility", request_key="message-1")
    with Store(tmp_path) as store:
        assert submit(store).id == task.id
        assert store.context(task.id)["inputs"][0]["text"] == "Preserve compatibility"
        store.add_input(task.id, "Preserve compatibility", request_key="message-1")
        assert store.get(task.id).revision == 2
        with pytest.raises(Conflict):
            store.submit("Different work", request_key="request-1")
        with pytest.raises(Conflict):
            store.add_input(task.id, "Different instruction", request_key="message-1")


def test_simultaneous_submit_and_claim_have_single_winner(tmp_path):
    with Store(tmp_path):
        pass

    def submit_and_claim(_):
        with Store(tmp_path) as store:
            return submit(store).id, store.claim("service")

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(submit_and_claim, range(4)))
    assert len({task_id for task_id, _ in results}) == 1
    assert sum(attempt is not None for _, attempt in results) == 1


def test_cancel_fences_tools_and_does_not_allow_revival(store):
    task = submit(store)
    assignment = store.claim("owner")
    store.request_stop(task.id, cancel=True)
    with pytest.raises(Fenced):
        store.exchange(assignment.id, "owner", "tool", "tool_start", {"operation_id": "x"})
    assert store.claim("owner") is None
    store.finish(assignment.id, "owner", complete=True, result="late", revision=1)
    assert store.get(task.id).status == "cancelled"
    with pytest.raises(Conflict):
        store.add_input(task.id, "wake", request_key="later")
    with pytest.raises(Conflict):
        store.resume(task.id)


def test_expired_lease_and_old_assignment_cannot_write(store, clock):
    task = submit(store)
    first = store.claim("owner", lease_seconds=10)
    clock[0] += 11
    with pytest.raises(Fenced):
        store.renew(first.id, "owner")
    store.finish(first.id, "owner", recovered=True)
    second = store.claim("new")
    with pytest.raises(Fenced):
        store.exchange(first.id, "owner", "stale", "checkpoint", {"actor": "main"})
    with pytest.raises(Fenced):
        store.finish(first.id, "owner", complete=True, revision=1)
    assert store.get(task.id).status == "running"
    assert second.id != first.id


def test_unconfirmed_side_effect_requires_human_reconciliation(store):
    task = submit(store)
    assignment = store.claim("owner")
    store.exchange(
        assignment.id,
        "owner",
        "intent",
        "tool_start",
        {"operation_id": "publish", "command": "external write"},
    )
    store.finish(assignment.id, "owner", recovered=True)
    assert store.get(task.id).status == "blocked"
    assert store.claim("owner") is None
    with pytest.raises(Conflict, match="reconcile"):
        store.resume(task.id)
    store.resume(task.id, reconciliation="Confirmed external write succeeded; do not repeat")
    assert not store.unresolved(task.id)
    assert store.context(task.id)["reconciliations"]


def test_checkpoint_and_confirmed_result_survive_recovery(store, clock):
    task = submit(store)
    assignment = store.claim("owner")
    store.exchange(assignment.id, "owner", "a", "tool_start", {"operation_id": "edit"})
    store.exchange(
        assignment.id, "owner", "b", "tool_end", {"operation_id": "edit", "result": "saved"}
    )
    store.exchange(
        assignment.id, "owner", "c", "checkpoint", {"actor": "main", "messages": ["evidence"]}
    )
    clock[0] += 12
    store.finish(assignment.id, "owner", recovered=True)
    assert store.get(task.id).seconds_used == 12
    assert store.context(task.id)["checkpoints"][0]["messages"] == ["evidence"]
    assert store.claim("new").ticket_id == task.id


def test_rpc_replay_does_not_double_charge_and_reuse_is_checked(store):
    task = submit(store, budget=Budget(model_calls=1))
    assignment = store.claim("owner")
    store.exchange(assignment.id, "owner", "same", "model_call", {})
    store.exchange(assignment.id, "owner", "same", "model_call", {})
    assert store.get(task.id).calls_used == 1
    with pytest.raises(Conflict):
        store.exchange(assignment.id, "owner", "same", "usage", {"tokens": 1})
    with pytest.raises(Fenced, match="budget"):
        store.exchange(assignment.id, "owner", "next", "model_call", {})
    # Using the last permitted call can still produce a valid final result.
    store.finish(assignment.id, "owner", result="done", complete=True, revision=1)
    assert store.get(task.id).status == "ready"


def test_wall_budget_does_not_reset_on_resume(store, clock):
    task = submit(store, budget=Budget(seconds=10))
    assignment = store.claim("owner")
    clock[0] += 4
    store.request_stop(task.id)
    store.finish(assignment.id, "owner")
    store.resume(task.id)
    second = store.claim("owner")
    assert second.deadline == clock[0] + 6
    clock[0] += 7
    store.finish(second.id, "owner", complete=True, result="late", revision=1)
    assert store.get(task.id).status == "paused"
    assert store.get(task.id).seconds_used == 10
    with pytest.raises(Conflict, match="Budget"):
        store.resume(task.id)
    store.resume(task.id, budget=Budget(seconds=20))


def test_final_result_racing_with_new_input_does_not_drop_input(store):
    task = submit(store)
    assignment = store.claim("owner")
    store.add_input(task.id, "Also cover the new constraint", request_key="late")
    store.finish(assignment.id, "owner", result="old result", complete=True, revision=1)
    assert store.get(task.id).status == "queued"
    assert store.context(task.id)["inputs"][0]["consumed_by"] is None


def test_inputs_are_not_consumed_by_reading_and_stale_results_cannot_be_accepted(store):
    task = submit(store)
    store.add_input(task.id, "constraint", request_key="input")
    assignment = store.claim("owner")
    incoming = store.exchange(assignment.id, "owner", "read", "inputs", {})
    assert len(incoming["inputs"]) == 1
    assert store.context(task.id)["inputs"][0]["consumed_by"] is None
    store.exchange(
        assignment.id, "owner", "ack", "consume", {"seqs": [incoming["inputs"][0]["seq"]]}
    )
    store.finish(assignment.id, "owner", result="result", complete=True, revision=2)
    assert store.get(task.id).status == "ready"
    store.accept(task.id)
    assert store.get(task.id).status == "completed"


def test_last_allowed_assignment_can_finish(store):
    task = submit(store, budget=Budget(attempts=1))
    assignment = store.claim("owner")
    store.finish(assignment.id, "owner", result="done", complete=True, revision=1)
    assert store.get(task.id).status == "ready"


@pytest.mark.parametrize("recovered", [True, False])
def test_durable_final_report_survives_crash_or_slow_cleanup(store, clock, recovered):
    task = submit(store)
    assignment = store.claim("owner")
    clock[0] += 2
    store.exchange(
        assignment.id,
        "owner",
        "report",
        "report",
        {"result": "already done", "complete": True, "revision": 1},
    )
    clock[0] += 100
    store.finish(assignment.id, "owner", recovered=recovered)
    assert store.get(task.id).status == "ready"
    assert store.get(task.id).result == "already done"
    assert store.get(task.id).seconds_used == 2


def test_transaction_failure_preserves_unknown_operation(store):
    task = submit(store, budget=Budget(attempts=1))
    assignment = store.claim("owner")
    store.exchange(assignment.id, "owner", "a", "job_start", {"operation_id": "job"})
    store.finish(assignment.id, "owner", recovered=True)
    with pytest.raises(Conflict, match="Budget"):
        store.resume(task.id, reconciliation="checked")
    assert store.unresolved(task.id)  # reconciliation rolled back with the failed command


def test_inputs_are_snapshotted_before_queue_visibility(store, tmp_path):
    source = tmp_path / "notes.txt"
    source.write_text("original")
    task = store.submit("read", request_key="input", inputs=[source])
    source.write_text("changed")
    assert (store.root / "tasks" / task.id / "inputs" / "notes.txt").read_text() == "original"
    assert store.submit("read", request_key="input", inputs=[source]).id == task.id
    assert store.claim("owner") is not None


def test_bad_input_cannot_leave_a_runnable_task(store, tmp_path):
    source = tmp_path / "link"
    source.symlink_to(tmp_path / "missing")
    with pytest.raises(ValueError, match="symlink"):
        store.submit("read", request_key="bad", inputs=[source])
    assert store.list() == []


@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_invalid_budgets_rejected(value):
    with pytest.raises(ValueError):
        Budget(seconds=value)
