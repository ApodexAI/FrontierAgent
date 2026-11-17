"""Real subprocess protocol tests, without a model endpoint or Docker dependency."""

from __future__ import annotations

import asyncio
import contextlib
import json
import sys

import pytest

from frontier_agent.superagent import Store
from frontier_agent.superagent.models import Conflict
from frontier_agent.superagent.service import DockerExecutor, Service, service_lock

_WORKER = r"""
import json, sys, time
context = json.loads(sys.stdin.readline())
def rpc(key, kind, payload):
    print(json.dumps(dict(key=key, kind=kind, payload=payload)), flush=True)
    reply = json.loads(sys.stdin.readline())
    if not reply["ok"]: sys.exit(2)
rpc("call", "model_call", {})
rpc("checkpoint", "checkpoint", {"actor":"main", "messages":["persisted work"]})
if context["ticket"]["goal"] == "hold":
    rpc("intent", "tool_start", {"operation_id":"uncertain"})
    time.sleep(60)
else:
    time.sleep(0.2)
    rpc("ack", "consume", {"seqs":[item["seq"] for item in context["inputs"]]})
    rpc("report", "report", {"result":"verified result", "complete":True,
                             "revision":context["ticket"]["revision"]})
"""


class TestExecutor:
    __test__ = False

    def __init__(self):
        self.processes = {}
        self.started = []
        self.stopped = []

    async def check(self):
        pass

    async def start(self, assignment, directory):
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-c",
            _WORKER,
            cwd=directory,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        self.processes[assignment.id] = process
        self.started.append((assignment.id, directory))
        return process

    async def stop(self, assignment):
        self.stopped.append(assignment.id)
        process = self.processes.get(assignment.id)
        if process and process.returncode is None:
            with contextlib.suppress(ProcessLookupError):
                process.kill()
            await process.wait()


async def until(predicate, timeout=5):
    async with asyncio.timeout(timeout):
        while not predicate():
            await asyncio.sleep(0.01)


async def test_two_tasks_parallel_and_ui_disconnection(tmp_path):
    with Store(tmp_path) as store:
        first = store.submit("one", request_key="one")
        second = store.submit("two", request_key="two")
        executor = TestExecutor()
        service = Service(store, executor, concurrency=2, poll_seconds=0.01)
        running = asyncio.create_task(service.run())
        try:
            await until(lambda: len(store.active()) == 2)
            # A new client can close without owning or cancelling either worker.
            with Store(tmp_path) as client:
                assert len(client.list()) == 2
            await until(lambda: all(task.status == "ready" for task in store.list()))
            assert store.get(first.id).result == store.get(second.id).result == "verified result"
            assert len({directory for _, directory in executor.started}) == 2
            assert store.get(first.id).calls_used == 1
        finally:
            service.stopping.set()
            await running


async def test_cancel_kills_worker_tree_and_preserves_task(tmp_path):
    with Store(tmp_path) as store:
        task = store.submit("hold", request_key="one")
        executor = TestExecutor()
        service = Service(store, executor, poll_seconds=0.01)
        running = asyncio.create_task(service.run())
        try:
            await until(lambda: bool(store.unresolved(task.id)))
            with Store(tmp_path) as client:
                client.request_stop(task.id, cancel=True)
            await until(lambda: store.get(task.id).status == "cancelled")
            assert executor.stopped
            assert all(process.returncode is not None for process in executor.processes.values())
        finally:
            service.stopping.set()
            await running


async def test_recovery_stops_old_environment_before_requeue(tmp_path):
    with Store(tmp_path) as store:
        task = store.submit("hold", request_key="one")
        assignment = store.claim("dead-service")
        store.exchange(
            assignment.id, "dead-service", "intent", "tool_start", {"operation_id": "write"}
        )
        executor = TestExecutor()
        service = Service(store, executor)
        await service.recover()
        assert executor.stopped == [assignment.id]
        assert store.get(task.id).status == "blocked"
        assert store.claim("replacement") is None


async def test_failed_environment_cleanup_prevents_reassignment(tmp_path):
    class Unreachable(TestExecutor):
        async def stop(self, assignment):
            raise RuntimeError("daemon unreachable")

    with Store(tmp_path) as store:
        task = store.submit("one", request_key="one")
        store.claim("old")
        with pytest.raises(RuntimeError, match="unreachable"):
            await Service(store, Unreachable()).recover()
        assert store.get(task.id).status == "running"
        assert store.claim("replacement") is None


def test_only_one_local_service_can_own_store(tmp_path):
    with service_lock(tmp_path), pytest.raises(Conflict), service_lock(tmp_path):
        pass


async def test_docker_mounts_only_one_task_and_keeps_secrets_off_argv(tmp_path, monkeypatch):
    commands = []
    executor = DockerExecutor(env_names=("OPENAI_API_KEY",))

    async def docker(*args):
        commands.append(list(args))
        return "created"

    async def launch(*args, **kwargs):
        commands.append(list(args))
        return object()

    monkeypatch.setenv("OPENAI_API_KEY", "do-not-put-this-on-argv")
    monkeypatch.setattr(executor, "_docker", docker)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", launch)
    with Store(tmp_path) as store:
        task = store.submit("one", request_key="one")
        assignment = store.claim("service")
        directory = tmp_path / "tasks" / task.id
        await executor.start(assignment, directory)
    command = commands[0]
    assert "do-not-put-this-on-argv" not in json.dumps(commands)
    mounts = [command[index + 1] for index, arg in enumerate(command) if arg == "--mount"]
    assert len(mounts) == 3
    assert all(f"source={directory}/" in mount for mount in mounts)
    assert "readonly" in mounts[-1]
    assert "tasks.sqlite" not in json.dumps(commands)
    assert "--privileged" not in command
    assert "/var/run/docker.sock" not in json.dumps(commands)


def test_cli_submit_read_resume_errors(tmp_path, capsys):
    from apodex.cli import main

    base = ["tasks", "--state-dir", str(tmp_path)]
    assert main([*base, "submit", "a task", "--request-key", "request"]) == 0
    ticket_id = json.loads(capsys.readouterr().out)["id"]
    assert main([*base, "pause", ticket_id]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "paused"
    assert main([*base, "resume", ticket_id]) == 0
    capsys.readouterr()
    assert main([*base, "accept", ticket_id]) == 1


async def test_dashboard_submits_and_pauses_without_owning_service(tmp_path):
    from textual.widgets import Input

    from apodex.tui.tasks import TaskDashboard

    with Store(tmp_path) as store:
        app = TaskDashboard(store)
        async with app.run_test(size=(120, 40)) as pilot:
            app.query_one(Input).value = "Inspect the project"
            await pilot.click("#submit")
            await pilot.pause()
            assert len(store.list()) == 1
            await pilot.click("#pause")
            assert store.list()[0].status == "paused"
        assert store.list()[0].goal == "Inspect the project"
