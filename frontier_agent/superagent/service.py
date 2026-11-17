"""Single-host scheduler with isolated, fenced workers and crash recovery."""

from __future__ import annotations

import asyncio
import contextlib
import fcntl
import json
import os
import signal
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Protocol

from frontier_agent.superagent.models import Assignment, Conflict, Fenced, Json
from frontier_agent.superagent.store import Store


@contextlib.contextmanager
def service_lock(root: Path) -> Iterator[None]:
    """The lock, not a PID file or a lease timeout, excludes another local scheduler."""
    with (root / "service.lock").open("a+") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise Conflict("A task service already owns this state directory") from exc
        stream.seek(0)
        stream.truncate()
        stream.write(str(os.getpid()))
        stream.flush()
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


class Executor(Protocol):
    async def check(self) -> None: ...
    async def start(
        self, assignment: Assignment, directory: Path
    ) -> asyncio.subprocess.Process: ...
    async def stop(self, assignment: Assignment) -> None: ...


class DockerExecutor:
    """Each attempt owns one named container; only its task's files are mounted."""

    def __init__(
        self,
        image: str = "apodex:local",
        *,
        env_names: tuple[str, ...] = (),
        memory: str = "4g",
        cpus: str = "2",
    ) -> None:
        self.image = image
        self.env_names = env_names
        self.memory = memory
        self.cpus = cpus

    @staticmethod
    def name(assignment: Assignment) -> str:
        return f"frontier-task-{assignment.id}"

    async def _docker(self, *args: str) -> str:
        process = await asyncio.create_subprocess_exec(
            "docker", *args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=30)
        except BaseException:
            with contextlib.suppress(ProcessLookupError):
                process.kill()
            await process.wait()
            raise
        if process.returncode:
            raise RuntimeError(stderr.decode(errors="replace").strip() or "Docker command failed")
        return stdout.decode().strip()

    async def check(self) -> None:
        await self._docker("info", "--format", "{{.ServerVersion}}")
        await self._docker("image", "inspect", self.image, "--format", "{{.Id}}")

    async def start(self, assignment: Assignment, directory: Path) -> asyncio.subprocess.Process:
        name = self.name(assignment)
        args = [
            "create",
            "--name",
            name,
            "--init",
            "--interactive",
            "--memory",
            self.memory,
            "--cpus",
            self.cpus,
            "--pids-limit",
            "256",
            "--cap-drop",
            "ALL",
            "--cap-add",
            "SETUID",
            "--cap-add",
            "SETGID",
            "--security-opt",
            "no-new-privileges",
            "--workdir",
            "/app",
            "--entrypoint",
            "python",
            "--label",
            f"frontier.assignment={assignment.id}",
        ]
        for folder in ("workspace", "outputs", "inputs"):
            path = directory / folder
            path.mkdir(parents=True, exist_ok=True)
            # The bundled image executes tools as an unprivileged uid.
            path.chmod(0o777 if folder != "inputs" else 0o755)
            mount = f"type=bind,source={path},target=/{folder}"
            args.extend(["--mount", mount + (",readonly" if folder == "inputs" else "")])
        for key, value in {
            "SANDBOX_BACKEND": "container",
            "APODEX_IN_CONTAINER": "1",
            "PYTHONPATH": "/app",
            "FRONTIER_AGENT_WORKSPACE_DIR": "/workspace",
            "FRONTIER_AGENT_OUTPUTS_DIR": "/outputs",
            "FRONTIER_AGENT_INPUTS_DIR": "/inputs",
        }.items():
            args.extend(["--env", f"{key}={value}"])
        for key in self.env_names:
            if key in os.environ:
                args.extend(["--env", key])
        args.extend([self.image, "-m", "frontier_agent.superagent.worker"])
        await self._docker(*args)
        return await asyncio.create_subprocess_exec(
            "docker",
            "start",
            "--attach",
            "--interactive",
            name,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            limit=8 * 1024 * 1024,
        )

    async def stop(self, assignment: Assignment) -> None:
        # A successful list distinguishes "absent" from an unreachable daemon.
        name = self.name(assignment)
        found = await self._docker("ps", "--all", "--quiet", "--filter", f"name=^/{name}$")
        if found:
            await self._docker("rm", "--force", name)


class Service:
    """Durable queue consumer. CLI/TUI commands use the same transactional store."""

    def __init__(
        self,
        store: Store,
        executor: Executor,
        *,
        concurrency: int = 2,
        poll_seconds: float = 0.25,
        lease_seconds: float = 30,
    ) -> None:
        if concurrency < 1 or poll_seconds <= 0 or lease_seconds <= poll_seconds * 2:
            raise ValueError("Invalid concurrency or heartbeat configuration")
        self.store = store
        self.executor = executor
        self.concurrency = concurrency
        self.poll_seconds = poll_seconds
        self.lease_seconds = lease_seconds
        self.owner = uuid.uuid4().hex
        self.stopping = asyncio.Event()
        self.running: dict[str, asyncio.Task[None]] = {}

    async def recover(self) -> None:
        """Stop named old environments before any reassignment or stale-result handling."""
        for assignment in self.store.active():
            await self.executor.stop(assignment)
            self.store.finish(
                assignment.id,
                assignment.owner,
                recovered=True,
                reason="Service restarted; continue from persisted context",
            )

    async def run(self) -> None:
        with service_lock(self.store.root):
            await self.executor.check()
            await self.recover()
            (self.store.root / "stop.request").unlink(missing_ok=True)
            (self.store.root / "service.ready").write_text(str(os.getpid()))
            try:
                while not self.stopping.is_set():
                    if (self.store.root / "stop.request").exists():
                        self.stopping.set()
                        break
                    for key, task in list(self.running.items()):
                        if task.done():
                            del self.running[key]
                            task.result()  # Do not silently launch over an unconfirmed stop.
                    while len(self.running) < self.concurrency:
                        assignment = self.store.claim(self.owner, lease_seconds=self.lease_seconds)
                        if assignment is None:
                            break
                        self.running[assignment.id] = asyncio.create_task(self._execute(assignment))
                    with contextlib.suppress(TimeoutError):
                        await asyncio.wait_for(self.stopping.wait(), self.poll_seconds)
            finally:
                self.stopping.set()
                # Each executor loop observes shutdown, kills its environment, then
                # persists a recoverable end. No new assignment can start meanwhile.
                await asyncio.gather(*self.running.values())
                (self.store.root / "service.ready").unlink(missing_ok=True)

    async def _execute(self, assignment: Assignment) -> None:
        process: asyncio.subprocess.Process | None = None
        reader: asyncio.Task[Json | None] | None = None
        logger: asyncio.Task[None] | None = None
        report: Json | None = None
        reason = "Worker exited without a final report"
        recovered = False
        directory = self.store.root / "tasks" / assignment.ticket_id
        directory.mkdir(parents=True, exist_ok=True)
        try:
            self.store.renew(assignment.id, self.owner, lease_seconds=self.lease_seconds)
            process = await self.executor.start(assignment, directory)
            assert process.stdin is not None
            process.stdin.write(
                (json.dumps(self.store.context(assignment.ticket_id)) + "\n").encode()
            )
            await process.stdin.drain()
            reader = asyncio.create_task(self._exchange(process, assignment))
            logger = asyncio.create_task(self._logs(process, directory / f"{assignment.id}.log"))
            while True:
                if self.stopping.is_set():
                    recovered, reason = True, "Service stopped; resume from recorded context"
                    break
                self.store.renew(assignment.id, self.owner, lease_seconds=self.lease_seconds)
                if reader.done():
                    report = reader.result()
                    break
                await asyncio.sleep(self.poll_seconds)
        except (Fenced, BrokenPipeError, ConnectionResetError) as exc:
            reason = str(exc)
        except Exception as exc:
            reason = f"{type(exc).__name__}: {exc}"
        finally:
            # Failure here must leave the assignment active. Restart will retry
            # stopping it, rather than risk two executors acting on one task.
            await self.executor.stop(assignment)
            if process is not None:
                if process.returncode is None:
                    with contextlib.suppress(ProcessLookupError):
                        process.kill()
                await process.wait()
            pending = [task for task in (reader, logger) if task is not None]
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
            self.store.finish(
                assignment.id,
                self.owner,
                result=str((report or {}).get("result", "")),
                complete=bool((report or {}).get("complete", False)),
                revision=(report or {}).get("revision"),
                reason=str((report or {}).get("reason") or reason),
                recovered=recovered,
            )

    async def _exchange(
        self, process: asyncio.subprocess.Process, assignment: Assignment
    ) -> Json | None:
        assert process.stdout is not None and process.stdin is not None
        while raw := await process.stdout.readline():
            request = json.loads(raw)
            key, kind, payload = request["key"], request["kind"], request["payload"]
            try:
                response = self.store.exchange(assignment.id, self.owner, key, kind, payload)
            except (Conflict, ValueError, KeyError) as exc:
                process.stdin.write(
                    (json.dumps({"key": key, "ok": False, "error": str(exc)}) + "\n").encode()
                )
                await process.stdin.drain()
                raise
            process.stdin.write(
                (json.dumps({"key": key, "ok": True, "payload": response}) + "\n").encode()
            )
            await process.stdin.drain()
            if kind == "report":
                return payload
        return None

    @staticmethod
    async def _logs(process: asyncio.subprocess.Process, path: Path) -> None:
        assert process.stderr is not None
        # Bound disk usage while always draining stderr so workers cannot deadlock.
        remaining = 8 * 1024 * 1024
        with path.open("wb") as stream:
            while data := await process.stderr.read(65536):
                if remaining > 0:
                    stream.write(data[:remaining])
                    remaining -= len(data)


async def serve(store: Store, executor: Executor, *, concurrency: int = 2) -> None:
    service = Service(store, executor, concurrency=concurrency)
    loop = asyncio.get_running_loop()
    for signum in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(signum, service.stopping.set)
    try:
        await service.run()
    finally:
        for signum in (signal.SIGINT, signal.SIGTERM):
            loop.remove_signal_handler(signum)
