"""The path gate is a denial, not a detour (handoff item 4a).

Three ways a refused write reached the host anyway, or a refusal was never
asked for:

* ``write_file`` / ``file_editor_create`` fall back to the sandbox when
  ``_path_auth`` refuses a local write, and for a ``CurrentSandbox`` that
  fallback is an ``open()`` in this process — as root, in container mode;
* a path interpolated unquoted into a sandbox shell command runs whatever it
  contains, without passing ``assess_bash_command``;
* a workspace root arriving through ExecutionScope metadata was trusted
  whatever it named, so ``{"workspace_root": "/etc"}`` authorized ``/etc``.

Nothing here executes a dangerous command: the shell-injection cases assert on
the command STRING that would have been sent.
"""

from __future__ import annotations

import asyncio
import importlib
from pathlib import Path

import pytest

from frontier_agent.core.execution_context import (
    ExecutionScope,
    reset_current_execution_scope,
    set_current_execution_scope,
)
from plugins.tools import _sandbox
from plugins.tools._path_auth import _configured_workspace_root, _is_path_allowed


class _scoped:
    def __init__(self, **metadata: object) -> None:
        self.metadata = metadata

    def __enter__(self) -> None:
        self._token = set_current_execution_scope(
            ExecutionScope(task_id="t", metadata=dict(self.metadata)),
        )

    def __exit__(self, *exc: object) -> None:
        reset_current_execution_scope(self._token)


# ── the in-process write branch must honour the gate ─────────────────────


@pytest.fixture
def container(tmp_path, monkeypatch):
    """A CurrentSandbox over a workspace, as container/native mode has."""
    workspace = tmp_path / "ws"
    workspace.mkdir()
    monkeypatch.setenv("SANDBOX_BACKEND", "native")
    monkeypatch.setenv("FRONTIER_AGENT_WORKSPACE_DIR", str(workspace))
    sandbox = _sandbox.CurrentSandbox(workspace)
    token = _sandbox.set_task_sandbox(sandbox)
    try:
        yield workspace
    finally:
        _sandbox.clear_task_sandbox(token)


def test_sandbox_write_refuses_what_the_path_gate_refused(container) -> None:
    target = container.parent / "not_authorized" / "x.txt"
    with _scoped(workspace_root=str(container)):
        assert not _is_path_allowed(str(target), write_access=True)[0]
        ok, reason = _sandbox.sandbox_write_file(
            _sandbox.get_existing_sandbox(), str(target), "payload",
        )
    assert not ok
    assert reason
    assert not target.exists(), "a refused path was written to the host"


def test_write_file_does_not_route_a_refused_path_through_the_sandbox(container) -> None:
    write_file = importlib.import_module("plugins.tools.write_file").write_file

    target = container.parent / "not_authorized" / "x.txt"
    with _scoped(workspace_root=str(container)):
        out = asyncio.run(write_file.func(path=str(target), content="payload"))
    assert "File written" not in out or str(target) not in out
    assert not target.exists()


def test_file_editor_create_does_not_route_a_refused_path_through_the_sandbox(
    container,
) -> None:
    editor = importlib.import_module("plugins.tools.file_editor")

    target = container.parent / "not_authorized" / "y.txt"
    with _scoped(workspace_root=str(container)):
        out = asyncio.run(
            editor.file_editor_create.func(path=str(target), content="payload"),
        )
    assert "File created" not in out
    assert not target.exists()


def test_an_authorized_write_still_succeeds_in_process(container) -> None:
    target = container / "report.md"
    with _scoped(workspace_root=str(container)):
        ok, reason = _sandbox.sandbox_write_file(
            _sandbox.get_existing_sandbox(), str(target), "body",
        )
    assert ok, reason
    assert target.read_text() == "body"


def test_a_remote_backend_is_not_gated_on_host_authorization() -> None:
    """The in-process branch is the only one that writes here; an E2B write
    targets a path that does not exist on this host and must not be judged by
    this host's gate."""
    written: dict[str, str] = {}

    class FakeFiles:
        def write(self, path: str, content: str) -> None:
            written[path] = content

    remote = type("Remote", (), {"files": FakeFiles()})()
    ok, reason = _sandbox.sandbox_write_file(remote, "/home/user/out.md", "body")
    assert ok, reason
    assert written == {"/home/user/out.md": "body"}


# ── paths interpolated into sandbox shell commands ───────────────────────


_INJECTION = "/workspace/a.txt; touch /tmp/pwned-by-path"


@pytest.fixture
def recorded(monkeypatch):
    """Record the command strings tools would run, and run none of them."""
    seen: list[str] = []

    class Result:
        exit_code = 0
        stdout = "FILE"
        stderr = ""

    async def fake_run(sandbox, command, **kwargs):
        seen.append(command)
        return Result()

    async def fake_sandbox(*_a, **_k):
        return object()

    async def fake_write(*_a, **_k):
        return True, ""

    for module in ("file_editor", "write_file", "read_file"):
        # ``plugins.tools.__init__`` re-exports the Tool objects under the same
        # names as their modules, so the module has to be fetched explicitly.
        mod = importlib.import_module(f"plugins.tools.{module}")
        for name, fn in (
            ("arun_sandbox_cmd", fake_run),
            ("aget_sandbox", fake_sandbox),
            ("asandbox_write_file", fake_write),
        ):
            if hasattr(mod, name):
                monkeypatch.setattr(mod, name, fn)
    return seen


def _assert_no_second_command(commands: list[str]) -> None:
    for command in commands:
        payload = command.split("touch", 1)
        assert len(payload) == 1 or "'" in command, (
            f"path reached the shell unquoted: {command!r}"
        )
        # The injected separator must sit inside quotes, never as shell syntax.
        for marker in ("; touch", "&& touch", "$(", "`"):
            if marker in command:
                before = command.split(marker, 1)[0]
                assert before.count("'") % 2 == 1, (
                    f"unquoted shell metacharacter in: {command!r}"
                )


def test_file_editor_view_quotes_the_path(recorded) -> None:
    editor = importlib.import_module("plugins.tools.file_editor")

    asyncio.run(editor.file_editor_view.func(path=_INJECTION))
    assert recorded
    _assert_no_second_command(recorded)


def test_file_editor_create_quotes_the_path(recorded) -> None:
    editor = importlib.import_module("plugins.tools.file_editor")

    asyncio.run(editor.file_editor_create.func(path=_INJECTION, content="x"))
    _assert_no_second_command(recorded)


def test_file_editor_str_replace_quotes_the_path(recorded) -> None:
    editor = importlib.import_module("plugins.tools.file_editor")

    asyncio.run(
        editor.file_editor_str_replace.func(path=_INJECTION, old_str="a", new_str="b"),
    )
    _assert_no_second_command(recorded)


def test_write_file_quotes_the_parent_directory(recorded, monkeypatch) -> None:
    module = importlib.import_module("plugins.tools.write_file")

    monkeypatch.setattr(module, "sandbox_available", lambda: True)
    monkeypatch.setattr(module, "resolve_sandbox_mode", lambda *a, **k: "container")
    asyncio.run(module.write_file.func(path=_INJECTION, content="x"))
    _assert_no_second_command(recorded)


def test_injection_marker_was_never_created() -> None:
    assert not Path("/tmp/pwned-by-path").exists()


# ── scope metadata cannot name a system directory as the workspace ───────


@pytest.mark.parametrize("root", [
    "/", "/etc", "/usr", "/var", "/opt", "/home", "/root", "/boot",
    "/usr/local", "/usr/bin", "/etc/cron.d",
])
@pytest.mark.parametrize("key", ["workspace_root", "coding_workspace_root"])
def test_a_system_directory_grants_nothing(root: str, key: str) -> None:
    if not Path(root).is_dir():
        pytest.skip(f"{root} does not exist here")
    with _scoped(**{key: root}):
        assert _configured_workspace_root() is None
        assert not _is_path_allowed(f"{root.rstrip('/')}/probe", write_access=True)[0]
    with _scoped(**{key: root}):
        assert not _is_path_allowed("/etc/hostname")[0]


def test_the_environment_variable_is_checked_the_same_way(monkeypatch) -> None:
    monkeypatch.setenv("CODING_WORKSPACE_ROOT", "/etc")
    with _scoped():
        assert _configured_workspace_root() is None


@pytest.mark.parametrize("key", ["workspace_root", "coding_workspace_root"])
def test_a_real_run_directory_is_still_accepted(tmp_path, key: str) -> None:
    # The shape a benchmark runner passes, and macOS $TMPDIR / a container
    # volume: deep under a protected-looking prefix, but the run's own.
    run_dir = tmp_path / "var" / "folders" / "ab" / "T" / "run-17" / "workspace"
    run_dir.mkdir(parents=True)
    with _scoped(**{key: str(run_dir)}):
        assert _configured_workspace_root() == run_dir
        assert _is_path_allowed(str(run_dir / "src.py"), write_access=True)[0]


def test_an_unset_root_is_unchanged(monkeypatch) -> None:
    monkeypatch.delenv("CODING_WORKSPACE_ROOT", raising=False)
    with _scoped():
        assert _configured_workspace_root() is None
