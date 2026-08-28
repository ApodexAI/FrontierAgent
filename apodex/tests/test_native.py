from __future__ import annotations

import asyncio
import os
from pathlib import Path

from apodex import cli, docker, sandbox
from apodex.native import prepare_native_runtime
from apodex.sandbox import BWRAP, CONTAINER, NATIVE, Strategy, resolve_strategy
from plugins.tools._sandbox import resolve_runtime_path


def test_native_runtime_keeps_mutable_state_under_workspace(tmp_path) -> None:
    workspace = tmp_path / "project"
    workspace.mkdir()
    env: dict[str, str] = {"APODEX_RUNS_ROOT_PINNED": "1"}

    root = prepare_native_runtime(
        str(workspace), "20260806-120000-react-ab12", environ=env,
    )

    assert root == workspace / ".apodex" / "runtime" / "native"
    assert env["APODEX_IN_NATIVE"] == "1"
    assert env["SANDBOX_BACKEND"] == "native"
    assert env["HOME"] == str(root / "home")
    assert env["UV_CACHE_DIR"] == str(root / "cache" / "uv")
    assert env["PIP_TARGET"] == str(root / "home" / ".local" / "site-packages")
    assert env["PYTHONPATH"].split(os.pathsep, 1)[0] == env["PIP_TARGET"]
    assert env["NPM_CONFIG_CACHE"] == str(root / "cache" / "npm")
    assert env["NPM_CONFIG_PREFIX"] == str(root / "dependencies" / "npm")
    run_workspace = (
        workspace / ".apodex" / "runs"
        / "20260806-120000-react-ab12" / "workspace"
    )
    workspace_link = root / "workspace"
    assert env["FRONTIER_AGENT_WORKSPACE_DIR"] == str(workspace_link)
    assert workspace_link.is_symlink()
    assert workspace_link.resolve() == run_workspace.resolve()
    assert env["APODEX_HOST_WORKSPACE_DIR"] == str(run_workspace)
    assert env["FRONTIER_AGENT_INPUTS_DIR"] == str(
        root / "inputs" / "20260806-120000-react-ab12"
    )
    assert env["FRONTIER_AGENT_OUTPUTS_DIR"] == str(
        workspace / ".apodex" / "runs" / "20260806-120000-react-ab12" / "outputs"
    )
    assert env["APODEX_RUNS_ROOT"] == str(workspace / ".apodex" / "runs")
    assert "APODEX_RUNS_ROOT_PINNED" not in env
    for key in (
        "HOME", "TMPDIR", "XDG_CACHE_HOME", "XDG_CONFIG_HOME",
        "XDG_STATE_HOME", "FRONTIER_AGENT_INPUTS_DIR",
        "FRONTIER_AGENT_OUTPUTS_DIR", "PIP_TARGET",
    ):
        assert Path(env[key]).is_dir()


def test_native_strategy_is_explicitly_not_os_isolated() -> None:
    strategy = Strategy(NATIVE, "test")

    assert not strategy.isolated
    assert "not an OS sandbox" in strategy.describe()


def test_native_runtime_resolves_canonical_mount_aliases(
    tmp_path, monkeypatch,
) -> None:
    workspace = tmp_path / "project"
    workspace.mkdir()
    env = os.environ.copy()
    prepare_native_runtime(
        str(workspace), "20260816-120000-react-ab12", environ=env,
    )

    for key, value in env.items():
        monkeypatch.setenv(key, value)
    assert resolve_runtime_path("/workspace/a.txt") == os.path.join(
        env["FRONTIER_AGENT_WORKSPACE_DIR"], "a.txt",
    )
    assert resolve_runtime_path("/outputs/report.docx") == os.path.join(
        env["FRONTIER_AGENT_OUTPUTS_DIR"], "report.docx",
    )
    assert resolve_runtime_path("/inputs/source.pdf") == os.path.join(
        env["FRONTIER_AGENT_INPUTS_DIR"], "source.pdf",
    )
    assert resolve_runtime_path("/outputs-old/report.docx") == (
        "/outputs-old/report.docx"
    )


def test_read_file_splits_an_image_batch_but_not_a_comma_in_a_filename(
    tmp_path, monkeypatch,
) -> None:
    """The comma-separated form is documented for images only.

    Splitting unconditionally also strips the space after the comma, so
    ``my report, final.pdf`` resolved to a path that does not exist.
    """
    from plugins.tools.read_file import _looks_like_batch

    workspace = tmp_path / "project"
    workspace.mkdir()
    env = os.environ.copy()
    prepare_native_runtime(
        str(workspace), "20260816-120000-react-ab12", environ=env,
    )
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    root = env["FRONTIER_AGENT_WORKSPACE_DIR"]

    assert _looks_like_batch("/workspace/a.png,/workspace/b.jpg") is True
    assert _looks_like_batch("/workspace/a.png, /workspace/b.JPEG") is True
    assert _looks_like_batch("/workspace/my report, final.pdf") is False
    assert _looks_like_batch("/workspace/a.png,/workspace/notes.md") is False
    assert _looks_like_batch("/workspace/one.png") is False

    batch = "/workspace/a.png, /workspace/b.jpg"
    assert ",".join(
        resolve_runtime_path(item.strip()) for item in batch.split(",")
    ) == f"{os.path.join(root, 'a.png')},{os.path.join(root, 'b.jpg')}"
    # The single path keeps its literal name, comma and space included.
    assert resolve_runtime_path("/workspace/my report, final.pdf") == (
        os.path.join(root, "my report, final.pdf")
    )


def test_linux_strategy_resolver_defaults_to_native_without_bwrap_probe(
    monkeypatch,
) -> None:
    monkeypatch.setattr("apodex.sandbox.sys.platform", "linux")
    monkeypatch.delenv("APODEX_SANDBOX", raising=False)
    monkeypatch.delenv("SANDBOX_BACKEND", raising=False)
    monkeypatch.delenv("APODEX_IN_CONTAINER", raising=False)
    monkeypatch.delenv("APODEX_IN_NATIVE", raising=False)
    monkeypatch.setattr(
        "apodex.sandbox._bwrap_usable",
        lambda: (_ for _ in ()).throw(AssertionError("must not probe bwrap")),
    )

    strategy = resolve_strategy()

    assert strategy.name == NATIVE
    assert strategy.reason == "default Linux host runtime"


def test_configured_container_backend_overrides_linux_native_default(
    monkeypatch,
) -> None:
    monkeypatch.setattr("apodex.sandbox.sys.platform", "linux")
    monkeypatch.delenv("APODEX_SANDBOX", raising=False)
    monkeypatch.setenv("SANDBOX_BACKEND", "container")

    strategy = resolve_strategy()

    assert strategy.name == CONTAINER
    assert strategy.reason == "SANDBOX_BACKEND=container"


def test_configured_bwrap_backend_remains_explicit(monkeypatch) -> None:
    monkeypatch.setattr("apodex.sandbox.sys.platform", "linux")
    monkeypatch.delenv("APODEX_SANDBOX", raising=False)
    monkeypatch.setenv("SANDBOX_BACKEND", "bwrap")
    monkeypatch.setattr(
        "apodex.sandbox._bwrap_usable", lambda: (True, "bubblewrap available"),
    )

    strategy = resolve_strategy()

    assert strategy.name == BWRAP
    assert strategy.reason == "SANDBOX_BACKEND=bwrap"


def test_bwrap_sandbox_rebuilds_when_workspace_changes(
    tmp_path, monkeypatch,
) -> None:
    class FakeBwrapSandbox:
        def __init__(self, *, workspace, binds):
            self.workspace = workspace
            self.binds = binds
            self.killed = False

        def kill(self):
            self.killed = True

    first_workspace = tmp_path / "first"
    second_workspace = tmp_path / "second"
    first_workspace.mkdir()
    second_workspace.mkdir()
    monkeypatch.setattr("plugins.tools._sandbox.BwrapSandbox", FakeBwrapSandbox)
    monkeypatch.setattr(sandbox, "_bwrap_sandbox", None)

    first = sandbox._get_bwrap_sandbox(str(first_workspace))
    same = sandbox._get_bwrap_sandbox(str(first_workspace / ".." / "first"))
    second = sandbox._get_bwrap_sandbox(str(second_workspace))

    assert same is first
    assert second is not first
    assert first.killed is True
    assert second.killed is False
    assert second.workspace == str(second_workspace.resolve())
    assert second.binds == ((str(second_workspace.resolve()),) * 2 + (False,),)


def test_macos_falls_back_to_native_when_docker_is_unavailable(
    tmp_path, monkeypatch, capsys,
) -> None:
    prepared: list[tuple[str, str]] = []
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "_load_env", lambda: None)
    monkeypatch.setattr(cli.sys, "platform", "darwin")
    monkeypatch.setattr(
        docker, "docker_available", lambda: (False, "daemon is stopped"),
    )
    monkeypatch.setattr(
        "apodex.native.prepare_native_runtime",
        lambda workspace, session_id: (
            prepared.append((workspace, session_id)) or tmp_path / ".apodex" / "runtime" / "native"
        ),
    )

    result = asyncio.run(cli._amain(["--mode", "invalid", "--no-tui"]))

    assert result == 2
    assert prepared and prepared[0][0] == str(tmp_path)
    stderr = capsys.readouterr().err
    assert "Docker is unavailable" in stderr
    assert "using native mode" in stderr
    assert "not a container or OS sandbox" in stderr


def test_workflow_honors_native_backend_selected_after_config_was_cached(
    monkeypatch,
) -> None:
    """The workflow must not resurrect cached ``auto`` after CLI fallback."""
    from frontier_agent.infra.config import get_config
    from plugins.tools._sandbox import resolve_sandbox_mode

    monkeypatch.setattr(get_config(), "sandbox_backend", "auto")
    monkeypatch.setenv("SANDBOX_BACKEND", "native")

    assert resolve_sandbox_mode({"sandbox_mode": "bwrap"}) == "native"


def test_linux_uses_native_runtime_by_default(
    tmp_path, monkeypatch, capsys,
) -> None:
    prepared: list[tuple[str, str]] = []
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "_load_env", lambda: None)
    monkeypatch.setattr(cli.sys, "platform", "linux")
    monkeypatch.delenv("APODEX_SANDBOX", raising=False)
    monkeypatch.delenv("SANDBOX_BACKEND", raising=False)
    monkeypatch.delenv("APODEX_IN_CONTAINER", raising=False)
    monkeypatch.delenv("APODEX_IN_NATIVE", raising=False)
    monkeypatch.setattr(
        "apodex.native.prepare_native_runtime",
        lambda workspace, session_id: (
            prepared.append((workspace, session_id))
            or tmp_path / ".apodex" / "runtime" / "native"
        ),
    )

    result = asyncio.run(cli._amain(["--mode", "invalid", "--no-tui"]))

    assert result == 2
    assert prepared and prepared[0][0] == str(tmp_path)
    assert "native mode" in capsys.readouterr().err


def test_linux_bwrap_is_explicit_and_skips_native_runtime(
    tmp_path, monkeypatch,
) -> None:
    prepared: list[tuple[str, str]] = []
    requested: list[str | None] = []
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "_load_env", lambda: None)
    monkeypatch.setattr(cli.sys, "platform", "linux")
    monkeypatch.setattr(
        "apodex.native.prepare_native_runtime",
        lambda workspace, session_id: prepared.append((workspace, session_id)),
    )

    def _resolve(name=None):
        requested.append(name)
        return Strategy(BWRAP, "test")

    monkeypatch.setattr("apodex.sandbox.resolve_strategy", _resolve)

    result = asyncio.run(cli._amain([
        "--bwrap", "--mode", "invalid", "--no-tui",
    ]))

    assert result == 2
    assert not prepared
    assert requested == [BWRAP]
