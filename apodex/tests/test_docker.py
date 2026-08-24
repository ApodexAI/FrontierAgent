from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from apodex import docker


def _stub_container(monkeypatch, tmp_path) -> list[list[str]]:
    """Capture the ``docker run`` argv instead of starting a container."""
    calls: list[list[str]] = []
    monkeypatch.setattr(docker, "docker_available", lambda: (True, "available"))
    monkeypatch.setattr(docker, "image_exists", lambda image: True)
    monkeypatch.setattr(
        docker.Path, "home", classmethod(lambda cls: tmp_path / "home"),
    )
    monkeypatch.setattr(
        docker.subprocess,
        "run",
        lambda command: calls.append(command) or SimpleNamespace(returncode=0),
    )
    return calls


def test_image_installs_pptx_capable_libreoffice_component() -> None:
    dockerfile = Path(__file__).resolve().parents[2] / "Dockerfile"
    assert "libreoffice-impress" in dockerfile.read_text(encoding="utf-8")


def test_image_bridges_baked_packages_into_tool_overlay() -> None:
    """Sandbox Python must see the packages installed in the harness venv."""
    dockerfile = Path(__file__).resolve().parents[2] / "Dockerfile"
    contents = dockerfile.read_text(encoding="utf-8")
    assert "/opt/tool-venv/bin/python" in contents
    assert "frontier_agent_baked.pth" in contents
    assert "baked_site" in contents
    # An ``addsitedir`` line, not a bare path: ``site.addpackage`` appends a
    # plain path to sys.path without processing the .pth files inside it, which
    # drops any distribution whose import hook ships as a .pth.
    assert 'site.addsitedir("%s")' in contents


def test_without_cwd_arg_removes_both_cli_spellings() -> None:
    assert docker._without_cwd_arg(
        [
            "--cwd", "/host/project", "--input", "/host/a.pdf",
            "-p", "hello", "--cwd=/other", "--input=/host/b.png",
        ]
    ) == ["-p", "hello"]


def test_output_session_id_follows_mode_and_sanitizes_resume() -> None:
    generated = docker._session_id_for_run(["--mode", "agent_team"])
    assert "-agent_team-" in generated
    assert docker._session_id_for_run(["--resume=../../legacy id"]) == "legacy-id"


def test_container_mounts_workflow_paths_and_rewrites_cwd(
    monkeypatch, tmp_path,
) -> None:
    workspace = tmp_path / "project"
    workspace.mkdir()
    calls: list[list[str]] = []

    monkeypatch.setattr(docker, "docker_available", lambda: (True, "available"))
    monkeypatch.setattr(docker, "image_exists", lambda image: True)
    monkeypatch.setattr(
        docker.Path, "home", classmethod(lambda cls: tmp_path / "home"),
    )
    monkeypatch.setattr(
        docker.subprocess,
        "run",
        lambda command: calls.append(command) or SimpleNamespace(returncode=0),
    )

    assert docker.run_in_container(
        ["--cwd", str(workspace), "-p", "make a report"],
        cwd=str(workspace),
        image="test-image",
    ) == 0

    command = calls[0]
    session_id = next(
        item.split("=", 1)[1]
        for item in command
        if item.startswith("APODEX_SESSION_ID=")
    )
    runs_root = workspace / ".apodex" / "runs"
    run_workspace = runs_root / session_id / "workspace"
    outputs = runs_root / session_id / "outputs"
    assert ["-v", f"{workspace}:/project"] == command[
        command.index("-v"):command.index("-v") + 2
    ]
    assert f"{runs_root}:/apodex-runs" in command
    inputs = tmp_path / "home" / ".apodex-inputs" / session_id
    assert f"{inputs}:/apodex-input-staging" in command
    assert f"{inputs}:/inputs:ro" in command
    assert command[command.index("-w") + 1] == "/project"
    assert "FRONTIER_AGENT_WORKSPACE_DIR=/workspace" in command
    assert "APODEX_SESSION_WORKSPACES_ROOT=/apodex-runs" in command
    assert "APODEX_WORKSPACE_LINK=/workspace" in command
    assert any(item.startswith("APODEX_LOCAL_UTC_OFFSET=") for item in command)
    assert "FRONTIER_AGENT_OUTPUTS_DIR=/outputs" in command
    assert "APODEX_RUNS_ROOT=/apodex-runs" in command
    assert "APODEX_SESSION_OUTPUTS_ROOT=/apodex-runs" in command
    assert "APODEX_OUTPUTS_LINK=/outputs" in command
    assert "FRONTIER_AGENT_INPUTS_DIR=/inputs" in command
    assert "APODEX_INPUT_STAGING_DIR=/apodex-input-staging" in command
    assert f"APODEX_HOST_OUTPUTS_DIR={outputs}" in command
    assert f"APODEX_HOST_OUTPUTS_ROOT={runs_root}" in command
    assert f"APODEX_HOST_WORKSPACE_ROOT={runs_root}" in command
    assert f"APODEX_HOST_WORKSPACE_DIR={run_workspace}" in command
    # The host root is what the TUI and follow-up prompts show the user, so it
    # must be the real host path rather than the container's mount point.
    assert f"APODEX_HOST_RUNS_ROOT={runs_root}" in command
    assert str(workspace) not in command[command.index("test-image") + 1:]
    assert outputs.is_dir()
    assert run_workspace.is_dir()
    assert inputs.is_dir()


def test_container_stages_cli_inputs_before_launch(monkeypatch, tmp_path) -> None:
    workspace = tmp_path / "project"
    workspace.mkdir()
    source = tmp_path / "evidence.pdf"
    source.write_bytes(b"evidence")
    calls: list[list[str]] = []

    monkeypatch.setattr(docker, "docker_available", lambda: (True, "available"))
    monkeypatch.setattr(docker, "image_exists", lambda image: True)
    monkeypatch.setattr(
        docker.Path, "home", classmethod(lambda cls: tmp_path / "home"),
    )
    monkeypatch.setattr(
        docker.subprocess,
        "run",
        lambda command: calls.append(command) or SimpleNamespace(returncode=0),
    )

    assert docker.run_in_container(
        ["--input", str(source), "-p", "review it"],
        cwd=str(workspace), image="test-image",
    ) == 0

    session_id = next(
        value.split("=", 1)[1] for value in calls[0]
        if value.startswith("APODEX_SESSION_ID=")
    )
    staged = tmp_path / "home" / ".apodex-inputs" / session_id / source.name
    assert staged.read_bytes() == b"evidence"
    assert str(source) not in calls[0][calls[0].index("test-image") + 1:]


def test_container_reuses_resume_id_for_output_directory(
    monkeypatch, tmp_path,
) -> None:
    workspace = tmp_path / "project"
    workspace.mkdir()
    calls: list[list[str]] = []

    monkeypatch.setattr(docker, "docker_available", lambda: (True, "available"))
    monkeypatch.setattr(docker, "image_exists", lambda image: True)
    monkeypatch.setattr(
        docker.Path, "home", classmethod(lambda cls: tmp_path / "home"),
    )
    monkeypatch.setattr(
        docker.subprocess,
        "run",
        lambda command: calls.append(command) or SimpleNamespace(returncode=0),
    )

    assert docker.run_in_container(
        ["--resume", "20260806-120000-react-ab12"],
        cwd=str(workspace),
        image="test-image",
    ) == 0

    runs_root = workspace / ".apodex" / "runs"
    outputs = runs_root / "20260806-120000-react-ab12" / "outputs"
    assert f"{runs_root}:/apodex-runs" in calls[0]
    assert outputs.is_dir()
    assert "APODEX_SESSION_ID=20260806-120000-react-ab12" in calls[0]


def test_container_keeps_the_image_entrypoint_and_names_the_host_identity(
    monkeypatch, tmp_path,
) -> None:
    """The direct ``--docker`` path must align the tool uid like Compose does.

    Replacing the entrypoint with ``apodex`` skipped docker/entrypoint.sh, which
    is the only thing that remaps agent-tool onto the invoking user and exports
    APODEX_TOOL_HOST_IDENTITY. Without that export the tool layer widens the
    mounts to 0777 — and /workspace is a bind mount of the user's own checkout.
    """
    workspace = tmp_path / "project"
    workspace.mkdir()
    calls = _stub_container(monkeypatch, tmp_path)

    assert docker.run_in_container(
        [], cwd=str(workspace), image="test-image",
    ) == 0

    command = calls[0]
    assert command[command.index("--entrypoint") + 1] == "/app/docker/entrypoint.sh"
    assert command[command.index("test-image") + 1] == "apodex"
    assert f"APODEX_HOST_UID={os.getuid()}" in command
    assert f"APODEX_HOST_GID={os.getgid()}" in command


def test_container_never_execs_a_bare_task_string_as_a_command(
    monkeypatch, tmp_path,
) -> None:
    """Why the entrypoint is still passed explicitly rather than dropped.

    docker/entrypoint.sh execs its arguments verbatim unless the first one looks
    like a flag, so a task that does not start with ``-`` has to arrive behind
    the ``apodex`` command word.
    """
    workspace = tmp_path / "project"
    workspace.mkdir()
    calls = _stub_container(monkeypatch, tmp_path)

    assert docker.run_in_container(
        ["fix the crash"], cwd=str(workspace), image="test-image",
    ) == 0

    command = calls[0]
    assert command[command.index("test-image") + 1:] == ["apodex", "fix the crash"]


def test_requested_image_is_pulled_rather_than_built_under_its_name(
    monkeypatch, tmp_path,
) -> None:
    workspace = tmp_path / "project"
    workspace.mkdir()
    calls = _stub_container(monkeypatch, tmp_path)
    monkeypatch.setattr(docker, "image_exists", lambda image: False)
    pulled: list[str] = []
    monkeypatch.setattr(
        docker, "pull_image", lambda image: pulled.append(image) or True,
    )
    monkeypatch.setattr(
        docker, "build_image",
        lambda *a, **k: pytest.fail("an explicit tag must not be built locally"),
    )

    assert docker.run_in_container(
        [], cwd=str(workspace), image="ghcr.io/apodexai/frontieragent:latest",
    ) == 0

    assert pulled == ["ghcr.io/apodexai/frontieragent:latest"]
    assert len(calls) == 1


def test_failed_pull_reports_instead_of_starting_a_container(
    monkeypatch, tmp_path,
) -> None:
    workspace = tmp_path / "project"
    workspace.mkdir()
    calls = _stub_container(monkeypatch, tmp_path)
    monkeypatch.setattr(docker, "image_exists", lambda image: False)
    monkeypatch.setattr(docker, "pull_image", lambda image: False)

    assert docker.run_in_container(
        [], cwd=str(workspace), image="registry.test/absent:1",
    ) == 1
    assert calls == []


def test_default_image_is_still_built_from_the_checkout(
    monkeypatch, tmp_path,
) -> None:
    workspace = tmp_path / "project"
    workspace.mkdir()
    _stub_container(monkeypatch, tmp_path)
    monkeypatch.setattr(docker, "image_exists", lambda image: False)
    built: list[str] = []
    monkeypatch.setattr(docker, "build_image", lambda image: built.append(image))
    monkeypatch.setattr(
        docker, "pull_image",
        lambda image: pytest.fail("the local default has no registry to pull from"),
    )

    assert docker.run_in_container(
        [], cwd=str(workspace), image=docker._DEFAULT_IMAGE,
    ) == 0
    assert built == [docker._DEFAULT_IMAGE]


def test_interactive_macos_container_receives_clipboard_broker(
    monkeypatch, tmp_path,
) -> None:
    workspace = tmp_path / "project"
    workspace.mkdir()
    calls: list[list[str]] = []
    lifecycle: list[str] = []

    class FakeBroker:
        port = 43210
        token = "session-token"

        def __init__(self, _manager) -> None:
            pass

        def start(self) -> None:
            lifecycle.append("start")

        def close(self) -> None:
            lifecycle.append("close")

    monkeypatch.setattr(docker, "docker_available", lambda: (True, "available"))
    monkeypatch.setattr(docker, "image_exists", lambda image: True)
    monkeypatch.setattr(docker.sys, "platform", "darwin")
    monkeypatch.setattr(docker.sys, "stdin", SimpleNamespace(isatty=lambda: True))
    monkeypatch.setattr(docker.sys, "stdout", SimpleNamespace(isatty=lambda: True))
    monkeypatch.setattr(
        docker.Path, "home", classmethod(lambda cls: tmp_path / "home"),
    )
    monkeypatch.setattr("apodex.clipboard.ClipboardBroker", FakeBroker)
    monkeypatch.setattr(
        docker.subprocess,
        "run",
        lambda command: calls.append(command) or SimpleNamespace(returncode=0),
    )

    assert docker.run_in_container([], cwd=str(workspace), image="test-image") == 0

    assert lifecycle == ["start", "close"]
    assert "APODEX_CLIPBOARD_BROKER_URL=http://host.docker.internal:43210" in calls[0]
    assert "APODEX_CLIPBOARD_BROKER_TOKEN=session-token" in calls[0]
