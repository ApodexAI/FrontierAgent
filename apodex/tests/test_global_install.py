"""A built wheel, installed as a ``uv tool``, launched from unrelated directories.

This is the one test that proves the "install once, launch anywhere" promise
with the real artifact: no source ``PYTHONPATH``, a throwaway ``HOME``, a
synthetic credential, and a local stub endpoint standing in for the model.
``--help`` alone would pass with a broken package; these runs go through
profile loading, the packaged provider registry, workflow dispatch, a tool
call against the project, and the configuration precedence.

``--native`` is passed so the same run works on a macOS runner with Docker
Desktop up; the platform defaults themselves are pinned separately in
``test_cli_runtime_selection.py``.

Skipped when ``uv`` is not on PATH. ``uv build`` and ``uv tool install`` reuse
the ambient uv cache, so a warmed CI runner installs in seconds.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path

import pytest

from deploy.huggingface.mock_llm import MockLLMServer, text_turn, tool_call_turn

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SECRET = "sk-synthetic-install-test-key-4242"

pytestmark = pytest.mark.skipif(
    shutil.which("uv") is None,
    reason="uv is not on PATH",
)


@dataclass(frozen=True)
class Installed:
    binary: Path  # <UV_TOOL_BIN_DIR>/frontier-agent
    python: Path  # the tool environment's own interpreter
    wheel: Path


@pytest.fixture(scope="module")
def installed(tmp_path_factory) -> Installed:
    """A wheel from this checkout, installed into an isolated tool directory."""
    root = tmp_path_factory.mktemp("global-install")
    dist, tool_dir, bin_dir = root / "dist", root / "tools", root / "bin"
    subprocess.run(
        ["uv", "build", "--wheel", "--out-dir", str(dist)],
        cwd=_REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    wheel = next(dist.glob("frontier_agent-*.whl"))
    env = {**os.environ, "UV_TOOL_DIR": str(tool_dir), "UV_TOOL_BIN_DIR": str(bin_dir)}
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
    subprocess.run(
        ["uv", "tool", "install", "--python", python_version, str(wheel)],
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    binary = bin_dir / "frontier-agent"
    assert binary.exists(), sorted(bin_dir.iterdir())
    assert (bin_dir / "apodex").exists()  # the compatibility alias
    # uv lays the tool environment out as <UV_TOOL_DIR>/<tool>/bin/python; the
    # console script's shebang is the authoritative pointer to it.
    shebang = binary.read_text(encoding="utf-8", errors="replace").splitlines()[0]
    python = Path(shebang.removeprefix("#!").strip())
    assert python.exists(), shebang
    assert tool_dir in python.parents
    return Installed(binary=binary, python=python, wheel=wheel)


def _launch(
    installed: Installed,
    args: list[str],
    *,
    cwd: Path,
    home: Path,
    extra: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run the installed CLI with a deliberately minimal environment."""
    system_path = os.pathsep.join(
        p for p in ("/usr/bin", "/bin", "/usr/sbin", "/sbin") if Path(p).is_dir()
    )
    env = {
        "PATH": f"{installed.binary.parent}{os.pathsep}{system_path}",
        "HOME": str(home),
        "TERM": "dumb",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PYTHONIOENCODING": "utf-8",
        **(extra or {}),
    }
    assert "PYTHONPATH" not in env
    return subprocess.run(
        [str(installed.binary), *args],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )


def _tool_result_texts(payload: dict) -> list[str]:
    return [
        str(m.get("content") or "") for m in payload.get("messages", []) if m.get("role") == "tool"
    ]


def _project(root: Path, name: str) -> Path:
    project = root / name
    project.mkdir()
    return project


# A ``bash`` turn: where model commands run, and which ``python3`` they get.
# ``pwd`` names the run-private scratch directory, which lives under the
# project's ``.apodex/runs/<session>/workspace``; the heredoc proves the
# interpreter is the tool environment (its ``sys.prefix`` plus a dependency
# a bare system Python does not have), not whatever ``/usr/bin`` offers.
_PROBE = "pwd\npython3 <<'PY'\nimport sys, textual\nprint('PREFIX=' + sys.prefix)\nPY\n"


def _probe_script(answer: str) -> list:
    return [tool_call_turn("bash", {"command": _PROBE}), text_turn(answer)]


def _session_records(project: Path) -> list[dict]:
    runs = project / ".apodex" / "runs"
    return [
        json.loads(path.read_text(encoding="utf-8")) for path in sorted(runs.glob("*/session.json"))
    ]


def _assert_task_ran_in(project: Path, installed: Installed, requests: list[dict]) -> None:
    """The run was bound to *project* and its tools saw the tool environment."""
    assert len(requests) >= 2, [r.get("model") for r in requests]
    results = _tool_result_texts(requests[1])
    assert any(f"PREFIX={installed.python.parent.parent}" in t for t in results), results
    real_project = Path(os.path.realpath(project))
    assert any(
        Path(line.strip()).is_relative_to(real_project)
        for t in results
        for line in t.splitlines()
        if line.startswith("/")
    ), results
    # The session record is the CLI's own statement of the workspace.
    records = _session_records(project)
    assert records, sorted((project / ".apodex").rglob("*"))
    assert Path(os.path.realpath(records[-1]["cwd"])) == real_project


def _write_user_env(home: Path, base_url: str) -> Path:
    user_env = home / ".config" / "apodex" / "env"
    user_env.parent.mkdir(parents=True, exist_ok=True)
    user_env.write_text(
        f"OPENAI_API_KEY={_SECRET}\nOPENAI_BASE_URL={base_url}\nOPENAI_MODEL=user-file-model\n",
        encoding="utf-8",
    )
    user_env.chmod(0o600)
    return user_env


# ── the artifact ──────────────────────────────────────────────────────────


def test_wheel_carries_the_runtime_resources_and_no_dockerfile(installed) -> None:
    names = set(zipfile.ZipFile(installed.wheel).namelist())

    for required in (
        "frontier_agent/infra/providers.yaml",  # packaged provider registry
        "frontier_agent/model_registry.yaml",
        "apodex/profiles/react.yaml",
        "apodex/profiles/agent_team.yaml",
        "workflows/stateful_react_agent/profiles/tui.yaml",
        "workflows/agent_team/profiles/tui.yaml",
    ):
        assert required in names, required
    # No image can be built from site-packages, which is why the Docker path
    # needs a checkout or an explicit context outside one.
    assert not any(name.endswith("Dockerfile") for name in names)


def test_installed_package_imports_without_the_checkout(installed, tmp_path) -> None:
    """Every runtime package imports from site-packages alone."""
    probe = (
        "import json, apodex.cli, apodex.docker, apodex.userenv, "
        "workflows.stateful_react_agent, workflows.agent_team, plugins.tools, "
        "frontier_agent.infra.providers as p; "
        "print(json.dumps({'providers': str(p._providers_path()), "
        "'names': sorted(p.load_providers())}))"
    )
    result = subprocess.run(
        [str(installed.python), "-c", probe],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        env={"HOME": str(tmp_path), "PATH": "/usr/bin:/bin"},
    )

    assert result.returncode == 0, result.stderr
    info = json.loads(result.stdout)
    assert info["providers"].endswith("frontier_agent/infra/providers.yaml")
    assert "openai" in info["names"]


def test_version_runs_without_any_configuration(installed, tmp_path) -> None:
    launch = tmp_path / "anywhere"
    launch.mkdir()

    result = _launch(installed, ["--version"], cwd=launch, home=tmp_path / "user-home")

    assert result.returncode == 0, result.stderr
    assert result.stdout.startswith("FrontierAgent ")


# ── the promised UX: cd into a project and run the bare command ──────────


def test_bare_launch_in_two_projects_shares_one_user_env_file(installed, tmp_path) -> None:
    home = tmp_path / "user-home"
    home.mkdir()
    first = _project(tmp_path, "first")
    second = _project(tmp_path, "second")

    with MockLLMServer(script=_probe_script("first done"), require_auth=True) as server:
        _write_user_env(home, server.base_url)
        result = _launch(
            installed,
            ["--native", "--no-tui", "--yes", "-p", "where am I"],
            cwd=first,
            home=home,  # no --cwd, no --model, nothing exported
        )
        first_requests = server.requests

    assert result.returncode == 0, result.stderr
    assert "first done" in result.stdout
    assert _SECRET not in result.stdout + result.stderr
    assert first_requests[0]["model"] == "user-file-model"
    _assert_task_ran_in(first, installed, first_requests)
    assert not (second / ".apodex").exists()

    with MockLLMServer(script=_probe_script("second done"), require_auth=True) as server:
        _write_user_env(home, server.base_url)  # same file, new stub port
        result = _launch(
            installed,
            ["--native", "--no-tui", "--yes", "-p", "where am I"],
            cwd=second,
            home=home,
        )
        second_requests = server.requests

    assert result.returncode == 0, result.stderr
    assert "second done" in result.stdout
    assert second_requests[0]["model"] == "user-file-model"
    _assert_task_ran_in(second, installed, second_requests)
    # Native mode redirected HOME under each project only after the user file
    # had been read from the real one: nothing else was written to it.
    assert sorted(p.name for p in home.iterdir()) == [".config"]


def test_model_flag_outranks_the_user_env_file(installed, tmp_path) -> None:
    home = tmp_path / "user-home"
    home.mkdir()
    project = _project(tmp_path, "flagged")

    with MockLLMServer(script=[text_turn("flag honoured")], require_auth=True) as server:
        _write_user_env(home, server.base_url)
        result = _launch(
            installed,
            ["--native", "--no-tui", "--model", "flag-model", "-p", "hello"],
            cwd=project,
            home=home,
        )
        requests = server.requests

    assert result.returncode == 0, result.stderr
    assert "flag honoured" in result.stdout
    assert requests and requests[0]["model"] == "flag-model"


# ── explicit --cwd from an unrelated directory, exported credentials ──────


def test_exported_credentials_and_cwd_from_an_unrelated_directory(installed, tmp_path) -> None:
    launch = tmp_path / "launch"  # unrelated to both the checkout and the project
    launch.mkdir()
    home = tmp_path / "user-home"
    home.mkdir()
    project = _project(tmp_path, "target")

    with MockLLMServer(script=_probe_script("target done"), require_auth=True) as server:
        result = _launch(
            installed,
            ["--native", "--no-tui", "--yes", "--cwd", str(project), "-p", "where am I"],
            cwd=launch,
            home=home,
            extra={
                "OPENAI_API_KEY": _SECRET,
                "OPENAI_BASE_URL": server.base_url,
                "OPENAI_MODEL": server.model,
            },
        )
        requests = server.requests

    assert result.returncode == 0, result.stderr
    assert "target done" in result.stdout
    assert _SECRET not in result.stdout + result.stderr
    assert requests[0]["model"] == server.model  # the exported model, via the profile
    _assert_task_ran_in(project, installed, requests)
    assert not (launch / ".apodex").exists()


def test_resume_listing_needs_no_credentials(installed, tmp_path) -> None:
    project = tmp_path / "project"
    project.mkdir()

    result = _launch(
        installed,
        ["--native", "--no-tui", "--resume"],
        cwd=project,
        home=tmp_path / "user-home",
    )

    assert result.returncode == 0, result.stderr
    assert "No saved sessions." in result.stdout
