"""Which execution boundary the CLI picks before any tool can run.

These cases pin the macOS branch on a Linux CI box: the choice is made from
``sys.platform`` and the flags, so it is decidable without a Mac. What they
cannot prove is that the container path still behaves on real Docker Desktop,
which is why the accompanying PR asks for a Mac run as well.
"""
from __future__ import annotations

import pytest

from apodex import cli, sandbox


@pytest.fixture
def macos(monkeypatch, tmp_path):
    """A macOS-looking CLI whose container launch is recorded, never executed."""
    monkeypatch.chdir(tmp_path)  # keep .env discovery away from the checkout
    monkeypatch.setattr(cli.sys, "platform", "darwin")
    entered: list[list[str]] = []
    monkeypatch.setattr(
        "apodex.docker.run_in_container",
        lambda argv, **kwargs: entered.append(list(argv)) or 0,
    )
    monkeypatch.setattr("apodex.docker.docker_available", lambda: (True, "available"))
    return entered


def test_macos_without_flags_still_runs_in_the_container(macos) -> None:
    # The implicit default, and the reason the macOS guide has to say so: an
    # unadorned invocation builds and enters the image when Docker is running.
    assert cli.main([]) == 0
    assert macos == [[]]


def test_macos_bwrap_never_detours_through_the_container(
    macos, monkeypatch, capsys,
) -> None:
    def _refuse(requested: str | None = None) -> sandbox.Strategy:
        raise sandbox.SandboxUnavailable("no bubblewrap here")

    # Stubbed so the outcome does not depend on whether the test host itself can
    # run bwrap; what matters is that the container was not entered on the way.
    monkeypatch.setattr("apodex.sandbox.resolve_strategy", _refuse)

    assert cli.main(["--bwrap"]) == 2
    assert macos == []
    assert "no bubblewrap here" in capsys.readouterr().err


def test_macos_global_install_with_docker_but_no_image_fails_closed(
    monkeypatch, tmp_path, capsys,
) -> None:
    """A wheel on a Mac with Docker running must not quietly go native.

    The platform default promised a container. With no image and no checkout
    to build one from, the honest outcome is a stop that names the options,
    not a native run the user never asked for.
    """
    from apodex import docker

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli.sys, "platform", "darwin")
    monkeypatch.setattr("apodex.docker.docker_available", lambda: (True, "available"))
    monkeypatch.setattr(docker, "image_exists", lambda image: False)
    monkeypatch.setattr(docker, "_REPO_ROOT", tmp_path / "site-packages")
    monkeypatch.delenv(docker.BUILD_CONTEXT_VAR, raising=False)
    monkeypatch.setattr(
        docker.subprocess, "run",
        lambda *a, **k: pytest.fail("no docker command may run without an image"),
    )
    monkeypatch.setattr(
        "apodex.native.prepare_native_runtime",
        lambda *a, **k: pytest.fail("must not fall back to the native runtime"),
    )

    assert cli.main([]) == 1

    err = capsys.readouterr().err
    assert "cannot use the Docker path" in err
    assert docker.BUILD_CONTEXT_VAR in err
    assert "--native" in err


def test_macos_container_launch_forwards_the_resolved_environment(
    monkeypatch, tmp_path,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli.sys, "platform", "darwin")
    monkeypatch.setattr("apodex.docker.docker_available", lambda: (True, "available"))
    seen: dict[str, object] = {}

    def _record(argv, **kwargs):
        seen["argv"] = list(argv)
        seen["forward_env"] = tuple(kwargs.get("forward_env", ()))
        return 0

    monkeypatch.setattr("apodex.docker.run_in_container", _record)
    (tmp_path / ".env").write_text("OPENAI_MODEL=project-model\n", encoding="utf-8")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-exported")
    monkeypatch.delenv("OPENAI_MODEL", raising=False)

    assert cli.main(["--docker", "-p", "hi"]) == 0

    assert seen["argv"] == ["-p", "hi"]
    assert "OPENAI_MODEL" in seen["forward_env"]     # from the launch .env
    assert "OPENAI_API_KEY" in seen["forward_env"]   # exported
    assert all("sk-exported" not in name for name in seen["forward_env"])
