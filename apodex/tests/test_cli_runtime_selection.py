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
