"""Launcher-pinned mounts: ``/workspace``, ``/outputs`` and ``/inputs``.

A launcher that binds the three canonical mount points itself — a container, an
external jail — has already put the real directories where the tools look.
``Session`` repointed all three aliases at run-local directories anyway, and
silently: a repointed ``/inputs`` is an *empty directory*, not an error. Measured
on an APEX benchmark run under an external jail, 188 of 188 trials logged
``container /inputs … is EMPTY`` and spent turns rediscovering their own inputs,
while ``bash`` — which does not go through ``resolve_runtime_path`` — read the
staged corpus fine.

``APODEX_PINNED_MOUNTS=1`` is the launcher's declaration that it owns the mounts.
Every test here has a mirror-image case without the flag, because the default
path — a local terminal session owning private per-session directories — must be
byte-identical to before.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import pytest

from apodex.attachments import AttachmentError, AttachmentManager
from apodex.run_layout import pinned_mounts
from apodex.session import TerminalSession
from plugins.tools._sandbox import resolve_mount_dirs, resolve_runtime_path

_MOUNT_VARS = (
    "APODEX_PINNED_MOUNTS",
    "APODEX_RUNS_ROOT",
    "APODEX_SESSION_OUTPUTS_ROOT",
    "APODEX_SESSION_WORKSPACES_ROOT",
    "APODEX_OUTPUTS_LINK",
    "APODEX_WORKSPACE_LINK",
    "APODEX_INPUT_STAGING_DIR",
    "APODEX_INPUT_STAGING_ROOT",
    "FRONTIER_AGENT_INPUTS_DIR",
    "FRONTIER_AGENT_INPUTS_ROOT",
    "FRONTIER_AGENT_OUTPUTS_DIR",
    "FRONTIER_AGENT_WORKSPACE_DIR",
    # Written by ``activate_run`` as a side effect rather than read by it. Listed
    # so a value inherited from the surrounding shell cannot reach the code under
    # test; the fixture's snapshot is what puts them back afterwards.
    "APODEX_SESSION_ID",
    "APODEX_RUN_DIR",
    "APODEX_RUNS_ROOT_PINNED",
    "APODEX_HOST_RUNS_ROOT",
    "APODEX_HOST_RUN_DIR",
    "APODEX_HOST_OUTPUTS_DIR",
    "APODEX_HOST_OUTPUTS_ROOT",
    "APODEX_HOST_WORKSPACE_DIR",
    "APODEX_HOST_WORKSPACE_ROOT",
)


@pytest.fixture(autouse=True)
def _isolate_mount_env() -> Iterator[None]:
    """Start from a clean mount environment and restore the real one after.

    A snapshot rather than ``monkeypatch.delenv`` because the code under test
    *writes* several of these as a side effect — ``activate_run`` assigns
    ``APODEX_SESSION_ID``, ``APODEX_RUNS_ROOT`` and ``APODEX_RUN_DIR``. On a name
    that is absent to begin with, ``delenv(..., raising=False)`` records nothing
    and so undoes nothing, which let all three escape this file. Measured: they
    did, and ``APODEX_SESSION_ID`` is the one that bites — ``Session.__init__``
    treats it as a session-id source, so an unrelated later test would adopt a
    session id set here.
    """
    saved = dict(os.environ)
    for name in _MOUNT_VARS:
        os.environ.pop(name, None)
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(saved)


def test_the_flag_is_off_unless_a_launcher_sets_it_to_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert pinned_mounts() is False
    monkeypatch.setenv("APODEX_PINNED_MOUNTS", "1")
    assert pinned_mounts() is True
    # Anything else is off: a half-set variable must not silently pin the run.
    for value in ("", "0", "true", "yes", "2"):
        monkeypatch.setenv("APODEX_PINNED_MOUNTS", value)
        assert pinned_mounts() is False, value


# ── /inputs ──────────────────────────────────────────────────────────────────


def test_pinned_inputs_keep_the_mount_the_launcher_declared(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The bug in one assertion.

    ``Session`` copies ``AttachmentManager.agent_dir`` into
    ``FRONTIER_AGENT_INPUTS_DIR``, which is what ``resolve_mount_dirs`` reads —
    so a manager that invents a session-scoped directory is what pointed
    ``read_file``/``glob_search``/``grep_search`` at nothing.
    """
    mount = tmp_path / "inputs"
    (mount / "filesystem").mkdir(parents=True)
    monkeypatch.setenv("APODEX_PINNED_MOUNTS", "1")
    monkeypatch.setenv("FRONTIER_AGENT_INPUTS_DIR", str(mount))

    manager = AttachmentManager(str(tmp_path), "20260821-152125+0000-agent_team-6acb")

    assert manager.agent_dir == mount
    assert manager.staging_dir == mount


def test_pinned_inputs_require_the_canonical_mount_point_to_exist(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pinning cannot mask an absent canonical mount with an empty directory.

    ``is_dir`` is stubbed to report the mount absent, because whether a real
    top-level ``/inputs`` happens to exist is a property of the host and not of
    the code under test. This machine has one left over from an unrelated setup,
    so validation passed, nothing raised, and the test failed here while passing
    on CI -- the same reach outside ``tmp_path`` that the sibling default-mount
    test already had to stub ``mkdir`` for.
    """
    monkeypatch.setenv("APODEX_PINNED_MOUNTS", "1")
    probed: list[Path] = []

    def _absent(self: Path) -> bool:
        probed.append(self)
        return False

    monkeypatch.setattr(Path, "is_dir", _absent)

    with pytest.raises(AttachmentError, match="does not exist: /inputs"):
        AttachmentManager(str(tmp_path), "session")
    # The path it validated was the default mount, not something tmp-derived.
    assert probed == [Path("/inputs")]
    assert resolve_mount_dirs()[2] == "/inputs"


def _refuse_mkdir(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make every ``mkdir`` fail the way a read-only filesystem does.

    Simulated rather than acted out: these tests run as root on this host, where
    an unwritable mode is not actually enforced, so a ``chmod``-based version
    passes for the wrong reason.
    """
    def _raise(self: Path, *args: object, **kwargs: object) -> None:
        raise OSError(30, "Read-only file system", str(self))

    monkeypatch.setattr(Path, "mkdir", _raise)


def test_a_pinned_read_only_inputs_mount_does_not_break_startup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``/inputs`` is bound read-only, and construction must survive that.

    Pinned paths are validated in place and never passed to ``mkdir``; the
    launcher, rather than the session, owns their creation.
    """
    mount = tmp_path / "inputs"
    mount.mkdir()
    monkeypatch.setenv("APODEX_PINNED_MOUNTS", "1")
    monkeypatch.setenv("FRONTIER_AGENT_INPUTS_DIR", str(mount))
    _refuse_mkdir(monkeypatch)

    manager = AttachmentManager(str(tmp_path), "session")

    assert manager.agent_dir == mount


def test_a_pinned_inputs_dir_that_is_genuinely_missing_still_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Tolerating a read-only mount must not tolerate a typo in the mount path —
    that would trade a loud failure for the silent empty-directory bug again."""
    monkeypatch.setenv("APODEX_PINNED_MOUNTS", "1")
    monkeypatch.setenv("FRONTIER_AGENT_INPUTS_DIR", str(tmp_path / "absent"))
    with pytest.raises(AttachmentError, match="pinned input staging directory"):
        AttachmentManager(str(tmp_path), "session")


def test_a_separate_pinned_agent_dir_must_also_exist(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    staging = tmp_path / "staging"
    staging.mkdir()
    monkeypatch.setenv("APODEX_PINNED_MOUNTS", "1")
    monkeypatch.setenv("APODEX_INPUT_STAGING_DIR", str(staging))
    monkeypatch.setenv("FRONTIER_AGENT_INPUTS_DIR", str(tmp_path / "absent-agent"))

    with pytest.raises(AttachmentError, match="pinned input agent directory"):
        AttachmentManager(str(tmp_path), "session")


def test_unpinned_mkdir_errors_are_not_swallowed_for_an_existing_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    staging = tmp_path / "staging"
    staging.mkdir()
    monkeypatch.setenv("APODEX_INPUT_STAGING_DIR", str(staging))
    _refuse_mkdir(monkeypatch)

    with pytest.raises(OSError, match="Read-only file system"):
        AttachmentManager(str(tmp_path), "session")


def test_unpinned_inputs_still_get_a_private_per_session_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The default a local terminal session relies on, unchanged.

    ``FRONTIER_AGENT_INPUTS_DIR`` is deliberately NOT inherited here, because
    ``TerminalSession`` rewrites it as sessions change in one process.
    """
    monkeypatch.setenv("FRONTIER_AGENT_INPUTS_DIR", "/inputs")
    monkeypatch.setenv("HOME", str(tmp_path))

    manager = AttachmentManager(str(tmp_path), "session-one")

    assert manager.agent_dir == tmp_path / ".apodex-inputs" / "session-one"
    assert manager.staging_dir == manager.agent_dir


# ── /outputs and /workspace ──────────────────────────────────────────────────


def test_pinned_outputs_and_workspace_are_left_on_their_mounts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``APODEX_RUNS_ROOT`` set means the repointing gate is open.

    ``Session.__init__`` calls ``activate_run`` before these two, and that sets
    the variable unconditionally — so inside a jail the gate is ALWAYS open and
    both aliases moved off the mounts. That is what handed publisher sub-agents
    ``<runs>/<session>/outputs/answer.md`` while the file tools told them only
    ``/workspace`` and ``/outputs`` were writable.
    """
    monkeypatch.setenv("APODEX_PINNED_MOUNTS", "1")
    monkeypatch.setenv("APODEX_RUNS_ROOT", str(tmp_path / "runs"))
    monkeypatch.setenv("FRONTIER_AGENT_OUTPUTS_DIR", "/outputs")
    monkeypatch.setenv("FRONTIER_AGENT_WORKSPACE_DIR", "/workspace")

    TerminalSession._activate_session_outputs("session", str(tmp_path))
    TerminalSession._activate_session_workspace("session", str(tmp_path))

    assert os.environ["FRONTIER_AGENT_OUTPUTS_DIR"] == "/outputs"
    assert os.environ["FRONTIER_AGENT_WORKSPACE_DIR"] == "/workspace"
    assert resolve_mount_dirs() == ("/workspace", "/outputs", "/inputs")


def test_pinned_workspace_activation_updates_the_run_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Session switches keep mount aliases but must advance run metadata."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.setenv("APODEX_PINNED_MOUNTS", "1")
    monkeypatch.setenv("FRONTIER_AGENT_WORKSPACE_DIR", str(workspace))

    TerminalSession._activate_session_workspace("session-two", str(workspace))

    assert os.environ["FRONTIER_AGENT_WORKSPACE_DIR"] == str(workspace)
    assert os.environ["APODEX_SESSION_ID"] == "session-two"
    assert os.environ["APODEX_RUN_DIR"] == str(
        workspace / ".apodex" / "runs" / "session-two"
    )


def test_unpinned_outputs_and_workspace_still_follow_the_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without the flag both aliases move into the run directory, as before —
    that is what keeps concurrent runs and an in-app ``/resume`` from writing
    into each other's trees."""
    runs = tmp_path / "runs"
    monkeypatch.setenv("APODEX_RUNS_ROOT", str(runs))

    TerminalSession._activate_session_outputs("session", str(tmp_path))
    TerminalSession._activate_session_workspace("session", str(tmp_path))

    run_dir = tmp_path / ".apodex" / "runs" / "session"
    assert os.environ["FRONTIER_AGENT_OUTPUTS_DIR"] == str(run_dir / "outputs")
    assert os.environ["FRONTIER_AGENT_WORKSPACE_DIR"] == str(run_dir / "workspace")


def test_pinned_mounts_make_path_rewriting_a_no_op(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The end state the whole pin exists for: the tools' canonical paths are
    the jail's real paths, so nothing is rewritten on the way to the filesystem."""
    monkeypatch.setenv("APODEX_PINNED_MOUNTS", "1")
    monkeypatch.setenv("FRONTIER_AGENT_INPUTS_DIR", "/inputs")
    monkeypatch.setenv("FRONTIER_AGENT_OUTPUTS_DIR", "/outputs")
    monkeypatch.setenv("FRONTIER_AGENT_WORKSPACE_DIR", "/workspace")

    for path in (
        "/inputs/filesystem/Industry Primer/report.pdf",
        "/outputs/answer.md",
        "/workspace/extract.txt",
    ):
        assert resolve_runtime_path(path) == path


def test_a_launchers_mounts_survive_a_whole_session_startup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The regression, asserted end to end rather than per-variable.

    Each piece above checks one call site. What actually went wrong was the
    *sequence*: a launcher set the three aliases correctly and ``Session``
    startup rewrote them afterwards. So drive a launcher's environment through
    the real startup steps in ``Session.__init__``'s order — ``activate_run``,
    then the two activators, then ``AttachmentManager`` (whose ``agent_dir``
    ``Session`` copies back into ``FRONTIER_AGENT_INPUTS_DIR``) — and check where
    the tools would then look.

    Note what ``activate_run`` does on the way past: it sets ``APODEX_RUNS_ROOT``
    unconditionally, which is the gate both activators test. That is why a
    launcher cannot fix this from the environment alone and needs the pin.
    """
    from apodex.run_layout import activate_run

    workspace = tmp_path / "workspace"
    inputs = tmp_path / "inputs"
    outputs = tmp_path / "outputs"
    (inputs / "filesystem").mkdir(parents=True)
    workspace.mkdir()
    outputs.mkdir()

    # What a launcher declares, mirroring apodex/docker.py's own env block.
    monkeypatch.setenv("APODEX_PINNED_MOUNTS", "1")
    monkeypatch.setenv("FRONTIER_AGENT_WORKSPACE_DIR", str(workspace))
    monkeypatch.setenv("FRONTIER_AGENT_OUTPUTS_DIR", str(outputs))
    monkeypatch.setenv("FRONTIER_AGENT_INPUTS_DIR", str(inputs))
    monkeypatch.setenv("APODEX_INPUT_STAGING_DIR", str(inputs))

    session_id = "20260821-152125+0000-agent_team-6acb"
    activate_run(session_id, str(workspace))
    assert os.environ["APODEX_RUNS_ROOT"]  # the gate is open, as it always is
    TerminalSession._activate_session_workspace(session_id, str(workspace))
    TerminalSession._activate_session_outputs(session_id, str(workspace))
    attachments = AttachmentManager(str(workspace), session_id)
    monkeypatch.setenv("FRONTIER_AGENT_INPUTS_DIR", str(attachments.agent_dir))

    assert resolve_mount_dirs() == (str(workspace), str(outputs), str(inputs))
    # The run record still follows the session, which is what a launcher that
    # collects traces out of the run directory depends on.
    assert os.environ["APODEX_RUN_DIR"] == str(
        workspace / ".apodex" / "runs" / session_id
    )


def test_without_the_pin_the_same_startup_loses_every_mount(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The failure this replaces, pinned down so it cannot come back quietly.

    Identical to the test above minus the flag. Every alias ends up somewhere the
    launcher never mounted: ``/inputs`` at an empty per-session directory (the
    "container /inputs … is EMPTY" line, 188 of 188 trials), and ``/outputs`` and
    ``/workspace`` inside the run directory — which is how a publisher came to be
    handed ``<runs>/<session>/outputs/answer.md`` while the file tools insisted
    only ``/workspace`` and ``/outputs`` were writable.
    """
    from apodex.run_layout import activate_run

    workspace = tmp_path / "workspace"
    inputs = tmp_path / "inputs"
    (inputs / "filesystem").mkdir(parents=True)
    workspace.mkdir()
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("FRONTIER_AGENT_WORKSPACE_DIR", str(workspace))
    monkeypatch.setenv("FRONTIER_AGENT_OUTPUTS_DIR", str(tmp_path / "outputs"))
    monkeypatch.setenv("FRONTIER_AGENT_INPUTS_DIR", str(inputs))

    session_id = "session"
    activate_run(session_id, str(workspace))
    TerminalSession._activate_session_workspace(session_id, str(workspace))
    TerminalSession._activate_session_outputs(session_id, str(workspace))
    attachments = AttachmentManager(str(workspace), session_id)
    monkeypatch.setenv("FRONTIER_AGENT_INPUTS_DIR", str(attachments.agent_dir))

    run_dir = workspace / ".apodex" / "runs" / session_id
    got_workspace, got_outputs, got_inputs = resolve_mount_dirs()
    assert got_workspace == str(run_dir / "workspace")
    assert got_outputs == str(run_dir / "outputs")
    assert got_inputs == str(tmp_path / "home" / ".apodex-inputs" / session_id)
    assert not list(Path(got_inputs).iterdir())  # the silent part: empty, not missing
