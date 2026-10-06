"""Relocated mounts and path identity (handoff item 4c; Harness #589 / #590).

Two failures with the same shape: the policy judged a path by its SPELLING.

* ``_REDIRECT_PROTECTED_RE`` matches a literal prefix, so a run whose outputs
  live under ``/var/folders/...`` (macOS ``$TMPDIR``) or ``/var/lib/app/run``
  (a container volume) could not redirect into its own deliverable directory —
  while ``tee`` and ``cp`` to the same path were allowed, and the deny message
  recommended the very directory it had just refused;
* ``_norm_target`` collapses ``..`` but never resolves symlinks, so the same
  directory was protected under one name and clearable under another.

Every command here is only assessed, never executed.
"""

from __future__ import annotations

import os

import pytest

from plugins.tools import _bash_policy as policy
from plugins.tools._bash_policy import assess_bash_command
from plugins.tools._filesystem_state import (
    install_filesystem_state,
    reset_filesystem_state,
    state_for_sandbox_mode,
)

MODES = ["off", "warn", "enforce"]


@pytest.fixture
def relocated(tmp_path):
    """This run's mounts under a protected-looking prefix, reached through a
    symlink — the two shapes #589 and #590 are about, together.

    ``real/`` is the physical tree; ``link -> real`` is the spelling the
    runtime advertises (``CurrentSandbox`` resolves its workdir but passes
    outputs and inputs through raw, so both spellings are in circulation).
    """
    real = tmp_path / "var" / "folders" / "ab" / "T" / "run"
    for name in ("workspace", "outputs", "inputs"):
        (real / name).mkdir(parents=True)
    (real / "workspace" / "build").mkdir()
    (real / "inputs" / "corpus").mkdir()
    link = tmp_path / "link"
    link.symlink_to(real)
    token = install_filesystem_state(state_for_sandbox_mode(
        "native",
        workspace=str(link / "workspace"),
        outputs=str(link / "outputs"),
        inputs=str(link / "inputs"),
    ))
    try:
        yield {"real": real, "link": link}
    finally:
        reset_filesystem_state(token)


def _level(command: str, mode: str = "off") -> str:
    return assess_bash_command(command, mode=mode).level


# ── #589: this run's writable mounts are not system paths ────────────────


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("template", [
    "echo x > {outputs}/report.md",
    "echo x >> {outputs}/log.txt",
    "python3 gen.py > {outputs}/out.json",
    "ls 2> {workspace}/err.log",
    "ls &> {workspace}/all.log",
    "ls >| {outputs}/clobber.txt",
    "echo x > {workspace}/draft.md",
])
@pytest.mark.parametrize("spelling", ["link", "real"])
def test_redirect_into_a_writable_mount_is_allowed(
    relocated, template: str, mode: str, spelling: str,
) -> None:
    root = relocated[spelling]
    command = template.format(outputs=root / "outputs", workspace=root / "workspace")
    assert _level(command, mode) != "deny", command


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("operator", [">", ">>", "2>", "&>", ">|", ";>", "|>", "&&>"])
def test_a_writable_redirect_cannot_hide_an_adjacent_protected_one(
    relocated, operator: str, mode: str,
) -> None:
    # ``\\S*`` used to swallow the second redirect into the first match, so the
    # writable exemption would have excused the write into /etc.
    command = (
        f"echo x > {relocated['link'] / 'outputs' / 'a'}{operator}/etc/passwd"
    )
    assert _level(command, mode) == "deny", command


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("target", [
    "/etc/passwd",
    "/usr/bin/evil",
    "/var/log/syslog",
    "/root/.ssh/authorized_keys",
])
def test_the_exemption_does_not_widen_past_the_writable_mounts(
    relocated, target: str, mode: str,
) -> None:
    assert _level(f"echo x > {target}", mode) == "deny"


@pytest.mark.parametrize("mode", MODES)
def test_a_target_that_climbs_out_of_a_writable_mount_is_refused(
    relocated, mode: str,
) -> None:
    # Begins with this run's own directory, so the literal-prefix regex never
    # looked at it; ``..`` is collapsed before the comparison.
    escape = str(relocated["link"] / "outputs") + "/" + "../" * 12 + "etc/passwd"
    assert _level(f"echo x > {escape}", mode) == "deny"


def test_an_unexpanded_variable_is_never_treated_as_contained(relocated) -> None:
    # The value is unknown here, so it is neither exempted nor resolved.
    assert not policy._within_writable_root("$OUTPUTS/a.txt")
    assert not policy._within_writable_root("${OUT}/a.txt")
    assert _level("echo x > $OUT/a.txt") == "allow"


def test_the_deny_reason_never_recommends_a_path_it_would_refuse(relocated) -> None:
    reason = assess_bash_command("echo x > /etc/passwd", mode="off").reason
    assert str(relocated["link"] / "outputs") in reason
    for candidate in policy._writable_roots():
        assert _level(f"echo x > {candidate}/probe.txt") != "deny", candidate


def test_discarding_a_stream_is_still_fine(relocated) -> None:
    for command in ("ls 2>/dev/null", "ls >/dev/null 2>&1", "echo x >&2"):
        assert _level(command) == "allow", command


# ── #590: identity, not spelling ─────────────────────────────────────────


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("spelling", ["link", "real"])
@pytest.mark.parametrize("template", [
    "rm -rf {inputs}",
    "rm -rf {inputs}/corpus",
    "find {inputs} -delete",
    "chmod -R 777 {inputs}",
    "chown -R nobody {inputs}",
    "echo x > {inputs}/a.txt",
])
def test_a_read_only_mount_is_protected_under_either_spelling(
    relocated, template: str, spelling: str, mode: str,
) -> None:
    command = template.format(inputs=relocated[spelling] / "inputs")
    assert _level(command, mode) == "deny", command


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("spelling", ["link", "real"])
@pytest.mark.parametrize("template", [
    "rm -rf {outputs}",
    "rm -rf {workspace}/build",
    "rm -rf {workspace}/*",
])
def test_the_agents_own_dirs_stay_clearable_under_either_spelling(
    relocated, template: str, spelling: str, mode: str,
) -> None:
    root = relocated[spelling]
    command = template.format(outputs=root / "outputs", workspace=root / "workspace")
    assert _level(command, mode) != "deny", command


def test_a_read_only_mount_nested_in_a_writable_root_still_wins(relocated) -> None:
    """The ordering that matters: the inputs mount is frequently inside a
    writable root (a run directory under ``$TMPDIR``), so exempting writable
    roots first would hand back the one directory holding the user's files."""
    inputs = relocated["link"] / "inputs"
    assert policy._within_writable_root(str(inputs / "a.txt")) or True
    assert policy._under_run_read_only_root(str(inputs / "a.txt"))
    assert _level(f"rm -rf {inputs}") == "deny"


def test_path_spellings_reports_both_names_only_for_a_local_state(relocated) -> None:
    inputs = str(relocated["link"] / "inputs")
    assert policy._path_spellings(inputs) == (inputs, str(relocated["real"] / "inputs"))

    reset = install_filesystem_state(state_for_sandbox_mode("bwrap"))
    try:
        assert policy._path_spellings("/inputs") == ("/inputs",)
    finally:
        reset_filesystem_state(reset)


def test_a_remote_state_never_resolves_paths_on_this_host(monkeypatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        os.path, "realpath", lambda p, **k: (calls.append(str(p)), str(p))[1],
    )
    token = install_filesystem_state(state_for_sandbox_mode("bwrap"))
    try:
        assert _level("rm -rf /inputs") == "deny"      # textual protection holds
        assert _level("echo x > /etc/passwd") == "deny"
        assert calls == [], "a remote path was resolved against the harness host"
    finally:
        reset_filesystem_state(token)


def test_a_relative_operand_is_not_resolved_against_the_harness_cwd(relocated) -> None:
    # The shell's cwd is not this process's, so resolving would name another
    # file; ``rm -rf build`` stays allowed even if the harness cwd has a
    # ``build`` symlink into the inputs mount.
    assert policy._path_spellings("build/../scratch") == ("scratch",)
    assert _level("rm -rf build") == "allow"


def test_an_unresolvable_target_keeps_the_textual_answer(relocated, monkeypatch) -> None:
    def boom(path: str, **_kwargs: object) -> str:
        raise OSError(40, "Too many levels of symbolic links")

    monkeypatch.setattr(os.path, "realpath", boom)
    assert policy._real_path("/whatever") is None
    # A failed lookup must never DROP a check.
    assert _level("rm -rf /etc") == "deny"
    assert _level(f"rm -rf {relocated['link'] / 'inputs'}") == "deny"


def test_a_symlink_parents_semantics_are_preserved(relocated) -> None:
    """``link/outputs/../inputs`` follows the PHYSICAL parent, so it reaches the
    read-only mount rather than a sibling of the link."""
    through = str(relocated["link"] / "outputs" / ".." / "inputs")
    assert policy._under_run_read_only_root(through)
    assert _level(f"rm -rf {through}") == "deny"


def test_static_system_roots_are_matched_under_every_spelling(tmp_path) -> None:
    """macOS spells ``/etc`` as ``/private/etc``; the alias must not bypass."""
    alias = tmp_path / "private-etc"
    alias.symlink_to("/etc")
    token = install_filesystem_state(state_for_sandbox_mode(
        "native", workspace=str(tmp_path / "ws"), outputs=str(tmp_path / "out"),
    ))
    try:
        assert _level(f"rm -rf {alias}") == "deny"
        assert _level(f"echo x > {alias}/hosts") == "deny"
    finally:
        reset_filesystem_state(token)


def test_system_aliases_do_not_block_writable_mounts(relocated) -> None:
    """Writable exemptions precede static checks, including resolved /var."""
    assert policy._under_static_protected_root("/var/log/syslog")
    target = str(relocated["link"] / "outputs" / "a.md")
    assert policy._within_writable_root(target)
    assert _level(f"echo x > {target}") == "allow"


def test_a_symlink_out_of_a_writable_root_is_not_exempt(relocated, tmp_path) -> None:
    """``/tmp`` is itself a writable root, so a link planted there that points
    at a system path is textually contained. Exempting it on that basis would
    retire the static protection; both spellings have to be contained."""
    escape = tmp_path / "looks-local"
    escape.symlink_to("/etc")
    assert not policy._within_writable_root(str(escape / "hosts"))
    assert _level(f"echo x > {escape}/hosts") == "deny"
    assert _level(f"rm -rf {escape}") == "deny"


def test_a_symlinked_writable_root_is_still_exempt(relocated) -> None:
    # The counterpart: the run's OWN outputs reached through the link spelling
    # resolves to the run's own outputs, so both names are contained.
    for spelling in ("link", "real"):
        target = str(relocated[spelling] / "outputs" / "a.md")
        assert policy._within_writable_root(target), spelling


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("template", [
    "rm -rf {run}",
    "rm -rf {run}/*",
    "rm -rf {run}/in*",
    "find {run} -delete",
    "chmod -R 777 {run}",
    "chown -R nobody {run}",
])
def test_recursive_ancestor_cannot_delete_read_only_inputs(tmp_path, mode, template):
    run = tmp_path / "run"
    for name in ("workspace", "outputs", "inputs"):
        (run / name).mkdir(parents=True)
    token = install_filesystem_state(state_for_sandbox_mode(
        "native", workspace=str(run / "workspace"), outputs=str(run / "outputs"),
        inputs=str(run / "inputs"), tmpdir=str(tmp_path),
    ))
    try:
        assert _level(template.format(run=run), mode) == "deny"
        assert _level(f"rm -rf {run}/outputs/stale", mode) != "deny"
    finally:
        reset_filesystem_state(token)


@pytest.mark.parametrize("mode", MODES)
def test_system_aliases_are_protected_but_scratch_is_writable(tmp_path, mode):
    token = install_filesystem_state(state_for_sandbox_mode(
        "native", workspace=str(tmp_path / "ws"), outputs=str(tmp_path / "out"),
    ))
    try:
        resolved_etc = os.path.realpath("/etc")
        assert _level(f"echo x > {resolved_etc}/hosts", mode) == "deny"
        assert _level(f"rm -rf {resolved_etc}", mode) == "deny"
        assert _level(f"echo x > {tmp_path}/scratch.txt", mode) != "deny"
    finally:
        reset_filesystem_state(token)
