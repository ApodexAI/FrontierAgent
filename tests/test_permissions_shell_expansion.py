"""Compare saved shell permissions with actual bash expansion semantics."""

from __future__ import annotations

import itertools
import shlex
import shutil
import subprocess

import pytest

from apodex.agent_tools import RISK_CONFIRM, RISK_DENY, assess_with_rules
from apodex.permissions import PermissionStore, _fallback_nested_shell
from plugins.tools._bash_policy import _extract_nested_shell, assess_bash_command


def _backtick(code: str) -> str:
    """Encode a shell body for one legacy command-substitution level."""
    escaped = "".join("\\" + c if c in "\\$`" else c for c in code)
    return "`" + escaped + "`"


@pytest.mark.parametrize("body,expected", [
    (r"echo \$(touch marker)", "echo $(touch marker)"),
    (r"echo \`touch marker\`", "echo `touch marker`"),
    (r"echo \\$(touch marker)", r"echo \$(touch marker)"),
    (r"echo \q", r"echo \q"),
    ("echo to\\\nuch", "echo touch"),
])
def test_backtick_scanners_apply_bash_escape_removal(body, expected):
    command = "echo `" + body + "`"
    assert _extract_nested_shell(command) == [expected]
    assert _fallback_nested_shell(command) == [expected]


@pytest.mark.parametrize("command", [
    r"echo `echo \$(foobarcmd)`",
    r"echo `echo \`foobarcmd\``",
    'echo "`echo \\$(foobarcmd)`"',
])
def test_backtick_escape_removal_reaches_enforced_policy(command):
    assert assess_bash_command(command, mode="enforce").level == "deny"


def test_saved_permissions_match_real_bash_nested_expansions(tmp_path):
    """Every generated command actually runs touch; none may inherit echo's allow.

    Mix both substitution syntaxes and double-quoted variants at each level.
    Execute only harmless commands that write one marker in this test's folder.
    """
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("bash is needed for the shell-semantics comparison")
    marker = tmp_path / "marker"
    marker_word = shlex.quote(marker.as_posix())
    allow = PermissionStore(allow={"Bash(echo)"})
    deny = PermissionStore(allow={"Bash(*)"}, deny={"Bash(touch)"})
    wrappers = (
        lambda code: "echo $(" + code + ")",
        lambda code: 'echo "$(' + code + ')"',
        lambda code: "echo " + _backtick(code),
        lambda code: 'echo "' + _backtick(code) + '"',
    )
    commands = []
    for depth in range(1, 4):
        for sequence in itertools.product(wrappers, repeat=depth):
            command = "touch " + marker_word
            for wrap in sequence:
                command = wrap(command)
            commands.append(command)
    # The concrete review examples also include a second evaluation level.
    commands.extend([
        r"echo `echo \$(touch " + marker_word + r")`",
        r"echo `echo \`touch " + marker_word + r"\``",
        'echo $(echo "$(echo ")\'")" $(touch ' + marker_word + '))',
    ])
    for command in commands:
        result = subprocess.run(
            [bash, "--noprofile", "--norc", "-c", command],
            cwd=tmp_path, capture_output=True, text=True, timeout=5,
        )
        assert result.returncode == 0, (command, result.stderr)
        assert marker.exists(), command
        marker.unlink()
        args = {"command": command}
        assert assess_with_rules("bash", args, str(tmp_path), allow).level == RISK_CONFIRM, command
        assert assess_with_rules("bash", args, str(tmp_path), deny).level == RISK_DENY, command
        assert _extract_nested_shell(command) == _fallback_nested_shell(command), command


@pytest.mark.parametrize("command", [
    "echo '$(touch marker)'",
    "echo '`touch marker`'",
    r"echo \`touch marker\`",
    r"echo `echo '\$(touch marker)'`",
    "echo $(echo literal)",
    "echo `echo literal`",
])
def test_literal_and_authorized_backtick_payloads_stay_allowed(command):
    assert PermissionStore(allow={"Bash(echo)"}).allows("bash", {"command": command})
