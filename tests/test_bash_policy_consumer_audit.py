"""Every data consumer must be audited for ways it runs code.

Tools in ``_DATA_CONSUMERS`` get their quoted words and heredoc bodies skipped
by the Layer-1 word screens (#628). A tool with an option, subcommand or
argument form that executes a program would turn that skipped text back into
an unscreened command. So each consumer must appear in exactly one table
below. Adding a tool to ``_DATA_CONSUMERS`` without classifying it here fails
this test on purpose: check its man page for exec-style options first.
"""
from __future__ import annotations

import pytest

from plugins.tools import _bash_policy as policy
from plugins.tools._bash_policy import assess_bash_command

# Guarded by a dedicated function in ``_treats_words_as_data`` (grammar or
# subcommand allowlist rather than an option pattern).
_GUARDED_BY_FUNCTION = {
    "sed": "positive grammar: s///, p/d/q only; e command and e flag fall back",
    "tar": "ordinary archive options only; checkpoint/to-command/-I fall back",
    "awk": "system(), pipes and getline fall back",
    "gawk": "same as awk",
    "mawk": "same as awk",
    "git": "data subcommands only; -x/--exec/--upload-pack/-u/-O/-c fall back",
    "uv": "pip/add/remove/sync/lock/venv/init/tree/version/python only",
}

# Audited: no option, subcommand or argument form that runs a program.
_AUDITED_NO_EXEC = {
    "echo": "prints", "cat": "prints", "tac": "prints", "tee": "writes files",
    "head": "prints", "tail": "prints", "wc": "counts", "uniq": "filters",
    "cut": "filters", "tr": "filters", "paste": "filters", "nl": "filters",
    "fold": "filters", "fmt": "filters", "column": "filters", "rev": "filters",
    "grep": "searches", "egrep": "searches", "fgrep": "searches",
    "diff": "compares", "cmp": "compares", "comm": "compares",
    "jq": "no system/exec builtin", "yq": "no exec builtin",
    "base64": "encodes", "md5sum": "hashes", "sha1sum": "hashes",
    "sha256sum": "hashes", "sha512sum": "hashes", "iconv": "converts",
    "ls": "lists", "stat": "reads metadata", "file": "reads magic",
    "du": "sizes", "df": "sizes", "mkdir": "creates dirs", "rmdir": "removes dirs",
    "touch": "creates files", "cp": "copies (tracked as written for write-then-run)",
    "mv": "moves (tracked)", "ln": "links (tracked)",
    "rm": "deletes (argv hard deny guards targets)",
    "chmod": "modes (argv hard deny guards targets)", "chown": "owners (same)",
    "find": "-exec/-execdir/-ok payloads are extracted and screened as code",
    "basename": "paths", "dirname": "paths", "realpath": "paths",
    "readlink": "paths", "mktemp": "creates temp file",
    "unzip": "extracts", "gzip": "compresses",
    "gunzip": "decompresses", "xz": "compresses",
    "pdftotext": "converts", "pdftoppm": "converts", "pdfinfo": "reads",
    "qpdf": "transforms PDFs",
    "curl": "transfers; no exec option",
    "systemctl": "shutdown operations recognised by _systemctl_requests_shutdown",
    "cd": "builtin", "pwd": "builtin", "test": "no subscript arithmetic (checked in bash)",
    "[": "same as test", "true": "builtin", "false": "builtin", ":": "builtin",
    "set": "builtin", "unset": "no subscript execution (checked in bash)",
    "shopt": "builtin", "exit": "no subscript execution (checked in bash)",
    "return": "same as exit", "sleep": "external", "date": "formats",
    "wait": "no subscript execution (checked in bash)",
}

# Accepted: can run arbitrary code by design; the sandbox bounds them, and
# blanking their text is the point of #628 (prose written into gen.py).
_ACCEPTED_INTERPRETERS = {
    "python": "interpreter", "python3": "interpreter", "node": "interpreter",
    "pip": "installs run package build code", "pip3": "same as pip",
}


def _classified() -> dict[str, set[str]]:
    return {
        "_CONSUMER_EXEC_OPTIONS": set(policy._CONSUMER_EXEC_OPTIONS),
        "_GUARDED_BY_FUNCTION": set(_GUARDED_BY_FUNCTION),
        "_AUDITED_NO_EXEC": set(_AUDITED_NO_EXEC),
        "_ACCEPTED_INTERPRETERS": set(_ACCEPTED_INTERPRETERS),
    }


def test_every_data_consumer_is_audited():
    classified = set().union(*_classified().values())
    unaudited = set(policy._DATA_CONSUMERS) - classified
    assert not unaudited, (
        f"{sorted(unaudited)} added to _DATA_CONSUMERS without an audit: check "
        "for options/subcommands/arguments that run a program, then add a guard "
        "to _CONSUMER_EXEC_OPTIONS (or a dedicated check) or record it in "
        "_AUDITED_NO_EXEC with a reason."
    )
    stale = classified - set(policy._DATA_CONSUMERS)
    assert not stale, f"{sorted(stale)} audited but no longer data consumers"


def test_each_consumer_is_classified_once():
    seen: dict[str, str] = {}
    for table, tools in _classified().items():
        for tool in tools:
            assert tool not in seen, f"{tool} in both {seen[tool]} and {table}"
            seen[tool] = table


def test_function_guards_are_dispatched():
    # A tool listed as function-guarded must actually reach its check: an
    # exec form of each is denied rather than blanked.
    examples = {
        "sed": "sed 's/x/halt/e' /tmp/in", "tar": "tar -cf o.tar --to-command='halt' i",
        "awk": "awk 'BEGIN{system(\"halt\")}'", "gawk": "gawk 'BEGIN{system(\"halt\")}'",
        "mawk": "mawk 'BEGIN{system(\"halt\")}'", "git": "git rebase -x 'halt' HEAD~1",
        "uv": "uv run 'halt'",
    }
    assert set(examples) == set(_GUARDED_BY_FUNCTION)
    for tool, command in examples.items():
        result = assess_bash_command(command, mode="off")
        assert result.level == "deny", (tool, result)


@pytest.mark.parametrize("command", [
    # arithmetic subscript evaluation runs $(...) even inside single quotes
    "[[ 'a[$(halt)]' -eq 1 ]]",
    "printf -v 'x[$(halt)]' '%s' v",
    "read 'x[$(halt)]' <<< v",
    "declare -i y='a[$(halt)]'",
    "typeset -i y='a[$(reboot)]'",
    "f(){ local -i z='a[$(halt)]'; }; f",
    "ag --pager 'halt' x",
])
def test_builtin_and_option_exec_forms_keep_the_raw_guard(command):
    result = assess_bash_command(command, mode="off")
    assert (result.level, result.reason) == ("deny", "Refuses host shutdown/reboot commands.")


@pytest.mark.parametrize("command", [
    "printf '%s\\n' 'market halt' > /workspace/a.md",
    "read -r line <<< 'exchange halt'",
    "declare note='exchange halt'",
    "[[ 'exchange halt' == *'halt'* ]]",
    "ag 'halt' /workspace",
])
def test_same_builtins_with_plain_prose_stay_data(command):
    result = assess_bash_command(command, mode="off")
    assert result.level in {"allow", "audit"}, (command, result)
