"""A redirection is not a command name.

Four denials across the two measured runs were the policy reading shell syntax
as a binary: ``diff <(a) <(b)`` and a ``for``/``while`` loop ending ``done >
file`` were refused as "`<` / `>` is not on the allowed-command list", and one
was refused as "`rag_fixed.md` is not on the allowed-command list" — the
redirect's target read as the command. Redirection was never the objection;
``cat a > b`` has always been allowed. Each cost the sub-agent a turn against an
unactionable message.

Two shapes produce a segment that is nothing but a redirection: ``done < f``,
where the loop keyword becomes the syntax sentinel, and each half of
``diff <(a) <(b)``, because the splitter breaks on the parenthesis.
"""

import pytest

from plugins.tools._bash_policy import assess_bash_command


def _level(command: str) -> str:
    return assess_bash_command(command, mode="enforce").level


def _reason(command: str) -> str:
    return assess_bash_command(command, mode="enforce").reason


# ── the regression ───────────────────────────────────────────────────────────

@pytest.mark.parametrize("command", [
    # process substitution — the splitter breaks on "("
    'diff <(head -100 a.txt) <(head -100 b.txt) | head -20',
    'comm -12 <(sort a.txt) <(sort b.txt)',
    # a compound command's trailing redirect, both directions
    'while IFS= read -r f; do echo "$f"; done < /workspace/inventory.txt',
    'for f in a b; do cat /workspace/$f.md; done > /workspace/merged.txt',
    'for f in a b; do cat $f; done >> /workspace/merged.txt',
    'if [ -f a ]; then cat a; fi > /workspace/out.txt',
    'while read -r line; do echo "$line"; done <<< "one line"',
])
def test_a_segment_that_is_only_a_redirection_is_not_a_command(command) -> None:
    assert _level(command) == "allow", _reason(command)


def test_the_redirect_target_is_not_read_as_a_command() -> None:
    """The "`rag_fixed.md` is not on the allowed-command list" case: skipping
    the operator but not its target just moves the mistake one token right."""
    reason = _reason('while read -r l; do echo "$l"; done < rag_fixed.md')

    assert "rag_fixed.md" not in reason
    assert _level('while read -r l; do echo "$l"; done < rag_fixed.md') == "allow"


# ── what must not change ─────────────────────────────────────────────────────

@pytest.mark.parametrize("command", [
    'head -60 /workspace/ragas_paper.txt',
    'cat /workspace/a.md > /workspace/b.md',
    'python3 x.py 2>&1 | tail -5',
    'echo hi &> /workspace/log.txt',
    'curl -sL https://example.test/a.pdf -o /workspace/a.pdf',
])
def test_ordinary_commands_are_still_allowed(command) -> None:
    assert _level(command) == "allow", _reason(command)


def test_a_command_hidden_after_a_redirection_is_still_assessed() -> None:
    """Skipping operator AND target follows shell semantics: in ``> out cmd``
    the command really is ``cmd``. Skipping to end of segment instead would
    have made a redirection prefix an escape hatch."""
    assert _level("> /workspace/out rm -rf /etc") == "deny"
    assert _level("> /workspace/out findmnt --target /") == "deny"


@pytest.mark.parametrize("command", [
    # A bare here-string / noclobber operator takes its target from the next
    # token. Matching it as ``<<`` / ``>`` with an attached target would assess
    # ``cat`` and let the real command after it bypass the allowlist.
    "<<< cat findmnt --target /",
    "2<<< cat sudo id",
    ">| cat findmnt --target /",
    "2>| cat sudo id",
])
def test_multi_character_redirect_cannot_hide_a_command(command) -> None:
    assert _level(command) == "deny"


@pytest.mark.parametrize("command", [
    "'>evil' cat /workspace/input",
    "'2>&1' cat /workspace/input",
    "env PATH=/workspace '>evil'",
])
def test_a_quoted_command_name_is_not_redirection_syntax(command) -> None:
    """Quote removal must not turn an executable name into a redirect token."""
    assert _level(command) == "deny"


def test_a_genuinely_absent_binary_is_still_denied() -> None:
    """The four fixed cases were parse artifacts. A real denial must survive,
    named accurately."""
    reason = _reason("findmnt --target /workspace | head")

    assert "`findmnt`" in reason
    assert _level("findmnt --target /workspace | head") == "deny"


def test_privilege_escalation_survives_a_redirection_prefix() -> None:
    assert _level("sudo tee /etc/hosts < /workspace/h") == "deny"


def test_both_scanners_agree_about_what_a_segment_runs() -> None:
    """``strip_command_prefixes`` feeds the deliverable policy and
    ``_resolve_exe`` feeds the allowlist. Its docstring exists because the two
    disagreeing is a bug class, so the redirection skip belongs to both."""
    from plugins.tools._bash_policy import (
        _SHELL_SYNTAX_TOKEN,
        _resolve_exe,
        strip_command_prefixes,
        tokenize_shell_segment,
    )

    argv = [
        _SHELL_SYNTAX_TOKEN,
        *tokenize_shell_segment("< /workspace/inventory.txt"),
    ]

    assert strip_command_prefixes(argv) == []
    assert _resolve_exe(argv) == (None, [])

    hidden = tokenize_shell_segment(
        "> /workspace/out cp /workspace/a /outputs/b",
    )
    assert strip_command_prefixes(hidden)[0] == "cp"
    assert _resolve_exe(hidden)[0] == "cp"
