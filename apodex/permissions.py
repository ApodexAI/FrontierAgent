"""Persistent per-command allow/deny rules (Claude-Code parity).

A session-global "auto-approve everything" (the ``[a]`` key) trades all future
safety for convenience. This adds granular, persisted rules — "always allow
``npm test``", "never allow ``git push``" — matched by command prefix, so the
gate stays livable without being all-or-nothing.

Rules are strings: ``Bash(npm test)`` / ``Bash(git push)`` for shell, or a bare
tool name (``write_file``) for everything else. Shell rules match by prefix per
``&&``/``|``/``;`` segment. An allow must cover every segment and a deny fires
on any one segment, so both fail safe.

Safety contract: this store only ever *downgrades a plain confirm to safe*, or
*forces a deny*. It is consulted in :func:`agent_tools.assess_tool_risk` AFTER
danger detection and the hard denylist — so a saved ``Bash(git)`` allow can
never green-light a dangerous ``git push --force``. Every ``$(...)`` or
backtick substitution the shell would run, unquoted or inside double quotes,
needs its own match against the saved allow prefixes. A deny prefix also fires
on a command nested inside one. A command carrying a ``danger`` label never
downgrades, so the typed-confirmation gate still fires.
"""

from __future__ import annotations

import json
import os
import re
import shlex
from dataclasses import dataclass, field

_DEFAULT_PATH = os.path.expanduser("~/.config/apodex/permissions.json")
# Commands whose first token alone is too coarse — keep two words so
# "always allow npm test" doesn't also allow "npm publish".
_MULTI_VERB = frozenset({
    "git", "npm", "pnpm", "yarn", "uv", "pip", "pip3", "cargo", "go", "docker",
    "poetry", "conda", "make", "apt", "apt-get", "brew", "kubectl", "gh",
})
_SEGMENT_SPLIT = re.compile(r"&&|\|\||\||;")
_HELPER_CMDS = frozenset({
    "cd", "pwd", "export", "set", "env", "echo", "mkdir", "clear", "true", "source", ".",
})


def _nested_shell_snippets(cmd: str) -> list[str]:
    """Shell-code strings nested in ``$(...)``/backticks the shell would run.

    Reuses :func:`plugins.tools._bash_policy._extract_nested_shell` (stdlib-only,
    no import cycle). It reads quotes the way bash does. A single-quoted span is
    skipped, so ``echo '$(rm -rf /)'`` is a harmless literal. A ``'`` inside
    double quotes is an ordinary character, so ``echo "'$(rm -rf /)'"`` still
    yields ``rm -rf /``. If the import fails it uses
    :func:`_fallback_nested_shell`, so matching never throws and never silently
    allows. ``test_nested_shell_extractors_agree`` checks the two give the same
    results.
    """
    try:
        from plugins.tools._bash_policy import (  # type: ignore
            _extract_nested_shell as _extract,
        )

        return list(_extract(cmd or ""))
    except Exception:
        return _fallback_nested_shell(cmd or "")


def _fallback_substitution_end(s: str, i: int) -> int:
    """Index of the ``)`` closing a ``$(`` whose body starts at ``i`` (``len``
    when unterminated); quoted or escaped parens don't count."""
    depth, n = 1, len(s)
    quote: str | None = None
    while i < n:
        c = s[i]
        if quote == "'":
            if c == "'":
                quote = None
        elif c == "\\":
            i += 1
        elif c == '"':
            quote = None if quote else '"'
        elif quote is None:
            if c == "'":
                quote = "'"
            elif c == "(":
                depth += 1
            elif c == ")":
                depth -= 1
                if depth == 0:
                    return i
        i += 1
    return n


def _fallback_nested_shell(s: str) -> list[str]:
    """Standalone copy of ``_extract_nested_shell`` for when its import fails.

    It tracks single quotes, double quotes and backslash escapes. An
    unterminated substitution yields the rest of the string, which fails closed.
    """
    out: list[str] = []
    i, n = 0, len(s)
    quote: str | None = None
    while i < n:
        c = s[i]
        if quote == "'":
            if c == "'":
                quote = None
            i += 1
            continue
        if c == "\\":
            i += 2
            continue
        if c == "'" and quote is None:
            quote = "'"
        elif c == '"':
            quote = None if quote else '"'
        elif c == "$" and s.startswith("(", i + 1):
            end = _fallback_substitution_end(s, i + 2)
            out.append(s[i + 2 : end])
            i = end + 1
            continue
        elif c == "`":
            j = i + 1
            while j < n and s[j] != "`":
                j += 2 if s[j] == "\\" else 1
            out.append(s[i + 1 : j])
            i = j + 1
            continue
        i += 1
    return out


def _nested_segments_authorized(nested: str, prefixes: set[str]) -> bool:
    """True when every ``&&``/``|``/``;`` piece of a nested snippet matches.

    Each piece must itself satisfy the same ``seg == p or seg.startswith(p)``
    prefix rule, transitively (a nested snippet containing further substitution
    must have that inner payload authorized too). Fail-closed: empty or
    unmatched pieces return False.
    """
    segs = _segments(nested)
    if not segs:
        return False
    for seg in segs:
        if not _seg_matches(seg, prefixes):
            return False
        # Transitive: ``echo $(foo $(bar))`` needs ``bar`` authorized as well.
        for inner in _nested_shell_snippets(seg):
            if not _nested_segments_authorized(inner, prefixes):
                return False
    return True


def _segments(cmd: str) -> list[str]:
    return [s.strip() for s in _SEGMENT_SPLIT.split(cmd or "") if s.strip()]


def _seg_matches(seg: str, prefixes: set[str]) -> bool:
    return any(seg == p or seg.startswith(p + " ") for p in prefixes)


def _denied_anywhere(cmd: str, prefixes: set[str]) -> bool:
    """True when any segment matches a deny prefix, at the top level or nested
    in a substitution at any depth.

    This is the opposite of the allow check, which needs every segment to
    match. A nested command that matches no rule never cancels a deny on its
    parent, so ``Bash(echo)`` still denies ``echo $(touch x)``.
    """
    if any(_seg_matches(seg, prefixes) for seg in _segments(cmd)):
        return True
    return any(_denied_anywhere(inner, prefixes) for inner in _nested_shell_snippets(cmd))


def _extract_prefix_from_segment(seg: str) -> str:
    try:
        toks = shlex.split(seg)
    except Exception:
        toks = seg.split()
    if not toks:
        return ""
    if toks[0] in _MULTI_VERB and len(toks) > 1:
        return f"{toks[0]} {toks[1]}"
    return toks[0]


def _bash_prefix(cmd: str) -> str:
    """A reusable prefix for a bash command (``cd /app && npm test`` → ``npm test``)."""
    segs = [s.strip() for s in _SEGMENT_SPLIT.split(cmd or "") if s.strip()]
    if not segs:
        return ""
    # Prefer the first non-helper segment so 'cd /foo && python script.py' saves Bash(python)
    for seg in segs:
        p = _extract_prefix_from_segment(seg)
        if p and p.split()[0] not in _HELPER_CMDS:
            return p
    return _extract_prefix_from_segment(segs[0])


def rule_for(name: str, args: dict) -> str:
    """The allow-rule string an 'always allow' on this call would create."""
    if name == "bash":
        p = _bash_prefix(str(args.get("command", "")))
        return f"Bash({p})" if p else "bash"
    return name


@dataclass
class PermissionStore:
    """Allow/deny rules, persisted to a JSON file."""

    allow: set[str] = field(default_factory=set)
    deny: set[str] = field(default_factory=set)
    path: str = _DEFAULT_PATH

    @classmethod
    def load(cls, path: str = _DEFAULT_PATH) -> PermissionStore:
        try:
            with open(path, encoding="utf-8") as f:
                d = json.load(f)
            return cls(set(d.get("allow") or []), set(d.get("deny") or []), path)
        except Exception:
            return cls(path=path)

    def save(self) -> None:
        try:
            os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump({"allow": sorted(self.allow), "deny": sorted(self.deny)}, f, indent=2)
        except Exception:
            pass

    def allows(self, name: str, args: dict) -> bool:
        """True only when every segment and every nested payload matches."""
        if _name_matches(self.allow, name):
            return True
        prefixes = _bash_rule_prefixes(self.allow) if name == "bash" else set()
        if not prefixes:
            return False
        cmd = str(args.get("command", ""))
        # Issue #39. On the raw string ``echo $(pip install x)`` looks like a
        # plain echo, so every payload the shell would run has to match a saved
        # prefix on its own. Payloads come from the whole command before the
        # helper filter runs. Otherwise ``echo $(touch x) && python -V`` would
        # pass, because the filter drops the ``echo`` segment.
        for nested in _nested_shell_snippets(cmd):
            if not _nested_segments_authorized(nested, prefixes):
                return False
        segs = _segments(cmd)
        # Filter out helper segments (e.g. 'cd /foo') unless all segments are helpers
        non_helpers = [
            s for s in segs
            # ``or [""]`` stops an empty-word segment (``"" && ...``) raising IndexError.
            if (_extract_prefix_from_segment(s).split() or [""])[0] not in _HELPER_CMDS
        ]
        check_segs = non_helpers if non_helpers else segs
        return bool(check_segs) and all(_seg_matches(s, prefixes) for s in check_segs)

    def denies(self, name: str, args: dict) -> bool:
        """True when any segment, top-level or nested, matches a deny rule."""
        if _name_matches(self.deny, name):
            return True
        prefixes = _bash_rule_prefixes(self.deny) if name == "bash" else set()
        return bool(prefixes) and _denied_anywhere(str(args.get("command", "")), prefixes)

    def add_allow(self, name: str, args: dict) -> str:
        """Persist 'always allow' for this call; returns the rule added."""
        rule = rule_for(name, args)
        self.allow.add(rule)
        self.save()
        return rule


def _name_matches(rules: set[str], name: str) -> bool:
    """A bare tool-name rule, or a whole-tool bash rule (``bash``/``Bash(*)``)."""
    return name in rules or (
        name == "bash" and ("Bash" in rules or "Bash(*)" in rules)
    )


def _bash_rule_prefixes(rules: set[str]) -> set[str]:
    return {r[5:-1] for r in rules if r.startswith("Bash(") and r.endswith(")")}


__all__ = ["PermissionStore", "rule_for"]
