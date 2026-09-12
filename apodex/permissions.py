"""Persistent per-command allow/deny rules (Claude-Code parity).

A session-global "auto-approve everything" (the ``[a]`` key) trades all future
safety for convenience. This adds granular, persisted rules — "always allow
``npm test``", "never allow ``git push``" — matched by command prefix, so the
gate stays livable without being all-or-nothing.

Rules are strings: ``Bash(npm test)`` / ``Bash(git push)`` for shell (matched by
prefix across every ``&&``/``|``/``;`` segment, fail-safe), or a bare tool name
(``write_file``) for everything else.

Safety contract: this store only ever *downgrades a plain confirm to safe*, or
*forces a deny*. It is consulted in :func:`agent_tools.assess_tool_risk` AFTER
danger detection and the hard denylist — so a saved ``Bash(git)`` allow can
never green-light a dangerous ``git push --force``. Unquoted ``$(...)`` and
backtick substitutions must be separately authorized against the same saved
prefixes, and a command carrying a ``danger`` label never downgrades (the
typed-confirmation gate still fires).
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
    """Shell-code strings nested in unquoted ``$(...)``/backticks.

    Reuses :func:`plugins.tools._bash_policy._extract_nested_shell` (stdlib-only,
    no import cycle). Single-quoted spans are skipped — the shell does not expand
    them, so ``echo '$(rm -rf /)'`` is a harmless literal. Falls back to a small
    self-contained scanner when the import fails so matching never throws and
    never silently allows.
    """
    try:
        from plugins.tools._bash_policy import (  # type: ignore
            _extract_nested_shell as _extract,
        )

        return list(_extract(cmd or ""))
    except Exception:
        pass
    out: list[str] = []
    s = cmd or ""
    n = len(s)
    i = 0
    sq = False
    while i < n:
        c = s[i]
        if sq:
            if c == "'":
                sq = False
            i += 1
            continue
        if c == "'":
            sq = True
            i += 1
            continue
        if c == "$" and i + 1 < n and s[i + 1] == "(":
            depth, j = 1, i + 2
            start = j
            while j < n and depth:
                if s[j] == "(":
                    depth += 1
                elif s[j] == ")":
                    depth -= 1
                j += 1
            if depth == 0:
                out.append(s[start : j - 1])
            i = j
            continue
        if c == "`":
            j = i + 1
            while j < n and s[j] != "`":
                j += 1
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
    segs = [p.strip() for p in _SEGMENT_SPLIT.split(nested or "") if p.strip()]
    if not segs:
        return False
    for seg in segs:
        if not any(seg == p or seg.startswith(p + " ") for p in prefixes):
            return False
        # Transitive: ``echo $(foo $(bar))`` needs ``bar`` authorized as well.
        for inner in _nested_shell_snippets(seg):
            if not _nested_segments_authorized(inner, prefixes):
                return False
    return True


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
        return self._matches(self.allow, name, args)

    def denies(self, name: str, args: dict) -> bool:
        return self._matches(self.deny, name, args)

    def add_allow(self, name: str, args: dict) -> str:
        """Persist 'always allow' for this call; returns the rule added."""
        rule = rule_for(name, args)
        self.allow.add(rule)
        self.save()
        return rule

    @staticmethod
    def _matches(rules: set[str], name: str, args: dict) -> bool:
        if name in rules:
            return True
        if name == "bash":
            if "bash" in rules or "Bash" in rules or "Bash(*)" in rules:
                return True
            prefixes = {r[5:-1] for r in rules if r.startswith("Bash(") and r.endswith(")")}
            if not prefixes:
                return False
            segs = [s.strip() for s in _SEGMENT_SPLIT.split(str(args.get("command", ""))) if s.strip()]
            # Filter out helper segments (e.g. 'cd /foo') unless all segments are helpers
            non_helpers = [
                s for s in segs
                if _extract_prefix_from_segment(s).split()[0] not in _HELPER_CMDS
            ]
            check_segs = non_helpers if non_helpers else segs
            if not check_segs:
                return False
            for seg in check_segs:
                if not any(seg == p or seg.startswith(p + " ") for p in prefixes):
                    return False
                # Nested-shell guard (issue #39): ``echo $(pip install x)`` is
                # "just an echo" only on the raw string. Each unquoted nested
                # payload must independently match a saved prefix, else the
                # whole command is not authorized.
                for nested in _nested_shell_snippets(seg):
                    if not _nested_segments_authorized(nested, prefixes):
                        return False
            return True
        return False


__all__ = ["PermissionStore", "rule_for"]
