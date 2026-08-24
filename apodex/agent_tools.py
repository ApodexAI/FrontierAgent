"""The local coding tool surface (Claude-Code parity) + risk classification.

All tools are FrontierAgent's existing ``plugins.tools`` LangChain tools,
passed straight to ``run_agent_loop(tools=...)``. We deliberately pick the
*local* file/shell tools (no E2B/sandbox, no web) so the agent works on
the user's real repository:

    bash · read_file · grep_search · glob_search ·
    file_editor_view / file_editor_create / file_editor_str_replace · write_file
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any

# Imported at module load (not per-call) so a transient import failure can't
# silently downgrade a denied command at risk-assessment time. ``None`` only
# if the whole tool module is unavailable (a broken install — loud elsewhere).
try:
    from plugins.tools.bash import assess_bash_command as _assess_bash_command
except Exception:  # pragma: no cover
    _assess_bash_command = None


def coding_tools() -> list[Any]:
    """Return the LangChain tool objects for the local coding agent.

    ``bash`` is our **local-cwd** bash (apodex.local_tools), not
    the shared ``plugins.tools.bash`` — the latter runs in an E2B/tempdir
    sandbox, on a different filesystem than the local file-edit tools.
    """
    from apodex.local_tools import (
        bash,
        delete_file,
        glob_search,
        grep_search,
        read_file,
    )
    from apodex.plan import exit_plan_mode
    from apodex.todo import todo_write
    from plugins.tools.file_editor import (
        file_editor_create,
        file_editor_str_replace,
        file_editor_view,
    )
    from plugins.tools.write_file import write_file

    return [
        bash,
        read_file,
        grep_search,
        glob_search,
        file_editor_view,
        file_editor_create,
        file_editor_str_replace,
        write_file,
        delete_file,
        todo_write,
        exit_plan_mode,
    ]


def research_tools() -> list[Any]:
    """Tools for the deep-research agent: web search/fetch + local compute.

    Uses our local ``bash`` (for ``python3 -c`` computation) instead of the
    shared ``run_python_code`` so it stays off the slow E2B/sandbox path.
    """
    from apodex.local_tools import bash, read_file
    from apodex.todo import todo_write
    from plugins.tools.web_fetch import web_fetch
    from plugins.tools.web_search import web_search

    return [web_search, web_fetch, bash, read_file, todo_write]


def terminal_tool_registry() -> dict[str, Any]:
    """Authoritative ``name -> tool`` map for what YAML profiles may request.

    The union of :func:`coding_tools` and :func:`research_tools` (LOCAL
    bash/read_file/etc — not the sandboxed ``plugins.tools`` variants; see
    ``coding_tools``), plus ``read_text`` so a skills-enabled profile can load
    full ``SKILL.md`` bodies (which live outside the workspace the local,
    path-gated ``read_file`` would refuse). Profiles resolve their ``tools:``
    list through this; an unknown name is a hard error at load time.
    """
    reg: dict[str, Any] = {}
    for t in (*coding_tools(), *research_tools()):
        name = getattr(t, "name", "")
        if name:
            reg.setdefault(name, t)
    try:  # optional — only needed by skills-enabled profiles
        from plugins.tools.read_text import read_text
        reg.setdefault("read_text", read_text)
    except Exception:  # pragma: no cover - broken install
        pass
    try:
        # In-process: it reads this run's own trajectory off the host, so the same
        # implementation serves the TUI and the sandboxed workflows unchanged.
        from plugins.tools.recover_result import recover_result
        reg.setdefault("recover_result", recover_result)
    except Exception:  # pragma: no cover - broken install
        pass
    return reg


# Tools that only read state / fetch / update the plan / manage tasks — never need approval.
# web_search/web_fetch are network reads; safe to auto-run like a local read.
_READ_ONLY = frozenset({
    "read_file", "grep_search", "glob_search", "file_editor_view", "todo_write",
    "web_search", "web_fetch", "read_text", "view_image", "recover_result",
    # Task board & planning built-ins
    "add_task", "update_task", "finish_planning",
    # Subagent & report workflow built-ins
    "create_subagent", "assign_task", "collect_reports", "stop_subagent",
    "submit_report", "finalize_answer",
})
# Tools that mutate the working tree — always confirmed (unless auto-approve)
# AND journaled (snapshot-before, so the change is diffable + revertable).
_WRITE_TOOLS = frozenset({
    "write_file", "file_editor_create", "file_editor_str_replace", "delete_file",
})
# Tools that write through the SANDBOX rather than the host cwd — the shipped
# react / agent_team workflows produce every deliverable with ``create_file``.
# They mutate state, so they must be confirmed with a visible target and locked
# by plan mode; they are deliberately outside _WRITE_TOOLS because their paths
# (``/outputs/report.docx``) legitimately live outside cwd, and the cwd deny
# there would refuse every deliverable.
_SANDBOX_WRITE_TOOLS = frozenset({"create_file", "download_file"})
# Public alias for the journal/observer layer. Only host-cwd writes can be
# snapshotted, so sandbox writes stay out (see the /revert help text).
MUTATING_TOOLS = _WRITE_TOOLS

# Risk levels, lowest → highest. The TUI gates anything above "safe".
RISK_SAFE = "safe"
RISK_CONFIRM = "confirm"
RISK_DENY = "deny"


@dataclass
class ToolRisk:
    level: str
    reason: str
    # Best-effort human-readable target (file path / command) for the prompt.
    target: str = ""
    # Non-empty label when the call is *destructive* (delete / dep-install /
    # dangerous shell). Unlike kimi's cosmetic red banner, this is wired into
    # the decision: the gate demands a deliberate typed confirmation.
    danger: str = ""


# Destructive patterns that warrant a SECOND (typed) confirmation, even though
# they're not auto-denied. Covers what kimi's table misses: dependency installs
# and destructive git. Matched against the full bash command string.
_DANGER_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\brm\s+(-\w*[rf]\w*|--recursive|--force)", re.I), "recursive/forced delete"),
    (re.compile(r"\bsudo\b", re.I), "sudo (root)"),
    (re.compile(r"\b(curl|wget)\b[^|]*\|\s*(sudo\s+)?(sh|bash|zsh|python\d?)\b", re.I), "pipe-to-shell"),
    (re.compile(r"\bdd\b[^|]*\bof=", re.I), "dd raw write"),
    (re.compile(r"\bmkfs\b", re.I), "mkfs (format)"),
    (re.compile(r">\s*/dev/(sd|nvme|disk|hd)", re.I), "write to raw device"),
    (re.compile(r"\bchmod\s+-?R?\s*0?777\b", re.I), "chmod 777"),
    (re.compile(r":\(\)\s*\{\s*:\|:&\s*\}", 0), "fork bomb"),
    (re.compile(r"\bgit\s+push\b[^|;&]*(--force\b|(?<!-)-f\b)", re.I), "git force-push"),
    (re.compile(r"\bgit\s+reset\s+--hard\b", re.I), "git reset --hard"),
    (re.compile(r"\bgit\s+clean\s+-\w*f", re.I), "git clean -f"),
    (re.compile(
        r"\b(pip3?|uv|npm|pnpm|yarn|poetry|conda|gem|cargo|go|apt|apt-get|brew)\b"
        r"[^|;&]*\b(install|add|sync)\b", re.I,
    ), "installs dependencies"),
]


def detect_danger(cmd: str) -> str:
    """Return a danger label if ``cmd`` matches a destructive pattern, else ''."""
    for pat, label in _DANGER_PATTERNS:
        if pat.search(cmd or ""):
            return label
    return ""


# Programs that only read/inspect — safe to auto-approve. Conservative
# allowlist (not a denylist): anything not here falls through to confirm.
_READONLY_CMDS = frozenset({
    "ls", "cat", "head", "tail", "wc", "find", "grep", "egrep", "fgrep", "rg",
    "tree", "pwd", "echo", "printf", "which", "type", "file", "stat", "du",
    "df", "printenv", "date", "whoami", "uname", "hostname", "id",
    "basename", "dirname", "realpath", "readlink", "uniq", "cut",
    "column", "tr", "nl", "tac", "diff", "cmp", "sha256sum", "md5sum",
    "ps", "top", "true", "test",
})
# NOTE: deliberately NOT read-only: ``env`` (can exec an arbitrary command),
# ``sort`` (``-o``/``--output`` writes a file), ``xargs`` (runs anything).
# git subcommands that don't mutate the repo / working tree.
_GIT_READONLY = frozenset({
    "status", "diff", "log", "show", "branch", "rev-parse", "ls-files",
    "remote", "blame", "describe", "tag", "config", "shortlog", "name-rev",
})
# Mutation/exec verbs + flags that veto auto-approve even under a read-only
# leading program: redirects, command substitution, **background ``&``**,
# write-capable ``find`` actions, package/exec verbs, etc. When in doubt the
# command falls through to "confirm" (fail-safe), never silent auto-run.
_MUTATION_GUARD = re.compile(
    r">>?|\$\(|`|(?<![&>])&(?!&)|"                       # redirect / subst / background &
    r"\b(rm|mv|cp|dd|mkfs|tee|truncate|chmod|chown|chgrp|ln|kill|pkill|"
    r"reboot|shutdown|install|pip|npm|pnpm|yarn|uv|apt|brew|make|sudo|"
    r"xargs|eval|exec|source|env|sort)\b|"               # exec-ish / write-capable
    r"-exec(dir)?\b|-ok(dir)?\b|-delete\b|-f(print|printf|ls)\b",  # find write/exec actions
    re.IGNORECASE,
)


def is_mutating_tool(name: str, args: dict) -> bool:
    """True if a tool call would change the working tree / system state — i.e.
    the calls Plan mode must block. A write tool, or a non-read-only ``bash``."""
    if name in _WRITE_TOOLS or name in _SANDBOX_WRITE_TOOLS:
        return True
    if name == "bash":
        return not is_read_only_bash(str(args.get("command", "")))
    return False


def is_read_only_bash(cmd: str) -> bool:
    """True only when ``cmd`` is confidently read-only (auto-approvable).

    Allowlist of inspection programs across every ``&&``/``||``/``|``/``;``
    segment, plus a guard that rejects redirections, command substitution, and
    mutation verbs/flags. Anything uncertain → False (→ confirm), fail-safe.
    """
    if not cmd or _MUTATION_GUARD.search(cmd):
        return False
    for seg in re.split(r"&&|\|\||\||;", cmd):
        toks = seg.strip().split()
        if not toks:
            return False
        prog = os.path.basename(toks[0])
        if prog == "git":
            if len(toks) < 2 or toks[1] not in _GIT_READONLY:
                return False
        elif prog not in _READONLY_CMDS:
            return False
    return True


# Tools whose ``path`` arg we localize (abs→rel) to keep them on the fast
# local path. read_file/file_editor try a (slow, fc-cache-bootstrapping)
# sandbox FIRST for ABSOLUTE paths — a relative path under cwd skips that
# entirely (~0.05s vs ~50s).
_PATH_TOOLS = frozenset({
    "read_file", "write_file", "file_editor_view", "file_editor_create",
    "file_editor_str_replace", "grep_search", "glob_search",
})


def localize_path_args(name: str, args: dict, cwd: str) -> dict | None:
    """If a tool's ``path`` is an absolute path *inside* cwd, return a copy of
    ``args`` with it rewritten relative to cwd; else ``None`` (no rewrite).

    Avoids the absolute-path sandbox slow path in read_file/file_editor.
    """
    if name not in _PATH_TOOLS:
        return None
    cwd_real = os.path.realpath(cwd)
    inputs_raw = os.environ.get("FRONTIER_AGENT_INPUTS_DIR", "").strip()
    inputs_real = os.path.realpath(inputs_raw) if inputs_raw else ""
    for key in ("path", "file_path"):
        p = args.get(key)
        if not isinstance(p, str) or not os.path.isabs(p):
            continue
        p_real = os.path.realpath(p)
        # Uploaded inputs are a separately-authorized read-only mount. Keep
        # their canonical absolute path: workflow tools can have a different
        # execution root from the terminal cwd, so cwd-relative rewriting is
        # both unnecessary and ambiguous for this namespace.
        if inputs_real and (
            p_real == inputs_real or p_real.startswith(inputs_real + os.sep)
        ):
            return None
        try:
            rel = os.path.relpath(p_real, cwd_real)
        except Exception:
            continue
        if not (rel == ".." or rel.startswith(".." + os.sep)):
            new = dict(args)
            new[key] = rel or "."
            return new
        # Absolute path OUTSIDE cwd: if the model meant a repo-root path like
        # "/README.md" and "<cwd>/README.md" exists, rewrite to that relative
        # form (avoids the absolute-path sandbox slow path for all file tools).
        alt = os.path.realpath(os.path.join(cwd_real, p.lstrip("/")))
        if (alt == cwd_real or alt.startswith(cwd_real + os.sep)) and os.path.exists(alt):
            new = dict(args)
            new[key] = os.path.relpath(alt, cwd_real)
            return new
    return None


def _arg_path(args: dict) -> str:
    for k in ("path", "file_path"):
        v = args.get(k)
        if isinstance(v, str) and v:
            return v
    return ""


def _sandbox_write_target(name: str, args: dict) -> str:
    """The destination to show for a sandbox write, as the tool will resolve it.

    ``create_file``'s ``path`` IS the destination. ``download_file``'s is not:
    the argument is optional, its directory components are ignored, the file
    lands in the workspace downloads directory, and a name collision renames it
    (``p.pdf`` → ``p-1.pdf``). Echoing the raw argument would name a file the
    tool never writes, which is worse than naming none.
    """
    if name != "download_file":
        return _arg_path(args)
    try:
        from plugins.tools.download_file import _download_dir

        directory = _download_dir()
    except Exception:
        directory = "<workspace>/downloads"
    requested = os.path.basename(str(args.get("path") or "").replace("\\", "/")).strip()
    if not requested:
        return f"{directory}/ (filename taken from the URL)"
    return f"{directory}/{requested} (renamed if it already exists)"


def _outside_cwd(path: str, cwd: str) -> bool:
    """True if ``path`` resolves outside ``cwd``. Fail-CLOSED: on any error
    (or empty path for a write) treat it as outside so the gate denies rather
    than silently allowing an unverifiable write."""
    if not path:
        return True
    try:
        base = os.path.realpath(cwd)
        target = os.path.realpath(path if os.path.isabs(path) else os.path.join(cwd, path))
        return os.path.commonpath([base, target]) != base
    except Exception:
        return True


def assess_tool_risk(name: str, args: dict, cwd: str) -> ToolRisk:
    """Classify a proposed tool call for the human-approval gate.

    Read-only tools are ``safe``. Writes are ``confirm`` (and ``deny`` if
    the path escapes the working directory). ``bash`` reuses FrontierAgent's
    own :func:`plugins.tools.bash.assess_bash_command` denylist.
    """
    if name in _READ_ONLY:
        return ToolRisk(RISK_SAFE, "read-only", _arg_path(args))

    if name in _WRITE_TOOLS:
        path = _arg_path(args)
        if _outside_cwd(path, cwd):
            return ToolRisk(RISK_DENY, "writes outside the working directory", path)
        if name == "delete_file":
            # Destructive (though journaled/revertable) → second confirmation.
            return ToolRisk(RISK_CONFIRM, "deletes a file", path, danger="deletes a file")
        return ToolRisk(RISK_CONFIRM, "modifies a file", path)

    if name in _SANDBOX_WRITE_TOOLS:
        return ToolRisk(
            RISK_CONFIRM, "writes a file in the sandbox",
            _sandbox_write_target(name, args),
        )

    if name == "bash":
        cmd = str(args.get("command", "")).strip()
        if _assess_bash_command is not None:
            try:
                a = _assess_bash_command(cmd)
            except Exception:
                # Fail closed: refuse rather than silently downgrading a
                # possibly destructive command to a confirmable one.
                return ToolRisk(RISK_DENY, "could not assess bash command safety", cmd)
            if a.level == "deny":
                return ToolRisk(RISK_DENY, a.reason, cmd)
        # Read-only inspection (ls/find/grep/tree/git status/…) runs without a
        # prompt so the agent isn't blocked on every harmless command. Anything
        # that could mutate state still requires confirmation.
        if is_read_only_bash(cmd):
            return ToolRisk(RISK_SAFE, "read-only command", cmd)
        # Destructive commands (rm -rf, dep-install, git force-push, …) keep
        # confirm-level but flag danger → the gate demands a typed confirmation.
        return ToolRisk(RISK_CONFIRM, "runs a shell command", cmd, danger=detect_danger(cmd))

    # Unknown tool → be cautious.
    return ToolRisk(RISK_CONFIRM, "non-read tool", "")


def assess_with_rules(
    name: str, args: dict, cwd: str, rules: Any = None, *, auto_for_me: bool = False,
) -> ToolRisk:
    """:func:`assess_tool_risk` plus persistent allow/deny rules and auto_for_me mode.

    Layering keeps the safety contract:
    1. A saved ``deny`` forces a block.
    2. Hard ``RISK_DENY`` (writes outside working directory, dangerous system blocks)
       is NEVER bypassed.
    3. If ``auto_for_me`` is enabled (Docker / trusted env mode), any non-denied call
       is treated as safe.
    4. If the user saved an explicit ``allow`` rule for this command/tool, downgrade
       ``RISK_CONFIRM`` to ``RISK_SAFE``.
    """
    base = assess_tool_risk(name, args, cwd)
    if rules is not None and rules.denies(name, args):
        return ToolRisk(RISK_DENY, "denied by a saved rule", base.target)
    if base.level == RISK_DENY:
        return base
    if auto_for_me:
        return ToolRisk(RISK_SAFE, "auto for me (docker/trusted env)", base.target)
    if base.level == RISK_CONFIRM and rules is not None and rules.allows(name, args):
        return ToolRisk(RISK_SAFE, "allowed by a saved rule", base.target)
    return base
