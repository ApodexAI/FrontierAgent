"""Bash command safety policy — argv-level allowlist + hard denylist.

Enforces command execution safety for sandboxed and host bash invocations.
Supports three policy modes ('off', 'warn', 'enforce'):
  - allow: safe read-only and routine operations
  - audit: logged operations that require extra tracking
  - confirm: destructive or sensitive actions requiring approval
  - deny: unsafe operations (destructive system commands, network bypasses)
"""

from __future__ import annotations

import contextlib
import contextvars
import fnmatch
import logging
import os
import re
import shlex
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BashCommandAssessment:
    level: str  # "allow" | "audit" | "confirm" | "deny"
    reason: str
    # Name of the always-denied group (``priv_esc`` / ``exfil`` /
    # ``process_kill``) that produced this verdict, else "". Lets an
    # interactive caller tell a group hit apart from an ordinary confirm.
    group: str = ""


# ── Mode resolution ─────────────────────────────────────────────────────

_VALID_MODES = ("off", "warn", "enforce")
_DEFAULT_MODE = "off"
# Ordered loosest → strictest, so a tighten-only source can be reconciled by
# taking the stricter of two modes.
_MODE_RANK = {"off": 0, "warn": 1, "enforce": 2}

# Per-run override, set by workflows adjacent to their per-task sandbox (mirrors
# the ``_task_sandbox`` contextvar pattern). Propagates into child asyncio tasks
# (agent_team sub-agents inherit the main agent's mode automatically).
_policy_mode_var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "mh_bash_policy_mode", default=None,
)


def set_policy_mode(mode: str) -> contextvars.Token:
    """Set the bash-policy mode for the current context. Returns a token for
    :func:`reset_policy_mode`.

    An unrecognised mode is refused with a warning and stores ``None`` (→ falls
    back to env / config / default). The warning matters: a typo used to fail
    SILENTLY toward ``_DEFAULT_MODE``, i.e. toward ``off`` — a caller asking for
    ``enfoce`` got the allowlist switched off, which is the one direction a
    mistake must never take.
    """
    if mode not in _VALID_MODES:
        logger.warning(
            "ignoring unknown bash policy mode (expected one of %s); "
            "falling back to env / config / %s",
            ", ".join(_VALID_MODES), _DEFAULT_MODE,
        )
        return _policy_mode_var.set(None)
    return _policy_mode_var.set(mode)


def reset_policy_mode(token: contextvars.Token) -> None:
    """Restore the previous bash-policy mode."""
    _policy_mode_var.reset(token)


def _env_mode() -> str:
    return (os.environ.get("BASH_ALLOWLIST_MODE") or "").strip().lower()


def _config_mode() -> str:
    try:
        from frontier_agent.infra.config import get_config
        return (getattr(get_config(), "bash_allowlist_mode", "") or "").strip().lower()
    except Exception:
        return ""


def _scope_mode() -> str:
    """Mode declared in the current ExecutionScope metadata (optional wiring —
    a workflow may pass ``scope_metadata={"bash_allowlist_mode": ...}``)."""
    try:
        from frontier_agent.core.execution_context import get_current_execution_scope
        scope = get_current_execution_scope()
        meta = scope.metadata if scope else {}
        return str(meta.get("bash_allowlist_mode") or "").strip().lower()
    except Exception:
        return ""


def resolve_mode(explicit: str | None = None) -> str:
    """Resolve the effective policy mode.

    Precedence: explicit arg → ``BASH_ALLOWLIST_MODE`` env (ops override) →
    per-run contextvar (workflow default) → config → ``off``.

    ExecutionScope metadata is workload input rather than a statement about the
    environment, so it can only TIGHTEN the mode the trusted sources settled on
    (``off`` → ``enforce``), never loosen it. An explicit argument comes from
    the calling code itself and is taken as is.
    """
    if explicit:
        candidate = explicit.strip().lower()
        if candidate in _VALID_MODES:
            return candidate
    resolved = _DEFAULT_MODE
    for candidate in (
        _env_mode(),
        _policy_mode_var.get() or "",
        _config_mode(),
    ):
        if candidate in _VALID_MODES:
            resolved = candidate
            break
    scope = _scope_mode()
    if scope in _VALID_MODES and _MODE_RANK[scope] > _MODE_RANK[resolved]:
        return scope
    return resolved


# ── Layer 1: hard denylist (all modes) ──────────────────────────────────

# Raw-string patterns kept from the original bash.py denylist. They stay as a
# cheap first screen; the argv analysis below is the robust complement.
_DENY_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"\bmkfs(\.\w+)?\b", "Refuses filesystem formatting commands."),
    (r"\bdd\s+if=.*\sof=/dev/", "Refuses raw writes to block devices."),
    (r"\b(shutdown|reboot|poweroff|halt)\b", "Refuses host shutdown/reboot commands."),
    (r":\(\)\s*\{\s*:\|:\s*&\s*\};:", "Refuses fork bombs."),
    (r"\bDROP\s+TABLE\b", "Refuses destructive database schema deletion."),
    (r">\s*/dev/sd[a-z]", "Refuses raw writes to block devices."),
)

_CONFIRM_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"\bgit\s+reset\s+--hard\b", "Requires confirmation for hard reset."),
    (r"\bgit\s+clean\s+-fdx\b", "Requires confirmation for destructive git clean."),
    (r"\bchmod\s+-R\s+777\b", "Requires confirmation for broad permission changes."),
    (r"\bchown\s+-R\b", "Requires confirmation for recursive ownership changes."),
)

_AUDIT_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"\b(npm|pnpm|yarn|uv|pip)\s+install\b", "Package installation should be audited."),
    (r"\bcurl\b.+\|\s*(sh|bash)\b", "Piped remote shell scripts should be audited."),
)

# Paths whose recursive deletion / mutation is refused outright. ``/workspace``,
# ``/outputs`` and ``/tmp`` are intentionally absent — an agent clearing its own
# scratch/output space is legitimate (and contained by the sandbox).
_PROTECTED_TARGETS = frozenset({
    "/", "/*", "~", "~/", "$HOME", "${HOME}", "$HOME/", "${HOME}/",
    "/inputs", "/etc", "/usr", "/bin", "/sbin", "/lib", "/lib64", "/boot",
    "/dev", "/proc", "/sys", "/var", "/root", "/home", "/opt",
})

# System / input roots under which recursive deletion is refused (``/etc/*``,
# ``/usr/local`` …). The writable sandbox dirs (/workspace, /outputs, /tmp) are
# deliberately excluded so an agent can clear its own scratch/output space.
_SYSTEM_ROOTS = (
    "/etc", "/usr", "/bin", "/sbin", "/lib", "/lib64", "/boot", "/dev",
    "/proc", "/sys", "/var", "/root", "/opt", "/inputs",
)


def _norm_target(arg: str) -> str:
    """Normalise a path argument for protected-target comparison: collapse
    repeated slashes and resolve ``..``/``.`` so ``/tmp/../etc`` -> ``/etc`` and
    ``//bin`` -> ``/bin`` can't dodge the protected-path checks."""
    a = arg.strip().strip("'\"")
    if not a:
        return "/"
    a = re.sub(r"/{2,}", "/", a)   # collapse repeated slashes (``//bin`` -> ``/bin``)
    a = os.path.normpath(a)        # resolve ``..``/``.`` (``/tmp/../etc`` -> ``/etc``)
    return a or "/"


# Shell parameter expansions: ``${VAR}`` / ``$VAR`` / ``$@`` / ``$*`` / ``$1``.
_VAR_RE = re.compile(r"\$\{[^}]*\}|\$[\w@*#?!-]+")


def _writable_roots() -> tuple[str, ...]:
    """This run's own writable directories, resolved per call.

    Read from the trusted runtime filesystem state, so they are exactly the
    directories the prompts name, the file tools authorize and the shell
    variables point at. Inputs are absent: that mount is read-only.
    """
    try:
        from plugins.tools._filesystem_state import current_filesystem_state

        return tuple(root for root in current_filesystem_state().scratch_roots() if root)
    except Exception:
        return ()


def _local_filesystem_paths() -> bool:
    """Whether these path spellings name THIS host's filesystem.

    The same flag the file-tool gate uses, so the two cannot disagree about
    whether resolving a path here describes the right machine.
    """
    try:
        from plugins.tools._filesystem_state import current_filesystem_state

        return current_filesystem_state().local_filesystem
    except Exception:
        return False


def _real_path(path: str) -> str | None:
    """``path`` with symlinks resolved, or ``None`` when it cannot be resolved.

    Non-strict: a path that does not exist still resolves. An ``OSError`` (a
    symlink loop, for instance) returns ``None`` and the caller keeps its
    textual answer, so a failed lookup never DROPS a check.
    """
    try:
        return os.path.realpath(path)
    except OSError:
        return None


def _path_spellings(path: str) -> tuple[str, ...]:
    """Every name this one path answers to: as written, and resolved.

    Comparing spelling alone made protection depend on how a path was typed:
    ``rm -rf /inputs`` was refused while ``rm -rf $(realpath /inputs)`` —
    the same directory — was allowed, and on macOS ``/private/etc`` reached
    ``/etc``. The runtime hands out both spellings itself, so this has to
    compare identity.

    Resolved only when the state says these paths are local AND the path is
    absolute with no unexpanded variable: the shell's cwd is not this process's,
    so resolving a relative operand would name a different file, and a remote
    path must never be interpreted by this host.
    """
    written = _norm_target(path)
    raw = path.strip().strip("'\"")
    if (
        not raw.startswith("/")
        or _VAR_RE.search(raw)
        or not _local_filesystem_paths()
    ):
        return (written,)
    # The RAW path is resolved, not the normalized one: collapsing ``..``
    # against a symlinked parent names a different file than the shell opens.
    resolved = _real_path(raw)
    return (written,) if resolved is None or resolved == written else (written, resolved)


def _within_writable_root(path: str) -> bool:
    """Whether ``path`` is strictly inside one of this run's writable roots.

    The exemption that keeps a relocated mount usable: ``_REDIRECT_PROTECTED_RE``
    matches a literal prefix, and a run directory legitimately sits under one —
    macOS ``$TMPDIR`` is ``/var/folders/...``, a container volume is
    ``/var/lib/app/run``. Refusing a redirect there while ``tee`` and ``cp`` to
    the same path were allowed meant the deny message recommended the very
    directory it had just refused, and the deliverable was lost.

    ``..`` is collapsed before comparing, so this cannot be used to climb out.
    An unexpanded variable is never treated as contained: its value is unknown
    here.
    """
    raw = path.strip().strip("'\"")
    if not raw.startswith("/") or _VAR_RE.search(raw):
        return False
    prefixes: set[str] = set()
    for root in _writable_roots():
        prefixes.add(_norm_target(root).rstrip("/") + "/")
        if _local_filesystem_paths():
            resolved_root = _real_path(root)
            if resolved_root:
                prefixes.add(resolved_root.rstrip("/") + "/")
    if not prefixes:
        return False
    # EVERY spelling must land inside a writable root, not merely one of them.
    # ``/tmp`` is itself a writable root, so a symlink there pointing at
    # ``/etc`` is textually contained; exempting it on that basis would retire
    # the static protection altogether. Requiring both names keeps #589's case
    # working — a file under a relocated outputs directory resolves to a file
    # under the resolved outputs directory, and both roots are in this set —
    # while a link that leads OUT of the run's directories is not exempt.
    candidates = _path_spellings(raw)
    return all(
        any(candidate.startswith(prefix) for prefix in prefixes)
        for candidate in candidates
    )


def _relocated_protected_roots() -> tuple[str, ...]:
    """Read-only roots of THIS run, wherever they were mounted.

    ``_SYSTEM_ROOTS`` names the canonical ones; a run whose inputs are mounted
    somewhere else entirely still must not have them deleted, and that mount
    holds the only copy of the user's files.
    """
    try:
        from plugins.tools._filesystem_state import current_filesystem_state

        return tuple(
            root for root in current_filesystem_state().read_only_roots() if root
        )
    except Exception:
        return ()


def _under_static_protected_root(path: str) -> bool:
    """Whether ``path`` names one of the canonical system roots.

    Local system roots include their resolved aliases (``/etc`` becomes
    ``/private/etc`` on macOS). Callers exempt this run's writable mounts
    first, so resolving ``/var`` does not block legitimate scratch files.
    Remote system paths are never resolved against the host.

    Loses to :func:`_within_writable_root`: a run directory legitimately sits
    under one of these prefixes.
    """
    roots: set[str] = set(_SYSTEM_ROOTS)
    if _local_filesystem_paths():
        roots.update(resolved for root in _SYSTEM_ROOTS if (resolved := _real_path(root)))
    return any(
        candidate in _PROTECTED_TARGETS or any(
            candidate == root or candidate.startswith(root + "/")
            for root in roots
        )
        for candidate in _path_spellings(path)
    )


def _under_run_read_only_root(path: str) -> bool:
    """Whether ``path`` is inside one of THIS run's read-only mounts.

    Wins over :func:`_within_writable_root`, which is the only ordering that
    works: the inputs mount is frequently nested inside a writable root (a run
    directory under ``$TMPDIR``, or anywhere under ``/tmp``), so an exemption
    applied first would hand back the one directory holding the user's own
    files — and it holds the only copy.
    """
    for root in _relocated_protected_roots():
        for root_spelling in {_norm_target(root), *_path_spellings(root)}:
            prefix = root_spelling.rstrip("/")
            if any(
                candidate == prefix or candidate.startswith(prefix + "/")
                for candidate in _path_spellings(path)
            ):
                return True
    return False


def _under_protected_root(path: str) -> bool:
    """Whether ``path`` names a protected root, under any of its spellings."""
    return _under_run_read_only_root(path) or _under_static_protected_root(path)


def _contains_run_read_only_root(path: str) -> bool:
    """Whether a recursive target selects a read-only root or its ancestor.

    Match ancestors too for glob operands such as ``/run/*``: deleting the
    parent or selecting inputs through a wildcard must not bypass protection.
    """
    targets = _path_spellings(path)
    for root in _relocated_protected_roots():
        for spelling in _path_spellings(root):
            ancestor = spelling
            while ancestor:
                if any(fnmatch.fnmatchcase(ancestor, target) for target in targets):
                    return True
                parent = os.path.dirname(ancestor)
                if parent == ancestor:
                    break
                ancestor = parent
    return False


def _is_delete_protected(arg: str) -> bool:
    """True if recursively deleting/mutating ``arg`` must be refused: the fs
    root, the home tree (``~`` / ``$HOME``), or anything under a system/input
    root. Writable sandbox dirs (/workspace, /outputs, /tmp) are allowed."""
    raw = arg.strip().strip("'\"")
    a = _norm_target(arg)
    if a in _PROTECTED_TARGETS or raw in _PROTECTED_TARGETS:
        return True
    if raw.startswith("~") or raw.startswith("$HOME") or raw.startswith("${HOME}"):
        return True
    # The RAW argument, so ``_path_spellings`` can resolve it: ``_norm_target``
    # has already collapsed ``..``, which loses symlink-parent semantics.
    target = raw if raw.startswith("/") else a
    if _under_run_read_only_root(target) or _contains_run_read_only_root(target):
        return True
    # This run's own workspace/outputs stay clearable even when they sit under
    # a protected prefix — ``rm -rf $OUTPUTS/stale`` with outputs under
    # ``/var/folders/...`` is ordinary cleanup, and refusing it while ``find
    # -delete`` on the same path was allowed only taught the model a detour.
    if _within_writable_root(target):
        return False
    if _under_static_protected_root(target):
        return True
    # An absolute-looking path whose variables strip to a protected root:
    # ``/$X`` -> ``/``, ``/$SYS/…`` -> ``/etc``. Requires a leading ``/`` so a
    # bare ``$SCRATCH`` (a legit workspace var) is NOT over-blocked.
    return bool(raw.startswith("/") and _VAR_RE.search(raw) and _under_protected_root(_norm_target(_VAR_RE.sub("", raw))))


def _short_flag_chars(argv: list[str]) -> set[str]:
    """Union of single-dash flag characters (``-rf`` → {r,f})."""
    chars: set[str] = set()
    for tok in argv:
        if tok.startswith("-") and not tok.startswith("--") and len(tok) > 1:
            chars.update(tok[1:])
    return chars


def _long_flags(argv: list[str]) -> set[str]:
    return {tok[2:] for tok in argv if tok.startswith("--")}


_PRIV_ESC = frozenset({"sudo", "su", "doas", "pkexec"})

# Shell control-flow leaders that precede a real command (``do rm``, ``if rm``,
# ``! rm``, ``{ rm``). ``_parse_commands`` recognizes them from the raw segment
# before shlex drops quoting and escaping. This distinction is security
# sensitive: ``done`` is syntax, while ``/tmp/done`` / ``'done'`` are external
# executables and must hit the allowlist.
_CONTROL_LEADERS = frozenset({
    "if", "while", "until", "then", "do", "else", "elif", "!", "{", "}",
    # Closing keywords are syntax, not commands — without them a well-formed
    # loop's trailing ``done`` was assessed as a binary named "done" and denied.
    "done", "fi", "esac",
})

# ``for x in a b c`` / ``select x in …`` headers contain no command at all —
# only the loop variable and a word list. Assessing them resolved to the loop
# VARIABLE as the executable (``for p in …`` → "`for` is not on the allowed
# -command list"), so a loop was unusable even when every command in its body
# was allowlisted. Any ``$(…)`` in the word list is still assessed separately
# by ``_extract_nested_shell``.
_LOOP_HEADERS = frozenset({"for", "select"})
_SHELL_SYNTAX_TOKEN = "__frontier_agent_shell_syntax__"

# A wrapper option's *separate* value token (``nice -n 10`` / ``timeout -s 9 10``
# / ``chrt -f 99``): a bare number, optionally with a decimal and/or a unit
# letter (``10s``, ``0.5``, ``5m``). NOT a command name, so it's safe to skip.
_WRAPPER_VALUE_RE = re.compile(r"\d+(\.\d+)?[a-zA-Z]?\Z")


# Options whose following word belongs to the wrapper, not its command
# (``env -u UNUSED bash`` / ``timeout --signal TERM 5 bash``).
_WRAPPER_OPTION_VALUES = {
    "env": {"-u", "--unset", "-C", "--chdir", "-S", "--split-string"},
    "sudo": {"-u", "--user", "-g", "--group", "-h", "--host", "-p", "--prompt"},
    "timeout": {"-s", "--signal", "-k", "--kill-after"},
    "nice": {"-n", "--adjustment"},
    "ionice": {"-c", "--class", "-n", "--classdata", "-p", "--pid"},
    "xargs": {"-I", "-n", "--max-args", "-P", "--max-procs", "-d", "--delimiter"},
}


def _skip_wrapper_args(argv: list[str], i: int, *, wrapper: str = "") -> int:
    """Advance ``i`` past a wrapper's option flags AND the separate value tokens
    they consume, so ``nice -n 10 rm`` / ``timeout -s 9 10 bash`` resolve to the
    real command (``rm`` / ``bash``) rather than the value token (``10``). This
    is the single home for wrapper-arg skipping — shared by
    strip_command_prefixes and
    _resolve_exe so they can't drift.

    ``wrapper`` names the wrapper being skipped, so a non-numeric option value
    (``env -u UNUSED``) is not read as the wrapped command."""
    n = len(argv)
    while i < n and (argv[i].startswith("-") or _WRAPPER_VALUE_RE.fullmatch(argv[i])):
        if argv[i] == "--":
            return i + 1
        if argv[i] in _WRAPPER_OPTION_VALUES.get(wrapper, ()):
            i += 2
            continue
        i += 1
    return i


# A redirection token as shlex leaves it: an optional fd or ``&`` prefix, the
# operator, and possibly the target already attached (``>file``, ``2>&1``).
# Keep longer operators first: otherwise bare ``<<<`` / ``>|`` look like
# ``<<`` / ``>`` with an attached target and can hide the real executable.
_REDIRECT_RE = re.compile(
    r"^(?:\d+|&)?(?:<<<|<<-?|<&|>&|>>|>\||<>|<|>)(.*)$",
)
_REDIRECT_SENTINEL = "\x1e"


_ANSI_C_SIMPLE_ESCAPES = {
    "a": "\a", "b": "\b", "e": "\x1b", "E": "\x1b", "f": "\f", "n": "\n",
    "r": "\r", "t": "\t", "v": "\v", "\\": "\\", "'": "'", '"': '"', "?": "?",
}


def _decode_ansi_c(body: str) -> str:
    """Decode the escapes bash applies inside ``$'...'``. Unknown escapes keep
    their backslash, as bash does, so decoding never fails."""
    out: list[str] = []
    i, n = 0, len(body)
    while i < n:
        c = body[i]
        if c != "\\" or i + 1 >= n:
            out.append(c)
            i += 1
            continue
        e = body[i + 1]
        if e in _ANSI_C_SIMPLE_ESCAPES:
            out.append(_ANSI_C_SIMPLE_ESCAPES[e])
            i += 2
        elif e in "01234567":
            m = re.match(r"[0-7]{1,3}", body[i + 1:])
            digits = m.group(0) if m else e
            out.append(chr(int(digits, 8) & 0xFF))
            i += 1 + len(digits)
        elif e in "xuU":
            limit = {"x": 2, "u": 4, "U": 8}[e]
            m = re.match(rf"[0-9A-Fa-f]{{1,{limit}}}", body[i + 2:])
            if m:
                try:
                    out.append(chr(int(m.group(0), 16)))
                except (ValueError, OverflowError):
                    out.append("\ufffd")
                i += 2 + len(m.group(0))
            else:
                out.append(body[i:i + 2])
                i += 2
        elif e == "c" and i + 2 < n:
            out.append(chr(ord(body[i + 2]) & 0x1F))
            i += 3
        else:
            out.append(body[i:i + 2])
            i += 2
    return "".join(out)


def _normalize_ansi_c_quotes(command: str) -> str:
    """Rewrite every unquoted ``$'...'`` word into the equivalent plain
    single-quoted word.

    ``shlex`` does not know ANSI-C quoting: it read ``bash -c $'sudo id'`` as
    the payload ``$sudo id`` — an executable named ``$sudo`` that matched no
    rule — and the ``\\'`` escape it allows also desynchronised every quote
    tracker in this module. Normalising once, before any scanner runs, gives
    all of them (and the recursion into payloads) the text bash executes.
    """
    if "$'" not in command:
        return command
    out: list[str] = []
    i, n = 0, len(command)
    quote: str | None = None
    while i < n:
        c = command[i]
        if quote:
            out.append(c)
            if c == "\\" and quote == '"' and i + 1 < n:
                out.append(command[i + 1])
                i += 2
                continue
            if c == quote:
                quote = None
            i += 1
            continue
        if c == "\\" and i + 1 < n:
            out.append(command[i:i + 2])
            i += 2
            continue
        if c == "$" and command[i + 1:i + 2] == "'":
            j = i + 2
            while j < n and command[j] != "'":
                j += 2 if command[j] == "\\" else 1
            body = command[i + 2:min(j, n)]
            decoded = _decode_ansi_c(body)
            out.append("'" + decoded.replace("'", "'\"'\"'") + "'")
            if j >= n:
                # Unterminated: keep the opening quote unbalanced so the parser
                # reports it instead of silently accepting the text.
                out.append("'")
            i = j + 1
            continue
        if c in ("'", '"'):
            quote = c
        out.append(c)
        i += 1
    return "".join(out)


def tokenize_shell_segment(segment: str) -> list[str]:
    """Split one shell segment while preserving which words are redirections.

    ``shlex.split`` removes quotes, so its output alone cannot distinguish the
    operator ``>evil`` from the quoted executable ``'>evil'``. Mark only
    unquoted redirect words before splitting; consumers can then skip syntax
    without allowing a quoted command name to bypass the executable policy.
    """
    if _REDIRECT_SENTINEL in segment:
        raise ValueError("shell segment contains a reserved control character")

    marked: list[str] = []
    quote: str | None = None
    i = 0
    while i < len(segment):
        char = segment[i]
        if quote:
            marked.append(char)
            if char == "\\" and quote == '"' and i + 1 < len(segment):
                marked.append(segment[i + 1])
                i += 2
                continue
            if char == quote:
                quote = None
            i += 1
            continue
        if char in ("'", '"'):
            quote = char
            marked.append(char)
            i += 1
            continue
        if char == "\\" and i + 1 < len(segment):
            marked.extend((char, segment[i + 1]))
            i += 2
            continue
        if char in ("<", ">"):
            word_start = len(marked)
            while word_start and not marked[word_start - 1].isspace():
                word_start -= 1
            prefix = "".join(marked[word_start:])
            if not prefix or prefix.isdigit() or prefix == "&":
                marked.insert(word_start, _REDIRECT_SENTINEL)
        marked.append(char)
        i += 1
    return shlex.split("".join(marked), comments=False)


def redirection_token(token: str) -> str | None:
    """The redirect spelling encoded in *token*, or ``None`` for a shell word."""
    if not token.startswith(_REDIRECT_SENTINEL):
        return None
    redirect = token[len(_REDIRECT_SENTINEL):]
    return redirect if _REDIRECT_RE.match(redirect) else None


def _skip_redirection(argv: list[str], i: int) -> int | None:
    """Index just past a redirection at ``argv[i]``, or ``None`` if not one.

    A segment can consist of nothing but a redirection. Two shapes reach here:
    ``done < f`` — the loop keyword becomes the syntax sentinel, leaving the
    redirect first — and each half of ``diff <(a) <(b)``, because
    :func:`_split_top_level` breaks on the parenthesis. Reading the operator as
    the command name denied ``diff`` and a ``for`` loop as "`<` is not on the
    allowed-command list", and reading the file after it denied one as
    "`rag_fixed.md` is not on the allowed-command list" — four wasted turns
    across two measured runs, on commands the policy actually permits, since
    ``cat a > b`` is allowed and redirection was never the objection.

    Skipping the target as well as the operator follows shell semantics rather
    than losing a command: in ``> out cmd args`` the command really is ``cmd``,
    and it still gets assessed.

    Sibling of the fix that made ``done`` itself syntax (see
    ``_CONTROL_LEADERS``). That one stopped the keyword being read as a binary;
    this stops its redirection being read as the next one.
    """
    redirect = redirection_token(argv[i])
    if redirect is None:
        return None
    match = _REDIRECT_RE.match(redirect)
    if match is None:
        return None
    # ``>file`` / ``2>&1`` carry their target; a bare operator takes the next.
    return i + 1 if match.group(1) else i + 2


def strip_command_prefixes(argv: list[str]) -> list[str]:
    """Return ``argv`` with leading ``VAR=val`` assignments, privilege-escalation
    prefixes (sudo …), command wrappers (env/timeout/xargs …) and control-flow
    leaders (do/then/if …) removed, so it starts at the *real* command. Lets the
    hard denylist catch ``sudo rm -rf /`` / ``do rm -rf /`` / ``xargs rm -rf /``
    regardless of prefix.

    Public because ``_deliverable_policy`` needs the same unwrapping before it
    can read a command's verb — ``env cp /workspace/x /outputs/leak.png`` hid
    the ``cp`` from its publisher check. One implementation so the two policies
    cannot disagree about what ``timeout 60 …`` actually runs."""
    i, n = 0, len(argv)
    while i < n:
        tok = argv[i]
        if _ASSIGN_RE.match(tok):
            i += 1
            continue
        after_redirect = _skip_redirection(argv, i)
        if after_redirect is not None:
            i = after_redirect
            continue
        base = _basename(tok)
        if base in _PRIV_ESC or base in _WRAPPERS:
            i = _skip_wrapper_args(argv, i + 1, wrapper=base)
            continue
        if tok == _SHELL_SYNTAX_TOKEN:
            i += 1
            continue
        break
    return argv[i:]


# Deleters whose target is unverifiable when fed from stdin via ``xargs``.
_STDIN_DELETERS = frozenset({"rm", "rmdir", "unlink", "shred"})
# ``find`` actions that run a command payload (analysed as a nested command).
_FIND_EXEC_FLAGS = frozenset({"-exec", "-execdir", "-ok", "-okdir"})


def _redirect_protection_reason(commands: list[list[str]]) -> str | None:
    """Why an output redirection in ``commands`` must be refused, or ``None``.

    The parsed counterpart of ``_REDIRECT_PROTECTED_RE``, and strictly stronger
    than it in two ways the regex cannot reach, because it matches a LITERAL
    prefix:

    * a target that climbs out — ``> $OUTPUTS/../../../../etc/passwd`` begins
      with this run's own directory, so the regex never looked at it, while
      ``_norm_target`` collapses it to ``/etc/passwd``;
    * this run's read-only mounts wherever they were mounted, which the regex's
      fixed list of system roots does not know about.

    Writable roots are exempt first (``_within_writable_root``), so a relocated
    outputs directory under ``/var`` stays usable.
    """
    for argv in commands:
        for raw_target in _redirect_targets(argv):
            target = raw_target.strip().strip("'\"")
            if not target.startswith("/") or _VAR_RE.search(target):
                continue
            if _REDIRECT_SAFE_DEVICE_RE.match(_norm_target(target)):
                continue
            # Read-only mounts first: see ``_under_run_read_only_root``.
            if not _under_run_read_only_root(target) and _within_writable_root(target):
                continue
            if _under_protected_root(target):
                writable = ", ".join(_writable_roots()) or "/workspace, /outputs or /tmp"
                return (
                    f"Refuses output redirection into a protected path "
                    f"(`{target}`). Write to {writable} instead; "
                    f"`>/dev/null` to discard is fine."
                )
    return None


def _argv_hard_deny(commands: list[list[str]]) -> str | None:
    """Robust dangerous-op detection on parsed argvs. Complements the raw
    regex — order-, flag-combination-, prefix- and quote-independent.
    """
    if any(_EXPANSION_SCREEN_FAILURE in argv for argv in commands):
        return "Cannot safely inspect shell code past the nesting limit; simplify the command."
    cd_into_protected = False
    for raw_argv in commands:
        argv = strip_command_prefixes(raw_argv)
        if not argv:
            continue
        exe = _basename(argv[0])
        args = argv[1:]

        # ``… | xargs rm -rf`` — targets arrive on stdin, invisible to argv, so
        # the target checks below can't see them. Refuse xargs feeding a deleter
        # (or a recursive chmod/chown) outright.
        if "xargs" in {_basename(t) for t in raw_argv} and (
            exe in _STDIN_DELETERS
            or exe in _SHELLS
            or (
                exe in ("chmod", "chown")
                and ("R" in _short_flag_chars(args) or "recursive" in _long_flags(args))
            )
        ):
                return (
                    f"Refuses `xargs` feeding `{exe}` — stdin targets/args can't "
                    "be validated; operate on explicit paths instead."
                )

        # ``cd /`` (or into any protected root) arms the next relative rm/find.
        if exe == "cd":
            cd_into_protected = bool(args) and _is_delete_protected(args[0])
            continue

        if exe == "rm":
            recursive = "r" in _short_flag_chars(args) or bool(
                _long_flags(args) & {"recursive"}
            )
            if recursive:
                targets = [a for a in args if not a.startswith("-")]
                if any(_is_delete_protected(t) for t in targets):
                    return "Refuses recursive deletion of a protected path."
                # ``cd / && rm -rf .`` / ``rm -rf *`` / ``rm -rf ..`` after a cd
                # into a root (_norm_target maps ``./``->``.`` and ``../``->``..``).
                if cd_into_protected and any(
                    _norm_target(t) in (".", "*", "..") for t in targets
                ):
                    return "Refuses recursive deletion of the current root directory."

        if exe == "find":
            starts = [a for a in args if not a.startswith("-")]
            deletes = ("-delete" in args) or (
                "-exec" in args and any(x in args for x in ("rm", "rmdir", "unlink"))
            )
            if deletes and any(_is_delete_protected(s) for s in starts):
                return "Refuses mass deletion via ``find`` on a protected path."

        if exe in ("chmod", "chown"):
            recursive = "R" in _short_flag_chars(args) or bool(
                _long_flags(args) & {"recursive"}
            )
            if recursive and any(_is_delete_protected(a) for a in args if not a.startswith("-")):
                return f"Refuses recursive {exe} on a protected path."

    return None


# ── Layer 2: command allowlist (warn / enforce) ─────────────────────────

# Read + analyse files, run python for compute, write to sandboxed dirs.
_ALLOWED_BINARIES = frozenset({
    # interpreters / compute
    "python", "python3", "python3.10", "python3.11", "python3.12", "python3.13",
    "ipython", "py", "bc", "dc", "expr", "numfmt",
    # file browsing / metadata
    "ls", "dir", "cat", "bat", "head", "tail", "file", "stat", "wc", "find",
    "fd", "tree", "realpath", "readlink", "basename", "dirname", "du", "df",
    "pwd", "cksum",
    # text processing
    "grep", "egrep", "fgrep", "rg", "ag", "sed", "awk", "gawk", "mawk", "cut",
    "sort", "uniq", "tr", "column", "jq", "yq", "comm", "join", "paste", "tac",
    "nl", "rev", "fold", "expand", "unexpand", "fmt", "tee", "strings", "od",
    "xxd", "hexdump", "diff", "cmp", "csvlook", "less", "more",
    # hashing / encoding
    "sha1sum", "sha224sum", "sha256sum", "sha384sum", "sha512sum", "md5sum",
    "base64", "base32",
    # archive read / extract
    "unzip", "zipinfo", "tar", "gzip", "gunzip", "zcat", "bzip2", "bunzip2",
    "bzcat", "xz", "unxz", "xzcat", "zstd", "unzstd", "7z", "7za", "unrar",
    # document / tabular extraction CLIs (present in the stateful-agent image)
    "pdftotext", "pdfinfo", "pdfimages", "pdftoppm", "pdftocairo",
    "in2csv", "csvcut", "csvgrep",
    "csvstat", "csvjson", "xlsx2csv", "csvtool", "antiword", "catdoc",
    "pandoc", "mutool", "ssconvert",
    # LibreOffice headless conversion. The agent-team / stateful-agent images
    # install libreoffice-{writer,calc,impress} precisely for this (see the
    # "headless convert/render (soffice)" note in both Dockerfiles), and it is
    # the ONLY office-format converter they carry — pandoc cannot write pptx
    # faithfully and ssconvert is spreadsheets only. Leaving it off the list
    # meant a task asking for a deck had no export path at all under the
    # profiles that do not expose ``create_file`` (apodex*, apex): the model
    # read "`soffice` is not on the allowed-command list", concluded the image
    # could not convert, and returned the source file or nothing.
    "soffice", "libreoffice", "lowriter", "localc", "loimpress", "lodraw",
    # network retrieval — task containers have a real filesystem and network.
    # Prefer the bounded download_file tool for documents; direct clients stay
    # available for APIs and compatibility with existing research skills.
    "curl", "wget", "aria2c",
    # dir/file mutation (contained to /workspace,/outputs,/tmp by the sandbox)
    "mkdir", "rmdir", "cp", "mv", "touch", "ln", "rm", "chmod", "split",
    "install", "truncate",
    # trivial builtins / navigation
    "cd", "echo", "printf", "true", "false", "test", "[", "[[", ":", "which",
    "type", "hash", "date", "seq", "sleep", "wait", "set", "unset", "read",
    "mapfile", "let", "export", "pushd", "popd", "dirs", "help",
    # package tooling — allowed but always audited (see _AUDIT_BINARIES).
    "pip", "pip3",
})

# Prefix wrappers — evaluate the command they wrap, not the wrapper itself.
_WRAPPERS = frozenset({
    "env", "nohup", "nice", "ionice", "time", "timeout", "stdbuf", "setsid",
    "chrt", "xargs", "command", "exec", "builtin",
})

# Denied binaries, split into NAMED GROUPS because they bind at different
# layers. ``_assess_allowlist`` (Layer 2, ``warn``/``enforce`` only) consults
# all of them. The groups in ``_ALWAYS_DENIED_GROUPS`` additionally bind in
# EVERY mode through ``_argv_group_deny`` (Layer 1.5): they used to be
# consulted only from Layer 2, which the default ``off`` mode never reaches, so
# ``sudo id`` / ``ssh host 'cat ~/.aws/credentials'`` / ``pkill -f python3``
# assessed as ``allow``. ``strip_command_prefixes`` deliberately *strips*
# ``sudo`` so the hard denylist can see the real command behind it, and no
# ``_DENY_PATTERNS`` entry covered the escalation itself.
#
# HTTP download clients are intentionally absent: current task containers have
# a writable filesystem and network access, and controlled document downloads
# use ``download_file``.
_DENY_GROUP_PRIV_ESC: dict[str, str] = {
    b: "Privilege escalation is not allowed." for b in ("sudo", "su", "doas", "pkexec")
}

_DENY_GROUP_NESTED_SHELL: dict[str, str] = {
    **{b: (
        "Nested/piped shells are not allowed — run the program directly or use "
        "a ``python3 <<'PY' ... PY`` heredoc."
    ) for b in ("bash", "sh", "zsh", "dash", "ksh", "csh", "tcsh", "fish", "ash")},
    "eval": "``eval`` of dynamic strings is not allowed.",
}

_DENY_GROUP_EXFIL: dict[str, str] = {
    b: (
        "Interactive network and remote-administration clients are not allowed. "
        "Use web/search/download tools or an HTTP client instead."
    ) for b in (
        "nc", "ncat", "netcat", "socat", "telnet",
        "ssh", "scp", "sftp", "ftp", "tftp", "rsync", "rclone",
    )
}

# Signal senders, split out of host administration so they bind in every mode.
# agent_team sub-agents share one uid and PID namespace, so ``pkill -f python3``
# from one sub-agent kills its siblings' commands, and the corpse surfaces as
# exit 137 that ``bash.py`` reports as a memory failure to the wrong agent.
_DENY_GROUP_PROCESS_KILL: dict[str, str] = {
    b: "System / host administration is not allowed." for b in ("kill", "killall", "pkill")
}

_DENY_GROUP_HOST_ADMIN: dict[str, str] = {
    b: "System / host administration is not allowed." for b in (
        "mount", "umount", "fdisk", "parted", "swapon", "systemctl", "service",
        "init", "kexec", "insmod", "modprobe", "sysctl", "iptables", "nft",
        "ip", "ifconfig", "route", "ufw",
        "crontab", "at", "batch",
    )
}

_DENY_GROUP_PKG_MGR: dict[str, str] = {
    b: "Installing system packages is not allowed." for b in (
        "apt", "apt-get", "aptitude", "yum", "dnf", "dpkg", "rpm", "pacman",
        "brew", "conda", "mamba", "snap",
    )
}

_DENY_GROUPS: dict[str, dict[str, str]] = {
    "priv_esc": _DENY_GROUP_PRIV_ESC,
    "nested_shell": _DENY_GROUP_NESTED_SHELL,
    "exfil": _DENY_GROUP_EXFIL,
    "process_kill": _DENY_GROUP_PROCESS_KILL,
    "host_admin": _DENY_GROUP_HOST_ADMIN,
    "pkg_mgr": _DENY_GROUP_PKG_MGR,
}

# Bound in every mode (Layer 1.5). Nested shells, host administration and
# package managers stay Layer-2 only: ``off`` is what the local coding CLI runs,
# where ``bash ./build.sh`` / ``make`` / ``apt-get`` are ordinary work behind a
# human approval gate.
_ALWAYS_DENIED_GROUPS = ("priv_esc", "exfil", "process_kill")

_DENIED_BINARIES: dict[str, str] = {
    binary: reason
    for group in _DENY_GROUPS.values()
    for binary, reason in group.items()
}

_ALWAYS_DENIED_BINARIES: dict[str, tuple[str, str]] = {
    binary: (group, reason)
    for group in _ALWAYS_DENIED_GROUPS
    for binary, reason in _DENY_GROUPS[group].items()
}

# Allowlisted but noteworthy → ``audit`` (runs, tagged).
_AUDIT_BINARIES = frozenset({"pip", "pip3"})

# Interpreters whose inline-code flag can't be introspected → ``audit``.
_INLINE_CODE_FLAGS = frozenset({"-c", "-e", "--command", "--eval"})

_ASSIGN_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")
# The target capture stops at shell punctuation rather than taking every
# non-space character: with ``\S*`` the single match ``>/outputs/a>/etc/passwd``
# swallowed the second redirect, so a writable-root exemption on the first one
# would have excused the write into ``/etc``.
_REDIRECT_PROTECTED_RE = re.compile(
    r"(?:&>>?|>\||>&|>>?)\s*"
    r"(/(?:etc|usr|bin|sbin|lib|lib64|boot|dev|proc|sys|var|root|opt)\b[^\s<>;&|()]*)"
)
# Character devices that every shell idiom redirects to. Discarding a stream
# (``2>/dev/null``) or pointing one at the terminal/an existing fd is not a
# write into a protected system path — denying them made ``2>/dev/null``
# unusable, and the deny reason never named the offending token, so the model
# could only guess (observed: 6 wasted turns in one trace). Raw block-device
# writes stay denied by the ``>\s*/dev/sd[a-z]`` and ``dd of=/dev/`` patterns
# in ``_DENY_PATTERNS`` above.
_REDIRECT_SAFE_DEVICE_RE = re.compile(
    r"^/dev/(?:null|zero|stdout|stderr|stdin|tty|fd/\d+)$"
)
_SHELLS = frozenset({"bash", "sh", "zsh", "dash", "ksh", "ash"})
_MAX_NEST = 4


class _ParseError(Exception):
    pass


def _basename(token: str) -> str:
    t = token.strip().strip("'\"")
    return t.rsplit("/", 1)[-1] if "/" in t else t


def _line_invokes_shell(pre: str) -> bool:
    """True if the command consuming a heredoc on this line is a shell (its body
    is shell CODE, not data) — e.g. ``bash <<EOF`` or ``printf x | sh <<EOF``.
    Checks the last simple command before the ``<<``."""
    segs = _split_top_level(pre)
    if not segs:
        return False
    try:
        toks = tokenize_shell_segment(_mask_nested_shell(segs[-1]))
    except ValueError:
        return False
    if not toks:
        return False
    keyword = _leading_shell_keyword(segs[-1])
    if keyword in _CONTROL_LEADERS and toks[0] == keyword:
        toks[0] = _SHELL_SYNTAX_TOKEN
    argv = strip_command_prefixes(toks)
    return bool(argv) and _basename(argv[0]) in _SHELLS


def _bracket_end(line: str, start: int, opener: str, closer: str) -> int:
    """Index just past the ``closer`` matching the ``opener`` before ``start``,
    or ``-1`` when it does not close on this line."""
    depth = 1
    i = start
    while i < len(line):
        c = line[i]
        if c == "\\":
            i += 2
            continue
        if c == opener:
            depth += 1
        elif c == closer:
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1
    return -1


def _heredoc_declarations(
    line: str, quote: str | None,
) -> tuple[list[tuple[str, bool, bool, bool]], str | None]:
    """Read only unquoted heredoc operators, preserving multiline quote state.

    Returns ``([(delimiter, quoted, strip_tabs, consumer_is_shell), ...],
    quote_state_after_line)``. Delimiter words undergo quote removal, not
    expansion. A ``<<`` in a comment, a quoted argument, a here-string or an
    arithmetic/substitution span is not a declaration and must never consume
    the lines after it — that would hide real commands as heredoc data.
    """
    declarations: list[tuple[str, bool, bool, bool]] = []
    expansion_ends = {start: end for start, end, *_ in _substitution_spans(line)}
    i = 0
    while i < len(line):
        if quote != "'" and i in expansion_ends:
            i = expansion_ends[i]
            continue
        c = line[i]
        if c == "\\" and quote != "'":
            i += 2
            continue
        if quote:
            if c == quote:
                quote = None
            i += 1
            continue
        if c in ("'", '"'):
            quote = c
            i += 1
            continue
        if c == "#" and (i == 0 or line[i - 1] in " \t;|&("):
            break
        if line.startswith("((", i):
            end = _find_expansion_end(line, i + 2, 2)
            if end >= 0:
                i = end
                continue
        # ``<<`` is a shift, not a heredoc, inside ``$[...]`` arithmetic,
        # ``${...}`` parameter expansion and ``name[...]`` subscripts. Mistaking
        # one for a heredoc would hide every following line as data, so skip
        # them; over-skipping only exposes more text as code.
        if line.startswith(("$[", "${"), i):
            end = _bracket_end(line, i + 2, line[i + 1], "]" if line[i + 1] == "[" else "}")
            if end >= 0:
                i = end
                continue
        if c == "[" and i and (line[i - 1].isalnum() or line[i - 1] == "_"):
            end = _bracket_end(line, i + 1, "[", "]")
            if end >= 0:
                i = end
                continue
        if line.startswith("<<<", i):
            i += 3
            continue
        if not line.startswith("<<", i):
            i += 1
            continue
        start = i
        i += 2
        tabs = line[i:i + 1] == "-"
        i += int(tabs)
        while i < len(line) and line[i] in " \t":
            i += 1
        word_start = i
        delimiter_quote = None
        quoted = False
        while i < len(line):
            c = line[i]
            if c == "\\" and delimiter_quote != "'":
                quoted = True
                i += 2
                continue
            if delimiter_quote:
                if c == delimiter_quote:
                    delimiter_quote = None
            elif c in ("'", '"'):
                quoted = True
                delimiter_quote = c
            elif c in " \t;<>&|()":
                break
            i += 1
        word = line[word_start:i]
        try:
            tokens = shlex.split(word)
        except ValueError:
            continue  # leave malformed headers for the command parser
        if len(tokens) == 1:
            declarations.append((
                tokens[0], quoted, tabs, _line_invokes_shell(line[:start]),
            ))
    return declarations, quote


def _strip_heredoc_bodies(command: str) -> tuple[str, list[str]]:
    """Drop here-document *bodies* (data, not commands) so they aren't parsed as
    top-level shell. Returns ``(stripped_command, shell_bodies)`` where
    ``shell_bodies`` is the code the shell still runs: bodies consumed by a
    shell (``bash <<EOF … EOF``), the ``$(...)`` in unquoted bodies, and — fail
    closed — the rest of a heredoc whose delimiter never arrives. The
    ``cmd <<'MARKER'`` line is kept.

    Multiple declarations on one line consume bodies in declaration order.
    Only ``<<-`` strips tabs; spaces and other indentation never close a
    heredoc."""
    lines = command.split("\n")
    out: list[str] = []
    shell_bodies: list[str] = []
    quote: str | None = None
    i = 0
    while i < len(lines):
        line = lines[i]
        out.append(line)
        declarations, quote = _heredoc_declarations(line, quote)
        i += 1
        for delim, quoted, tabs, consumer_is_shell in declarations:
            body: list[str] = []
            closed = False
            while i < len(lines):
                current = lines[i].lstrip("\t") if tabs else lines[i]
                i += 1
                if current == delim:
                    closed = True
                    break
                body.append(current)
            body_text = "\n".join(body)
            # An empty delimiter closes on an empty line, which the caller's
            # ``strip()`` removes when it ends the command.
            if not closed and delim:
                # A delimiter that never closes usually means ``<<`` was not a
                # heredoc at all (a shift we failed to recognise). Treating the
                # rest of the command as data would hide it from every check,
                # so screen it as code instead.
                shell_bodies.append(body_text)
            elif consumer_is_shell:
                shell_bodies.append(body_text)
            elif not quoted:
                # Unquoted delimiter: the shell still expands $()/`` in the
                # body before the (non-shell) consumer sees it, and quotes in
                # the body are ordinary characters there.
                shell_bodies.extend(
                    _extract_nested_shell(body_text, literal_quotes=True)
                )
    return "\n".join(out), shell_bodies


def _ends_with_redirect(buf: list[str]) -> bool:
    """True if the pending segment ends in a redirection operator (``>``/``<``),
    ignoring trailing spaces — i.e. the next ``&``/``|`` belongs to that
    redirection (``2>&1``, ``>&2``, ``>| file``) and is not a separator."""
    for ch in reversed(buf):
        if ch in (" ", "\t"):
            continue
        return ch in (">", "<")
    return False


def _strip_comments(command: str) -> str:
    """Drop ``#`` comments the shell itself would never execute.

    bash starts a comment only where ``#`` begins a word — at the start of the
    input or after whitespace / ``;`` / ``|`` / ``&`` / ``(``. A URL fragment
    (``curl http://x/#frag``) is therefore untouched. Without this, a leading
    ``# note`` line was tokenised as a command named ``#`` and denied as "not on
    the allowed-command list", which tells the model nothing about the real
    problem. Quoted spans are preserved verbatim.
    """
    out: list[str] = []
    i, n = 0, len(command)
    quote: str | None = None
    while i < n:
        c = command[i]
        if quote:
            out.append(c)
            if c == "\\" and quote == '"' and i + 1 < n:
                out.append(command[i + 1])
                i += 2
                continue
            if c == quote:
                quote = None
            i += 1
            continue
        if c in ("'", '"'):
            quote = c
            out.append(c)
            i += 1
            continue
        if c == "\\" and i + 1 < n:
            out.append(c)
            out.append(command[i + 1])
            i += 2
            continue
        if c == "#" and (not out or out[-1] in (" ", "\t", "\n", ";", "|", "&", "(")):
            while i < n and command[i] != "\n":
                i += 1
            continue
        out.append(c)
        i += 1
    return "".join(out)


def _inside_shell_quote(command: str, position: int) -> bool:
    """Whether ``position`` falls inside a single- or double-quoted span."""
    quote: str | None = None
    i = 0
    while i < min(position, len(command)):
        c = command[i]
        if c == "\\" and quote != "'" and i + 1 < position:
            i += 2
            continue
        if quote:
            if c == quote:
                quote = None
        elif c in ("'", '"'):
            quote = c
        i += 1
    return quote is not None


def _leading_shell_keyword(segment: str) -> str | None:
    """Return an unquoted shell keyword at the start of ``segment``.

    This runs before ``shlex.split`` intentionally. Once shlex has removed
    quotes and backslashes, a reserved word is indistinguishable from an
    external executable deliberately named ``for`` or ``done``.
    """
    text = segment.lstrip()
    for keyword in _LOOP_HEADERS | _CONTROL_LEADERS:
        if not text.startswith(keyword):
            continue
        end = len(keyword)
        if end == len(text) or text[end].isspace():
            return keyword
    return None


def _split_top_level(command: str) -> list[str]:
    """Split into simple commands on top-level ``; & | && || ( ) \\n`` respecting
    single/double quotes and backslash escapes.

    Command substitutions ``$(...)`` and backtick spans are copied verbatim (NOT
    split) — their inner commands are assessed separately via
    :func:`_extract_nested_shell`, so a top-level ``;``/``|`` inside a ``$(...)``
    must not fragment the outer command. Their extent comes from
    :func:`_find_expansion_end`, the same scanner extraction uses, so a quoted
    ``)`` cannot end one early here while extraction reads it differently.
    Bare ``(``/``)`` (subshell grouping) DO split, so ``(rm -rf /)`` is
    analysed. Redirections (``>`` ``<``) do not split.
    """
    segs: list[str] = []
    buf: list[str] = []
    i, n = 0, len(command)
    quote: str | None = None
    while i < n:
        c = command[i]
        if quote != "'" and c == "$" and command[i + 1:i + 2] == "(":
            arithmetic = command[i + 2:i + 3] == "("
            end = _find_expansion_end(
                command, i + (3 if arithmetic else 2), 2 if arithmetic else 1,
            )
            end = n if end < 0 else end
            buf.append(command[i:end])
            i = end
            continue
        if quote != "'" and c == "`":  # backtick span — copy to its closer
            _body, end = _backtick_body(command, i + 1)
            end = n if end < 0 else end
            buf.append(command[i:end])
            i = end
            continue
        if quote:
            buf.append(c)
            if c == "\\" and quote == '"' and i + 1 < n:
                buf.append(command[i + 1])
                i += 2
                continue
            if c == quote:
                quote = None
            i += 1
            continue
        if c in ("'", '"'):
            quote = c
            buf.append(c)
            i += 1
            continue
        if c == "\\" and i + 1 < n:
            buf.append(c)
            buf.append(command[i + 1])
            i += 2
            continue
        # Redirection operators that CONTAIN a control character must not split:
        # ``2>&1`` / ``>&2`` (fd duplication) and ``&>file`` / ``&>>file``
        # (stdout+stderr) are one redirection, not a command boundary. Without
        # this, ``python3 x.py 2>&1`` split at the ``&`` and left a phantom
        # segment ``1``, denied as "`1` is not on the allowed-command list".
        if c in ("&", "|") and _ends_with_redirect(buf):
            buf.append(c)
            i += 1
            continue
        if command[i:i + 2] == "&>":
            buf.append(c)
            i += 1
            continue
        if command[i:i + 2] in ("&&", "||"):
            segs.append("".join(buf))
            buf = []
            i += 2
            continue
        if c in (";", "\n", "|", "&", "(", ")"):
            segs.append("".join(buf))
            buf = []
            i += 1
            continue
        buf.append(c)
        i += 1
    segs.append("".join(buf))
    return [s.strip() for s in segs if s.strip()]


# Inert stand-ins for expansions in the OUTER view of a command. Never shown to
# the model: a deny reason that would name one describes it in words instead.
_SUBSTITUTION_SENTINEL = "__FA_COMMAND_SUBSTITUTION__"
_ARITHMETIC_SENTINEL = "__FA_ARITHMETIC_EXPANSION__"

#: One expansion the shell evaluates: its outer extent, its body, and what kind
#: it is. ``body`` is non-empty only when the text the shell executes is not a
#: verbatim slice of the input (a backtick span, whose escapes bash removes
#: before parsing); otherwise the body is ``command[inner_start:inner_end]``.
_Span = tuple[int, int, int, int, str, str]


def _find_expansion_end(command: str, start: int, depth: int) -> int:
    """Index just past the parens closing an expansion that opened at ``start``,
    or ``-1`` when it is unterminated.

    Quoted parens do not count: ``$(echo ")")`` ends at the last ``)``, not the
    quoted one.

    Like bash, every nested ``$(`` / ``<(`` / ``>(`` starts with its OWN quote
    state, so the quotes in ``$(echo "$(echo ")'")" $(sudo id))`` pair up inside
    the inner substitution and the scan still reaches ``sudo id``. A single
    quote state shared across levels desynchronised there and ended the outer
    span early, which hid the last command from every check. The levels live on
    a list rather than the call stack, so deep nesting cannot hit the recursion
    limit. A backtick span is skipped whole.
    """
    n = len(command)
    quotes: list[str | None] = [None]  # quote state of each open level
    depths = [depth]                   # unquoted "(" nesting inside each level
    i = start
    while i < n:
        c = command[i]
        quote = quotes[-1]
        if quote == "'":
            if c == "'":
                quotes[-1] = None
        elif c == "\\":
            i += 1
        elif command.startswith("(", i + 1) and (c == "$" or (c in "<>" and quote is None)):
            quotes.append(None)
            depths.append(1)
            i += 1
        elif c == "`":
            i += 1
            while i < n and command[i] != "`":
                i += 2 if command[i] == "\\" else 1
        elif c == '"':
            quotes[-1] = None if quote else '"'
        elif quote is None:
            if c == "'":
                quotes[-1] = "'"
            elif c == "(":
                depths[-1] += 1
            elif c == ")":
                depths[-1] -= 1
                if depths[-1] == 0:
                    if len(depths) == 1:
                        return i + 1
                    depths.pop()
                    quotes.pop()
        i += 1
    return -1


def _backtick_body(command: str, i: int) -> tuple[str, int]:
    """Read a backtick body and apply bash's first-pass escape removal.

    Inside backticks, a backslash before ``$``, a backtick, a backslash or a
    newline is removed before the body is parsed as shell code; other
    backslashes are retained. Without this, ``echo `echo \\`sudo id\\``` left
    the escapes in place and the inner command was never recognised.

    Returns the decoded body and the index just past the closing backtick, or
    ``-1`` for that index when it never closes.
    """
    body: list[str] = []
    n = len(command)
    while i < n:
        c = command[i]
        if c == "`":
            return "".join(body), i + 1
        if c == "\\" and i + 1 < n and command[i + 1] in "$`\\\n":
            i += 1
            if command[i] != "\n":
                body.append(command[i])
        else:
            body.append(c)
        i += 1
    return "".join(body), -1


def _dup_redirect_word(command: str, i: int) -> str:
    """The target word of a ``>&`` redirect that starts at ``i``, with its
    quotes and backslashes removed.

    bash expands that word a SECOND time after quote removal, so
    ``echo x >&'$(sudo id)'`` runs ``sudo id`` even though the substitution is
    single-quoted. The stripped text is roughly what that second pass sees;
    dropping every backslash can only expose more ``$(``, never hide one.
    """
    n = len(command)
    while i < n and command[i] in " \t":
        i += 1
    start = i
    quote: str | None = None
    while i < n:
        c = command[i]
        if quote == "'":
            if c == "'":
                quote = None
        elif c == "\\":
            i += 1
        elif command.startswith("$(", i):
            end = _find_expansion_end(command, i + 2, 1)
            i = n if end < 0 else end - 1
        elif c == "`":
            i += 1
            while i < n and command[i] != "`":
                i += 2 if command[i] == "\\" else 1
        elif c == '"':
            quote = None if quote else '"'
        elif quote is None:
            if c == "'":
                quote = "'"
            elif c.isspace() or c in ";&|<>()":
                break
        i += 1
    return re.sub(r"[\\'\"]", "", command[start:i])


def _substitution_spans(
    command: str, *, literal_quotes: bool = False,
) -> list[_Span]:
    """Locate every expansion the shell evaluates, as ``(start, end, inner
    start, inner end, kind, body)`` with ``kind`` in
    ``{"command", "arithmetic"}``.

    This is the single source of truth for *where an expansion begins and ends*.
    Masking (:func:`_mask_nested_shell`) and extraction
    (:func:`_extract_nested_shell`) are two views of the same spans; when they
    scanned independently they disagreed on quoted delimiters and a nested
    command could end up assessed by neither view (``x=$(echo ")"; sudo id)``).

    Recognised: ``$(...)``, backticks, and unquoted process substitution
    ``<(...)`` / ``>(...)`` — the shell runs all of them. Only top-level spans
    are returned; a substitution nested inside another is reached by assessing
    the outer one's body recursively. Single-quoted spans are skipped (the shell
    does not expand them), but double-quoted ones are not: ``"$(sudo id)"``
    still runs, and so does ``"'$(sudo id)'"`` — inside double quotes a ``'`` is
    an ordinary character. An unterminated expansion extends to the end of the
    text, so its body is still assessed (fail-closed).

    ``literal_quotes`` is for unquoted here-document bodies, where ``'`` and
    ``"`` are ordinary characters and never suppress expansion.
    """
    spans: list[_Span] = []
    i, n = 0, len(command)
    quote: str | None = None
    while i < n:
        c = command[i]
        if quote == "'":
            if c == "'":
                quote = None
            i += 1
            continue
        if c == "\\" and i + 1 < n:
            i += 2
            continue
        if quote is None and not literal_quotes:
            if c in ("'", '"'):
                quote = c
                i += 1
                continue
        elif quote == '"' and c == '"':
            quote = None
            i += 1
            continue
        if command.startswith("(", i + 1) and (
            c == "$" or (c in "<>" and quote is None)
        ):
            # ``$((`` is arithmetic: it expands a number rather than running a
            # command, so its body is not shell code (but may still contain a
            # real substitution, which the caller reaches by recursing).
            arithmetic = c == "$" and command[i + 2:i + 3] == "("
            inner_start = i + 3 if arithmetic else i + 2
            end = _find_expansion_end(command, inner_start, 2 if arithmetic else 1)
            if end < 0:
                spans.append((i, n, inner_start, n, "command", ""))
                break
            inner_end = max(inner_start, end - (2 if arithmetic else 1))
            spans.append((
                i, end, inner_start, inner_end,
                "arithmetic" if arithmetic else "command", "",
            ))
            i = end
            continue
        if c == "`":
            # The body is DECODED (bash removes a layer of escapes before
            # parsing it), so it is carried rather than sliced back out.
            body, end = _backtick_body(command, i + 1)
            if end < 0:
                spans.append((i, n, i + 1, n, "command", body))
                break
            spans.append((i, end, i + 1, end - 1, "command", body))
            i = end
            continue
        i += 1
    return spans


def _mask_nested_shell(command: str) -> str:
    """Replace expansions with inert tokens for outer parsing.

    Nested code is parsed independently by :func:`_extract_nested_shell`. If
    it is also left verbatim for tokenization, an assignment such as
    ``version=$(nginx -V)`` becomes ``['version=$(nginx', '-V)']`` and the
    option is falsely assessed as an executable. Masking only the outer view
    preserves both checks: the assignment stays an assignment, while the real
    ``nginx -V`` command is still assessed recursively.
    """
    spans = _substitution_spans(command)
    if not spans:
        return command
    out: list[str] = []
    prev = 0
    for start, end, _inner_start, _inner_end, kind, _body in spans:
        out.append(command[prev:start])
        out.append(_ARITHMETIC_SENTINEL if kind == "arithmetic" else _SUBSTITUTION_SENTINEL)
        prev = end
    out.append(command[prev:])
    return "".join(out)


def _dup_redirect_bodies(command: str) -> list[str]:
    """Code reachable through a ``>&`` target's second expansion.

    A separate pass rather than a span: this text is not an expansion present in
    the input but a RE-expansion of a word after quote removal, so it has no
    extent in the original string to mask.
    """
    out: list[str] = []
    i, n = 0, len(command)
    quote: str | None = None
    while i < n:
        c = command[i]
        if quote == "'":
            if c == "'":
                quote = None
            i += 1
            continue
        if c == "\\" and i + 1 < n:
            i += 2
            continue
        if c == "'" and quote is None:
            quote = "'"
        elif c == '"':
            quote = None if quote else '"'
        elif c == ">" and quote is None and command.startswith("&", i + 1):
            word = _dup_redirect_word(command, i + 2)
            if word != command:
                out.extend(_extract_nested_shell(word))
        i += 1
    return out


def _extract_nested_shell(command: str, *, literal_quotes: bool = False) -> list[str]:
    """Return shell-code strings nested in ``$(...)``, backticks and unquoted
    process substitution (which the shell expands+executes). Single-quoted
    spans are skipped — the shell does not expand them, so
    ``echo '$(rm -rf /)'`` is a harmless literal. The one exception is the
    target of ``>&``, which bash expands twice (see
    :func:`_dup_redirect_word`)."""
    out: list[str] = []
    for _start, _end, inner_start, inner_end, kind, body in _substitution_spans(
        command, literal_quotes=literal_quotes,
    ):
        inner = body or command[inner_start:inner_end]
        if kind == "arithmetic":
            # The arithmetic body is not shell code, but ``$(( $(id) + 1 ))``
            # still runs ``id``.
            out.extend(_extract_nested_shell(inner))
        elif inner:
            out.append(inner)
    out.extend(_dup_redirect_bodies(command))
    return out


def _find_exec_payloads(argv: list[str]) -> list[list[str]]:
    """For a ``find`` argv, return the command payload(s) of its
    ``-exec``/``-execdir``/``-ok``/``-okdir`` actions as argv lists (terminator
    ``;``/``+`` and ``{}`` placeholders dropped). So ``find … -exec bash -c
    '<code>' \\;`` surfaces ``bash -c <code>`` for assessment + recursion."""
    if not argv or _basename(argv[0]) != "find":
        return []
    out: list[list[str]] = []
    i, n = 0, len(argv)
    while i < n:
        if argv[i] in _FIND_EXEC_FLAGS:
            j = i + 1
            payload: list[str] = []
            while j < n and argv[j] not in (";", "+"):
                if argv[j] != "{}":
                    payload.append(argv[j])
                j += 1
            if payload:
                out.append(payload)
            i = j + 1
        else:
            i += 1
    return out


def _blank_quoted(command: str) -> str:
    """Replace the contents of quoted spans with spaces, keeping the quotes.

    Quoted text is an argument, not a command name, so the raw hard-deny screen
    should not match words inside it. Expansions that still run inside double
    quotes are screened separately via :func:`_extract_nested_shell`.
    """
    out: list[str] = []
    quote: str | None = None
    i, n = 0, len(command)
    while i < n:
        c = command[i]
        if quote is None:
            if c == "\\" and i + 1 < n:
                out.append(command[i:i + 2])
                i += 2
                continue
            if c in ("'", '"'):
                quote = c
            out.append(c)
        elif c == quote:
            quote = None
            out.append(c)
        elif c == "\\" and quote == '"' and i + 1 < n:
            out.append("  ")
            i += 2
            continue
        else:
            out.append("\n" if c == "\n" else " ")
        i += 1
    return "".join(out)


def _shell_code_args(tokens: list[str]) -> list[str]:
    """Code strings a simple command hands to a shell: the command string of
    ``bash -c`` (also combined short flags such as ``-lc`` / ``-ec``) and the
    arguments of ``eval``."""
    argv = strip_command_prefixes(tokens)
    if not argv:
        return []
    base = _basename(argv[0])
    if base == "eval":
        return [" ".join(argv[1:])] if len(argv) > 1 else []
    if base not in _SHELLS:
        return []
    k = 1
    while k < len(argv):
        tok = argv[k]
        if tok == "--" or not tok.startswith("-"):
            break
        if tok in {"-o", "-O", "--rcfile", "--init-file"}:
            k += 2
            continue
        if tok.startswith("-") and not tok.startswith("--") and "c" in tok[1:]:
            return argv[k + 1:k + 2]
        k += 1
    return []


def _systemctl_requests_shutdown(args: list[str]) -> bool:
    """Recognize shutdown verbs and activation of shutdown units.

    systemctl accepts options before or after the verb. Option values are not
    units; queries such as status/show must not be mistaken for activation.
    """
    values = {
        "-H", "--host", "-M", "--machine", "-t", "--type", "--state",
        "-p", "--property", "--root", "--image", "--boot-loader-entry",
        "--boot-loader-menu", "--kill-whom", "-s", "--signal",
    }
    operands: list[str] = []
    now = False
    i = 0
    while i < len(args):
        arg = args[i]
        if arg == "--":
            operands.extend(args[i + 1:])
            break
        if not arg.startswith("-"):
            operands.append(arg)
        elif arg == "--now":
            now = True
        i += 2 if arg in values else 1
    if not operands:
        return False
    operation, *units = operands
    if operation in {"halt", "reboot", "poweroff"}:
        return True
    activates = operation in {
        "start", "restart", "try-restart", "reload-or-restart", "reload-or-try-restart",
        "isolate",
    } or (operation in {"enable", "reenable"} and now)
    shutdown_units = {
        "halt.target", "reboot.target", "poweroff.target", "shutdown.target",
        "runlevel0.target", "runlevel6.target", "ctrl-alt-del.target",
        "systemd-halt.service", "systemd-reboot.service", "systemd-poweroff.service",
    }
    return activates and any(unit in shutdown_units for unit in units)


# Commands that run their string arguments as shell code (``watch 'cmd'``,
# ``tmux new -d 'cmd'``, ``ssh host 'cmd'``). Known forms have payload adapters;
# unknown forms retain the unblanked screen rather than guessing option arity.
_STRING_EVALUATORS = frozenset({
    "watch", "tmux", "screen", "ssh", "script", "su", "runuser", "sg", "parallel",
})
# Commands that read shell code from stdin regardless of their arguments.
_STDIN_CODE_READERS = frozenset({"at", "batch"})


def _evaluator_options(
    args: list[str], flags: str, values: str,
    long_flags: frozenset[str] = frozenset(),
    long_values: frozenset[str] = frozenset(),
) -> tuple[list[str], dict[str, str]] | None:
    """Consume documented options without losing operand boundaries.

    Unknown options return None: their arity is unknown, so the caller must
    retain the unblanked screen instead of guessing where executable code starts.
    """
    options: dict[str, str] = {}
    i = 0
    while i < len(args):
        token = args[i]
        if token == "--":
            return args[i + 1:], options
        if token == "-" or not token.startswith("-"):
            break
        if token.startswith("--"):
            name, sep, value = token.partition("=")
            if name in long_values:
                if not sep:
                    i += 1
                    if i >= len(args):
                        return None
                    value = args[i]
                options[name] = value
            elif name in long_flags and not sep:
                options[name] = ""
            else:
                return None
        else:
            j = 1
            while j < len(token):
                flag = token[j]
                if flag in values:
                    value = token[j + 1:]
                    if not value:
                        i += 1
                        if i >= len(args):
                            return None
                        value = args[i]
                    options["-" + flag] = value
                    break
                if flag not in flags:
                    return None
                options["-" + flag] = ""
                j += 1
        i += 1
    return args[i:], options


def _evaluator_payloads(tokens: list[str]) -> tuple[list[str], bool]:
    """Return actual code arguments and whether conservative screening is needed.

    Do not concatenate option values, host/session names or tmux subcommands
    with the payload: doing so hides its executable behind an invented prefix.
    Unsupported evaluator forms retain the whole-text protection.
    """
    argv = tokens
    while argv:
        exe = _basename(argv[0])
        if exe in _STRING_EVALUATORS:
            break
        after_redirect = _skip_redirection(argv, 0)
        if _ASSIGN_RE.match(argv[0]) or argv[0] == _SHELL_SYNTAX_TOKEN:
            argv = argv[1:]
        elif after_redirect is not None:
            argv = argv[after_redirect:]
        elif exe in _WRAPPERS or exe in _PRIV_ESC:
            argv = argv[_skip_wrapper_args(argv, 1, wrapper=exe):]
        else:
            return [], False
    if not argv:
        return [], False
    exe = _basename(argv[0])
    # Redirections are marked by ``tokenize_shell_segment``, so dropping them
    # cannot erase a quoted code argument that merely starts with ``>``.
    args = _without_redirections(argv[1:])
    parsed = None
    direct = False
    if exe == "watch":
        parsed = _evaluator_options(
            args, "bcdgpetwx", "nq",
            frozenset({"--beep", "--color", "--differences", "--chgexit", "--errexit",
                       "--precise", "--no-title", "--no-wrap", "--exec"}),
            frozenset({"--interval", "--equexit"}),
        )
        if parsed:
            direct = "-x" in parsed[1] or "--exec" in parsed[1]
    elif exe == "tmux":
        global_options = _evaluator_options(args, "2CDluUvV", "cfLST")
        if not global_options or not global_options[0]:
            return [], True
        args, _ = global_options
        subcommand, *args = args
        if subcommand in {"new", "new-session"}:
            parsed = _evaluator_options(args, "AdDEPX", "ceFnstxy")
        elif subcommand in {"neww", "new-window"}:
            parsed = _evaluator_options(args, "abdkPS", "ceFnt")
        elif subcommand in {"splitw", "split-window"}:
            parsed = _evaluator_options(args, "bdfhIvPZ", "ceFlt")
        elif subcommand in {"respawnp", "respawn-pane", "respawnw", "respawn-window"}:
            parsed = _evaluator_options(args, "k", "cet")
        else:
            return [], True
        if parsed:
            # tmux executes a single string via a shell, but multiple operands
            # are an argv vector (preserve the nested shell's -c argument).
            direct = len(parsed[0]) > 1
    elif exe == "screen":
        parsed = _evaluator_options(args, "AdmDqRx", "cSept")
        direct = True
    elif exe == "ssh":
        parsed = _evaluator_options(args, "46AaCfGgKkMNnqsTtVvXxYy", "BbcDEeFIiJLlmOopQRSWw")
        if parsed:
            operands, options = parsed
            parsed = (operands[1:], options) if operands else None
    elif exe == "script":
        parsed = _evaluator_options(
            args, "aeqf", "cOITB",
            frozenset({"--append", "--return", "--quiet", "--flush"}),
            frozenset({"--command", "--log-out", "--log-in", "--log-timing", "--log-io"}),
        )
        if parsed:
            options = parsed[1]
            payload = options.get("-c", options.get("--command"))
            return ([payload], False) if payload is not None else ([], True)
    elif exe in {"su", "runuser", "sg", "parallel"}:
        # Account/shell options differ between implementations. Preserve the
        # whole-text guard for these forms until a dedicated adapter exists.
        return [], True
    if parsed is None:
        return [], True
    operands, _ = parsed
    if not operands:
        return [], True
    # Every direct-exec argument is data, even when shlex.quote would omit its
    # quotes (e.g. the word 'halt' in `screen echo halt`). The command name and
    # shell -c payload will be recovered independently by the recursive parser.
    payload = (
        " ".join("'" + word.replace("'", "'\"'\"'") + "'" for word in operands)
        if direct else " ".join(operands)
    )
    return [payload], False


def _without_redirections(argv: list[str]) -> list[str]:
    """Drop redirection operators and their targets, including here-strings
    (``<<< word``), so only real operands remain."""
    out: list[str] = []
    i = 0
    while i < len(argv):
        after_redirect = _skip_redirection(argv, i)
        if after_redirect is not None:
            i = after_redirect
            continue
        out.append(argv[i])
        i += 1
    return out


def _reads_code_from_stdin(argv: list[str]) -> bool:
    """Whether a (prefix-stripped) command executes shell code read from stdin:
    a shell with no ``-c`` string and no script operand, or with ``-s``
    (``… | bash``, ``bash <<< 'cmd'``, ``sh -s <<EOF``), or ``at`` / ``batch``."""
    if not argv:
        return False
    exe = _basename(argv[0])
    if exe in _STDIN_CODE_READERS:
        return True
    # A "script file" that is really stdin, an fd or a process substitution
    # (``source /dev/stdin``, ``bash /proc/self/fd/0``, ``. <(echo …)``).
    if any(_is_stdin_script(p) for p in _executed_script_operands(argv)):
        return True
    if exe not in _SHELLS or _shell_code_args(argv):
        return False
    args = _without_redirections(argv[1:])
    k = 0
    while k < len(args):
        tok = args[k]
        if tok == "--":
            k += 1
            break
        if not tok.startswith(("-", "+")):
            break
        if tok in {"-o", "-O", "+o", "+O", "--rcfile", "--init-file"}:
            k += 2
            continue
        if not tok.startswith("--") and "s" in tok[1:]:
            return True
        k += 1
    return k >= len(args)


# Commands whose operands are files they write (besides redirections).
_FILE_WRITERS = frozenset({"tee"})
# Commands whose operands name the files they create (``cp a x.sh``).
_FILE_COPIERS = frozenset({"cp", "mv", "ln", "install"})


def _redirect_targets(tokens: list[str]) -> list[str]:
    """Targets of the output redirections in ``tokens`` (``> x`` / ``>>x`` /
    ``&> x``), quote-removed. fd duplications (``2>&1``) are not files."""
    out: list[str] = []
    for k, tok in enumerate(tokens):
        redirect = redirection_token(tok)
        if redirect is None:
            continue
        match = _REDIRECT_RE.match(redirect)
        if match is None:
            continue
        operator = redirect[:len(redirect) - len(match.group(1))]
        if ">" not in operator or operator.endswith("&"):
            continue
        target = match.group(1) or (tokens[k + 1] if k + 1 < len(tokens) else "")
        if target:
            out.append(target)
    return out


def _written_paths(tokens: list[str]) -> set[str]:
    """Basenames of files a simple command writes: redirection targets and
    ``tee`` / ``cp``-style operands."""
    out = {_basename(t) for t in _redirect_targets(tokens)}
    argv = strip_command_prefixes(tokens)
    if argv and _basename(argv[0]) in _FILE_WRITERS:
        out.update(_basename(t) for t in _without_redirections(argv[1:]) if not t.startswith("-"))
    if argv and _basename(argv[0]) in _FILE_COPIERS:
        operands = [t for t in _without_redirections(argv[1:]) if not t.startswith("-")]
        out.update(_basename(t) for t in operands)  # sources too: ``mv x.sh dir/``
    return out


def _is_stdin_script(operand: str) -> bool:
    return operand in {"-", "/dev/stdin"} or operand.startswith(
        ("<(", "/dev/fd/", "/proc/self/fd/", "/proc/thread-self/fd/"),
    )


def _executed_script_operands(argv: list[str]) -> list[str]:
    """Raw script operands a (prefix-stripped) command runs as shell code."""
    if not argv:
        return []
    exe = _basename(argv[0])
    if exe in {"source", "."}:
        return _without_redirections(argv[1:])[:1]
    if exe in _SHELLS and not _shell_code_args(argv):
        args = _without_redirections(argv[1:])
        k = 0
        while k < len(args) and args[k].startswith(("-", "+")) and args[k] != "--":
            k += 2 if args[k] in {"-o", "-O", "+o", "+O", "--rcfile", "--init-file"} else 1
        if k < len(args) and args[k] == "--":
            k += 1
        return args[k:k + 1]
    return []


def _executed_script_paths(argv: list[str]) -> set[str]:
    """Basenames of files a (prefix-stripped) command runs as shell code:
    ``bash x.sh``, ``source x.sh`` / ``. x.sh`` and path execution ``./x.sh``.
    Interpreters such as ``python3`` are deliberately absent: their scripts
    are not shell, and writing ``gen.py`` then running it is ordinary work."""
    if not argv:
        return set()
    operands = _executed_script_operands(argv)
    if operands:
        return {_basename(operands[0])}
    if "/" in argv[0]:
        return {_basename(argv[0])}
    return set()


# Commands that treat their quoted arguments, heredoc bodies and stdin as data.
# Only these get the raw-screen blanking; any other command in the input keeps
# the whole-text screen, so an unknown way of turning data back into code (a
# runner such as ``systemd-run 'x'``, ``trap 'x' EXIT``, ``xargs``, ``alias``)
# stays as strict as before instead of needing its own adapter. ``python*`` and
# ``node`` are included on purpose: writing prose into ``gen.py`` and running it
# is ordinary work, and an interpreter can build any string anyway -- the
# sandbox, not this regex, bounds what its code does.
_DATA_CONSUMERS = frozenset({
    # output / text tools
    "echo", "printf", "cat", "tac", "tee", "head", "tail", "wc", "sort", "uniq",
    "cut", "tr", "paste", "nl", "fold", "fmt", "column", "rev", "grep", "egrep",
    "fgrep", "rg", "ag", "diff", "cmp", "comm", "jq", "yq", "base64", "md5sum",
    "sha1sum", "sha256sum", "sha512sum", "iconv", "sed",
    # files and paths
    "ls", "stat", "file", "du", "df", "mkdir", "rmdir", "touch", "cp", "mv",
    "ln", "rm", "chmod", "chown", "find", "basename", "dirname", "realpath",
    "readlink", "mktemp", "tar", "zip", "unzip", "gzip", "gunzip", "xz",
    # document tooling
    "pandoc", "pdftotext", "pdftoppm", "pdfinfo", "qpdf", "soffice", "libreoffice",
    "xmllint", "pdflatex", "xelatex", "lualatex",
    # guarded below: they can execute code through specific options
    "git", "awk", "gawk", "mawk",
    # service queries; shutdown operations are recognised separately
    # (``_systemctl_requests_shutdown``), unit names are otherwise data
    "systemctl",
    # network retrieval (arguments are URLs / headers / bodies)
    "curl", "wget",
    # interpreters and package tooling (see above)
    "python", "python3", "node", "pip", "pip3", "uv",
    # builtins whose words are data
    "cd", "pwd", "test", "[", "[[", "true", "false", ":", "export", "declare",
    "typeset", "local", "readonly", "read", "set", "unset", "shopt", "exit",
    "return", "sleep", "date", "wait",
})
# ``awk`` runs commands via ``system()``, ``| "cmd"`` / ``"cmd" | getline``.
_AWK_EXEC_RE = re.compile(r"\bsystem\s*\(|\|\s*&?|\bgetline\b")


def _sed_data_expression(expression: str) -> bool:
    """Recognize a small non-executing sed grammar, never infer it from no 'e'.

    Everything outside simple substitutions and print/delete/quit commands
    falls back to the whole-text screen. In particular addresses, custom
    delimiters and escaping must not conceal an execution command or
    substitution flag.
    """
    expression = expression.strip()
    expression = re.sub(r"^(?:\d+|\$)(?:,(?:\d+|\$))?!?", "", expression)
    if expression in {"p", "P", "d", "D", "q", "Q", "="}:
        return True
    if len(expression) < 2 or expression[0] != "s":
        return False
    delimiter = expression[1]
    if delimiter.isalnum() or delimiter.isspace() or delimiter == "\\":
        return False
    i = 2
    for _ in range(2):  # pattern and replacement, each closed by the delimiter
        while i < len(expression):
            char = expression[i]
            if char == "\n":
                return False
            if char == "\\":
                if i + 1 >= len(expression) or expression[i + 1] == "\n":
                    return False
                i += 2
                continue
            i += 1
            if char == delimiter:
                break
        else:
            return False
    return re.fullmatch(r"[gIpMm0-9]*", expression[i:]) is not None


def _sed_treats_words_as_data(args: list[str]) -> bool:
    expressions: list[str] = []
    operands: list[str] = []
    i = 0
    options = True
    while i < len(args):
        token = args[i]
        if options and token == "--":
            options = False
        elif options and token.startswith("--expression="):
            expressions.append(token.partition("=")[2])
        elif options and token == "--expression":
            i += 1
            if i >= len(args):
                return False
            expressions.append(args[i])
        elif options and token.startswith("--"):
            if token not in {"--quiet", "--silent", "--regexp-extended", "--sandbox",
                             "--unbuffered", "--null-data", "--posix", "--in-place"} and not (
                token.startswith("--in-place=")
            ):
                return False
        elif options and token.startswith("-") and token != "-":
            j = 1
            while j < len(token):
                flag = token[j]
                if flag == "e":
                    expr = token[j + 1:]
                    if not expr:
                        i += 1
                        if i >= len(args):
                            return False
                        expr = args[i]
                    expressions.append(expr)
                    break
                if flag == "i":
                    break  # the remainder is the optional backup suffix
                if flag not in "nEruzb":
                    return False  # includes -f: unseen script files aren't data
                j += 1
        else:
            operands.append(token)
        i += 1
    if not expressions:
        expressions = operands[:1]
    return bool(expressions) and all(_sed_data_expression(expr) for expr in expressions)


def _tar_treats_words_as_data(args: list[str]) -> bool:
    """Only ordinary archive options may opt out of whole-text screening.

    Unknown/abbreviated options retain the raw guard. This includes checkpoint
    actions, external compressors, remote shell commands and --to-command.
    """
    # tar also accepts traditional option words such as `tar czf archive ...`.
    if args and args[0] and not args[0].startswith("-"):
        if all(c in "ctxrvuzjpJOfCThv" for c in args[0]):
            args = ["-" + args[0], *args[1:]]
        else:
            return False
    while args:
        parsed = _evaluator_options(
            args, "ctxrvuzjpJOh", "fCT",
            frozenset({"--create", "--extract", "--get", "--list", "--append", "--update",
                       "--verbose", "--gzip", "--bzip2", "--xz", "--zstd", "--no-recursion",
                       "--dereference", "--numeric-owner", "--null"}),
            frozenset({"--file", "--directory", "--files-from", "--exclude"}),
        )
        if parsed is None:
            return False
        operands, _ = parsed
        if "--" in args[:len(args) - len(operands)]:
            return True
        if not operands:
            return True
        args = operands[1:]  # GNU tar also accepts options after file operands
    return True


# Bash evaluates array subscripts arithmetically, and that evaluation runs any
# ``$(...)`` inside -- even in a single-quoted word: ``[[ 'a[$(cmd)]' -eq 1 ]]``,
# ``printf -v 'x[$(cmd)]'``, ``read 'x[$(cmd)]'``, ``declare -i v='a[$(cmd)]'``.
_ARITH_SUBSCRIPT_EXEC = r"\$\(|`"
# Options through which an otherwise data-only tool runs a program or code.
# Searched in each argument; a hit keeps the whole-text screen. Every entry of
# _DATA_CONSUMERS must be classified in tests/test_bash_policy_consumer_audit.py.
_CONSUMER_EXEC_OPTIONS: dict[str, re.Pattern[str]] = {
    tool: re.compile(pattern) for tool, pattern in {
        "[[": _ARITH_SUBSCRIPT_EXEC,
        "printf": _ARITH_SUBSCRIPT_EXEC,
        "read": _ARITH_SUBSCRIPT_EXEC,
        "declare": _ARITH_SUBSCRIPT_EXEC,
        "typeset": _ARITH_SUBSCRIPT_EXEC,
        "local": _ARITH_SUBSCRIPT_EXEC,
        "readonly": _ARITH_SUBSCRIPT_EXEC,
        "export": _ARITH_SUBSCRIPT_EXEC,
        "ag": r"\A--pager",
        "pandoc": r"\A(?:-F|--filter|-L|--lua-filter|--pdf-engine)",
        "pdflatex": r"\A--?(?:shell-escape|enable-write18)\Z",
        "xelatex": r"\A--?(?:shell-escape|enable-write18)\Z",
        "lualatex": r"\A--?(?:shell-escape|enable-write18)\Z",
        "rg": r"\A--pre(?:=|\Z)",
        "sort": r"\A--compress-program",
        "zip": r"\A(?:-TT|--unzip-command)",
        "wget": r"\A(?:--use-askpass|-e|--execute)",
        "soffice": r"\A(?:macro:|vnd\.sun\.star\.script:)",
        "libreoffice": r"\A(?:macro:|vnd\.sun\.star\.script:)",
        "xmllint": r"\A--shell\Z",
    }.items()
}
# git: subcommands that only read/write repository data, and the options that
# still hand git a program to run (hooks and config-driven drivers aside).
_GIT_DATA_SUBCOMMANDS = frozenset({
    "add", "commit", "status", "log", "diff", "show", "init", "rm", "mv",
    "restore", "checkout", "switch", "branch", "tag", "stash", "rev-parse",
    "ls-files", "blame", "reset", "remote", "merge", "fetch", "pull", "push",
    "clone", "describe", "shortlog",
})
_GIT_EXEC_OPTIONS = re.compile(
    r"\A(?:-u\Z|--upload-pack|--receive-pack|--exec|-x\Z|--ext-diff|-O|"
    r"--open-files-in-pager|--extcmd|--tool|-c|--config-env)"
)
_GIT_GLOBAL_VALUE_OPTIONS = frozenset({"-C", "--git-dir", "--work-tree", "--namespace"})
# uv: project/package management only; ``uv run`` / ``uv tool run`` execute.
_UV_DATA_SUBCOMMANDS = frozenset({
    "pip", "add", "remove", "sync", "lock", "venv", "init", "tree", "version", "python",
})
# Environment variables whose value is a command or code another program runs
# (``GIT_SSH_COMMAND='x' git fetch``, ``export PAGER='x'``).
_EXEC_ENV_ASSIGN_RE = re.compile(
    r"\A(?:GIT_[A-Z_]*|PAGER|MANPAGER|EDITOR|VISUAL|SSH_ASKPASS|BROWSER|SHELL|"
    r"BASH_ENV|ENV|PROMPT_COMMAND|LD_PRELOAD|PYTHONSTARTUP|NODE_OPTIONS|PERL5OPT|"
    r"[A-Z_]*_(?:COMMAND|CMD|EDITOR|PAGER))="
)


def _git_treats_words_as_data(args: list[str]) -> bool:
    i = 0
    while i < len(args) and args[i].startswith("-"):
        if _GIT_EXEC_OPTIONS.match(args[i]):
            return False
        i += 2 if args[i] in _GIT_GLOBAL_VALUE_OPTIONS else 1
    if i >= len(args) or args[i] not in _GIT_DATA_SUBCOMMANDS:
        return False
    return not any(_GIT_EXEC_OPTIONS.match(t) for t in args[i + 1:])


def _is_dynamic_name(token: str) -> bool:
    """The command name comes from an expansion (``$x`` / ``$(...)``)."""
    return (
        "$" in token or "`" in token
        or _SUBSTITUTION_SENTINEL in token or _ARITHMETIC_SENTINEL in token
    )


def _treats_words_as_data(argv: list[str]) -> bool:
    """Whether blanking this (prefix-stripped) command's quoted words and
    heredoc bodies is safe: a known data consumer, used without a code flag."""
    if not argv:
        return True  # assignments / redirections only
    exe = _basename(argv[0])
    if exe not in _DATA_CONSUMERS:
        return False
    args = _without_redirections(argv[1:])
    if exe == "sed":
        return _sed_treats_words_as_data(args)
    if exe == "tar":
        return _tar_treats_words_as_data(args)
    if exe in {"awk", "gawk", "mawk"}:
        return not any(_AWK_EXEC_RE.search(t) for t in args)
    if exe == "git":
        return _git_treats_words_as_data(args)
    if exe == "uv":
        operands = [t for t in args if not t.startswith("-")]
        return bool(operands) and operands[0] in _UV_DATA_SUBCOMMANDS
    guard = _CONSUMER_EXEC_OPTIONS.get(exe)
    return guard is None or not any(guard.search(t) for t in args)


def _raw_screen_views(command: str, depth: int = 0, root: str | None = None) -> list[str]:
    """Texts the Layer-1 raw regexes should see: only what the shell executes.

    Heredoc bodies fed to a non-shell and the contents of quoted arguments are
    data (``cat > a.md <<'MD'`` / ``echo "market halt"`` /
    ``python3 -c "print('reboot')"``), so they are blanked. Code the shell does
    run is recursed into: ``$(...)`` / backticks (also inside double quotes and
    unquoted heredoc bodies), ``bash -c`` / ``-lc`` strings, ``eval`` and
    heredocs consumed by a shell. Past the nesting limit, or when a segment
    cannot be tokenised, the unblanked text is screened (fail-closed).

    Blanking applies only when every command is a known data consumer or a form
    whose code is recovered exactly (shell ``-c``, evaluator adapters,
    ``find -exec``, a shell given a script file). An unknown command or a
    dynamic command name (``$x``, ``$(...)``) at any depth screens ``root``, the
    whole original command.
    """
    command = _normalize_ansi_c_quotes(command)
    root = _strip_comments(command) if root is None else root
    if depth > _MAX_NEST:
        return [_strip_comments(command)]
    stripped, bodies = _strip_heredoc_bodies(command)
    stripped = _strip_comments(stripped)
    views = [_blank_quoted(stripped)]
    nested = _extract_nested_shell(stripped) + bodies
    written: set[str] = set()
    executed: set[str] = set()
    for seg in _split_top_level(stripped):
        try:
            tokens = tokenize_shell_segment(_mask_nested_shell(seg))
        except ValueError:
            views.append(seg)
            continue
        keyword = _leading_shell_keyword(seg)
        if tokens and keyword in _CONTROL_LEADERS and tokens[0] == keyword:
            tokens[0] = _SHELL_SYNTAX_TOKEN
        shell_code = _shell_code_args(tokens)
        nested.extend(shell_code)
        evaluator_code, evaluator_fallback = _evaluator_payloads(tokens)
        nested.extend(evaluator_code)
        env_code = _env_split_payloads(tokens)
        nested.extend(env_code)
        if evaluator_fallback:
            views.append(seg)
        resolved = strip_command_prefixes(tokens)
        for payload in _find_exec_payloads(resolved):
            nested.append(shlex.join(payload))
        if any(_EXEC_ENV_ASSIGN_RE.match(t) for t in tokens):
            views.append(root)
        # A dynamic command name, or a command not known to treat its words as
        # data, gets the whole original text screened.
        if (resolved and _is_dynamic_name(resolved[0])) or not (
            _treats_words_as_data(resolved)
            or shell_code
            or evaluator_code
            or env_code
            # Path execution is useful for write/run correlation, but an
            # arbitrary /usr/bin/tool is not thereby a known data consumer.
            or _executed_script_operands(resolved)
        ) or (tokens and _basename(tokens[0]) == "xargs" and not resolved):
            views.append(root)
        if _reads_code_from_stdin(resolved):
            # Whatever feeds this command's stdin -- a pipe, a here-string or a
            # heredoc, quoted or not -- is code, and it can come from anywhere
            # in the command. Screen the unblanked text.
            views.append(_strip_comments(command))
        written |= _written_paths(tokens)
        executed |= _executed_script_paths(resolved)
        # Quote removal happens before the shell opens a redirect target, so
        # ``> "/dev/sda"`` must be screened by its real spelling.
        views.extend("> " + target for target in _redirect_targets(tokens))
        if resolved:
            # Quote removal happens before the shell looks the command up, so
            # ``"halt"`` / ``m''kfs`` must be screened by their real spelling.
            # The name is a command, not data; ``dd`` also needs its operands
            # (``of="/dev/sda"``).
            exe = _basename(resolved[0])
            views.append(" ".join([exe, *resolved[1:]]) if exe == "dd" else exe)
            if exe == "systemctl" and _systemctl_requests_shutdown(
                _without_redirections(resolved[1:]),
            ):
                views.append("shutdown")
    if written & executed:
        # A file this command writes and then runs as a shell script: whatever
        # was written into it (echoed text, a heredoc body) is code. Matching
        # by basename is deliberately loose -- it only errs towards denying.
        views.append(_strip_comments(command))
    for sub in nested:
        if sub.strip():
            views.extend(_raw_screen_views(sub, depth + 1, root))
    return views


# ``DROP TABLE`` keeps screening the whole text: SQL reaches its engine through
# quoted arguments and heredocs (``psql -c "…"`` / ``sqlite3 db <<SQL``), so
# blanking data would silently drop that guard.
_WHOLE_TEXT_DENY_PATTERNS = frozenset({r"\bDROP\s+TABLE\b"})


def _env_split_payloads(tokens: list[str]) -> list[str]:
    """Command strings ``env -S`` / ``--split-string`` splits and RUNS.

    ``env -S 'sudo id'`` executes ``sudo id``, but as one shell word it looked
    like a single argument and the real command was never assessed. The
    payload is returned as an ``env`` command line with the remaining words
    appended, as env runs it.
    """
    if _is_command_lookup(tokens):
        return []
    for k, tok in enumerate(tokens):
        # Only an env in command position can run its split string. A word in
        # ``echo env -S 'sudo id'`` is data, and words after an earlier -S are
        # arguments to the command that env starts.
        if _basename(tok) != "env" or _resolve_exe(tokens[:k])[0] is not None:
            continue
        i = k + 1
        while i < len(tokens):
            t = tokens[i]
            payload: str | None = None
            if t in ("-S", "--split-string"):
                if i + 1 >= len(tokens):
                    break
                payload, i = tokens[i + 1], i + 2
            elif t.startswith("--split-string="):
                payload, i = t.partition("=")[2], i + 1
            elif t.startswith("-") and not t.startswith("--") and "S" in t[1:]:
                rest = t[t.index("S", 1) + 1:]
                if rest:
                    payload, i = rest, i + 1
                elif i + 1 < len(tokens):
                    payload, i = tokens[i + 1], i + 2
                else:
                    break
            if payload is not None:
                # Re-prefixed with ``env``: the split string may itself start
                # with env options (``env -S '-i sudo id'``, the shebang idiom
                # ``env -S -i python3``), which only env's own option skipping
                # resolves to the real executable.
                return [" ".join(["env", payload, *(shlex.quote(w) for w in tokens[i:])])]
            if t == "--" or not t.startswith("-"):
                break
            i += 2 if t in _WRAPPER_OPTION_VALUES["env"] else 1
    return []


def _dynamic_env_split_reason(commands: list[list[str]]) -> str | None:
    """Refuse env -S when expansion determines the executable it will start.

    env expands ``${VAR}`` inside its split string after the shell has passed
    the argument to it. Without the resulting value, Layer 1.5 cannot tell
    whether that executable belongs to an always-denied group.
    """
    for argv in commands:
        for payload in _env_split_payloads(argv):
            try:
                words = tokenize_shell_segment(payload)
            except ValueError:
                return "Cannot safely inspect the executable in `env -S`."
            exe, _ = _resolve_exe(words)
            if exe is not None and _is_dynamic_name(exe):
                return (
                    "Refuses `env -S` with a dynamically generated executable "
                    "name; use a fixed command name instead."
                )
    return None


def _unknown_evaluator_words(argv: list[str]) -> list[str]:
    """Words of an evaluator whose argument form is not recognised
    (``su``, ``parallel``, an unknown ``tmux`` subcommand …).

    Its payload cannot be separated from its options, so the always-denied
    group check looks at every word instead of guessing: ``parallel ::: 'sudo
    id'`` is refused rather than allowed.
    """
    code, fallback = _evaluator_payloads(argv)
    if not fallback or code:
        return []
    words: list[str] = []
    for tok in argv:
        try:
            words.extend(shlex.split(tok))
        except ValueError:
            words.extend(tok.split())
    return words


def _is_command_lookup(argv: list[str]) -> bool:
    """``command -v X`` / ``command -V X`` asks whether ``X`` exists; it does
    not run it."""
    argv = [t for t in argv if t != _SHELL_SYNTAX_TOKEN and not _ASSIGN_RE.match(t)]
    return (
        len(argv) > 1 and _basename(argv[0]) == "command"
        and argv[1] in ("-v", "-V")
    )


def _argv_group_deny(commands: list[list[str]]) -> tuple[str, str] | None:
    """Layer 1.5 — ``(group, reason)`` for a command whose executable is in one
    of the always-denied groups (see ``_ALWAYS_DENIED_GROUPS``).

    Unlike :func:`_assess_allowlist` this runs in EVERY mode, which is the
    point: under the default ``off`` mode a plain ``sudo …`` / ``ssh host '…'``
    / ``pkill -f python3`` used to be assessed ``allow``.

    Resolution goes through :func:`_resolve_exe`, so wrappers are unwrapped
    (``env ssh``, ``timeout 5 sudo``), and the argv list handed in has already
    been flattened by :func:`_parse_commands` — nested ``$(...)``, ``bash -c``
    bodies and heredoc bodies are separate entries here (bounded by
    ``_MAX_NEST``).
    """
    for argv in commands:
        if _is_command_lookup(argv):
            continue
        for word in _unknown_evaluator_words(argv):
            base = _basename(word)
            if base in _PRIV_ESC:
                return "priv_esc", _DENY_GROUP_PRIV_ESC[base]
            hit = _ALWAYS_DENIED_BINARIES.get(base)
            if hit is not None:
                group, reason = hit
                return group, f"`{base}`: {reason}"
        exe, rest = _resolve_exe(argv)
        if exe is None:
            continue
        if exe == "__DENY__":
            # Privilege-escalation prefix, recognised during resolution.
            reason = rest[0] if rest else _DENY_GROUP_PRIV_ESC["sudo"]
            return "priv_esc", reason
        hit = _ALWAYS_DENIED_BINARIES.get(exe)
        if hit is not None:
            group, reason = hit
            return group, f"`{exe}`: {reason}"
    return None


#: Expansion punctuation, blanked before the bounded fallback screen.
_EXPANSION_PUNCT_RE = re.compile(r"\$\(\(|\$\(|<\(|>\(|[`()]")
_EXPANSION_SCREEN_FAILURE = "__FA_EXPANSION_SCREEN_FAILURE__"
_MAX_SCREEN_PASSES = 8
_RESIDUAL_SHELL_SYNTAX_RE = re.compile(r"[\s;|&'\"\\`()]|\$'")


def _expansion_commands(bodies: list[str]) -> list[list[str]]:
    """Conservatively screen code past the parser's nesting limit.

    Flatten expansion punctuation once per pass, split shell separators, and
    retain both complete argvs (for argument-sensitive hard denials) and each
    word as a candidate executable. Quoted code is decoded by tokenization;
    words still containing shell syntax are screened again, without recursion.
    This intentionally treats code-looking data as code at excessive depth.

    Both the pass count and total input scanned are bounded. If quoting cannot
    be resolved within that budget, or tokenization fails, record a hard denial
    rather than dropping the remaining code. A plain substitution chain of any
    depth needs just one pass, so its cost stays linear in the input size.
    """
    commands: list[list[str]] = []
    candidates: list[list[str]] = []
    pending = bodies
    remaining = max(1_024, _MAX_SCREEN_PASSES * sum(map(len, bodies)))
    for _ in range(_MAX_SCREEN_PASSES):
        next_pass: list[str] = []
        for body in pending:
            remaining -= len(body)
            if remaining < 0:
                return [*commands, *candidates, [_EXPANSION_SCREEN_FAILURE]]
            flat = _EXPANSION_PUNCT_RE.sub(" ", _normalize_ansi_c_quotes(body))
            for segment in _split_top_level(flat):
                try:
                    words = tokenize_shell_segment(segment)
                except ValueError:
                    return [*commands, *candidates, [_EXPANSION_SCREEN_FAILURE]]
                if not words:
                    continue
                keyword = _leading_shell_keyword(segment)
                if keyword in _CONTROL_LEADERS and words[0] == keyword:
                    words[0] = _SHELL_SYNTAX_TOKEN
                commands.append(words)
                commands.extend(_find_exec_payloads(strip_command_prefixes(words)))
                candidates.extend([word] for word in words if word.strip())
                for word in words:
                    # Encoded redirection syntax is already checked in the
                    # complete argv; feeding its sentinel back to shlex would
                    # turn an ordinary redirect into a parse failure.
                    if redirection_token(word) is None and _RESIDUAL_SHELL_SYNTAX_RE.search(word):
                        next_pass.append(word)
        if not next_pass:
            # Keep complete commands in execution order. Interspersing a
            # singleton `cd` would reset the protected-directory state before
            # a following relative deletion could be checked.
            return [*commands, *candidates]
        pending = next_pass
    return [*commands, *candidates, [_EXPANSION_SCREEN_FAILURE]]


def _parse_commands(command: str, depth: int = 0) -> list[list[str]]:
    """Parse into a list of argv lists (one per simple command), recursively
    including commands nested in ``$(...)`` / backticks, in the code argument of
    ``eval`` / ``bash -c`` (also ``-lc``), in ``find -exec`` payloads, and in
    shell heredoc bodies. Raises :class:`_ParseError` when a top-level segment
    can't be tokenised (unbalanced quotes)."""
    command = _normalize_ansi_c_quotes(command)
    stripped, heredoc_bodies = _strip_heredoc_bodies(command)
    # After heredoc bodies are out of the way (their ``#`` lines are data/code,
    # not shell comments) drop the shell's own comments.
    stripped = _strip_comments(stripped)
    argvs: list[list[str]] = []
    for seg in _split_top_level(stripped):
        try:
            # Nested substitutions are assessed recursively below; mask them
            # from the outer argv so their spaces/options cannot create phantom
            # executables (``x=$(nginx -V)`` would otherwise yield ``-V)``).
            tokens = tokenize_shell_segment(_mask_nested_shell(seg))
        except ValueError as exc:
            raise _ParseError(str(exc)) from exc
        if tokens:
            keyword = _leading_shell_keyword(seg)
            if keyword in _LOOP_HEADERS:
                # The header is syntax plus a variable/word list, not a
                # command. Nested ``$(...)`` is still collected below from the
                # original string.
                continue
            if keyword in _CONTROL_LEADERS and tokens[0] == keyword:
                tokens[0] = _SHELL_SYNTAX_TOKEN
            argvs.append(tokens)

    # find -exec/-execdir/-ok payloads become their own commands to assess.
    for argv in list(argvs):
        argvs.extend(_find_exec_payloads(argv))

    nested = _extract_nested_shell(stripped) + list(heredoc_bodies)
    for raw_argv in list(argvs):
        # ``_shell_code_args`` unwraps prefixes first, so ``env bash -c …`` /
        # ``timeout 10 bash -lc …`` / ``xargs sh -c …`` are recognised as
        # nested shells (not just bare ``bash``/``eval`` at argv[0]).
        nested.extend(_shell_code_args(raw_argv))
        # Code other runners execute: ``watch 'x'`` / ``script -c 'x'`` /
        # ``tmux new 'x'`` / ``ssh host 'x'``, and ``env -S 'x'``. The word
        # screens already recursed into these; without this the group check
        # (Layer 1.5) and the allowlist never saw ``watch 'sudo id'``.
        nested.extend(_evaluator_payloads(raw_argv)[0])
        nested.extend(_env_split_payloads(raw_argv))
    if depth >= _MAX_NEST:
        # Deep code is screened without further parser recursion. The fallback
        # must retain separators, quoted payloads and argument-sensitive rules;
        # anything it cannot inspect within bounded work fails closed.
        argvs.extend(_expansion_commands(nested))
        return argvs

    for sub in nested:
        if sub.strip():
            # Unparseable nested code — the outer parse already recorded it.
            with contextlib.suppress(_ParseError):
                argvs.extend(_parse_commands(sub, depth + 1))
    return argvs


def _resolve_exe(argv: list[str]) -> tuple[str | None, list[str]]:
    """Return ``(executable_basename, remaining_args)`` after stripping leading
    ``VAR=val`` assignments and unwrapping prefix wrappers (``env``, ``timeout``,
    ``xargs``, ``command``, ``exec`` …, skipping their option flags so
    ``command -v python3`` resolves to ``python3``). ``("__DENY__", [reason])``
    for privilege-escalation prefixes."""
    i = 0
    n = len(argv)
    while i < n:
        tok = argv[i]
        if _ASSIGN_RE.match(tok):
            i += 1
            continue
        after_redirect = _skip_redirection(argv, i)
        if after_redirect is not None:
            i = after_redirect
            continue
        base = _basename(tok)
        if base in ("sudo", "su", "doas", "pkexec"):
            return "__DENY__", [_DENIED_BINARIES.get(base, "Privilege escalation is not allowed.")]
        if base in _WRAPPERS:
            # skip the wrapper's option flags AND their separate values (so
            # ``nice -n 10 rm`` / ``timeout -s 9 10 bash`` resolve past ``10``).
            i = _skip_wrapper_args(argv, i + 1, wrapper=base)
            continue
        if tok == _SHELL_SYNTAX_TOKEN:
            i += 1
            continue
        return base, argv[i + 1:]
    return None, []


def _assess_allowlist(commands: list[list[str]], *, mode: str) -> BashCommandAssessment:
    """Layer 2. ``mode`` is ``warn`` or ``enforce``."""
    worst = BashCommandAssessment(level="allow", reason="Command allowed.")

    def _raise(level: str, reason: str) -> None:
        nonlocal worst
        order = {"allow": 0, "audit": 1, "confirm": 2, "deny": 3}
        if order[level] > order[worst.level]:
            worst = BashCommandAssessment(level=level, reason=reason)

    for argv in commands:
        exe, rest = _resolve_exe(argv)
        if exe == "__DENY__":
            return BashCommandAssessment(level="deny", reason=rest[0] if rest else "Denied.")
        if exe is None:
            continue
        if exe in _DENIED_BINARIES:
            return BashCommandAssessment(
                level="deny", reason=f"`{exe}`: {_DENIED_BINARIES[exe]}",
            )
        if exe in _ALLOWED_BINARIES:
            if exe in _AUDIT_BINARIES:
                _raise("audit", f"`{exe}` (package tooling) is audited.")
            elif exe.startswith("python") and any(f in rest for f in _INLINE_CODE_FLAGS):
                _raise("audit", (
                    "`python3 -c` inline code can't be inspected; prefer a "
                    "`python3 <<'PY' ... PY` heredoc."
                ))
            continue
        if _SUBSTITUTION_SENTINEL in exe or _ARITHMETIC_SENTINEL in exe:
            reason = (
                "A shell expansion appears in executable position, so the "
                "command name is generated dynamically and cannot be checked "
                "against the allowed-command list. Invoke a fixed allowlisted "
                "executable instead."
            )
            if mode == "enforce":
                return BashCommandAssessment(level="deny", reason=reason)
            _raise("audit", f"[allowlist:warn] {reason}")
            continue
        # Not on the allowlist.
        # Name every allowed category, `pip` included. The message used to omit
        # package tooling even though pip has been allowlisted since the layer
        # shipped, so a model whose `pip` command was denied for an unrelated
        # reason (a `cd &&` prefix, a stray token) read "pip is not allowed",
        # reached for workarounds, and re-downloaded packages the image bakes.
        #
        # The closing hint deliberately does NOT name packages: this module is
        # shared by every image, and what each bakes differs (worker-shell /
        # standalone carry only the `sandbox` extra, which excludes matplotlib
        # and the office stack by design — see pyproject.toml). Naming them here
        # would send those models to an ImportError, which is the same
        # misinformation-driven thrash, inverted.
        reason = (
            f"`{exe}` is not on the allowed-command list for this sandboxed "
            f"eval agent. Allowed: file inspection (ls/cat/head/grep/find/…), "
            f"text tools (sed/awk/jq/sort/…), archive extraction (unzip/tar/…), "
            f"document conversion (soffice/pandoc/pdftotext/pdftoppm/…), "
            f"HTTP retrieval (curl/wget/aria2c), package installs (pip/pip3), "
            f"and `python3` for computation. This image bakes a scientific and "
            f"document stack — check with `python3 -c 'import <pkg>'` before "
            f"installing anything."
        )
        if mode == "enforce":
            return BashCommandAssessment(level="deny", reason=reason)
        _raise("audit", f"[allowlist:warn] {reason}")

    return worst


# ── Public entry point ──────────────────────────────────────────────────


def assess_bash_command(
    command: str, *, mode: str | None = None, interactive: bool = False,
) -> BashCommandAssessment:
    """Classify a bash command into ``allow`` / ``audit`` / ``confirm`` / ``deny``.

    ``mode`` overrides the resolved policy mode (see :func:`resolve_mode`); left
    ``None`` it is resolved from env / contextvar / config / scope. Regardless
    of mode, the Layer-1 hard denylist always runs first and the always-denied
    groups (privilege escalation, remote/exfil clients, signal senders) run
    right after it (Layer 1.5).

    ``interactive`` is for a caller that puts every non-``allow`` verdict in
    front of a human (the local coding CLI). A Layer-1.5 group hit is then
    ``confirm`` (with :attr:`BashCommandAssessment.group` set) instead of
    ``deny``, so the human decides; the caller must not let auto-approval or a
    saved rule answer it. Layer 1 stays ``deny`` either way. Only calling code
    can pass this — no config, profile or request field reaches it.
    """
    normalized = command.strip()
    if not normalized:
        return BashCommandAssessment(level="deny", reason="Empty command.")

    effective_mode = mode if mode in _VALID_MODES else resolve_mode(mode)

    # ── Layer 1: hard denylist (all modes) ──
    # Raw screens must see only shell code. In particular, a protected-looking
    # redirect in ``# > /etc/passwd`` is inert comment text, not an attempted
    # write. The argv parser below performs the same stripping independently.
    #
    # The same holds for data: heredoc bodies and quoted arguments routinely
    # carry prose ("an exchange halt applies"), so the word screens only see
    # the text the shell executes (see ``_raw_screen_views``).
    executable_text = _strip_comments(_normalize_ansi_c_quotes(normalized))
    screen_text = "\n".join(_raw_screen_views(normalized))
    for pattern, reason in _DENY_PATTERNS:
        text = executable_text if pattern in _WHOLE_TEXT_DENY_PATTERNS else screen_text
        if re.search(pattern, text, re.IGNORECASE):
            return BashCommandAssessment(level="deny", reason=reason)

    for match in _REDIRECT_PROTECTED_RE.finditer(executable_text):
        # A redirect-shaped string passed as data (``echo '> /etc/passwd'``)
        # is not shell syntax and must not be treated as a write.
        if _inside_shell_quote(executable_text, match.start()):
            continue
        # ``\S*`` swallows any shell punctuation glued to the target
        # (``2>/dev/null;`` / ``>/dev/null)``) — trim it before classifying.
        # Only quotes are stripped now: the pattern no longer swallows shell
        # punctuation, so trimming ``;&|)`` would cut into a real filename.
        target = match.group(1).strip("\"'")
        if _REDIRECT_SAFE_DEVICE_RE.match(target):
            continue
        # This run's own writable directories, wherever they were mounted.
        # Containment is computed after ``..`` collapsing, so this cannot be
        # used to climb out of a writable root into a real system path.
        if not _under_run_read_only_root(target) and _within_writable_root(target):
            continue
        # Name the directories this run actually has rather than the canonical
        # mounts: this text reaches the model right after it tried a path that
        # was refused, which is exactly where a self-teaching error earns its
        # keep. Every directory named here must itself accept a redirect —
        # ``test_the_deny_reason_never_recommends_a_path_it_would_refuse``.
        writable = ", ".join(_writable_roots()) or "/workspace, /outputs or /tmp"
        return BashCommandAssessment(
            level="deny",
            reason=(
                f"Refuses output redirection into a protected system path "
                f"(`{target}`). Write to {writable} instead; "
                f"`>/dev/null` to discard is fine."
            ),
        )

    parse_error: _ParseError | None = None
    commands: list[list[str]] = []
    try:
        commands = _parse_commands(normalized)
    except _ParseError as exc:
        parse_error = exc

    if commands:
        argv_reason = _argv_hard_deny(commands)
        if argv_reason:
            return BashCommandAssessment(level="deny", reason=argv_reason)

        redirect_reason = _redirect_protection_reason(commands)
        if redirect_reason:
            return BashCommandAssessment(level="deny", reason=redirect_reason)

        env_reason = _dynamic_env_split_reason(commands)
        if env_reason:
            return BashCommandAssessment(level="deny", reason=env_reason)

        # ── Layer 1.5: always-denied groups (every mode) ──
        group_hit = _argv_group_deny(commands)
        if group_hit:
            group, reason = group_hit
            return BashCommandAssessment(
                level="confirm" if interactive else "deny", reason=reason, group=group,
            )

    # ── off mode: legacy denylist-only behaviour ──
    if effective_mode == "off":
        for pattern, reason in _CONFIRM_PATTERNS:
            if re.search(pattern, normalized, re.IGNORECASE):
                return BashCommandAssessment(level="confirm", reason=reason)
        for pattern, reason in _AUDIT_PATTERNS:
            if re.search(pattern, normalized, re.IGNORECASE):
                return BashCommandAssessment(level="audit", reason=reason)
        return BashCommandAssessment(level="allow", reason="Command allowed.")

    # ── warn / enforce: allowlist ──
    if parse_error is not None:
        reason = (
            f"Could not parse command safely ({parse_error}); simplify it "
            "(one operation per call, balanced quotes)."
        )
        if effective_mode == "enforce":
            return BashCommandAssessment(level="deny", reason=reason)
        return BashCommandAssessment(level="audit", reason=f"[allowlist:warn] {reason}")

    return _assess_allowlist(commands, mode=effective_mode)
