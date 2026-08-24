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


# ── Mode resolution ─────────────────────────────────────────────────────

_VALID_MODES = ("off", "warn", "enforce")
_DEFAULT_MODE = "off"

# Per-run override, set by workflows adjacent to their per-task sandbox (mirrors
# the ``_task_sandbox`` contextvar pattern). Propagates into child asyncio tasks
# (agent_team sub-agents inherit the main agent's mode automatically).
_policy_mode_var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "mh_bash_policy_mode", default=None,
)


def set_policy_mode(mode: str) -> contextvars.Token:
    """Set the bash-policy mode for the current context. Returns a token for
    :func:`reset_policy_mode`. An invalid mode stores ``None`` (→ falls back to
    env / config / default)."""
    return _policy_mode_var.set(mode if mode in _VALID_MODES else None)


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
    per-run contextvar (workflow default) → config → ExecutionScope metadata →
    ``off``.
    """
    for candidate in (
        (explicit or "").strip().lower() if explicit else "",
        _env_mode(),
        _policy_mode_var.get() or "",
        _config_mode(),
        _scope_mode(),
    ):
        if candidate in _VALID_MODES:
            return candidate
    return _DEFAULT_MODE


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


def _under_protected_root(path: str) -> bool:
    return path in _PROTECTED_TARGETS or any(
        path == root or path.startswith(root + "/") for root in _SYSTEM_ROOTS
    )


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
    if _under_protected_root(a):
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


def _skip_wrapper_args(argv: list[str], i: int) -> int:
    """Advance ``i`` past a wrapper's option flags AND the separate value tokens
    they consume, so ``nice -n 10 rm`` / ``timeout -s 9 10 bash`` resolve to the
    real command (``rm`` / ``bash``) rather than the value token (``10``). This
    is the single home for wrapper-arg skipping — shared by
    strip_command_prefixes and
    _resolve_exe so they can't drift."""
    n = len(argv)
    while i < n and (argv[i].startswith("-") or _WRAPPER_VALUE_RE.fullmatch(argv[i])):
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
            i = _skip_wrapper_args(argv, i + 1)
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


def _argv_hard_deny(commands: list[list[str]]) -> str | None:
    """Robust dangerous-op detection on parsed argvs. Complements the raw
    regex — order-, flag-combination-, prefix- and quote-independent.
    """
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

# Denied in ``warn`` and ``enforce`` — but NOT in ``off``, which is
# ``_DEFAULT_MODE`` and therefore what most deployments run. This table is
# consulted from ``_assess_allowlist``, which the ``off`` branch returns before
# reaching; in ``off`` the only binding checks are ``_DENY_PATTERNS`` and
# ``_argv_hard_deny`` above. Anything that must hold unconditionally (e.g.
# ``curl … | bash``) belongs there, not here.
#
# Privilege escalation / nested shells / remote administration / host +
# package management. HTTP download clients are intentionally absent: current
# task containers have a writable filesystem and network access, and
# controlled document downloads use ``download_file``.
_DENIED_BINARIES: dict[str, str] = {
    **{b: "Privilege escalation is not allowed." for b in ("sudo", "su", "doas", "pkexec")},
    **{b: (
        "Nested/piped shells are not allowed — run the program directly or use "
        "a ``python3 <<'PY' ... PY`` heredoc."
    ) for b in ("bash", "sh", "zsh", "dash", "ksh", "csh", "tcsh", "fish", "ash")},
    "eval": "``eval`` of dynamic strings is not allowed.",
    **{b: (
        "Interactive network and remote-administration clients are not allowed. "
        "Use web/search/download tools or an HTTP client instead."
    ) for b in (
        "nc", "ncat", "netcat", "socat", "telnet",
        "ssh", "scp", "sftp", "ftp", "tftp", "rsync", "rclone",
    )},
    **{b: "System / host administration is not allowed." for b in (
        "mount", "umount", "fdisk", "parted", "swapon", "systemctl", "service",
        "init", "kexec", "insmod", "modprobe", "sysctl", "iptables", "nft",
        "ip", "ifconfig", "route", "ufw", "kill", "killall", "pkill",
        "crontab", "at", "batch",
    )},
    **{b: "Installing system packages is not allowed." for b in (
        "apt", "apt-get", "aptitude", "yum", "dnf", "dpkg", "rpm", "pacman",
        "brew", "conda", "mamba", "snap",
    )},
}

# Allowlisted but noteworthy → ``audit`` (runs, tagged).
_AUDIT_BINARIES = frozenset({"pip", "pip3"})

# Interpreters whose inline-code flag can't be introspected → ``audit``.
_INLINE_CODE_FLAGS = frozenset({"-c", "-e", "--command", "--eval"})

_ASSIGN_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")
_HEREDOC_RE = re.compile(r"<<-?\s*(['\"]?)([A-Za-z_]\w*)\1")
_REDIRECT_PROTECTED_RE = re.compile(
    r"(?:&>>?|>\||>&|>>?)\s*"
    r"(/(?:etc|usr|bin|sbin|lib|lib64|boot|dev|proc|sys|var|root|opt)\b\S*)"
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
        toks = shlex.split(segs[-1], comments=False)
    except ValueError:
        return False
    if not toks:
        return False
    exe, _ = _resolve_exe(toks)
    return exe in _SHELLS


def _strip_heredoc_bodies(command: str) -> tuple[str, list[str]]:
    """Drop here-document *bodies* (data, not commands) so they aren't parsed as
    top-level shell. Returns ``(stripped_command, shell_bodies)`` where
    ``shell_bodies`` are the bodies whose consuming command is a shell (e.g.
    ``bash <<EOF … EOF`` — the body is shell code the shell executes, so it is
    recursed into by :func:`_parse_commands`). The ``cmd <<'MARKER'`` line is
    kept."""
    lines = command.split("\n")
    out: list[str] = []
    shell_bodies: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        out.append(line)
        m = _HEREDOC_RE.search(line)
        if m:
            delim = m.group(2)
            quoted = bool(m.group(1))  # <<'EOF' / <<"EOF" suppress expansion
            consumer_is_shell = _line_invokes_shell(line[:m.start()])
            i += 1
            body: list[str] = []
            while i < len(lines) and lines[i].strip() != delim:
                body.append(lines[i])
                i += 1
            i += 1  # skip the closing delimiter line
            body_text = "\n".join(body)
            if body_text:
                if consumer_is_shell:
                    # The whole body is shell code the shell executes.
                    shell_bodies.append(body_text)
                elif not quoted:
                    # Unquoted delimiter: the shell still expands $()/`` in the
                    # body before the (non-shell) consumer sees it.
                    shell_bodies.extend(_extract_nested_shell(body_text))
            continue
        i += 1
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
    must not fragment the outer command. Bare ``(``/``)`` (subshell grouping) DO
    split, so ``(rm -rf /)`` is analysed. Redirections (``>`` ``<``) do not split.
    """
    segs: list[str] = []
    buf: list[str] = []
    i, n = 0, len(command)
    quote: str | None = None
    subst = 0  # depth inside $(...)
    while i < n:
        c = command[i]
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
        if subst > 0:  # inside $(...): copy verbatim, track nesting, never split
            buf.append(c)
            if c == "(":
                subst += 1
            elif c == ")":
                subst -= 1
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
        if c == "$" and i + 1 < n and command[i + 1] == "(":
            buf.append("$(")
            subst = 1
            i += 2
            continue
        if c == "`":  # backtick span — copy verbatim to the closing backtick
            buf.append(c)
            i += 1
            while i < n and command[i] != "`":
                buf.append(command[i])
                i += 1
            if i < n:
                buf.append(command[i])
                i += 1
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


def _extract_nested_shell(command: str) -> list[str]:
    """Return shell-code strings nested in ``$(...)`` and backticks (which the
    shell expands+executes). Single-quoted spans are skipped — the shell does
    not expand them, so ``echo '$(rm -rf /)'`` is a harmless literal."""
    out: list[str] = []
    i, n = 0, len(command)
    sq = False
    while i < n:
        c = command[i]
        if sq:
            if c == "'":
                sq = False
            i += 1
            continue
        if c == "'":
            sq = True
            i += 1
            continue
        if c == "$" and i + 1 < n and command[i + 1] == "(":
            depth, j = 1, i + 2
            start = j
            while j < n and depth:
                if command[j] == "(":
                    depth += 1
                elif command[j] == ")":
                    depth -= 1
                j += 1
            if depth == 0:
                out.append(command[start:j - 1])
            i = j
            continue
        if c == "`":
            j = i + 1
            while j < n and command[j] != "`":
                j += 1
            out.append(command[i + 1:j])
            i = j + 1
            continue
        i += 1
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


def _parse_commands(command: str, depth: int = 0) -> list[list[str]]:
    """Parse into a list of argv lists (one per simple command), recursively
    including commands nested in ``$(...)`` / backticks, in the code argument of
    ``eval`` / ``bash -c``, in ``find -exec`` payloads, and in shell heredoc
    bodies. Raises :class:`_ParseError` when a top-level segment can't be
    tokenised (unbalanced quotes)."""
    stripped, heredoc_bodies = _strip_heredoc_bodies(command)
    # After heredoc bodies are out of the way (their ``#`` lines are data/code,
    # not shell comments) drop the shell's own comments.
    stripped = _strip_comments(stripped)
    argvs: list[list[str]] = []
    for seg in _split_top_level(stripped):
        try:
            tokens = tokenize_shell_segment(seg)
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

    if depth >= _MAX_NEST:
        return argvs

    nested = _extract_nested_shell(stripped) + list(heredoc_bodies)
    for raw_argv in list(argvs):
        # Unwrap prefixes first so ``env bash -c …`` / ``timeout 10 bash -c …`` /
        # ``xargs sh -c …`` are recognised as nested shells (not just bare
        # ``bash``/``eval`` at argv[0]).
        argv = strip_command_prefixes(raw_argv)
        if not argv:
            continue
        base = _basename(argv[0])
        if base == "eval" and len(argv) > 1:
            nested.append(" ".join(argv[1:]))
        elif base in _SHELLS and "-c" in argv:
            k = argv.index("-c")
            if k + 1 < len(argv):
                nested.append(argv[k + 1])
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
            i = _skip_wrapper_args(argv, i + 1)
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


def assess_bash_command(command: str, *, mode: str | None = None) -> BashCommandAssessment:
    """Classify a bash command into ``allow`` / ``audit`` / ``confirm`` / ``deny``.

    ``mode`` overrides the resolved policy mode (see :func:`resolve_mode`); left
    ``None`` it is resolved from env / contextvar / config / scope. Regardless
    of mode, the Layer-1 hard denylist always runs first.
    """
    normalized = command.strip()
    if not normalized:
        return BashCommandAssessment(level="deny", reason="Empty command.")

    effective_mode = mode if mode in _VALID_MODES else resolve_mode(mode)

    # ── Layer 1: hard denylist (all modes) ──
    # Raw screens must see only shell code. In particular, a protected-looking
    # redirect in ``# > /etc/passwd`` is inert comment text, not an attempted
    # write. The argv parser below performs the same stripping independently.
    executable_text = _strip_comments(normalized)
    for pattern, reason in _DENY_PATTERNS:
        if re.search(pattern, executable_text, re.IGNORECASE):
            return BashCommandAssessment(level="deny", reason=reason)

    for match in _REDIRECT_PROTECTED_RE.finditer(executable_text):
        # A redirect-shaped string passed as data (``echo '> /etc/passwd'``)
        # is not shell syntax and must not be treated as a write.
        if _inside_shell_quote(executable_text, match.start()):
            continue
        # ``\S*`` swallows any shell punctuation glued to the target
        # (``2>/dev/null;`` / ``>/dev/null)``) — trim it before classifying.
        target = match.group(1).rstrip(";&|)\"'")
        if _REDIRECT_SAFE_DEVICE_RE.match(target):
            continue
        return BashCommandAssessment(
            level="deny",
            reason=(
                f"Refuses output redirection into a protected system path "
                f"(`{target}`). Write to /workspace, /outputs or /tmp instead; "
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
