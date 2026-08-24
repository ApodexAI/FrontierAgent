"""Demo-safe boundaries: secret redaction, tool policy, download containment.

Three independent jobs, deliberately kept in one small module so a reviewer
can see the whole public-exposure surface at once:

1. **Secrets never leave the process** — not in logs, not in the UI, not in a
   traceback, not in a downloadable file. Three distinct escapes have to be
   closed for that claim to hold: the answer *stream* (where a secret can
   straddle two SSE chunks, so :class:`StreamRedactor` is stateful), *tool
   arguments* (which reach the tool verbatim, hence the rewrite in
   ``containment.SecretArgumentObserver``), and *file contents* (hence the
   optional scan in :func:`list_output_files`).
2. **The agent gets a demo-safe toolset** — real research and real file
   deliverables, no arbitrary shell, no package installs, no subprocess.
3. **Downloads cannot escape the session's ``outputs/``** — not via ``..``,
   not via an absolute path, not via a symlink.
"""

from __future__ import annotations

import logging
import os
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

#: Shortest secret we will substring-match. Below this, a "secret" would match
#: so much ordinary text that redaction would destroy the logs it protects.
_MIN_SECRET_LEN = 8

logger = logging.getLogger(__name__)

REDACTED = "***REDACTED***"

#: Credential shapes worth catching even when we were never told the value —
#: e.g. a key the *upstream endpoint* echoes back inside an error body.
_SECRET_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bsk-[A-Za-z0-9_\-]{16,}"),
    re.compile(r"\bhf_[A-Za-z0-9]{16,}"),
    re.compile(r"\b[Bb]earer\s+[A-Za-z0-9._\-]{16,}"),
    # The key name may be quoted (JSON), so allow a closing quote before the
    # separator: {"api_key": "…"} must match as readily as api_key=…
    re.compile(
        r"(?i)\b(api[_-]?key|authorization|token)\b[\"']?\s*[:=]\s*[\"']?"
        r"([A-Za-z0-9._\-]{12,})",
    ),
)

#: Tools the demo refuses regardless of what the allowlist asks for. This is
#: the fail-closed half of the policy: ``allowed_tools`` can only ever narrow
#: the set further, never widen it past this line.
HARD_DENIED_TOOLS: frozenset[str] = frozenset({
    # Arbitrary command / code execution.
    "bash",
    "run_python_code",
    # Unbounded network writes to the local filesystem.
    "download_file",
    # Sub-agent orchestration: out of P0 scope and multiplies model spend.
    "create_subagent",
    "assign_task",
    "stop_subagent",
    "collect_reports",
    "submit_report",
    # Extra editing surface with no P0 use. ``write_file`` is the one write path
    # the demo keeps: it writes in-process through ``plugins/tools/_path_auth``,
    # which is fail-closed and confined to the session's authorised tree.
    "file_editor_create",
    "file_editor_str_replace",
})

#: Names never offered for download even when they land in ``outputs/``.
_BLOCKED_DOWNLOAD_NAMES: frozenset[str] = frozenset({
    ".env", ".env.local", "config.yaml", "id_rsa", ".netrc", ".git-credentials",
})

#: Substrings that make a filename undownloadable (defence in depth with the
#: runtime's own ``plugins/tools/_path_auth._BLOCKED_PATTERNS``).
_BLOCKED_DOWNLOAD_SUBSTRINGS: tuple[str, ...] = (
    "secret", "credential", "password", "token", ".pem", ".key", ".crt",
)


class DownloadDenied(PermissionError):
    """A requested download is outside the session's ``outputs/`` directory."""


@dataclass(frozen=True)
class Redactor:
    """Replaces known secret values (and credential-shaped text) with a mask."""

    literals: tuple[str, ...] = ()

    @classmethod
    def for_secrets(cls, secrets: Iterable[str]) -> Redactor:
        literals = tuple(sorted(
            {s for s in (str(x or "").strip() for x in secrets) if len(s) >= _MIN_SECRET_LEN},
            key=len,
            reverse=True,  # longest first, so a prefix never masks a superstring
        ))
        return cls(literals=literals)

    def __call__(self, value: str) -> str:
        return self.redact(value)

    def redact(self, value: str) -> str:
        if not value:
            return value
        out = self.redact_literals(value)
        for pattern in _SECRET_PATTERNS:
            out = pattern.sub(_mask_match, out)
        return out

    def redact_literals(self, value: str) -> str:
        """Literal-only pass, for rewriting content rather than previewing it.

        Used where the text is data the agent will act on (tool arguments, file
        content), so only *configured* secrets are touched. Credential-shaped
        text that is not one of our secrets is left alone — mangling it would
        corrupt legitimate deliverables to protect nothing.
        """
        if not value:
            return value
        out = value
        for literal in self.literals:
            if literal in out:
                out = out.replace(literal, REDACTED)
        return out

    def contains_secret(self, value: str) -> bool:
        """True when ``value`` carries a configured secret verbatim."""
        return bool(value) and any(lit in value for lit in self.literals)

    @property
    def longest_literal(self) -> int:
        return max((len(lit) for lit in self.literals), default=0)


#: Minimum tail a :class:`StreamRedactor` holds back. Sized so the regex
#: patterns above also get a chance to match across a chunk boundary, not just
#: the known literals.
_MIN_STREAM_HOLD = 48


class StreamRedactor:
    """Redacts a token stream, tolerating secrets split across chunks.

    SSE chunk boundaries are arbitrary. Redacting each delta independently is
    therefore unsound: a secret split as ``"sk-fake-demo-"`` + ``"key-1234"``
    matches neither half, both are emitted verbatim, and whoever concatenates
    the deltas — the browser, a log, an event consumer — gets the whole thing
    back.

    So this keeps a rolling buffer and withholds the last ``hold`` characters
    until more text arrives or :meth:`flush` is called. The cost is that the
    visible answer trails the stream by at most ``hold`` characters.
    """

    def __init__(self, redactor: Redactor, *, hold: int | None = None) -> None:
        self._redactor = redactor
        # A literal of length L can only be recognised if at least L-1 of its
        # characters are still buffered when the rest arrives.
        needed = max(redactor.longest_literal - 1, 0)
        self._hold = max(hold if hold is not None else needed, _MIN_STREAM_HOLD)
        self._buffer = ""

    @property
    def pending(self) -> int:
        return len(self._buffer)

    def feed(self, chunk: str) -> str:
        """Absorb ``chunk`` and return the text that is now safe to emit."""
        if not chunk:
            return ""
        redacted = self._redactor.redact(self._buffer + chunk)
        if len(redacted) <= self._hold:
            self._buffer = redacted
            return ""
        self._buffer = redacted[-self._hold:]
        return redacted[: -self._hold]

    def flush(self) -> str:
        """Emit whatever is still held back. Call this at end of stream."""
        remaining = self._redactor.redact(self._buffer)
        self._buffer = ""
        return remaining

    def discard(self) -> None:
        """Throw away the held-back tail.

        For an attempt the runtime discarded: that text is not part of the
        answer, so flushing it would splice a dead draft onto the retry.
        """
        self._buffer = ""


def redact_deep(value: Any, redactor: Redactor) -> Any:
    """Redact configured secrets in every string inside a nested structure.

    Tool arguments are not always flat — ``create_file`` takes a nested ops
    program, for instance — so a shallow pass would miss the interesting cases.
    """
    if isinstance(value, str):
        return redactor.redact_literals(value)
    if isinstance(value, dict):
        return {k: redact_deep(v, redactor) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        rebuilt = [redact_deep(v, redactor) for v in value]
        return type(value)(rebuilt) if isinstance(value, tuple) else rebuilt
    return value


def _mask_match(match: re.Match[str]) -> str:
    """Keep any label the pattern captured; mask only the credential itself."""
    if match.re.groups >= 2:
        return f"{match.group(1)}={REDACTED}"
    return REDACTED


class RedactingLogFilter(logging.Filter):
    """Scrub secrets from every log record that passes through a handler.

    Installed on the *root* logger's handlers, so it also covers third-party
    libraries (openai/httpx) that may echo a request header on error.
    """

    def __init__(self, redactor: Redactor) -> None:
        super().__init__()
        self._redactor = redactor

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            record.msg = self._redactor.redact(str(record.msg))
            if record.args:
                if isinstance(record.args, dict):
                    record.args = {
                        k: self._redactor.redact(v) if isinstance(v, str) else v
                        for k, v in record.args.items()
                    }
                elif isinstance(record.args, tuple):
                    record.args = tuple(
                        self._redactor.redact(a) if isinstance(a, str) else a
                        for a in record.args
                    )
        except Exception:
            pass
        return True


def install_log_redaction(redactor: Redactor) -> None:
    """Attach ``redactor`` to root logging (idempotent)."""
    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(level=logging.WARNING)
    for handler in root.handlers:
        if any(isinstance(f, RedactingLogFilter) for f in handler.filters):
            continue
        handler.addFilter(RedactingLogFilter(redactor))


def demo_safe_tool_names(
    allowed: Sequence[str], *, public_mode: bool = True,
) -> tuple[str, ...]:
    """Intersect the requested allowlist with what a demo may expose."""
    denied = HARD_DENIED_TOOLS if public_mode else frozenset()
    return tuple(
        name for name in dict.fromkeys(allowed)
        if name and name.lower() not in denied
    )


def demo_safe_tool_policy(
    allowed: Sequence[str], *, public_mode: bool = True,
) -> Any:
    """Build the runtime's process-wide tool policy for this demo.

    Returns a ``ToolPermissionContext`` combining an allowlist (only these
    tools may bind) with the hard deny list (fail-closed even if a profile,
    a role definition, or a future allowlist edit tries to re-add one).
    """
    from frontier_agent.core.runtime.resources.tool_permission import (
        ToolPermissionContext,
    )

    names = demo_safe_tool_names(allowed, public_mode=public_mode)
    return ToolPermissionContext.from_iterables(
        allow_names=set(names),
        deny_names=set(HARD_DENIED_TOOLS) if public_mode else set(),
    )


def is_downloadable(path: Path) -> bool:
    """True when ``path`` is a plain file safe to offer to a browser."""
    name = path.name
    if name.startswith("."):
        return False
    if name in _BLOCKED_DOWNLOAD_NAMES:
        return False
    lowered = name.lower()
    if any(token in lowered for token in _BLOCKED_DOWNLOAD_SUBSTRINGS):
        return False
    try:
        return path.is_file() and not path.is_symlink()
    except OSError:
        return False


#: Largest file whose *contents* are scanned for secrets before being offered
#: for download. Beyond this, reading the file to check it would cost more than
#: the check is worth; the tool-call boundary is the primary defence.
_MAX_SCAN_BYTES = 4 * 1024 * 1024


def list_output_files(
    outputs_dir: Path, *, limit: int = 200, redactor: Redactor | None = None,
) -> list[Path]:
    """Every downloadable file under ``outputs_dir``, sorted, symlinks excluded.

    When ``redactor`` is supplied, files whose *contents* carry a configured
    secret are withheld as well. This is the second line: secrets are already
    stripped from tool arguments before a file can be written, but a filename
    check alone would let any other write path leak a key to a visitor.
    """
    root = Path(outputs_dir)
    if not root.is_dir():
        return []
    found: list[Path] = []
    try:
        real_root = root.resolve(strict=True)
    except OSError:
        return []
    for candidate in sorted(root.rglob("*")):
        if len(found) >= limit:
            break
        if candidate.is_symlink() or not is_downloadable(candidate):
            continue
        try:
            # A file reached through a symlinked *directory* must not escape.
            candidate.resolve(strict=True).relative_to(real_root)
        except (OSError, ValueError):
            continue
        if redactor is not None and _holds_secret(candidate, redactor):
            logger.warning(
                "withholding %s from download: it contains a configured secret",
                candidate.name,
            )
            continue
        found.append(candidate)
    return found


def _holds_secret(path: Path, redactor: Redactor) -> bool:
    """True when ``path`` looks like text and contains a configured secret."""
    if not redactor.literals:
        return False
    try:
        if path.stat().st_size > _MAX_SCAN_BYTES:
            return False
        body = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return True  # unreadable: fail closed rather than serve it blind
    return redactor.contains_secret(body)


def resolve_download(outputs_dir: Path, requested: str) -> Path:
    """Resolve ``requested`` to a real file inside ``outputs_dir``.

    Raises :class:`DownloadDenied` for anything else — traversal, absolute
    paths, symlinks out of the tree, non-files, and blocked names. This is the
    only function the UI may use to turn user input into a served path.
    """
    root = Path(outputs_dir)
    try:
        real_root = root.resolve(strict=True)
    except OSError as exc:
        raise DownloadDenied("this session has no outputs directory") from exc

    raw = str(requested or "").strip()
    if not raw:
        raise DownloadDenied("no file requested")
    if raw.startswith(("/", "\\")) or (os.name == "nt" and ":" in raw[:3]):
        raise DownloadDenied("absolute paths are not downloadable")

    candidate = (real_root / raw).resolve()
    try:
        candidate.relative_to(real_root)
    except ValueError as exc:
        raise DownloadDenied(
            "the requested path is outside this session's outputs directory",
        ) from exc
    if candidate.is_symlink() or not candidate.is_file():
        raise DownloadDenied("not a downloadable file")
    if not is_downloadable(candidate):
        raise DownloadDenied(f"{candidate.name!r} is not offered for download")
    return candidate


__all__ = [
    "HARD_DENIED_TOOLS",
    "REDACTED",
    "DownloadDenied",
    "RedactingLogFilter",
    "Redactor",
    "StreamRedactor",
    "demo_safe_tool_names",
    "demo_safe_tool_policy",
    "install_log_redaction",
    "is_downloadable",
    "list_output_files",
    "redact_deep",
    "resolve_download",
]
