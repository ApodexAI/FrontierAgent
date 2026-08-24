"""Read-before-edit guard (Claude-Code parity).

The agent must Read a file before editing or overwriting it, and the file must
not have changed on disk since that read. This prevents two real failure modes:
(a) blindly rewriting a file it never saw, and (b) silently clobbering a
concurrent edit by the user or a linter.

State is a single session-lived map ``realpath -> mtime-at-read``. Reads are
recorded centrally (by the observer, for every successful read tool), and edits
are checked at the approval gate — so no file tool needs modifying.
"""

from __future__ import annotations

import os

_READ_AT: dict[str, float] = {}  # realpath -> file mtime when last read


def _abs(path: str, cwd: str) -> str:
    return os.path.realpath(path if os.path.isabs(path) else os.path.join(cwd, path))


def record_read(path: str, cwd: str) -> None:
    """Remember that ``path`` was read (with its current mtime)."""
    try:
        ap = _abs(path, cwd)
        if os.path.isfile(ap):
            _READ_AT[ap] = os.path.getmtime(ap)
    except Exception:
        pass


def check_can_edit(path: str, cwd: str) -> str | None:
    """Return an error message if editing ``path`` should be blocked, else None.

    A brand-new file (doesn't exist yet) is always editable. An existing file
    must have been read this session and not changed since.
    """
    try:
        ap = _abs(path, cwd)
        if not os.path.exists(ap):
            return None  # creating a new file is fine
        if ap not in _READ_AT:
            return (
                f"'{path}' has not been read this session. Read it first "
                "(read_file / file_editor_view) before editing it, so you don't "
                "overwrite content you haven't seen."
            )
        if os.path.getmtime(ap) > _READ_AT[ap] + 1e-6:
            return (
                f"'{path}' has changed on disk since you read it (a user or tool "
                "may have edited it). Read it again before editing."
            )
    except Exception:
        return None  # never block on an unexpected error
    return None


def clear() -> None:
    """Reset on /clear or a cwd change."""
    _READ_AT.clear()
