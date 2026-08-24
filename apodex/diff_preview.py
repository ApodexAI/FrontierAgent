"""Compute a unified diff preview for a proposed write/edit tool call.

Lets the approval gate show *exactly what will change on disk* before the
edit is applied (apodex-style per-hunk preview), instead of only echoing
the tool's raw arguments.

Pure + stdlib (``difflib``) so it is fully unit-testable and never touches
the network or a sandbox.
"""

from __future__ import annotations

import difflib
import os


def _abspath(path: str, cwd: str) -> str:
    return path if os.path.isabs(path) else os.path.join(cwd, path)


def _read(path: str) -> str | None:
    """Read a text file. Returns ``""`` for a missing file, ``None`` for a
    binary file (NUL byte in the first 8 KB) so callers skip previewing it
    rather than rendering replacement-character garbage.
    """
    try:
        if not os.path.exists(path):
            return ""
        with open(path, "rb") as fb:
            head = fb.read(8192)
        if b"\x00" in head:
            return None
        with open(path, encoding="utf-8", errors="replace") as f:
            return f.read()
    except Exception:
        return ""


def proposed_change(name: str, args: dict, cwd: str) -> tuple[str, str, str] | None:
    """Return ``(path, old_content, new_content)`` for a write/edit call.

    ``None`` for tools that don't modify a file, or when the change can't be
    previewed (e.g. ``str_replace`` whose ``old_str`` isn't found).
    """
    path = args.get("path") or args.get("file_path")
    if not path:
        return None
    # Normalize so diff headers match what the real tools operate on
    # (read_file/file_editor call os.path.normpath before resolving).
    disp = os.path.normpath(str(path))
    abspath = os.path.normpath(_abspath(str(path), cwd))
    old = _read(abspath)
    if old is None:  # binary file — don't preview
        return None

    if name == "write_file":
        content = str(args.get("content", ""))
        new = (old + content) if args.get("append") else content
    elif name == "file_editor_create":
        new = str(args.get("content", ""))
    elif name == "file_editor_str_replace":
        old_str = str(args.get("old_str", ""))
        new_str = str(args.get("new_str", ""))
        # The tool requires old_str to occur EXACTLY once; only preview when
        # that holds (count 0 → not found, >1 → tool errors). Both unpreviewable.
        if not old_str or old.count(old_str) != 1:
            return None
        new = old.replace(old_str, new_str, 1)
    else:
        return None

    return (disp, old, new)


def unified_diff(name: str, args: dict, cwd: str, *, context: int = 3) -> str | None:
    """Return a unified-diff string for the proposed change, or ``None``.

    ``None`` when there is nothing to preview (non-edit tool, no net change,
    or an unpreviewable str_replace).
    """
    change = proposed_change(name, args, cwd)
    if change is None:
        return None
    path, old, new = change
    if old == new:
        return None
    is_new = not old
    # Plain path labels (this diff is for display, not ``git apply``), so an
    # absolute path doesn't render as ``b//abs/path``.
    diff = difflib.unified_diff(
        old.splitlines(),
        new.splitlines(),
        fromfile=("/dev/null" if is_new else path),
        tofile=path,
        n=context,
        lineterm="",
    )
    return "\n".join(diff)


def change_stats(name: str, args: dict, cwd: str) -> tuple[int, int] | None:
    """Return ``(added_lines, removed_lines)`` for the proposed change."""
    change = proposed_change(name, args, cwd)
    if change is None:
        return None
    _, old, new = change
    added = removed = 0
    for line in difflib.ndiff(old.splitlines(), new.splitlines()):
        if line.startswith("+ "):
            added += 1
        elif line.startswith("- "):
            removed += 1
    return (added, removed)
