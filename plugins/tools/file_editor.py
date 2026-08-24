"""File editor tools — view, create, and edit files in E2B sandbox."""

from __future__ import annotations

import logging

from frontier_agent.core.tool import tool
from plugins.tools._deliverable_policy import output_write_error
from plugins.tools._path_auth import _authorized_local_path
from plugins.tools._sandbox import (
    aget_sandbox,
    arun_sandbox_cmd,
    asandbox_write_file,
)

logger = logging.getLogger(__name__)

_MAX_OUTPUT_CHARS = 16_000


def _truncate(text: str, max_chars: int = _MAX_OUTPUT_CHARS) -> str:
    """Truncate output if too long."""
    if len(text) > max_chars:
        return text[:max_chars] + "\n\n[... output truncated]"
    return text


@tool
async def file_editor_view(path: str, view_range: str = "") -> str:
    """View file contents or list directory in E2B sandbox.

    Args:
        path: Absolute path to file or directory.
        view_range: Optional line range like "1-50" to view specific lines.

    Returns:
        File contents with line numbers, or directory listing.
    """
    if not path or not path.strip():
        return "Error: path is required."

    local_path, _reason = _authorized_local_path(path)
    if local_path is not None:
        from plugins.tools.ignore_rules import discover_repo_root, should_ignore_path
        if local_path.is_dir():
            repo_root = discover_repo_root(local_path)
            entries: list[str] = []
            for entry in sorted(local_path.rglob("*")):
                if should_ignore_path(entry, repo_root):
                    continue
                try:
                    rel = entry.relative_to(local_path)
                except ValueError:
                    rel = entry
                if len(rel.parts) > 2:
                    continue
                entries.append(str(rel))
                if len(entries) >= 100:
                    break
            return "\n".join(entries) if entries else f"(empty directory: {path})"
        if not local_path.is_file():
            return f"Error: File not found: {path}"
        try:
            lines = local_path.read_text(encoding="utf-8").splitlines()
        except Exception as e:
            return f"Error viewing {path}: {e}"
        if view_range and "-" in view_range:
            start_s, end_s = view_range.split("-", 1)
            start = max(int(start_s.strip() or "1"), 1)
            end = max(int(end_s.strip() or str(start)), start)
        else:
            start, end = 1, len(lines)
        rendered = "\n".join(f"{idx}: {line}" for idx, line in enumerate(lines[start - 1:end], start=start))
        return _truncate(rendered or "(empty file)")

    try:
        sandbox = await aget_sandbox()
    except RuntimeError as e:
        return f"Error: {e}"

    try:
        check = await arun_sandbox_cmd(
            sandbox,
            f"test -d {path} && echo DIR || echo FILE", timeout=10,
        )
        is_dir = "DIR" in check.stdout

        if is_dir:
            result = await arun_sandbox_cmd(
                sandbox,
                f"find {path} -maxdepth 2 -not -path '*/.git/*' "
                f"-not -path '*/node_modules/*' -not -path '*/__pycache__/*' "
                f"| head -100 | sort",
                timeout=15,
            )
            return _truncate(result.stdout or f"(empty directory: {path})")

        # File view
        if view_range:
            parts = view_range.split("-")
            if len(parts) == 2:
                start, end = parts[0].strip(), parts[1].strip()
                cmd = f"sed -n '{start},{end}p' {path} | cat -n"
            else:
                cmd = f"cat -n {path}"
        else:
            cmd = f"cat -n {path}"

        result = await arun_sandbox_cmd(sandbox, cmd, timeout=15)

        if result.exit_code != 0:
            return f"Error: {result.stderr or 'File not found'}"

        return _truncate(result.stdout or "(empty file)")

    except Exception as e:
        return f"Error viewing {path}: {e}"


@tool
async def file_editor_create(path: str, content: str) -> str:
    """Create a new file in E2B sandbox with the given content.

    Use this to create new files. For modifying existing files, use file_editor_str_replace instead.

    Args:
        path: Absolute path for the new file.
        content: The file content to write.

    Returns:
        Confirmation message.
    """
    if not path or not path.strip():
        return "Error: path is required."

    deliverable_error = output_write_error(path)
    if deliverable_error:
        return f"Error: {deliverable_error}"
    local_path, reason = _authorized_local_path(path, write_access=True)
    if local_path is not None:
        try:
            local_path.parent.mkdir(parents=True, exist_ok=True)
            local_path.write_text(content, encoding="utf-8")
            lines = content.count("\n") + (1 if content else 0)
            return f"File created: {path} ({lines} lines, {len(content)} bytes)"
        except Exception as e:
            return f"Error creating {path}: {e}"

    try:
        sandbox = await aget_sandbox()
    except RuntimeError as e:
        return f"Error: {e}" if "allowed" not in reason.lower() else f"Access denied: {reason}"

    try:
        import os
        parent = os.path.dirname(path)
        if parent:
            await arun_sandbox_cmd(
                sandbox, f"mkdir -p {parent}", timeout=10,
            )

        # Write via python in sandbox (avoids files.write permission issues)
        ok, err = await asandbox_write_file(sandbox, path, content)
        if not ok:
            return f"Error creating {path}: {err}"

        result = await arun_sandbox_cmd(
            sandbox, f"wc -l < {path}", timeout=5,
        )
        lines = result.stdout.strip() if result.stdout else "?"

        return f"File created: {path} ({lines} lines, {len(content)} bytes)"

    except Exception as e:
        return f"Error creating {path}: {e}"


@tool
async def file_editor_str_replace(path: str, old_str: str, new_str: str) -> str:
    """Replace a string occurrence in a file in E2B sandbox.

    The old_str must appear exactly once in the file (for safety).
    Use file_editor_view first to see the file contents.

    Args:
        path: Absolute path to the file.
        old_str: The exact string to find and replace.
        new_str: The replacement string.

    Returns:
        Confirmation with a diff summary.
    """
    if not path or not old_str:
        return "Error: path and old_str are required."

    deliverable_error = output_write_error(path)
    if deliverable_error:
        return f"Error: {deliverable_error}"
    local_path, reason = _authorized_local_path(path, write_access=True)
    if local_path is not None:
        if not local_path.is_file():
            return f"Error: Cannot read {path}: file not found"
        try:
            content = local_path.read_text(encoding="utf-8")
        except Exception as e:
            return f"Error editing {path}: {e}"

        count = content.count(old_str)
        if count == 0:
            return (
                f"Error: old_str not found in {path}. "
                "Make sure the string matches exactly (including whitespace and indentation)."
            )
        if count > 1:
            return (
                f"Error: old_str found {count} times in {path}. "
                "Provide a more specific string that appears exactly once."
            )
        try:
            local_path.write_text(content.replace(old_str, new_str, 1), encoding="utf-8")
        except Exception as e:
            return f"Error editing {path}: {e}"

        old_lines = old_str.count("\n") + 1
        new_lines = new_str.count("\n") + 1
        return f"Replaced in {path}: {old_lines} line(s) → {new_lines} line(s)"

    try:
        sandbox = await aget_sandbox()
    except RuntimeError as e:
        return f"Error: {e}" if "allowed" not in reason.lower() else f"Access denied: {reason}"

    try:
        # Read current content via cat (more reliable than files.read for all paths)
        result = await arun_sandbox_cmd(sandbox, f"cat {path}", timeout=10)
        if result.exit_code != 0:
            return f"Error: Cannot read {path}: {result.stderr or 'file not found'}"
        content = result.stdout

        count = content.count(old_str)
        if count == 0:
            return (
                f"Error: old_str not found in {path}. "
                "Make sure the string matches exactly (including whitespace and indentation)."
            )
        if count > 1:
            return (
                f"Error: old_str found {count} times in {path}. "
                "Provide a more specific string that appears exactly once."
            )

        # Replace and write back via python in sandbox (avoids files.write permission issues)
        new_content = content.replace(old_str, new_str, 1)
        ok, err = await asandbox_write_file(sandbox, path, new_content)
        if not ok:
            return f"Error writing {path}: {err}"

        # Show diff stats
        old_lines = old_str.count("\n") + 1
        new_lines = new_str.count("\n") + 1
        return f"Replaced in {path}: {old_lines} line(s) → {new_lines} line(s)"

    except Exception as e:
        return f"Error editing {path}: {e}"
