"""read_text tool — plain-text / line-range reader for allowed local dirs or E2B sandbox.

This is the cheap host-local reader (formerly named ``read_file``): it reads a
file's text directly off the host (``plugins/skills/``, ``data/``,
``/tmp/agent-outputs/`` + an explicit workspace root), with an E2B sandbox
fallback for absolute paths, applies an optional 1-indexed line range and
truncates at 15 KB. No subprocess, no structured parsing — for office/PDF/xlsx
documents use the ``read_file`` tool (the sandbox-backed structured reader).

The path-authorization gate lives in :mod:`plugins.tools._path_auth`; this
module only does the reading. :func:`read_text_file` is the importable async
util; :data:`read_text` is the same function exposed as a registered tool.

References:
- DeerFlow: sandbox/tools.py read_file_tool()
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from frontier_agent.core.tool import tool
from plugins.tools._path_auth import (
    _ALLOWED_ABSOLUTE_PREFIXES,
    _authorized_local_path,
)
from plugins.tools._paths import virtual_to_physical

logger = logging.getLogger(__name__)

_MAX_CONTENT_BYTES = 15_000
# Document files written by write_file always land here (host-local); see
# the matching prefix in :mod:`plugins.tools._path_auth`.
_LOCAL_OUTPUT_DIR = "/tmp/agent-outputs"


async def read_text_file(path: str, start_line: int = 0, end_line: int = 0) -> str:
    """Read a file's plain text content from allowed directories or E2B sandbox.

    Local mode: reads from plugins/skills/, data/, /tmp/agent-outputs/.
    Sandbox mode: reads any file from the E2B sandbox.
    For skills, use paths like: plugins/skills/deep-research/SKILL.md
    For office/PDF/spreadsheet documents, use `read_file` instead (structured).

    Args:
        path: File path (relative for local, absolute for sandbox).
        start_line: Optional 1-indexed start line (0 = from beginning).
        end_line: Optional 1-indexed end line (0 = to end of file).

    Returns:
        File content as text, or error message.
    """
    if not path or not path.strip():
        return "Error: file path is required."
    path = virtual_to_physical(path.strip())

    def _finish(content: str) -> str:
        """Apply the line range, then cap the result at 15 KB."""
        if start_line > 0 or end_line > 0:
            lines = content.splitlines(keepends=True)
            s = max(0, start_line - 1) if start_line > 0 else 0
            e = end_line if end_line > 0 else len(lines)
            sliced = "".join(lines[s:e])
            if s > 0 or e < len(lines):
                content = f"[lines {s + 1}-{min(e, len(lines))} of {len(lines)}]\n" + sliced
            else:
                content = sliced
        if len(content) > _MAX_CONTENT_BYTES:
            content = content[:_MAX_CONTENT_BYTES] + "\n\n... (truncated at 15KB)"
        return content

    # ── Try host /tmp/agent-outputs/ first ──────────────────────
    # Check here BEFORE the sandbox to find persistent write_file outputs.
    if os.path.isabs(path) and not path.startswith(_LOCAL_OUTPUT_DIR):
        alt_resolved = Path(os.path.join(_LOCAL_OUTPUT_DIR, os.path.basename(path))).resolve()
        if alt_resolved.is_file():
            try:
                return _finish(alt_resolved.read_text(encoding="utf-8"))
            except Exception:
                pass  # Fall through to other strategies

    # ── Try E2B sandbox for absolute paths ────────────────────────────
    # Skip sandbox for paths that are known to live on the host (e.g. skills).
    _is_host_path = any(
        os.path.normpath(path).startswith(os.path.normpath(p))
        for p in _ALLOWED_ABSOLUTE_PREFIXES
    )
    if os.path.isabs(path) and not _is_host_path:
        try:
            from plugins.tools._sandbox import (
                aget_sandbox,
                arun_sandbox_cmd,
                sandbox_available,
                shell_quote,
            )
            if sandbox_available():
                sandbox = await aget_sandbox()
                result = await arun_sandbox_cmd(
                    sandbox,
                    f"cat -- {shell_quote(path)}",
                    timeout=15,
                )
                if result.exit_code != 0:
                    raise FileNotFoundError(result.stderr)
                return _finish(result.stdout)
        except Exception as e:
            logger.debug("Sandbox read failed for '%s': %s, trying local", path, e)

    # ── Local file read with security checks ──────────────────────────
    resolved, reason = _authorized_local_path(path)
    if resolved is None:
        return f"Access denied: {reason}"
    if not resolved.is_file():
        return f"File not found: {path}"

    try:
        return _finish(resolved.read_text(encoding="utf-8"))
    except UnicodeDecodeError:
        return f"Cannot read '{path}': not a text file"
    except Exception as e:
        logger.warning("read_text error for '%s': %s", path, e)
        return f"Error reading file: {e}"


# The util IS the tool — decorate it directly so there is one body + docstring.
read_text = tool(name="read_text")(read_text_file)
