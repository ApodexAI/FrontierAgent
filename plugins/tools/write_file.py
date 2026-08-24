"""File writing tool — writes content to files in E2B sandbox or local fallback.

References:
- DeerFlow: sandbox/tools.py write_file_tool()
"""

from __future__ import annotations

import logging
import os

from frontier_agent.core.tool import tool
from plugins.tools._deliverable_policy import output_write_error
from plugins.tools._path_auth import _authorized_local_path
from plugins.tools._sandbox import (
    aget_sandbox,
    arun_sandbox_cmd,
    asandbox_write_file,
    resolve_runtime_path,
    resolve_sandbox_mode,
    sandbox_available,
)

logger = logging.getLogger(__name__)

_MAX_CONTENT_BYTES = 1_048_576  # 1MB
_LOCAL_OUTPUT_DIR = "/tmp/agent-outputs"


@tool
async def write_file(path: str, content: str, append: bool = False) -> str:
    """Write plain-text content or source code to a file.

    In sandbox mode, writes to the E2B sandbox filesystem.
    Without sandbox, writes to /tmp/agent-outputs/ only.
    This tool only writes bytes; it does not execute generated scripts or load
    runtime packages. Use `bash` to run a generated JavaScript file. Use
    `create_file`, not this tool, for .docx/.xlsx/.pptx deliverables.

    Args:
        path: Absolute file path (e.g., /tmp/output.html, /root/chart.png).
        content: The text content to write.
        append: If True, append to existing file instead of overwriting.

    Returns:
        Confirmation message with the file path.
    """
    if not path or not path.strip():
        return "Error: file path is required."

    deliverable_error = output_write_error(path)
    if deliverable_error:
        return f"Error: {deliverable_error}"
    path = resolve_runtime_path(path)

    if len(content) > _MAX_CONTENT_BYTES:
        return f"Error: content exceeds maximum size of {_MAX_CONTENT_BYTES // 1024}KB."

    local_path, _reason = _authorized_local_path(path, write_access=True)

    # Prefer local writes for allowed repo paths
    if local_path is not None:
        try:
            local_path.parent.mkdir(parents=True, exist_ok=True)
            mode = "a" if append else "w"
            with open(local_path, mode, encoding="utf-8") as f:
                f.write(content)
            return f"File written: {local_path} ({len(content)} bytes)"
        except Exception as e:
            return f"Error writing file: {e}"

    if sandbox_available():
        try:
            sandbox = await aget_sandbox()
            parent = os.path.dirname(path)
            if parent:
                await arun_sandbox_cmd(
                    sandbox, f"mkdir -p {parent}", timeout=10,
                )

            mode = "a" if append else "w"
            ok, err = await asandbox_write_file(
                sandbox, path, content, mode=mode,
            )
            if not ok:
                raise RuntimeError(err)

            return f"File written: {path} ({len(content)} bytes)"
        except Exception as e:
            # Container mode: /outputs & /workspace are REAL mounts inside the
            # isolated task container, so a write failure is a genuine error —
            # NOT a cue to silently redirect the deliverable to /tmp (which
            # would make it vanish from /outputs and emit no file_delta). Fail
            # loudly so the misconfiguration surfaces.
            if resolve_sandbox_mode() == "container":
                return (
                    f"Error writing file {path!r}: {e}. "
                    "(container mode: writes must target the mounted "
                    "/outputs or /workspace; no /tmp fallback)"
                )
            logger.warning("Sandbox write_file failed, trying local: %s", e)

    # Local fallback — restricted to /tmp/agent-outputs/
    if not path.startswith(_LOCAL_OUTPUT_DIR):
        path = os.path.join(_LOCAL_OUTPUT_DIR, os.path.basename(path))

    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        mode = "a" if append else "w"
        with open(path, mode, encoding="utf-8") as f:
            f.write(content)
        return f"File written: {path} ({len(content)} bytes)"
    except Exception as e:
        return f"Error writing file: {e}"
