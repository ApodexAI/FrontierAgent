"""Controlled document download tool."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

from frontier_agent.core.tool import tool
from plugins.tools._sandbox import (
    aget_sandbox,
    arun_sandbox_cmd,
    resolve_mount_dirs,
    shell_quote,
)

logger = logging.getLogger(__name__)

_MIB = 1024 * 1024
_DEFAULT_FILE_LIMIT = 256 * _MIB
_DEFAULT_TASK_LIMIT = 512 * _MIB
_DEFAULT_DISK_RESERVE = 128 * _MIB
# The in-sandbox budget and the outer wall clock are derived from one constant
# so they cannot drift into outer < inner, where the caller's timeout would
# pre-empt the runner's structured JSON error.
_TOTAL_TIMEOUT = 600
_TIMEOUT = _TOTAL_TIMEOUT + 60
# Piped through stdin rather than embedded in argv: keeps the command short
# (execve's 128KB single-arg limit) and out of shell traces. Same convention
# as ``read_file``'s parser bundle.
_RUNNER_SOURCE = Path(__file__).with_name("_download_runner.py").read_text(encoding="utf-8")


def _download_dir() -> str:
    """Controlled download directory inside the task workspace.

    Derived from ``resolve_mount_dirs()`` rather than a literal ``/workspace``
    so deployments that relocate the mount (``FRONTIER_AGENT_WORKSPACE_DIR``, or a
    Harbor-provisioned working dir attached via ``CurrentSandbox``) keep the
    downloads inside the directory the platform actually mounts — otherwise the
    files land outside it and never surface as workspace files.
    """
    explicit = (os.getenv("FRONTIER_AGENT_DOWNLOAD_DIR") or "").strip()
    if explicit:
        return explicit
    workspace, _outputs, _inputs = resolve_mount_dirs()
    return str(Path(workspace) / "downloads")


def _positive_env_bytes(name: str, default: int) -> int:
    raw = (os.getenv(name) or "").strip()
    if raw:
        try:
            value = int(raw)
            if value > 0:
                return value
        except ValueError:
            pass
    return default


@tool
async def download_file(url: str, path: str = "") -> str:
    """Download a bounded document or small archive into the task workspace.

    Use this for PDFs, Word/Excel/PowerPoint files, text/data documents,
    images, and small ZIP/TAR/GZIP archives. The tool validates and pins every
    redirect hop, rejects private/internal destinations, preflights declared
    size when available, enforces the actual streamed byte count against
    per-file and per-task budgets, and atomically publishes the completed
    file. It does not extract archives.

    Args:
        url: Public HTTP(S) URL of the file to download.
        path: Optional destination filename. Directory components are ignored;
            downloads always land in the workspace ``downloads`` directory.

    Returns:
        JSON metadata including path, actual size, SHA-256, content type, and
        final URL, or a bounded-download error.
    """
    if not url or not url.strip():
        return json.dumps({"status": "error", "error": "url is required"})

    file_limit = _positive_env_bytes(
        "DOWNLOAD_FILE_MAX_BYTES", _DEFAULT_FILE_LIMIT,
    )
    task_limit = _positive_env_bytes(
        "DOWNLOAD_TASK_MAX_BYTES", _DEFAULT_TASK_LIMIT,
    )
    reserve_bytes = _positive_env_bytes(
        "DOWNLOAD_DISK_RESERVE_BYTES", _DEFAULT_DISK_RESERVE,
    )

    try:
        sandbox = await aget_sandbox()
    except RuntimeError as exc:
        return json.dumps({"status": "error", "error": str(exc)})

    command = (
        f"python3 - "
        f"{shell_quote(url.strip())} "
        f"--path {shell_quote(path.strip())} "
        f"--dir {shell_quote(_download_dir())} "
        f"--max-bytes {file_limit} "
        f"--task-max-bytes {task_limit} "
        f"--reserve-bytes {reserve_bytes} "
        f"--total-timeout {_TOTAL_TIMEOUT}"
    )
    try:
        result = await arun_sandbox_cmd(
            sandbox,
            command,
            timeout=_TIMEOUT,
            input=_RUNNER_SOURCE,
            allow_net=True,
        )
    except TimeoutError:
        return json.dumps({
            "status": "error",
            "error": f"download timed out after {_TIMEOUT}s",
        })
    except Exception as exc:
        logger.warning("download_file failed: %s", exc)
        return json.dumps({
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
        })

    output = (result.stdout or "").strip()
    if output:
        return output
    return json.dumps({
        "status": "error",
        "error": (result.stderr or "download runner returned no output").strip(),
    })


__all__ = ["download_file"]
