"""Virtual path mapping — hide host filesystem paths from agents."""

from __future__ import annotations

import os
from pathlib import Path

# ── Virtual → Physical mapping ──────────────────────────────────────────

# Virtual root that agents see
VIRTUAL_PREFIX = "/mnt/agent"

# Pre-rename virtual root: old sessions/traces still reference it, so reads
# keep resolving. Never emitted in new output.
LEGACY_VIRTUAL_PREFIX = "/mnt/frontier_agent"

# Virtual path segments
VIRTUAL_WORKSPACE = f"{VIRTUAL_PREFIX}/workspace"
VIRTUAL_OUTPUTS = f"{VIRTUAL_PREFIX}/outputs"
VIRTUAL_SKILLS = f"{VIRTUAL_PREFIX}/skills"
VIRTUAL_DATA = f"{VIRTUAL_PREFIX}/data"

# Physical roots (resolved at runtime)
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)


def _get_physical_roots() -> dict[str, str]:
    """Map virtual prefixes to physical directories."""
    return {
        VIRTUAL_WORKSPACE: os.path.join(_PROJECT_ROOT, "data", "workspace"),
        VIRTUAL_OUTPUTS: "/tmp/agent-outputs",
        VIRTUAL_SKILLS: os.path.join(_PROJECT_ROOT, "plugins", "skills"),
        VIRTUAL_DATA: os.path.join(_PROJECT_ROOT, "data"),
    }


def virtual_to_physical(virtual_path: str) -> str:
    """Translate a virtual path to physical path.

    Args:
        virtual_path: Path starting with /mnt/agent/...

    Returns:
        Physical path on host filesystem.
        If not a virtual path, returns as-is.
    """
    if virtual_path.startswith(LEGACY_VIRTUAL_PREFIX):
        virtual_path = VIRTUAL_PREFIX + virtual_path[len(LEGACY_VIRTUAL_PREFIX):]
    if not virtual_path.startswith(VIRTUAL_PREFIX):
        return virtual_path

    for v_prefix, p_prefix in _get_physical_roots().items():
        if virtual_path.startswith(v_prefix):
            relative = virtual_path[len(v_prefix):].lstrip("/")
            return os.path.join(p_prefix, relative)

    return virtual_path


def physical_to_virtual(physical_path: str) -> str:
    """Translate a physical path back to virtual path.

    Args:
        physical_path: Actual host filesystem path.

    Returns:
        Virtual path if it maps to a known root, otherwise as-is.
    """
    resolved = str(Path(physical_path).resolve()) if physical_path else physical_path

    for v_prefix, p_prefix in _get_physical_roots().items():
        p_resolved = str(Path(p_prefix).resolve()) if os.path.exists(p_prefix) else p_prefix
        if resolved.startswith(p_resolved):
            relative = resolved[len(p_resolved):].lstrip("/")
            return f"{v_prefix}/{relative}" if relative else v_prefix

    return physical_path


def mask_paths_in_output(output: str) -> str:
    """Replace any physical paths in tool output with virtual equivalents.

    Scans for known physical root patterns and replaces them.
    This prevents the LLM from seeing host filesystem structure.

    Args:
        output: Raw tool output string.

    Returns:
        Output with physical paths replaced by virtual paths.
    """
    if not output:
        return output

    for v_prefix, p_prefix in _get_physical_roots().items():
        if p_prefix in output:
            output = output.replace(p_prefix, v_prefix)

    # Also mask the project root itself
    if _PROJECT_ROOT in output:
        output = output.replace(_PROJECT_ROOT, VIRTUAL_PREFIX)

    return output
