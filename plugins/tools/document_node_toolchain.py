"""Runtime discovery and prompt guidance for the baked document Node toolchain."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from plugins.tools._sandbox import is_e2b_available

_DOCUMENT_PACKAGES = ("docx", "pptxgenjs")
_BWRAP_VISIBLE_ROOTS = (
    Path("/usr"),
    Path("/bin"),
    Path("/lib"),
    Path("/lib64"),
    Path("/etc"),
)


@dataclass(frozen=True)
class DocumentNodeToolchain:
    """Versions proven available to the active model-authored command backend."""

    node_version: str
    docx_version: str
    pptxgenjs_version: str


def _is_within(path: Path, roots: tuple[Path, ...]) -> bool:
    resolved = path.resolve()
    for root in roots:
        try:
            resolved.relative_to(root)
            return True
        except ValueError:
            continue
    return False


def _package_version(package: str, roots: tuple[Path, ...]) -> tuple[str, Path] | None:
    for root in roots:
        manifest = root / package / "package.json"
        if not manifest.is_file():
            continue
        try:
            data = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, TypeError):
            continue
        version = str(data.get("version") or "").strip()
        if version:
            return version, manifest
    return None


def discover_document_node_toolchain(
    *,
    sandbox_mode: str,
) -> DocumentNodeToolchain | None:
    """Return the document Node stack visible to model-authored commands.

    ``container`` executes in the current image. ``bwrap`` (and ``auto``
    without E2B) can see only paths mounted into the jail, so both the Node
    binary and module roots must live under its read-only system mounts.
    Explicit/auto E2B executes off-host and must never inherit a capability
    merely because the serving process itself has it installed.
    """
    mode = (sandbox_mode or "auto").strip().lower()
    if mode not in {"container", "native", "bwrap"} and is_e2b_available():
        return None

    node_path = os.environ.get("NODE_PATH", "").strip()
    path_env = os.environ.get("PATH", "").strip()
    if not node_path or not path_env:
        return None

    roots = tuple(
        Path(raw.strip()).expanduser()
        for raw in node_path.split(os.pathsep)
        if raw.strip()
    )
    node = shutil.which("node", path=path_env)
    if not roots or not node:
        return None

    packages = {
        package: _package_version(package, roots)
        for package in _DOCUMENT_PACKAGES
    }
    if any(value is None for value in packages.values()):
        return None

    if mode not in {"container", "native"}:
        visible_paths = [
            Path(node),
            *(value[1] for value in packages.values() if value is not None),
        ]
        if not all(_is_within(path, _BWRAP_VISIBLE_ROOTS) for path in visible_paths):
            return None

    try:
        result = subprocess.run(
            [node, "--version"],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
            env={"PATH": path_env, "NODE_PATH": node_path},
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    node_version = result.stdout.strip().removeprefix("v")
    if not node_version:
        return None

    docx = packages["docx"]
    pptxgenjs = packages["pptxgenjs"]
    assert docx is not None and pptxgenjs is not None
    return DocumentNodeToolchain(
        node_version=node_version,
        docx_version=docx[0],
        pptxgenjs_version=pptxgenjs[0],
    )


def render_document_node_toolchain_note(
    *,
    sandbox_mode: str,
    tool_names: list[str] | tuple[str, ...],
    audience: str = "agent",
) -> str:
    """Render a truthful prompt note for agents that can execute ``bash``."""
    names = {str(name) for name in tool_names}
    if "bash" not in names:
        return ""
    capability = discover_document_node_toolchain(sandbox_mode=sandbox_mode)
    if capability is None:
        return ""

    if "create_file" in names:
        fallback = (
            "Use this fallback order: (1) `create_file`; (2) Python libraries "
            "such as `python-docx` or `python-pptx` only after `create_file` "
            "explicitly reports the required operation unsupported; (3) these "
            "Node packages only when Python still cannot cover the operation, "
            "or when the user requires accurate preservation of an existing "
            "template that the Python path would not preserve."
        )
    else:
        fallback = (
            "`create_file` is not available in this tool set. Use Python "
            "libraries first; use these Node packages only when Python cannot "
            "cover the required operation, or when the user requires accurate "
            "preservation of an existing template that Python would not "
            "preserve."
        )
    scope = "sub-agent runtime" if audience == "coordinator" else "runtime-discovered"
    return (
        f"\n\nDOCUMENT TOOLCHAIN ({scope}): `bash` has Node.js "
        f"{capability.node_version}, `docx@{capability.docx_version}`, and "
        f"`pptxgenjs@{capability.pptxgenjs_version}`. `NODE_PATH` is already "
        "configured: load them as `require('docx')` and "
        "`require('pptxgenjs')`; do not hard-code `/usr/lib/node_modules`. "
        f"{fallback}"
    )
