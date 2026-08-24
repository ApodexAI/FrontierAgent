"""Deterministic GDPval deliverable validation (no agentic pairwise judge)."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

_OFFICE_ZIPS = {".docx", ".xlsx", ".pptx"}


def _valid_file(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    if path.suffix.lower() in _OFFICE_ZIPS:
        try:
            with zipfile.ZipFile(path) as archive:
                return bool(archive.namelist())
        except zipfile.BadZipFile:
            return False
    if path.suffix.lower() == ".pdf":
        return path.read_bytes().startswith(b"%PDF-")
    return True


def score_gdpval_outputs(outputs_dir: Path, target: str) -> tuple[int, float]:
    """Validate non-empty, structurally readable deliverables deterministically."""
    files = sorted(path for path in outputs_dir.rglob("*") if path.is_file()) if outputs_dir.is_dir() else []
    if not files or not all(_valid_file(path) for path in files):
        return 0, 0.0
    try:
        raw = json.loads(target) if target else {}
    except json.JSONDecodeError:
        raw = {}
    references = raw.get("reference_files", []) if isinstance(raw, dict) else []
    expected_extensions = {Path(str(path)).suffix.lower() for path in references if Path(str(path)).suffix}
    actual_extensions = {path.suffix.lower() for path in files}
    if expected_extensions and not expected_extensions.intersection(actual_extensions):
        return 0, 0.0
    return 1, 1.0
