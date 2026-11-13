"""Snapshot explicit inputs before a task becomes visible to the scheduler."""

from __future__ import annotations

import shutil
from pathlib import Path

_EXCLUDED = {".git", ".venv", ".apodex", "node_modules", "__pycache__", ".pytest_cache"}


def snapshot_inputs(
    sources: list[Path], destination: Path, *, max_bytes: int = 256 * 1024 * 1024
) -> None:
    """Copy regular files only, with bounded size; never follow input symlinks."""
    names = [source.name for source in sources]
    if len(set(names)) != len(names):
        raise ValueError("Input names must be unique")
    files: list[tuple[Path, Path]] = []
    total = 0
    for source in sources:
        if source.is_symlink() or not source.exists():
            raise ValueError(f"Input must exist and cannot be a symlink: {source}")
        candidates = source.rglob("*") if source.is_dir() else [source]
        for path in candidates:
            relative = path.relative_to(source) if source.is_dir() else Path()
            if any(part in _EXCLUDED for part in relative.parts):
                continue
            if path.is_symlink():
                raise ValueError(f"Input contains a symlink: {path}")
            if path.is_dir():
                continue
            if not path.is_file():
                raise ValueError(f"Input contains a non-regular file: {path}")
            total += path.stat().st_size
            if total > max_bytes:
                raise ValueError("Input snapshot exceeds 256 MiB; attach a smaller selection")
            files.append((path, destination / source.name / relative))
    try:
        destination.mkdir(parents=True, exist_ok=True)
        for source, target in files:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target, follow_symlinks=False)
            if target.is_symlink():
                raise ValueError(f"Input changed to a symlink during snapshot: {source}")
            target.chmod(0o644)
    except BaseException:
        shutil.rmtree(destination, ignore_errors=True)
        raise
