"""Safely stage an APEX world snapshot and its per-task overlay."""

from __future__ import annotations

import shutil
import zipfile
from pathlib import Path


def _safe_extract(archive: zipfile.ZipFile, destination: Path) -> None:
    root = destination.resolve()
    for member in archive.infolist():
        target = (root / member.filename).resolve()
        if target != root and root not in target.parents:
            raise ValueError(f"unsafe zip member path: {member.filename}")
    archive.extractall(root)


def populate(
    world_id: str,
    task_id: str,
    dataset_root: str | Path,
    worktree: str | Path,
) -> tuple[str, str]:
    """Populate one world and return its filesystem and app-state paths."""
    root = Path(dataset_root)
    work = Path(worktree)
    work.mkdir(parents=True, exist_ok=True)
    world_zip = root / "world_files_zipped" / f"{world_id}.zip"
    if world_zip.is_file():
        with zipfile.ZipFile(world_zip) as archive:
            _safe_extract(archive, work)
    task_root = root / "task_files" / task_id
    for subdir in ("filesystem", ".apps_data"):
        source = task_root / subdir
        if source.is_dir():
            shutil.copytree(source, work / subdir, dirs_exist_ok=True)
    filesystem = work / "filesystem"
    apps_data = work / ".apps_data"
    filesystem.mkdir(parents=True, exist_ok=True)
    apps_data.mkdir(parents=True, exist_ok=True)
    return str(filesystem), str(apps_data)
