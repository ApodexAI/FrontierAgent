#!/usr/bin/env python3
"""Resolve the exact FrontierChallenge task set for an evaluation run."""

from __future__ import annotations

import argparse
import fnmatch
import json
from dataclasses import dataclass
from pathlib import Path

VALID_ENVIRONMENTS = {"open", "licensed-orca"}


@dataclass(frozen=True)
class SelectedTask:
    task_id: str
    environment: str
    path: Path


def select_tasks(
    tasks_root: Path,
    *,
    track: str,
    include: tuple[str, ...] = (),
    exclude: tuple[str, ...] = (),
    registry_path: Path | None = None,
) -> list[SelectedTask]:
    """Select from verified solve tasks, never from persistent staging."""
    if track not in {"open", "full"}:
        raise ValueError(f"unsupported track: {track}")

    registry_environments: dict[str, str] | None = None
    if registry_path is not None:
        registry = json.loads(registry_path.read_text(encoding="utf-8"))
        registry_environments = {}
        for record in registry.get("tasks", []):
            task_id = record.get("id")
            environment = record.get("image")
            if not isinstance(task_id, str) or environment not in VALID_ENVIRONMENTS:
                raise ValueError(f"invalid registry task record: {record!r}")
            if task_id in registry_environments:
                raise ValueError(f"duplicate registry task ID: {task_id}")
            registry_environments[task_id] = environment

    selected: list[SelectedTask] = []
    for task_path in sorted(tasks_root.iterdir()):
        if not task_path.is_dir() or not (task_path / "task.toml").is_file():
            continue
        metadata_path = task_path / "task.json"
        if not metadata_path.is_file():
            raise ValueError(f"task metadata missing: {metadata_path}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        task_id = metadata.get("task_id")
        environment = metadata.get("environment")
        if task_id != task_path.name:
            raise ValueError(f"task ID/path mismatch: {task_path.name} != {task_id!r}")
        if environment not in VALID_ENVIRONMENTS:
            raise ValueError(f"invalid environment for {task_id}: {environment!r}")
        if registry_environments is not None:
            registered = registry_environments.get(task_id)
            if registered != environment:
                raise ValueError(
                    f"task/registry environment mismatch for {task_id}: "
                    f"{environment!r} != {registered!r}"
                )
        if track == "open" and environment != "open":
            continue
        if include and not any(fnmatch.fnmatchcase(task_id, pattern) for pattern in include):
            continue
        if any(fnmatch.fnmatchcase(task_id, pattern) for pattern in exclude):
            continue
        selected.append(SelectedTask(task_id, environment, task_path.resolve()))
    return selected


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks-root", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--track", choices=("open", "full"), required=True)
    parser.add_argument("--include", action="append", default=[])
    parser.add_argument("--exclude", action="append", default=[])
    args = parser.parse_args()

    try:
        selected = select_tasks(
            args.tasks_root,
            track=args.track,
            include=tuple(args.include),
            exclude=tuple(args.exclude),
            registry_path=args.registry,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    if not selected:
        parser.error("task selection is empty")
    for task in selected:
        path = str(task.path)
        if any(character in path for character in ("\t", "\n", "\r")):
            parser.error(f"task path contains a control character: {task.path}")
        print(f"{task.task_id}\t{task.environment}\t{path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
