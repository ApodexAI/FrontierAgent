from __future__ import annotations

import json
from pathlib import Path

import pytest
import task_selection


def make_task(root: Path, task_id: str, environment: str, instruction: str = "") -> Path:
    task = root / task_id
    (task / "environment").mkdir(parents=True)
    (task / "task.toml").write_text('schema_version = "1.1"\n')
    (task / "task.json").write_text(
        json.dumps({"task_id": task_id, "environment": environment})
    )
    (task / "instruction.md").write_text(instruction)
    (task / "environment" / "Dockerfile").write_text("FROM example/open\n")
    return task


def test_open_task_can_mention_orca_without_requiring_licensed_runtime(tmp_path):
    make_task(
        tmp_path,
        "task_098_orca_claisen_thermochemistry",
        "open",
        "Read the supplied ORCA output files; do not execute ORCA.",
    )

    selected = task_selection.select_tasks(tmp_path, track="open")

    assert [(task.task_id, task.environment) for task in selected] == [
        ("task_098_orca_claisen_thermochemistry", "open")
    ]


def test_open_track_excludes_declared_licensed_tasks(tmp_path):
    make_task(tmp_path, "task_011_open", "open")
    make_task(tmp_path, "task_199_orca", "licensed-orca")

    assert [task.task_id for task in task_selection.select_tasks(tmp_path, track="open")] == [
        "task_011_open"
    ]
    assert [task.task_id for task in task_selection.select_tasks(tmp_path, track="full")] == [
        "task_011_open",
        "task_199_orca",
    ]


def test_include_and_exclude_use_glob_semantics(tmp_path):
    for task_id in ("task_011_alpha", "task_012_beta", "task_199_orca"):
        make_task(tmp_path, task_id, "open")

    selected = task_selection.select_tasks(
        tmp_path,
        track="open",
        include=("task_0*",),
        exclude=("*_beta",),
    )

    assert [task.task_id for task in selected] == ["task_011_alpha"]


def test_selection_reads_solve_source_not_stale_stage(tmp_path):
    solve = tmp_path / "solve"
    stale_stage = tmp_path / "stage"
    make_task(solve, "task_011_open", "open")
    make_task(stale_stage, "task_199_orca", "licensed-orca")

    selected = task_selection.select_tasks(solve, track="open")

    assert [task.task_id for task in selected] == ["task_011_open"]


def test_invalid_environment_is_rejected(tmp_path):
    make_task(tmp_path, "task_011_bad", "orca-by-text-search")

    with pytest.raises(ValueError, match="invalid environment"):
        task_selection.select_tasks(tmp_path, track="full")


def test_task_environment_must_match_registry_commitment(tmp_path):
    make_task(tmp_path, "task_098_orca_claisen_thermochemistry", "open")
    registry = tmp_path / "source_registry.json"
    registry.write_text(
        json.dumps(
            {
                "tasks": [
                    {
                        "id": "task_098_orca_claisen_thermochemistry",
                        "image": "licensed-orca",
                    }
                ]
            }
        )
    )

    with pytest.raises(ValueError, match="task/registry environment mismatch"):
        task_selection.select_tasks(
            tmp_path,
            track="full",
            registry_path=registry,
        )
