"""Exercise the shell runner with real archives and stubbed Docker/Harbor APIs.

This verifies orchestration, not container isolation or real model execution.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import reference_archive

ROOT = Path(__file__).parents[1]
OPEN = "task_098_orca_claisen_thermochemistry"
LICENSED = "task_199_orca"


@pytest.fixture
def runtime(tmp_path):
    runtime = tmp_path / "runtime"
    shutil.copytree(ROOT / "scripts", runtime / "scripts")
    solve, reference = tmp_path / "solve", tmp_path / "reference"
    records = []
    for name, environment in [(OPEN, "open"), (LICENSED, "licensed-orca")]:
        task = solve / "tasks" / name
        (task / "environment/data").mkdir(parents=True)
        (task / "task.toml").write_text('[agent]\ntimeout_sec = 600\n')
        (task / "instruction.md").write_text("Analyze supplied ORCA output.")
        (task / "task.json").write_text(json.dumps({
            "task_id": name, "environment": environment, "source_task_sha256": "fixture",
        }, indent=2))
        (task / "environment/Dockerfile").write_text("FROM fixture/open\n")
        blob = tmp_path / f"{name}.blob"
        blob.write_text("precomputed output")
        (task / "environment/data/output.txt").symlink_to(os.path.relpath(blob, task / "environment/data"))
        verifier = reference / "tasks" / name
        (verifier / "tests").mkdir(parents=True)
        (verifier / "tests/test.sh").write_text("#!/bin/sh\nexit 0\n")
        (verifier / "tests/run_frontier_verifier.py").write_text(
            'def main():\n    reward = {"passed": 1.0 if passed is True else 0.0,}\n'
            '\nif __name__ == "__main__":\n    main()\n'
        )
        reference_archive.pack(verifier, reference_archive.ARCHIVE_BY_KIND["verifier"],
                               "frontier-challenge-reference", force=True)
        reference_archive.strip(verifier, reference_archive.ARCHIVE_BY_KIND["verifier"])
        records.append({"id": name, "image": environment})
    registry = {"name": "fixture", "n_tasks": 2, "tasks": records}
    for root, filename in [(runtime, "registry.json"), (solve, "source_registry.json"),
                           (reference, "source_registry.json")]:
        (root / filename).write_text(json.dumps(registry))
    envfile = tmp_path / "credentials.env"
    envfile.write_text("ANTHROPIC_API_KEY=fixture-not-a-key\n")
    binaries = tmp_path / "bin"
    binaries.mkdir()
    (binaries / "python3").symlink_to(sys.executable)
    docker = binaries / "docker"
    docker.write_text('#!/bin/sh\nprintf "%s\\n" "$*" >> "$DOCKER_LOG"\n'
                      'case "$*" in *orca-user-local*) exit 1;; esac\nexit 0\n')
    docker.chmod(0o755)
    harbor = binaries / "harbor"
    harbor.write_text('#!/bin/sh\nprintf "%s\\n" "$@" > "$HARBOR_LOG"\n')
    harbor.chmod(0o755)
    env = {**os.environ, "PATH": f"{binaries}:{os.environ['PATH']}",
           "DOCKER_LOG": str(tmp_path / "docker.log"), "HARBOR_LOG": str(tmp_path / "harbor.log"),
           "FRONTIER_CONFIG_FILE": str(tmp_path / "no-config")}
    stage = tmp_path / "stage"
    cmd = ["bash", str(runtime / "scripts/run_eval.sh"), "--agent", "claude-code",
           "--model", "fixture", "--solve-dir", str(solve), "--reference-dir", str(reference),
           "--stage-dir", str(stage), "--env-file", str(envfile), "--no-judge-override", "--no-summary",
           "--jobs-dir", str(tmp_path / "jobs"), "--job-name", "fixture"]
    return tmp_path, solve, stage, cmd, env


def test_open_selection_migrates_legacy_cache_and_ignores_stale_orca(runtime):
    root, solve, stage, cmd, env = runtime
    stale = stage / OPEN
    shutil.copytree(solve / "tasks" / OPEN, stale, symlinks=True)
    # Exactly the identity the old runner wrote, but nested input symlink now broken.
    (stale / ".frontier-source").write_text(
        f"{solve}|source_task_sha256:fixture|frontierchallenge/cpu-open:2026.08\n")
    assert (stale / "environment/data/output.txt").is_symlink()
    assert not (stale / "environment/data/output.txt").exists()
    shutil.copytree(solve / "tasks" / LICENSED, stage / LICENSED, symlinks=True)
    result = subprocess.run(cmd + ["--track", "open"], env=env, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Staged 1 task(s), reused 0" in result.stdout
    assert (stale / "environment/data/output.txt").read_text() == "precomputed output"
    assert not (stale / "environment/data/output.txt").is_symlink()
    assert (stale / "tests/test.sh").is_file()
    adapter = (stale / "tests/run_frontier_verifier.py").read_text()
    assert '"passed": 1.0 if complete and 0.999 < float(score or 0.0) / 100.0 <= 1.0 else 0.0' in adapter
    assert "orca-user-local" not in (root / "docker.log").read_text()
    args = (root / "harbor.log").read_text().splitlines()
    assert args[args.index("--include-task-name") + 1] == OPEN
    assert LICENSED not in args
    result = subprocess.run(cmd, env=env, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Staged 0 task(s), reused 1" in result.stdout


def test_full_track_exclusion_prevents_orca_preflight(runtime):
    root, _, _, cmd, env = runtime
    result = subprocess.run(cmd + ["--track", "full", "--exclude-task-name", LICENSED],
                            env=env, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "orca-user-local" not in (root / "docker.log").read_text()


def test_selected_licensed_task_still_requires_orca(runtime):
    root, _, _, cmd, env = runtime
    result = subprocess.run(cmd + ["--track", "full", "--include-task-name", LICENSED],
                            env=env, capture_output=True, text=True, check=False)
    assert result.returncode != 0
    assert "1 task(s) require ORCA" in result.stderr
    assert not (root / "harbor.log").exists()


def test_legacy_job_is_refused_before_staging(runtime):
    root, _, stage, cmd, env = runtime
    job = root / "jobs/fixture"
    job.mkdir(parents=True)
    (job / "config.json").write_text("{}")
    result = subprocess.run(cmd, env=env, capture_output=True, text=True, check=False)
    assert result.returncode != 0
    assert "Use a fresh --job-name" in result.stderr
    assert not (root / "harbor.log").exists()
    assert not (stage / OPEN).exists()
    assert (job / "config.json").read_text() == "{}"


def test_current_policy_job_resumes(runtime):
    import job_policy
    root, _, _, cmd, env = runtime
    job = root / "jobs/fixture"
    job_policy.record_policy(job, [OPEN])
    job.mkdir(parents=True)
    (job / "config.json").write_text("{}")
    (job / "lock.json").write_text(json.dumps({"trials": [{"task": {"name": OPEN}}]}))
    result = subprocess.run(cmd, env=env, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    assert (root / "harbor.log").read_text().splitlines()[:2] == ["job", "resume"]
