"""Prevent resuming or mixing jobs produced by incompatible scoring policies."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

POLICY = {"schema": 1, "metric_definition": "score-gt-0.999", "diagnostics": "no-native-pass"}


def marker_path(job: Path) -> Path:
    if not job.name:
        raise ValueError("job must be a named directory")
    return job.parent / ".frontierchallenge-policies" / f"{job.name}.json"


def expected_policy(task_ids: list[str]) -> dict:
    return {**POLICY, "task_ids": sorted(set(task_ids))}


def check_job(job: Path, task_ids: list[str]) -> str:
    if not job.exists() or (job.is_dir() and not any(job.iterdir())):
        return "new"
    try:
        stored = json.loads(marker_path(job).read_text())
        if stored != expected_policy(task_ids):
            raise ValueError("scoring/log policy or task selection changed")
        if not (job / "config.json").is_file():
            raise ValueError("job config is missing")
        lock = json.loads((job / "lock.json").read_text())
        recorded = {trial["task"]["name"] for trial in lock["trials"]}
        if recorded != set(task_ids):
            raise ValueError("recorded task selection differs")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise ValueError(
            f"cannot resume {job}: missing/incompatible FrontierChallenge policy or job metadata. "
            "Use a fresh --job-name (or --jobs-dir); existing results were not modified. "
            "Old results may be summarized separately, but must not be mixed into a new-policy job."
        ) from exc
    return "resume"


def record_policy(job: Path, task_ids: list[str]) -> None:
    check_job(job, task_ids)
    target = marker_path(job)
    target.parent.mkdir(parents=True, exist_ok=True)
    document = expected_policy(task_ids)
    if target.exists() and json.loads(target.read_text()) == document:
        return
    # Only absent/empty jobs can acquire a new marker; populated old jobs fail
    # the check above. No old result or old log is rewritten during an upgrade.
    target.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("check", "record"))
    parser.add_argument("job", type=Path)
    parser.add_argument("--task-id", action="append", required=True)
    args = parser.parse_args()
    try:
        if args.action == "check":
            print(check_job(args.job.absolute(), args.task_id))
        else:
            record_policy(args.job.absolute(), args.task_id)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
