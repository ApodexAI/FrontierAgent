#!/usr/bin/env python3
"""Aggregate a Harbor job directory into a Pass Rate + Score summary.

Reads each trial's ``config.json`` (to recover which task it ran) and
``verifier/reward.json`` (``task_score``, ``evaluation_complete``)
under a Harbor jobs-dir job, and writes a per-task CSV plus an overall JSON
summary next to it.

    python3 scripts/summarize_results.py results/harbor/<job-name>
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

#: 100 tasks minus task_065 (needs a GPU) and task_047 / task_049 (their
#: deterministic grader only runs in a nested container, unavailable here).
EXPECTED_TOTAL_TASKS = 97
METRIC_DEFINITION = "exact-full-score"


def valid_score(score: Any) -> bool:
    return (
        isinstance(score, (int, float))
        and not isinstance(score, bool)
        and 0.0 <= score <= 1.0
        and math.isfinite(score)
    )


def official_pass(row: dict[str, Any]) -> bool:
    """The sole pass decision: exact full credit on a completed evaluation."""
    return (
        row.get("evaluation_complete") == 1
        and valid_score(row.get("task_score"))
        and row["task_score"] == 1.0
    )

#: Judge stderr wording (both casings occur across the frozen graders) for the
#: one "verifier failed" cause that is not an infrastructure fault at all: the
#: submission simply does not contain a file the task requires. That is the
#: agent failing the task, so it scores 0 and belongs in the denominator -
#: dropping it would quietly reward a model for producing nothing.
MISSING_ARTIFACT_MARKERS = (
    "required CANDIDATE artifact is missing",
    "required candidate artifact missing",
)

#: The native grader's wording when the agent produced no output whatsoever;
#: it exits before the Judge runs, so no judge log exists to inspect.
EMPTY_SUBMISSION_MARKER = "submission directory not found"


def load_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


def missing_required_artifact(trial_dir: Path) -> bool:
    """True if this trial failed grading because the submission is inadequate.

    Two shapes of the same thing, both a genuine 0 rather than an
    infrastructure fault:

    * the Judge ran and reported a required file absent, and
    * the agent wrote *nothing at all*, so the native grader exits with
      "submission directory not found" before the Judge is even reached.

    The second is by far the more common, and treating it as a verifier
    failure would drop the task from the denominator entirely - i.e. hand a
    model that produced no output the same treatment as a broken endpoint.
    """
    verifier_dir = trial_dir / "verifier"
    if not verifier_dir.is_dir():
        return False
    for log in verifier_dir.glob("native_judge_*.stderr.txt"):
        try:
            text = log.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if any(marker in text for marker in MISSING_ARTIFACT_MARKERS):
            return True

    grader_log = verifier_dir / "native_grader.stderr.txt"
    try:
        grader_text = grader_log.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False
    if EMPTY_SUBMISSION_MARKER not in grader_text:
        return False
    # Only when the submission really is empty - the same message would be a
    # pipeline bug, not a model failure, if artifacts had in fact been produced.
    output_dir = trial_dir / "artifacts" / "app" / "output"
    return not (output_dir.is_dir() and any(output_dir.iterdir()))


def collect_rows(job_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for trial_dir in sorted(p for p in job_dir.iterdir() if p.is_dir()):
        config = load_json(trial_dir / "config.json")
        if config is None:
            continue
        task_path = config.get("task", {}).get("path", "")
        task_id = Path(task_path).name if task_path else trial_dir.name
        agent = config.get("agent", {})
        row: dict[str, Any] = {
            "task_id": task_id,
            "trial_name": trial_dir.name,
            "agent": agent.get("name"),
            "model": agent.get("model_name"),
            "task_score": None,
            "passed": None,
            "evaluation_complete": None,
            "error": None,
            "scored_zero_missing_artifact": False,
        }
        reward = load_json(trial_dir / "verifier" / "reward.json")
        if reward is not None:
            row["task_score"] = reward.get("task_score")
            row["evaluation_complete"] = reward.get("evaluation_complete")
            if not valid_score(row["task_score"]):
                row["task_score"] = None
                row["evaluation_complete"] = 0.0
                row["error"] = "invalid task_score: expected a finite number in [0, 1]"
        else:
            row["error"] = "no reward.json (trial errored or is still running)"
        if row["evaluation_complete"] != 1.0 and missing_required_artifact(trial_dir):
            row["task_score"] = 0.0
            row["passed"] = 0.0
            row["evaluation_complete"] = 1.0
            row["scored_zero_missing_artifact"] = True
            row["error"] = "required artifact missing from submission - genuine 0"
        row["passed"] = float(official_pass(row))
        rows.append(row)
    return rows


def summarize(rows: list[dict[str, Any]], expected_total: int) -> dict[str, Any]:
    if expected_total <= 0:
        raise ValueError("expected_total must be positive")
    ids = [r["task_id"] for r in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate task trials: select one predeclared attempt per task; do not select by score")
    if len(rows) > expected_total:
        raise ValueError("more tasks than expected_total; check the evaluation scope")
    # Incomplete/invalid evaluations contribute zero to the fixed denominator,
    # but remain visible as failures rather than successful zero-score grading.
    graded = [r for r in rows if r["evaluation_complete"] == 1.0 and valid_score(r["task_score"])]
    ungraded = [r for r in rows if r["evaluation_complete"] == 0.0]
    n = len(rows)
    mean_score = sum(r["task_score"] for r in graded) / expected_total
    n_passed = sum(official_pass(r) for r in rows)
    pass_rate = n_passed / expected_total
    return {
        "metric_definition": METRIC_DEFINITION,
        "pass_rule": "evaluation_complete == 1 and task_score == 1.0",
        "n_tasks_expected": expected_total,
        "n_trials_found": n,
        "n_graded": len(graded),
        "n_zero_missing_artifact": sum(1 for r in graded if r["scored_zero_missing_artifact"]),
        "n_verifier_failed": len(ungraded),
        "n_missing_or_errored": expected_total - len(graded) - len(ungraded),
        "complete": n == expected_total and len(graded) == expected_total,
        "n_passed": n_passed,
        "pass_rate": pass_rate,
        "mean_task_score": mean_score,
        "mean_task_score_100": (mean_score * 100) if mean_score is not None else None,
    }


def write_outputs(job_dir: Path, rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    csv_path = job_dir / "summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "task_id",
                "trial_name",
                "agent",
                "model",
                "task_score",
                "passed",
                "evaluation_complete",
                "scored_zero_missing_artifact",
                "error",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    json_path = job_dir / "summary.json"
    json_path.write_text(
        json.dumps({"job_dir": str(job_dir), **summary, "rows": rows}, indent=2),
        encoding="utf-8",
    )


def print_report(job_dir: Path, summary: dict[str, Any]) -> None:
    print(f"Job: {job_dir}")
    print(f"Trials found: {summary['n_trials_found']} / {summary['n_tasks_expected']} expected")
    if not summary["complete"]:
        print("WARNING: incomplete run - treat Pass Rate / Score below as partial, not final.")
    if summary["n_zero_missing_artifact"]:
        print(
            f"NOTE: {summary['n_zero_missing_artifact']} trial(s) scored 0 because the "
            "submission was missing a required artifact - a real task failure, "
            "counted in Pass Rate / Score below, not excluded."
        )
    if summary["n_verifier_failed"]:
        print(
            f"NOTE: {summary['n_verifier_failed']} trial(s) had the verifier itself "
            "fail (e.g. a Judge request the endpoint rejected, a missing-artifact "
            "hard error) - contribute 0 to the fixed-denominator metrics below."
        )
    if summary["pass_rate"] is not None:
        print(
            f"Pass Rate (task_score == 1): {summary['n_passed']}/{summary['n_tasks_expected']} "
            f"= {summary['pass_rate'] * 100:.1f}%"
        )
    else:
        print("Pass Rate: n/a (no graded trials)")
    if summary["mean_task_score_100"] is not None:
        print(f"Mean Score: {summary['mean_task_score_100']:.1f} / 100 (denominator {summary['n_tasks_expected']})")
    else:
        print("Mean Score: n/a (no graded trials)")
    print(f"Wrote {job_dir / 'summary.csv'} and {job_dir / 'summary.json'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job_dir", type=Path, help="Harbor jobs-dir job, e.g. results/harbor/<job-name>")
    parser.add_argument(
        "--task-id", action="append",
        help="only include this task ID (repeatable); excludes stale trials from a reused job",
    )
    parser.add_argument(
        "--expected-total",
        type=int,
        default=EXPECTED_TOTAL_TASKS,
        help=f"Expected task count for a complete run (default: {EXPECTED_TOTAL_TASKS})",
    )
    args = parser.parse_args()

    job_dir = args.job_dir.resolve()
    if not job_dir.is_dir():
        raise SystemExit(f"not a directory: {job_dir}")

    rows = collect_rows(job_dir)
    if args.task_id is not None:
        selected = set(args.task_id)
        rows = [row for row in rows if row["task_id"] in selected]
    try:
        summary = summarize(rows, args.expected_total)
    except ValueError as exc:
        parser.error(str(exc))
    write_outputs(job_dir, rows, summary)
    print_report(job_dir, summary)


if __name__ == "__main__":
    main()
