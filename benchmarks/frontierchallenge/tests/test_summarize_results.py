from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest
import summarize_results as metrics


def trial(root, name, score, native_passed=1, complete=1):
    directory = root / name
    (directory / "verifier").mkdir(parents=True)
    (directory / "config.json").write_text(json.dumps({"task": {"path": f"/tasks/{name}"}}))
    (directory / "verifier/reward.json").write_text(json.dumps({
        "task_score": score, "passed": native_passed, "evaluation_complete": complete,
    }))
    return directory


@pytest.mark.parametrize("score,native,complete,passed", [
    (1.0, 0, 1, 1), (1, 1, 1, 1), (0.999999999, 1, 1, 0),
    (0.999, 1, 1, 0), (0.8, 1, 1, 0), (0.0, 1, 1, 0),
    (1.0, 1, 0, 0), (1.0, 1, None, 0),
])
def test_official_pass_ignores_native_threshold(tmp_path, score, native, complete, passed):
    directory = trial(tmp_path, "task_one", score, native, complete)
    before = (directory / "verifier/reward.json").read_bytes()
    row = metrics.collect_rows(tmp_path)[0]
    assert row["passed"] == passed
    assert row["native_passed"] == native
    assert (directory / "verifier/reward.json").read_bytes() == before
    assert metrics.summarize([row], 97)["pass_rate"] == passed / 97


@pytest.mark.parametrize("score", [None, "1", True, -0.1, 1.001, 10**400, float("inf"), float("nan")])
def test_invalid_scores_are_not_passes_or_partial_credit(tmp_path, score):
    trial(tmp_path, "task_bad", score)
    rows = metrics.collect_rows(tmp_path)
    assert rows[0]["task_score"] is None
    assert rows[0]["evaluation_complete"] == 0
    summary = metrics.summarize(rows, 97)
    assert summary["n_passed"] == summary["mean_task_score"] == 0
    assert summary["n_verifier_failed"] == 1


def test_fixed_denominator_and_partial_credit(tmp_path):
    trial(tmp_path, "task_one", 1, 0)
    trial(tmp_path, "task_two", 0.7, 1)
    trial(tmp_path, "task_failed", 1, 1, 0)
    summary = metrics.summarize(metrics.collect_rows(tmp_path), 97)
    assert summary["n_passed"] == 1
    assert summary["pass_rate"] == 1 / 97
    assert summary["mean_task_score_100"] == pytest.approx(170 / 97)
    assert summary["n_missing_or_errored"] == 94
    assert summary["complete"] is False
    assert summary["metric_definition"] == "exact-full-score"


def test_empty_run_is_zero_and_incomplete():
    summary = metrics.summarize([], 97)
    assert summary["n_missing_or_errored"] == 97
    assert summary["pass_rate"] == summary["mean_task_score"] == 0
    assert not summary["complete"]


def test_repeated_trials_and_wrong_denominator_are_rejected(tmp_path):
    trial(tmp_path, "task_one", 1)
    rows = metrics.collect_rows(tmp_path)
    with pytest.raises(ValueError, match="duplicate"):
        metrics.summarize(rows * 2, 97)
    with pytest.raises(ValueError, match="positive"):
        metrics.summarize(rows, 0)
    trial(tmp_path, "task_two", 0)
    with pytest.raises(ValueError, match="more tasks"):
        metrics.summarize(metrics.collect_rows(tmp_path), 1)


def test_missing_submission_still_counts_as_zero(tmp_path):
    directory = trial(tmp_path, "task_one", 0, 0, 0)
    (directory / "verifier/native_grader.stderr.txt").write_text("submission directory not found")
    summary = metrics.summarize(metrics.collect_rows(tmp_path), 1)
    assert summary["complete"] is True
    assert summary["n_zero_missing_artifact"] == 1
    assert summary["pass_rate"] == 0


def test_cli_outputs_official_and_native_flags_separately(tmp_path):
    trial(tmp_path, "task_one", 0.9, 1)
    trial(tmp_path, "task_two", 1, 0)
    subprocess.run([sys.executable, str(Path(metrics.__file__)), str(tmp_path),
                    "--expected-total", "2"], check=True, capture_output=True)
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["pass_rate"] == 0.5
    assert summary["mean_task_score_100"] == 95
    assert summary["complete"] is True
    with (tmp_path / "summary.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert [(r["passed"], r["native_passed"]) for r in rows] == [("0.0", "1"), ("1.0", "0")]


def test_cli_excludes_stale_trials_from_reused_job(tmp_path):
    trial(tmp_path, "task_selected", 0.5)
    trial(tmp_path, "task_stale", 1)
    subprocess.run([sys.executable, str(Path(metrics.__file__)), str(tmp_path),
                    "--task-id", "task_selected"], check=True, capture_output=True)
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["n_trials_found"] == 1
    assert summary["n_passed"] == 0
    assert summary["mean_task_score"] == 0.5 / 97


@pytest.mark.parametrize("value", [[], 1, "broken", None])
def test_non_object_reward_is_reported_as_missing(tmp_path, value):
    directory = trial(tmp_path, "task_bad", 1)
    (directory / "verifier/reward.json").write_text(json.dumps(value))
    rows = metrics.collect_rows(tmp_path)
    assert rows[0]["task_score"] is None
    assert metrics.summarize(rows, 97)["n_passed"] == 0
