import json

import pytest
from job_policy import check_job, marker_path, record_policy


def existing(job):
    job.mkdir(parents=True)
    (job / "config.json").write_text("{}")
    (job / "lock.json").write_text(json.dumps({"trials": [{"task": {"name": "task_one"}}]}))


def test_new_job_marker_does_not_create_harbor_job(tmp_path):
    job = tmp_path / "new"
    assert check_job(job, ["task_one"]) == "new"
    record_policy(job, ["task_one"])
    assert not job.exists()
    assert marker_path(job).is_file()
    existing(job)
    assert check_job(job, ["task_one"]) == "resume"


@pytest.mark.parametrize("case", ["missing", "old-metric", "old-logs", "different-tasks", "invalid-lock"])
def test_unsafe_resume_does_not_mutate_results(tmp_path, case):
    job = tmp_path / "old"
    record_policy(job, ["task_one"])
    existing(job)
    marker = marker_path(job)
    doc = json.loads(marker.read_text())
    if case == "missing":
        marker.unlink()
    elif case == "old-metric":
        doc["metric_definition"] = "exact-full-score"
        marker.write_text(json.dumps(doc))
    elif case == "old-logs":
        doc.pop("diagnostics")
        marker.write_text(json.dumps(doc))
    elif case == "invalid-lock":
        (job / "lock.json").write_text("broken")
    tasks = ["other"] if case == "different-tasks" else ["task_one"]
    before = {p.name: p.read_bytes() for p in job.iterdir()}
    for operation in (check_job, record_policy):
        with pytest.raises(ValueError, match="fresh --job-name"):
            operation(job, tasks)
    assert before == {p.name: p.read_bytes() for p in job.iterdir()}
