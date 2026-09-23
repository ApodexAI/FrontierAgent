from __future__ import annotations

import pytest
from apply_score_policy import OLD_RULE, apply_policy


@pytest.mark.parametrize("score,native,complete,expected", [
    (100, False, True, 1), (80, True, True, 0), (99.999, True, True, 1),
    # The frozen normalization emits 99.9 / 100 as 0.9990000000000001.
    # Compare that actual task_score, not a rounded display of 0.999.
    (99.9, True, True, 1), (99.91, False, True, 1), (99.899, True, True, 0),
    (99.91, True, False, 0), (100.1, True, True, 0), (float("inf"), True, True, 0),
    (100, True, False, 0), (0, True, True, 0), (100, None, True, 1),
])
def test_staged_reward_uses_strict_score_threshold(tmp_path, score, native, complete, expected):
    (tmp_path / "tests").mkdir()
    adapter = tmp_path / "tests/run_frontier_verifier.py"
    adapter.write_text('def main():\n    global reward\n    reward = {' + OLD_RULE
                       + '}\n\nif __name__ == "__main__":\n    main()\n')
    apply_policy(tmp_path)
    namespace = {"score": score, "passed": native, "complete": complete, "__name__": "test_adapter"}
    exec(compile(adapter.read_text(), str(adapter), "exec"), namespace)
    namespace["_run_native_verifier"]()
    assert namespace["reward"] == {"passed": float(expected)}


def test_unknown_adapter_fails_closed(tmp_path):
    (tmp_path / "tests").mkdir()
    adapter = tmp_path / "tests/run_frontier_verifier.py"
    adapter.write_text("reward = {}\n")
    with pytest.raises(ValueError, match="unsupported"):
        apply_policy(tmp_path)
    assert adapter.read_text() == "reward = {}\n"


@pytest.mark.parametrize("failure", [False, True])
def test_adapter_publishes_clean_logs_on_success_and_failure(tmp_path, failure):
    import json
    adapter_dir = tmp_path / "task/tests"
    adapter_dir.mkdir(parents=True)
    adapter = adapter_dir / "run_frontier_verifier.py"
    adapter.write_text('''from pathlib import Path
import json
def main():
    global private_path
    private_path = LOGS
    LOGS.mkdir(parents=True, exist_ok=True)
    (LOGS / "native_grader_result.json").write_text(json.dumps({"score": 80, "passed": True}))
    (LOGS / "native_grader.stdout.txt").write_text("PASSED=true\\nscore:80\\n")
    if failure:
        raise RuntimeError("fixture failure")
    reward = {"task_score": 0.8, ''' + OLD_RULE + ''' "evaluation_complete": 1}
    (LOGS / "reward.json").write_text(json.dumps(reward))

if __name__ == "__main__":
    main()
''')
    apply_policy(tmp_path / "task")
    public = tmp_path / "public"
    ns = {"__name__": "fixture", "LOGS": public, "failure": failure,
          "score": 80, "passed": True, "complete": True}
    exec(compile(adapter.read_text(), str(adapter), "exec"), ns)
    if failure:
        with pytest.raises(RuntimeError, match="fixture failure"):
            ns["main"]()
    else:
        ns["main"]()
        assert json.loads((public / "reward.json").read_text())["passed"] == 0
    assert json.loads((public / "native_grader_result.json").read_text()) == {"score": 80}
    assert "PASSED" not in (public / "native_grader.stdout.txt").read_text()
    assert ns["LOGS"] == public
    assert not ns["private_path"].exists()
