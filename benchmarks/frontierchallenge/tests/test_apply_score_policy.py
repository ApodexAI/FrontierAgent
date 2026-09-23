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
    adapter.write_text('reward = {' + OLD_RULE + '}\n')
    apply_policy(tmp_path)
    namespace = {"score": score, "passed": native, "complete": complete}
    exec(compile(adapter.read_text(), str(adapter), "exec"), namespace)
    assert namespace["reward"] == {"passed": float(expected)}


def test_unknown_adapter_fails_closed(tmp_path):
    (tmp_path / "tests").mkdir()
    adapter = tmp_path / "tests/run_frontier_verifier.py"
    adapter.write_text("reward = {}\n")
    with pytest.raises(ValueError, match="unsupported"):
        apply_policy(tmp_path)
    assert adapter.read_text() == "reward = {}\n"
