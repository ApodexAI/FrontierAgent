import json

from verifier_log_policy import clean_json, clean_text, publish_logs


def test_nested_decisions_are_removed_but_scores_remain():
    source = {"score": 80, "passed": True, "detail": [{"PASSED": True, "score": 5}],
              "evaluation_complete": True, "pass_threshold": 70, "verdict": "PASS"}
    assert clean_json(source) == {"score": 80, "detail": [{"score": 5}],
                                  "evaluation_complete": True, "verdict": "[decision omitted]"}
    assert source["passed"] is True  # The adapter can still parse its in-memory payload.


def test_text_decisions_are_removed_without_losing_ordinary_errors():
    text = 'score: 80\nPASSED = true\nscore 80 -> PASS\nfailed to open input.csv\n'
    assert clean_text(text) == 'score: 80\n[decision omitted]\n[decision omitted]\nfailed to open input.csv\n'
    assert json.loads(clean_text('{"score": 80, "passed": true}')) == {"score": 80}


def test_only_reward_keeps_canonical_pass(tmp_path):
    private, public = tmp_path / "private", tmp_path / "public"
    private.mkdir()
    (private / "native_grader_result.json").write_text('{"score":80,"passed":true}')
    (private / "native_grader.stdout.txt").write_text('PASSED=true\nscore: 80\n')
    reward = '{"task_score":0.8,"passed":0,"evaluation_complete":1}'
    (private / "reward.json").write_text(reward)
    publish_logs(private, public)
    assert (public / "reward.json").read_text() == reward
    assert json.loads((public / "native_grader_result.json").read_text()) == {"score": 80}
    assert "PASSED" not in (public / "native_grader.stdout.txt").read_text()
