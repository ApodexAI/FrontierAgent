from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from benchmarks.public.core.registry import DatasetConfig, _ground_truth, get_config, load_questions
from benchmarks.public.runner.export_frontier_search import export_run
from benchmarks.public.runner.export_frontier_search import main as export_main
from benchmarks.public.runner.run_one_task import _score_answer
from benchmarks.public.runner.run_subprocess import _failure_result
from benchmarks.public.runner.score_frontier_search import (
    _scorer_env,
    _seed_previous_scores,
    run_official_scorer,
)


def test_bundled_frontier_search_dataset_loads_canonical_queries():
    cfg = get_config("frontier_search")
    questions = load_questions("frontier_search")

    assert cfg.scoring_mode == "external"
    assert cfg.source_format == "json"
    assert len(questions) == 41
    assert [question.id for question in questions] == [str(i) for i in range(1, 42)]
    assert all(question.answer_type == "report" for question in questions)
    assert all(question.ground_truth == "" for question in questions)


@pytest.mark.asyncio
async def test_external_scoring_marks_answer_pending_without_judge_call():
    reward, method, error, rubric = await _score_answer(
        "a complete research report",
        "",
        "question",
        "report",
        "1",
        "frontier_search",
    )

    assert reward is None
    assert method == "external_pending"
    assert error is None
    assert rubric is None


def test_exporter_uses_full_canonical_query_text(tmp_path: Path):
    run_dir = tmp_path / "run"
    trial = run_dir / "trials" / "1"
    trial.mkdir(parents=True)
    (trial / "result.json").write_text(
        json.dumps(
            {
                "question_id": "1",
                "question": "truncated display text",
                "predicted_answer": "full model report",
                "is_correct": None,
                "reward": None,
                "judge_method": "external_pending",
            }
        ),
        encoding="utf-8",
    )

    output = tmp_path / "answers.json"
    summary = export_run(run_dir, output)
    rows = json.loads(output.read_text(encoding="utf-8"))
    canonical = load_questions("frontier_search")[0]

    assert summary["exported"] == 1
    assert summary["missing_ids"] == list(range(2, 42))
    assert rows == [
        {
            "id": 1,
            "query": canonical.question,
            "report_content": "full model report",
            "response": "",
        }
    ]


def test_scorer_wrapper_maps_existing_judge_credentials(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_BASE_URL", raising=False)
    monkeypatch.setenv("JUDGE_API_KEY", "judge-key")
    monkeypatch.setenv("JUDGE_BASE_URL", "https://judge.example/v1")

    env = _scorer_env()

    assert env["OPENROUTER_API_KEY"] == "judge-key"
    assert env["OPENROUTER_BASE_URL"] == "https://judge.example/v1"


def test_official_scorer_dry_run_isolated_from_source_tree(tmp_path: Path):
    bench_root = (
        Path(__file__).resolve().parents[1] / "benchmarks" / "frontier_search_bench"
    )
    template = bench_root / "queries" / "answer_template.json"
    args = argparse.Namespace(
        models=[f"frontier_agent={template}"],
        out=tmp_path / "scores",
        only="1,5,16,20,31,40,41",
        force=False,
        force_rerun=False,
        allow_query_mismatch=False,
        dry_run=True,
        per_query_timeout=60,
    )

    scorers_root = bench_root / "eval" / "verifiable" / "scorers"
    assert scorers_root.is_dir()  # guard: an empty glob below must mean "clean"

    assert run_official_scorer(args) == 0
    assert not list(scorers_root.glob("query_*/auto_scores"))


def test_crashed_trial_stays_unscored_under_external_scoring():
    """A trial that never produced an answer has no verdict to report.

    Marking it WRONG would make check_progress compute an accuracy for a
    collection-only run, which is exactly what external scoring avoids.
    """
    external = _failure_result(
        "7", error="boom", duration=1.0,
        judge_method="worker_crash", external_scoring=True,
    )
    inline = _failure_result(
        "7", error="boom", duration=1.0,
        judge_method="worker_crash", external_scoring=False,
    )

    assert external["is_correct"] is None
    assert external["reward"] is None
    assert inline["is_correct"] is False
    assert inline["reward"] == 0


def test_missing_answer_column_fails_loudly():
    cfg = DatasetConfig(name="X", key="x", answer_field="ground_truth")

    with pytest.raises(KeyError, match="ground_truth"):
        _ground_truth({"id": 1, "answer": "42"}, cfg, "x", 0)


def test_ground_truth_preserves_falsy_and_structured_values():
    cfg = DatasetConfig(name="X", key="x", answer_field="ground_truth")

    assert _ground_truth({"ground_truth": 0}, cfg, "x", 0) == "0"
    assert _ground_truth({"ground_truth": False}, cfg, "x", 0) == "false"
    assert _ground_truth({"ground_truth": None}, cfg, "x", 0) == ""
    assert _ground_truth({"ground_truth": ""}, cfg, "x", 0) == ""
    # WideSearch ships a structured eval_spec; it must survive as parseable JSON.
    spec = {"gold_table": [{"a": 1}]}
    assert json.loads(_ground_truth({"ground_truth": spec}, cfg, "x", 0)) == spec
    # No answer field configured at all (external scoring) is not an error.
    assert _ground_truth({}, DatasetConfig(name="X", key="x", answer_field=""), "x", 0) == ""


def _write_partial_run(tmp_path: Path) -> Path:
    run_dir = tmp_path / "run"
    trial = run_dir / "trials" / "1"
    trial.mkdir(parents=True)
    (trial / "result.json").write_text(
        json.dumps({"question_id": "1", "predicted_answer": "report"}),
        encoding="utf-8",
    )
    return run_dir


def test_partial_export_exits_nonzero(tmp_path: Path, monkeypatch, capsys):
    run_dir = _write_partial_run(tmp_path)
    out = tmp_path / "answers.json"
    monkeypatch.setattr(
        "sys.argv",
        ["export_frontier_search", str(run_dir), "--out", str(out)],
    )

    with pytest.raises(SystemExit) as excinfo:
        export_main()

    assert excinfo.value.code == 3
    assert "canonical queries have no result" in capsys.readouterr().err


def test_partial_export_allowed_explicitly(tmp_path: Path, monkeypatch):
    run_dir = _write_partial_run(tmp_path)
    out = tmp_path / "answers.json"
    monkeypatch.setattr(
        "sys.argv",
        ["export_frontier_search", str(run_dir), "--out", str(out), "--allow-partial"],
    )

    export_main()  # no SystemExit

    assert len(json.loads(out.read_text(encoding="utf-8"))) == 1


def test_scorer_env_does_not_mix_providers(monkeypatch):
    """An explicit OpenRouter key must not be paired with a judge gateway URL."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "openrouter-key")
    monkeypatch.delenv("OPENROUTER_BASE_URL", raising=False)
    monkeypatch.setenv("JUDGE_API_KEY", "judge-key")
    monkeypatch.setenv("JUDGE_BASE_URL", "https://internal-gateway/v1")

    env = _scorer_env()

    assert env["OPENROUTER_API_KEY"] == "openrouter-key"
    assert env["OPENROUTER_BASE_URL"] == "https://openrouter.ai/api/v1"


def test_scorer_env_falls_back_to_openai_pair(monkeypatch):
    for var in ("OPENROUTER_API_KEY", "OPENROUTER_BASE_URL",
                "JUDGE_API_KEY", "JUDGE_BASE_URL"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://openai.example/v1")

    env = _scorer_env()

    assert env["OPENROUTER_API_KEY"] == "openai-key"
    assert env["OPENROUTER_BASE_URL"] == "https://openai.example/v1"


def test_previous_scores_are_seeded_back_into_the_working_tree(tmp_path: Path):
    output_dir = tmp_path / "official_scores"
    (output_dir / "per_query" / "query_01").mkdir(parents=True)
    (output_dir / "per_query" / "query_01" / "scores.json").write_text(
        '{"frontier_agent": {}}', encoding="utf-8"
    )
    # A per_query entry with no matching scorer must be ignored, not crash.
    (output_dir / "per_query" / "query_99").mkdir(parents=True)

    eval_root = tmp_path / "work" / "eval" / "verifiable"
    (eval_root / "scorers" / "query_01").mkdir(parents=True)

    assert _seed_previous_scores(output_dir, eval_root) == 1
    seeded = eval_root / "scorers" / "query_01" / "auto_scores" / "scores.json"
    assert seeded.read_text(encoding="utf-8") == '{"frontier_agent": {}}'
