from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.public.judges.apex import _parse_result, score_apex
from benchmarks.public.judges.gdpval import score_gdpval_outputs
from benchmarks.public.judges.officeqa import score_officeqa
from benchmarks.public.judges.onemillion import _parse_rubric_verdicts, score_onemillion


@pytest.mark.asyncio
async def test_officeqa_uses_tagged_direct_answer() -> None:
    verdict, score = await score_officeqa(
        "unused",
        "$1.5 billion",
        "<REASONING>checked source</REASONING><FINAL_ANSWER>$1.5 billion</FINAL_ANSWER>",
    )
    assert (verdict, score) == ("CORRECT", 1.0)


def test_gdpval_deterministic_artifact_validation(tmp_path: Path) -> None:
    output = tmp_path / "result.txt"
    output.write_text("deliverable", encoding="utf-8")
    assert score_gdpval_outputs(tmp_path, "{}") == (1, 1.0)

    output.unlink()
    (tmp_path / "broken.docx").write_text("not a zip", encoding="utf-8")
    assert score_gdpval_outputs(tmp_path, "{}") == (0, 0.0)


def test_gdpval_requires_expected_deliverable_extension(tmp_path: Path) -> None:
    (tmp_path / "answer.txt").write_text("answer", encoding="utf-8")
    target = json.dumps({"reference_files": ["gold/answer.xlsx"]})
    assert score_gdpval_outputs(tmp_path, target) == (0, 0.0)


def test_widesearch_url_match_ignores_trailing_prose_punctuation() -> None:
    pytest.importorskip("pandas")  # widesearch pulls the eval extras in at import
    from benchmarks.public.judges.widesearch import _url_netlocs, metric_url_match

    # A parenthesized or sentence-final URL used to carry ")"/"." into the netloc
    # and so never matched its counterpart.
    assert _url_netlocs("see (https://example.com)") == {"example.com"}
    assert _url_netlocs("source: https://example.com.") == {"example.com"}
    assert metric_url_match("see (https://example.com).", "https://example.com")[0] == 1.0

    # Dots inside the URL body must survive — only trailing punctuation is cut.
    assert _url_netlocs("https://a.example.com/x.html, next") == {"a.example.com"}
    assert metric_url_match("https://a.com/x", "https://b.com/x")[0] == 0.0


def test_rubric_json_parsers_tolerate_wrapping() -> None:
    assert _parse_result('result: {"result": 1, "reason": "ok"}') is True
    assert _parse_rubric_verdicts(
        '```json\n[{"rubric_id": 1, "status": "yes"}, '
        '{"rubric_id": 2, "status": "no"}]\n```'
    ) == {1}


@pytest.mark.asyncio
async def test_empty_rubric_answers_fail_without_llm_calls() -> None:
    apex_target = json.dumps({"rubric": [{"criteria": "must answer"}]})
    one_million_target = json.dumps([
        {"rubric_number": 1, "rubric_detail": "must answer", "rubric_weight": 1},
    ])
    assert await score_apex("q", apex_target, "") == ("INCORRECT", 0.0)
    assert await score_onemillion("q", one_million_target, "") == ("INCORRECT", 0.0)
