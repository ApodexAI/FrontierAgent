"""OneMillion-Bench's signed, weighted rubric judge."""

from __future__ import annotations

import contextlib
import json
import logging
import os
import re

from benchmarks.public.judges._common import (
    Verdict,
    _build_judge_kwargs,
    _judge_call_with_effort_fallback,
)

logger = logging.getLogger(__name__)


def _parse_rubric_verdicts(content: str) -> set[int]:
    match = re.search(r"\[[\s\S]*\]", content or "")
    if not match:
        return set()
    try:
        values = json.loads(re.sub(r",\s*([}\]])", r"\1", match.group(0)))
    except (json.JSONDecodeError, TypeError, ValueError):
        return set()
    hits: set[int] = set()
    for value in values:
        if isinstance(value, dict) and str(value.get("status", "")).lower() in {"yes", "true", "是"}:
            with contextlib.suppress(TypeError, ValueError):
                hits.add(int(value.get("rubric_id", value.get("id")) or 0))
    return hits


async def score_onemillion(
    question: str,
    target: str,
    predicted: str,
) -> tuple[Verdict, float | None]:
    """Return pass/fail and the official earned/positive-weight fraction."""
    try:
        raw = json.loads(target) if isinstance(target, str) else target
        items = [
            {
                "id": int(item.get("rubric_number", index + 1)),
                "detail": str(item.get("rubric_detail", "")),
                "weight": int(item.get("rubric_weight", 0)),
            }
            for index, item in enumerate(raw or [])
        ]
    except (json.JSONDecodeError, TypeError, ValueError):
        logger.warning("OneMillion: malformed rubrics target")
        return "NOT_ATTEMPTED", None
    if not items:
        return "NOT_ATTEMPTED", None
    if not predicted.strip():
        return "INCORRECT", 0.0

    rubrics = "\n\n".join(
        f"Rubric {item['id']} (weight {item['weight']:+d})\n{item['detail']}"
        for item in items
    )
    prompt = (
        "For each rubric, output a JSON array of objects with rubric_id and status "
        f"(yes or no).\n\nQuestion\n{question}\n\nResponse\n{predicted}\n\nRubrics\n{rubrics}"
    )
    content = await _judge_call_with_effort_fallback(
        _build_judge_kwargs(
            prompt,
            max_completion_tokens=16_384,
            system="You are a strict rubric grader. Reply only with a JSON array.",
            temperature=0,
        ),
        label="OneMillion",
    )
    if not content:
        return "NOT_ATTEMPTED", None
    hits = _parse_rubric_verdicts(content)
    max_positive = sum(item["weight"] for item in items if item["weight"] > 0)
    earned = sum(item["weight"] for item in items if item["id"] in hits)
    fraction = earned / max_positive if max_positive else 0.0
    threshold = float(os.environ.get("ONEMILLION_PASS", "0.5"))
    return ("CORRECT" if fraction >= threshold else "INCORRECT"), round(fraction, 4)
