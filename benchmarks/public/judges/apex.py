"""Official-style independent-criterion APEX rubric grading."""

from __future__ import annotations

import asyncio
import json
import logging
import re

from benchmarks.public.judges._common import (
    Verdict,
    _build_judge_kwargs,
    _judge_call_with_effort_fallback,
)

logger = logging.getLogger(__name__)
_MAX_RETRIES = 10


def _parse_result(content: str) -> bool | None:
    for match in re.finditer(r"\{[\s\S]*?\}", content or ""):
        try:
            value = json.loads(match.group(0)).get("result")
        except (AttributeError, json.JSONDecodeError):
            continue
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return int(value) == 1
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "pass"}
    return None


async def _judge_criterion(criterion: str, solution: str) -> bool | None:
    prompt = (
        f"Criterion to evaluate: {criterion}\n\nResponse to evaluate: {solution}\n\n"
        "Return JSON only: {\"result\": 1 or 0, \"reason\": \"brief explanation\"}."
    )
    for _attempt in range(_MAX_RETRIES):
        content = await _judge_call_with_effort_fallback(
            _build_judge_kwargs(
                prompt,
                max_completion_tokens=8192,
                system="Evaluate the response against one criterion strictly.",
            ),
            label="APEX-criterion",
        )
        result = _parse_result(content)
        if result is not None:
            return result
    return None


async def score_apex(
    question: str,
    target: str,
    predicted: str,
) -> tuple[Verdict, float | None]:
    """Judge every criterion independently; all criteria must pass."""
    del question
    try:
        raw = json.loads(target) if isinstance(target, str) else target
        rubric = (raw or {}).get("rubric") or []
    except (AttributeError, json.JSONDecodeError, TypeError):
        logger.warning("APEX: malformed rubric target")
        return "NOT_ATTEMPTED", None
    if not rubric:
        return "NOT_ATTEMPTED", None
    if not predicted.strip():
        return "INCORRECT", 0.0
    results = await asyncio.gather(*(
        _judge_criterion(str(item.get("criteria", "")), predicted)
        for item in rubric
    ))
    if all(result is None for result in results):
        return "NOT_ATTEMPTED", None
    fraction = sum(result is True for result in results) / len(rubric)
    return ("CORRECT" if fraction == 1.0 else "INCORRECT"), fraction
