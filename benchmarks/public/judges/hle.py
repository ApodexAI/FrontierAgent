"""HLE judge — Humanity's Last Exam (also used by SUPERChem-Text MCQ).

Pins ``o3-mini-2025-01-31`` and uses ``beta.chat.completions.parse``
with ``response_format=_HLEExtractedAnswer`` for structured output.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Literal

from pydantic import BaseModel

from benchmarks.public.judges._common import (
    Verdict,
    _make_client,
    _resolve_judge_model,
)

logger = logging.getLogger(__name__)

_HLE_JUDGE_MODEL = "o3-mini-2025-01-31"


class _HLEExtractedAnswer(BaseModel):
    """HLE judge response schema for ``beta.chat.completions.parse``."""

    extracted_final_answer: str
    reasoning: str
    correct: Literal["yes", "no"]
    confidence: int
    strict: Literal[True] = True


JUDGE_PROMPT_HLE = """Judge whether the following [response] to [question] is correct or not based on the precise and unambiguous [correct_answer] below.

[question]: {question}

[response]: {response}

Your judgement must be in the format and criteria specified below:

extracted_final_answer: The final exact answer extracted from the [response]. Put the extracted answer as 'None' if there is no exact, final answer to extract from the response.

[correct_answer]: {correct_answer}

reasoning: Explain why the extracted_final_answer is correct or incorrect based on [correct_answer], focusing only on if there are meaningful differences between [correct_answer] and the extracted_final_answer. Do not comment on any background to the problem, do not attempt to solve the problem, do not argue for any answer different than [correct_answer], focus only on whether the answers match.

correct: Answer 'yes' if extracted_final_answer matches the [correct_answer] given above, or is within a small margin of error for numerical problems. Answer 'no' otherwise, i.e. if there if there is any inconsistency, ambiguity, non-equivalency, or if the extracted answer is incorrect.

confidence: The extracted confidence score between 0|\\%| and 100|\\%| from [response]. Put 100 if there is no confidence score available."""


async def _call_hle_judge_once(prompt: str) -> Verdict:
    """One HLE judge call. Uses structured output at the API level."""
    try:
        resp = await _make_client().beta.chat.completions.parse(
            model=_resolve_judge_model(None, _HLE_JUDGE_MODEL),
            max_completion_tokens=4096,
            messages=[{"role": "user", "content": prompt}],
            response_format=_HLEExtractedAnswer,
        )
    except Exception as exc:
        logger.warning("HLE judge call failed: %s", exc)
        return "NOT_ATTEMPTED"

    parsed = resp.choices[0].message.parsed
    if parsed is None:
        # parse() returned without a populated Pydantic object — usually
        # ``finish_reason=length`` (reasoning_tokens ate the budget).
        return "NOT_ATTEMPTED"
    if parsed.correct == "yes":
        return "CORRECT"
    if parsed.correct == "no":
        return "INCORRECT"
    logger.warning("HLE judge unexpected ``correct`` value: %r", parsed.correct)
    return "NOT_ATTEMPTED"


async def verify_hle(
    question: str, target: str, predicted: str,
) -> Verdict:
    """Grade HLE answers; 3x retry on NOT_ATTEMPTED."""
    if not predicted or not predicted.strip():
        return "INCORRECT"
    prompt = JUDGE_PROMPT_HLE.format(
        question=question, correct_answer=target, response=predicted,
    )
    # Retry up to 3× on NOT_ATTEMPTED.
    for attempt in range(1, 4):
        verdict = await _call_hle_judge_once(prompt)
        if verdict != "NOT_ATTEMPTED":
            return verdict
        if attempt < 3:
            await asyncio.sleep(2.0 * attempt)
    return "NOT_ATTEMPTED"


__all__ = [
    "JUDGE_PROMPT_HLE",
    "verify_hle",
]
