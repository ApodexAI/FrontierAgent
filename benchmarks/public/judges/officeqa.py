"""Deterministic OfficeQA official reward wrapper."""

from __future__ import annotations

from benchmarks.public.judges._common import Verdict
from benchmarks.public.judges.reward import score_answer as official_score


async def score_officeqa(
    question: str,
    target: str,
    predicted: str,
) -> tuple[Verdict, float | None]:
    del question
    score = official_score(target or "", predicted or "")
    return ("CORRECT" if score == 1 else "INCORRECT"), score
