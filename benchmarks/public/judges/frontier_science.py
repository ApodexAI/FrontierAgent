"""FrontierScience judge — research (rubric) + olympiad (answer-match)."""

from __future__ import annotations

import logging
import re

from benchmarks.public.judges._common import (
    Verdict,
    _build_judge_kwargs,
    _judge_call_with_effort_fallback,
)

logger = logging.getLogger(__name__)

_FS_DEFAULT_JUDGE_MODEL = "openai/gpt-5"     # rubric / olympiad answer matching
_FS_DEFAULT_REASONING_EFFORT = "high"        # gpt-5 thinking on
_FS_PASS_THRESHOLD = 7.0
_FS_JUDGE_TRIALS = 1


async def _call_rubric_judge_once(prompt: str) -> float | None:
    """One rubric judge call expecting a ``VERDICT: X`` line at the end.

    ``max_completion_tokens`` is set high (64K) so chain-of-thought
    over ~10 rubric items has room before the verdict.

    Effort fallback: empty completions step down high → medium → low.
    """
    kwargs = _build_judge_kwargs(
        prompt, max_completion_tokens=65536,
        default_model=_FS_DEFAULT_JUDGE_MODEL,
        default_reasoning_effort=_FS_DEFAULT_REASONING_EFFORT,
    )
    content = await _judge_call_with_effort_fallback(kwargs, label="Rubric")
    if not content:
        return None
    match = re.search(r"VERDICT:\s*(\d+(?:\.\d+)?)", content)
    if not match:
        logger.warning("Rubric judge output missing VERDICT: %r", content[-200:])
        return None
    return float(match.group(1))


# ─── FrontierScience prompt (from paper Appendix B) ───────────────────────

JUDGE_PROMPT_FS = """You are grading a science exam. You will be given the problem, attempted answer, and a rubric to grade the answer. The rubric will total up to 10 points. Evaluate the attempted answer against the provided rubric. Pay close attention to detail and grade it strictly, but fairly. Only evaluate against the rubric, as you yourself should not make any judgements (e.g., even if you think the answer is correct but rubric is wrong, you should treat the rubric as the gold standard). Return the absolute total number of points earned (it can be a decimal based on the rubric). First, think step-by-step about each rubric item. Explain your reasoning for each rubric item. Then, tally the points up and write VERDICT: {{total_points}} in the last line of your response, no other text. For example, VERDICT: 2.5 or VERDICT: 8.

Problem:
{question}

Rubric:
{rubric}

Attempted Answer:
{answer}""".strip()


# Olympiad split — direct answer matching (vs Research split's rubric scoring).
# Prompt follows FrontierScience paper Appendix B.
JUDGE_PROMPT_FS_OLYMPIAD = (
    "You are grading an attempted answer to a science olympiad problem. "
    "You will be given the problem, attempted answer, and reference answer. "
    "Evaluate the solution against the provided reference solution, ensuring "
    "it is complete and matches the reference solution. Pay close attention "
    "to detail and grade it strictly, but fairly.\n\n"
    "The reference answer is either a single number or expression in latex "
    "formatting, a chemical formula, a compound name, or a phrase referring "
    "to a specific name, entity, or method.\n\n"
    "Mark the attempted answer as correct if it fully matches the reference "
    "answer or is otherwise equivalent (e.g., an equivalent algebraic "
    "expression, a numerical number within 1 decimal place rounding of the "
    "reference answer (e.g., 6.69 ≈ 6.7), an equivalent name for a "
    "compound/formula, equivalent when accounting for units, etc.). Mark it "
    "as incorrect if it is not equivalent to the reference answer.\n"
    "***\n"
    "The problem: {problem}\n"
    "***\n"
    "The reference answer: {reference_answer}\n"
    "***\n"
    "The attempted answer: {answer}\n"
    "***\n"
    "First, think step-by-step about whether the attempted answer matches "
    "the reference answer. If the attempted answer is correct, write "
    '"VERDICT: CORRECT" in the last line of your response, with no other '
    'text or formatting. If it is incorrect, write "VERDICT: INCORRECT".'
)


async def verify_frontier_science(
    question: str, target: str, predicted: str,
) -> Verdict:
    """Grade FrontierScience-Research answers via rubric scoring.

    Pass threshold: avg score >= 7.0 over ``_FS_JUDGE_TRIALS`` independent
    judge calls.
    """
    if not predicted or not predicted.strip():
        return "INCORRECT"

    n_trials = _FS_JUDGE_TRIALS
    prompt = JUDGE_PROMPT_FS.format(
        question=question, rubric=target, answer=predicted,
    )

    scores: list[float] = []
    for _ in range(n_trials):
        score = await _call_rubric_judge_once(prompt)
        if score is not None:
            scores.append(score)

    if not scores:
        logger.warning("FrontierScience judge: all %d trial(s) failed to parse", n_trials)
        return "NOT_ATTEMPTED"

    avg = sum(scores) / len(scores)
    logger.info(
        "FrontierScience judge: trials=%d scores=%s avg=%.2f pass=%s",
        len(scores), scores, avg, avg >= _FS_PASS_THRESHOLD,
    )
    return "CORRECT" if avg >= _FS_PASS_THRESHOLD else "INCORRECT"


async def _call_olympiad_verdict_once(prompt: str) -> Verdict:
    """One Olympiad-style judge call expecting ``VERDICT: CORRECT|INCORRECT``."""
    kwargs = _build_judge_kwargs(
        prompt, max_completion_tokens=32768,
        default_model=_FS_DEFAULT_JUDGE_MODEL,
        default_reasoning_effort=_FS_DEFAULT_REASONING_EFFORT,
    )
    content = await _judge_call_with_effort_fallback(kwargs, label="Olympiad")
    if not content:
        return "NOT_ATTEMPTED"
    # Walk lines bottom-up looking for the VERDICT marker.
    for line in reversed(content.splitlines()):
        up = line.strip().upper()
        if "VERDICT" not in up:
            continue
        if "INCORRECT" in up:
            return "INCORRECT"
        if "CORRECT" in up:
            return "CORRECT"
    logger.warning("Olympiad judge output missing VERDICT: %r", content[-200:])
    return "NOT_ATTEMPTED"


async def verify_frontier_science_olympiad(
    question: str, target: str, predicted: str,
) -> Verdict:
    """Grade FrontierScience-Olympiad answers via direct answer matching.

    Binary CORRECT/INCORRECT; majority vote over ``_FS_JUDGE_TRIALS`` calls.
    """
    if not predicted or not predicted.strip():
        return "INCORRECT"
    n_trials = _FS_JUDGE_TRIALS
    prompt = JUDGE_PROMPT_FS_OLYMPIAD.format(
        problem=question, reference_answer=target, answer=predicted,
    )
    verdicts: list[Verdict] = []
    for _ in range(n_trials):
        v = await _call_olympiad_verdict_once(prompt)
        if v != "NOT_ATTEMPTED":
            verdicts.append(v)
    if not verdicts:
        logger.warning("Olympiad judge: all %d trial(s) failed to parse", n_trials)
        return "NOT_ATTEMPTED"
    correct = sum(1 for v in verdicts if v == "CORRECT")
    logger.info(
        "Olympiad judge: trials=%d correct=%d/%d → %s",
        len(verdicts), correct, len(verdicts),
        "CORRECT" if correct * 2 > len(verdicts) else "INCORRECT",
    )
    return "CORRECT" if correct * 2 > len(verdicts) else "INCORRECT"


async def score_frontier_science(
    question: str, target: str, predicted: str,
) -> tuple[Verdict, float | None]:
    """Same as ``verify_frontier_science`` plus the raw rubric score (0-10)."""
    if not predicted or not predicted.strip():
        return "INCORRECT", 0.0

    n_trials = _FS_JUDGE_TRIALS
    prompt = JUDGE_PROMPT_FS.format(
        question=question, rubric=target, answer=predicted,
    )

    scores: list[float] = []
    for _ in range(n_trials):
        score = await _call_rubric_judge_once(prompt)
        if score is not None:
            scores.append(score)

    if not scores:
        logger.warning("FrontierScience judge: all %d trial(s) failed to parse", n_trials)
        return "NOT_ATTEMPTED", None

    avg = sum(scores) / len(scores)
    logger.info(
        "FrontierScience judge: trials=%d scores=%s avg=%.2f pass=%s",
        len(scores), scores, avg, avg >= _FS_PASS_THRESHOLD,
    )
    return ("CORRECT" if avg >= _FS_PASS_THRESHOLD else "INCORRECT"), avg


__all__ = [
    "JUDGE_PROMPT_FS",
    "JUDGE_PROMPT_FS_OLYMPIAD",
    "score_frontier_science",
    "verify_frontier_science",
    "verify_frontier_science_olympiad",
]
