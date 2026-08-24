"""DeepSearchQA judge."""

from __future__ import annotations

import asyncio
import logging
import re

from benchmarks.public.judges._common import (
    Verdict,
    _build_judge_kwargs,
    _judge_call_with_effort_fallback,
)

logger = logging.getLogger(__name__)

_DSQA_DEFAULT_JUDGE_MODEL = "google/gemini-2.5-flash"  # fast, cheap, set-aware


# DSQA splits questions into ``Single Answer`` (one expected answer, lenient
# match) and ``Set Answer`` (a list of items; *every* expected item must be
# present and there must be no excessive answers). The judge emits a JSON
# block with ``Correctness Details`` (per-item booleans) and
# ``Excessive Answers`` (extras the model added). CORRECT iff *all* expected
# items are true AND ``Excessive Answers`` is empty.

JUDGE_PROMPT_DEEPSEARCHQA = """Your task is to evaluate whether a given "AI Response" for a specific "User Prompt" arrived at the correct answer.

**Answer Correctness Task**

*   **Purpose:** Assess whether the AI response provides the correct answer(s) based on the provided "Correct Answer" and "Prompt Type".
*   **Process:**
    *   Identify the "Prompt Type": "<prompt_type>".
    *   Refer to the "Correct Answer": "<answer>".
    *   Based on the "Prompt Type", determine if the "AI Response" contains the expected answer(s).
        *   **'Single Answer'**: Check if the response provides the answer that addresses the user's question. It does not have to match the exact wording of the provided answer.
        *   **'Set Answer'**: Check if the response includes *each* item from the provided ground truth answers. The order might not matter unless specified otherwise. The response might include more answers than the list. Determine the correctness *only* based on the list first and then check if the response includes answers not in the list.
    *   **Explanation:** Provide a brief explanation justifying your assessment of answer correctness, referencing specific parts of the AI response and the correct answer.
    *   **Correctness Details:** Provide a dictionary, one key for each expected answer part, and value is a boolean indicating whether each expected answer part was found.
        *   For 'Set Answer', this will be a list of attributes, one for each item/part in the "Correct Answer". Each key will be a string indicating the expected answer part, and the value will be a boolean indicating whether that part was found in the response.
    *   **Excessive Answers:** Provide a list of strings, each indicating an excessive answer part. If the response provides answers that are **not** in the "Correct Answer" list, add these answers as excessive answers. Return an empty list when there's no excessive answers in the response.


**Output Format:**

Your evaluation *must* be structured as a nested JSON dictionary with the following top-level keys: `"Answer Correctness"`. Please return NULL if any of "Prompt", "AI Response" or "Correct Answer" is empty.
The value for `"Answer Correctness"` should be a dictionary containing `"Explanation"` (a string), `"Correctness Details"` (a dictionary where each key is the expected correct answer, and the value is a boolean indicating whether the response contains the correct answer), and `"Excessive Answers"` (a list of strings indicating the excessive answers).

Make sure you return a valid JSON string. Pay special attention to quotes, commas and special characters in the JSON string. Make sure to escape all special characters and quotes in the JSON string.


**Example (Partial):**

"```json
{{
  "Answer Correctness": {{
    "Explanation": "The response correctly identified Belgium and France but also includes an excessive answer, Italy.",
    "Correctness Details": {{
      "Belgium": true,
      "France": true,
    }},
    "Excessive Answers": [ "Italy" ]
  }}
}}
```"

**Now, proceed with the evaluation using the provided User Prompt, AI Response, and Correct Answer.**

User Prompt (Wrapped in <prompt> and </prompt>):
<prompt>
{prompt}
</prompt>
--------------------
**  Correct Answer (Wrapped in <answer> and </answer>):
Prompt Type: {prompt_type}
<answer>
{answer}
</answer>
--------------------
AI assistant response (Wrapped in <response> and </response>):
<response>
{response}
</response>

--------------------
Rating:"""


async def _call_deepsearchqa_judge_once(prompt: str) -> tuple[Verdict, dict | None]:
    """One DSQA judge call. Returns ``(verdict, details_dict)`` where
    ``details_dict`` carries the parsed per-item info needed by
    ``score_deepsearchqa`` (None on parse failure).

    ``details_dict`` keys: ``correctness`` (dict[str, bool]) and
    ``excessive`` (list[str]). CORRECT iff every value in ``correctness``
    is True AND ``excessive`` is empty (matches the paper's "Fully Correct"
    category, S = G).
    """
    import json as _json

    kwargs = _build_judge_kwargs(
        prompt, max_completion_tokens=8192,
        default_model=_DSQA_DEFAULT_JUDGE_MODEL,
    )
    content = await _judge_call_with_effort_fallback(kwargs, label="DSQA")
    if not content:
        return "NOT_ATTEMPTED", None

    text = content
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence:
        text = fence.group(1)
    elif (start := text.find("{")) != -1 and (end := text.rfind("}")) != -1 and end > start:
        text = text[start:end + 1]

    try:
        parsed = _json.loads(text)
    except _json.JSONDecodeError:
        logger.warning("DSQA judge output not parseable JSON: %r", content[:300])
        return "NOT_ATTEMPTED", None

    ac = parsed.get("Answer Correctness")
    if not isinstance(ac, dict):
        logger.warning("DSQA judge missing 'Answer Correctness': %r", parsed)
        return "NOT_ATTEMPTED", None
    details_raw = ac.get("Correctness Details") or {}
    excessive_raw = ac.get("Excessive Answers") or []
    if not isinstance(details_raw, dict) or not details_raw:
        return "NOT_ATTEMPTED", None
    correctness = {str(k): bool(v) for k, v in details_raw.items()}
    excessive = [str(x) for x in excessive_raw] if isinstance(excessive_raw, list) else []
    all_correct = all(correctness.values())
    verdict: Verdict = "CORRECT" if (all_correct and not excessive) else "INCORRECT"
    return verdict, {"correctness": correctness, "excessive": excessive}


def _classify_dsqa(tp: int, fp: int, fn: int) -> str:
    """Classify a DSQA trial into one of four disjoint categorical buckets
    per the paper (Section 3.1, "Categorical Classification"):

    * ``fully_correct``  — S == G          (tp >= 1, fp == 0, fn == 0)
    * ``fully_incorrect`` — S ∩ G == ∅      (tp == 0)
    * ``partially_correct`` — ∅ ≠ S∩G ⊂ G  (tp >= 1, fn >= 1)
    * ``extraneous``     — G ⊂ S            (tp >= 1, fp >= 1, fn == 0)
    """
    if tp == 0:
        return "fully_incorrect"
    if fn == 0 and fp == 0:
        return "fully_correct"
    if fn == 0 and fp > 0:
        return "extraneous"
    return "partially_correct"


def _compute_dsqa_metrics(details: dict) -> dict:
    """Turn a parsed judge response into P / R / F1 + categorical label.

    ``details`` is the second element of ``_call_deepsearchqa_judge_once``'s
    return: ``{"correctness": {item: bool}, "excessive": [str]}``.

    Returns ``{precision, recall, f1, category, tp, fp, fn}``.
    """
    correctness = details["correctness"]
    excessive = details["excessive"]
    tp = sum(1 for v in correctness.values() if v)
    fn = sum(1 for v in correctness.values() if not v)
    fp = len(excessive)
    submitted = tp + fp        # |S_i|
    expected = tp + fn         # |G_i|
    precision = tp / submitted if submitted else 0.0
    recall = tp / expected if expected else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "category": _classify_dsqa(tp, fp, fn),
        "tp": tp, "fp": fp, "fn": fn,
    }


async def verify_deepsearchqa(
    question: str, target: str, predicted: str, *, prompt_type: str = "Single Answer",
) -> Verdict:
    """Binary CORRECT/INCORRECT verdict (= the paper's "Fully Correct" category).

    Retries on NOT_ATTEMPTED up to 3 times, then gives up.
    """
    if not predicted or not predicted.strip():
        return "INCORRECT"
    prompt = JUDGE_PROMPT_DEEPSEARCHQA.format(
        prompt_type=prompt_type,
        prompt=question,
        answer=target,
        response=predicted,
    )
    for attempt in range(1, 4):
        verdict, _ = await _call_deepsearchqa_judge_once(prompt)
        if verdict != "NOT_ATTEMPTED":
            return verdict
        if attempt < 3:
            await asyncio.sleep(2.0 * attempt)
    return "NOT_ATTEMPTED"


async def score_deepsearchqa(
    question: str, target: str, predicted: str,
    *, prompt_type: str = "Single Answer",
) -> tuple[Verdict, float | None]:
    """Returns ``(verdict, f1)`` per the DeepSearchQA paper, Section 3.1.

    F1 is the paper's primary ranking metric. The accompanying per-trial
    breakdown — precision, recall, categorical label, TP/FP/FN — is
    accessible via :func:`score_deepsearchqa_full` for callers that need it.
    """
    verdict, metrics = await score_deepsearchqa_full(
        question, target, predicted, prompt_type=prompt_type,
    )
    return verdict, (metrics["f1"] if metrics else None)


async def score_deepsearchqa_full(
    question: str, target: str, predicted: str,
    *, prompt_type: str = "Single Answer",
) -> tuple[Verdict, dict | None]:
    """Same evaluator as ``score_deepsearchqa`` but returns the full metrics
    dict (precision, recall, f1, category, tp, fp, fn). Returns ``(verdict, None)``
    when the judge fails to produce a parseable response after retries.
    """
    if not predicted or not predicted.strip():
        return "INCORRECT", {
            "precision": 0.0, "recall": 0.0, "f1": 0.0,
            "category": "fully_incorrect", "tp": 0, "fp": 0, "fn": 0,
        }
    prompt = JUDGE_PROMPT_DEEPSEARCHQA.format(
        prompt_type=prompt_type, prompt=question, answer=target, response=predicted,
    )
    for attempt in range(1, 4):
        verdict, details = await _call_deepsearchqa_judge_once(prompt)
        if verdict != "NOT_ATTEMPTED" and details is not None:
            return verdict, _compute_dsqa_metrics(details)
        if attempt < 3:
            await asyncio.sleep(2.0 * attempt)
    return "NOT_ATTEMPTED", None


__all__ = [
    "JUDGE_PROMPT_DEEPSEARCHQA",
    "score_deepsearchqa",
    "score_deepsearchqa_full",
    "verify_deepsearchqa",
]
