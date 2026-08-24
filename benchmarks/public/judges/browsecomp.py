"""BrowseComp judges (English + Chinese).

Prompts follow the published BrowseComp / BrowseComp-ZH templates.
Chinese variant pins ``gpt-4o`` with the standard sampling params.
"""

from __future__ import annotations

import asyncio
import logging
import re

from benchmarks.public.judges._common import (
    Verdict,
    _build_judge_kwargs,
    _judge_call_with_effort_fallback,
    _make_client,
)

logger = logging.getLogger(__name__)


JUDGE_PROMPT_BROWSECOMP_OFFICIAL = """Judge whether the following [response] to [question] is correct or not based on the precise and unambiguous [correct_answer] below.

[question]: {question}

[response]: {response}

Your judgement must be in the format and criteria specified below:

extracted_final_answer: The final exact answer extracted from the [response]. Put the extracted answer as 'None' if there is no exact, final answer to extract from the response.

[correct_answer]: {correct_answer}

reasoning: Explain why the extracted_final_answer is correct or incorrect based on [correct_answer], focusing only on if there are meaningful differences between [correct_answer] and the extracted_final_answer. Do not comment on any background to the problem, do not attempt to solve the problem, do not argue for any answer different than [correct_answer], focus only on whether the answers match.

correct: Answer 'yes' if extracted_final_answer matches the [correct_answer] given above, or is within a small margin of error for numerical problems. Answer 'no' otherwise, i.e. if there if there is any inconsistency, ambiguity, non-equivalency, or if the extracted answer is incorrect.

confidence: The extracted confidence score between 0|\\%| and 100|\\%| from [response]. Put 100 if there is no confidence score available."""


JUDGE_PROMPT_BC_ZH = """根据以下精确且明确的[response]，判断以下对[question]的[correct_answer]是否正确。

[question]:  {question}

[response]:  {response}

您的判断必须符合以下指定的格式和标准：

extracted_final_answer: 从[response]中提取的最终准确答案。如果无法从答案中提取出准确的最终答案，则将提取的答案填写为"None"。

[correct_answer]: {correct_answer}

reasoning: 根据[correct_answer]解释提取的最终答案正确或错误的原因， 仅关注[correct_answer]和提取的最终答案之间是否存在有意义的差异。请勿评论问题的任何背景，请勿尝试解决问题，请勿争论任何与[correct_answer]不同的答案，仅关注答案是否匹配。

correct: 如果提取的最终答案与上面给出的[correct_answer]相符，或者在数值问题的误差范围内，则回答"yes"。否则，例如，如果存在任何不一致、歧义、不等同，或者提取的答案不正确，则回答"no"。

confidence: 从[response]中提取的置信度分数，介于0% 到100% 之间。如果没有可用的置信度分数，则填写100%。

"""


_BC_JUDGE_MODEL = "gpt-4.1-2025-04-14"
_BC_ZH_JUDGE_MODEL = "gpt-4o"


async def _call_browsecomp_official_once(prompt: str) -> Verdict:
    """One judge call. Parses ``correct: yes|no`` from the structured output."""
    kwargs = _build_judge_kwargs(
        prompt, max_completion_tokens=16384, model_override=_BC_JUDGE_MODEL,
    )
    content = await _judge_call_with_effort_fallback(kwargs, label="BrowseComp")
    if not content:
        return "NOT_ATTEMPTED"
    m = re.search(r"correct\s*:\s*(yes|no)\b", content, re.IGNORECASE)
    if not m:
        logger.warning(
            "BrowseComp judge missing 'correct: yes|no' line: %r",
            content[-300:],
        )
        return "NOT_ATTEMPTED"
    return "CORRECT" if m.group(1).lower() == "yes" else "INCORRECT"


async def verify_browsecomp(
    question: str, target: str, predicted: str,
) -> Verdict:
    """Grade BrowseComp answers; 10x retry on NOT_ATTEMPTED."""
    prompt = JUDGE_PROMPT_BROWSECOMP_OFFICIAL.format(
        question=question, correct_answer=target, response=predicted,
    )
    for attempt in range(1, 11):
        verdict = await _call_browsecomp_official_once(prompt)
        if verdict != "NOT_ATTEMPTED":
            return verdict
        if attempt < 10:
            logger.info(
                "BrowseComp judge NOT_ATTEMPTED — retrying %d/10", attempt,
            )
            await asyncio.sleep(5.0)
    return "NOT_ATTEMPTED"


async def _call_browsecomp_zh_once(prompt: str) -> Verdict:
    """One BC-ZH judge call."""
    try:
        resp = await _make_client().chat.completions.create(
            model=_BC_ZH_JUDGE_MODEL,
            messages=[
                {"role": "system", "content": "you are a helpful assistant!"},
                {"role": "user", "content": prompt},
            ],
            temperature=0.6,
            top_p=0.95,
            max_completion_tokens=16384,
        )
        content = (resp.choices[0].message.content or "").strip()
    except Exception as exc:
        logger.error("BC-ZH judge call error: %s", exc)
        return "NOT_ATTEMPTED"
    if not content:
        return "NOT_ATTEMPTED"
    m = re.search(r"correct\s*:\s*(yes|no)\b", content, re.IGNORECASE)
    if not m:
        logger.warning(
            "BC-ZH judge missing 'correct: yes|no' line: %r", content[-300:],
        )
        return "NOT_ATTEMPTED"
    return "CORRECT" if m.group(1).lower() == "yes" else "INCORRECT"


async def verify_browsecomp_zh(
    question: str, target: str, predicted: str,
) -> Verdict:
    """Grade BrowseComp-ZH answers; 10x retry on NOT_ATTEMPTED."""
    prompt = JUDGE_PROMPT_BC_ZH.format(
        question=question, response=predicted, correct_answer=target,
    )
    for attempt in range(1, 11):
        verdict = await _call_browsecomp_zh_once(prompt)
        if verdict != "NOT_ATTEMPTED":
            return verdict
        if attempt < 10:
            logger.info(
                "BC-ZH judge NOT_ATTEMPTED — retrying %d/10", attempt,
            )
            await asyncio.sleep(5.0)
    return "NOT_ATTEMPTED"


__all__ = [
    "JUDGE_PROMPT_BC_ZH",
    "JUDGE_PROMPT_BROWSECOMP_OFFICIAL",
    "verify_browsecomp",
    "verify_browsecomp_zh",
]
