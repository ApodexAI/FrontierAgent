"""xbench-DeepResearch judge."""

from __future__ import annotations

import asyncio
import logging
import re

from benchmarks.public.judges._common import Verdict, _make_client, _resolve_judge_model

logger = logging.getLogger(__name__)

_XBENCH_JUDGE_MODEL = "google/gemini-2.0-flash-001"


LLM_JUDGE_PROMPT = """
你是一个通用人工智能助手。根据下面给出的[正确答案], 判断以下对[原问题]的[回答]的回答是否正确。

[原问题]: {question}

[正确答案]: {correct_answer}

[回答]:{response}

你的判断必须按照以下格式和标准进行:

最终答案: 从[回答]中提取出的最终准确答案。如果[回答]中没有明确的最终答案, 则填写'无'。

解释: 根据[正确]解释为什么[最终答案]是正确的或错误的。只关注[最终答案]与[正确答案]之间是否存在实质性差异, 不要评论题目的背景, 不要尝试重新解题, 不要为任何不同于[正确答案]的答案辩护, 只专注于判断答案是否一致。

结论: 如果[最终答案]与上方给出的[正确答案]一致, 或者在数值题目中处于可接受的微小误差范围内, 则填写'正确'; 否则（即存在任何不一致、歧义、不等价或提取出的答案错误的情况）填写'错误'。
""".strip()


def _parse_match(m: re.Match | None) -> str | None:
    if m is None:
        return None
    raw = m.group(0)
    try:
        return raw.split(":", 1)[1].strip()
    except (IndexError, ValueError):
        return raw


async def verify_xbench(
    question: str, target: str, predicted: str,
) -> Verdict:
    """xbench grader.

    Try strict match on the final-answer marker first; on miss, call
    the LLM judge and parse the verdict marker.
    """
    if not predicted or not predicted.strip():
        return "INCORRECT"

    simple = _parse_match(re.search(r"最终答案:*(.*)", predicted))
    if simple is not None and simple == target:
        return "CORRECT"

    prompt = LLM_JUDGE_PROMPT.format(
        question=question, correct_answer=target, response=predicted,
    )

    for attempt in range(1, 4):
        try:
            resp = await _make_client().chat.completions.create(
                model=_resolve_judge_model(None, _XBENCH_JUDGE_MODEL),
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=2048,
            )
            content = (resp.choices[0].message.content or "").strip()
        except Exception as exc:
            logger.warning(
                "xbench judge call failed (attempt %d/3, model=%s): %s",
                attempt, _XBENCH_JUDGE_MODEL, exc,
            )
            content = ""

        if content:
            m = re.search(r"结论\s*[:：]\s*[\s\S]?\s*(正确|错误)", content)
            if m:
                return "CORRECT" if m.group(1) == "正确" else "INCORRECT"
            logger.warning(
                "xbench judge missing 结论 line: %r", content[-200:],
            )
        if attempt < 3:
            await asyncio.sleep(2.0 * attempt)
    return "NOT_ATTEMPTED"


__all__ = ["LLM_JUDGE_PROMPT", "verify_xbench"]
