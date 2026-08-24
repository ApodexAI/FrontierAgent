"""
Extraction Pipeline v2 — verifiable eval framework

Flow:
  Step 1-2: Primary (claude-sonnet-4) + Secondary (gpt-5) per entity (concurrent).
  Step 3:   Span grounding (pure string).
  Step 4:   If needs_review → parallel re-extract with 3 more models, then the
            analyzer (default: anthropic/claude-opus-4.6, configurable) judges
            from 5 extractions + raw response.

Top-level states: "confirmed" | "needs_review".
Sub-reasons on needs_review: disagreement / possibly_hallucinated /
possibly_missed / primary_ungrounded / secondary_ungrounded / analyzer_uncertain.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import UTC, datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

try:
    from openai import OpenAI
except ImportError:
    sys.exit("[ERROR] pip install openai")


# ═══════════════════════════════════════════════════════════════════════════
# Model configuration (override via run_pipeline kwargs)
# ═══════════════════════════════════════════════════════════════════════════

DEFAULTS = {
    # Default model slate (override via run_pipeline kwargs).
    "primary_model": "anthropic/claude-sonnet-4",
    "secondary_model": "openai/gpt-5",
    "parallel_models": [
        "google/gemini-2.5-pro",
        "x-ai/grok-3",
        "deepseek/deepseek-v3.2",
    ],
    "analyzer_model": "anthropic/claude-opus-4.6",
    "temperature": 0.0,
    # Reasoning models + list-type schemas need long outputs;
    # 16384 leaves headroom for ~20-item structured lists.
    "max_tokens_extract": 16384,
    "max_tokens_analyzer": 16384,
    "concurrency": 8,
    "retry_max": 3,
    "retry_backoff_s": 2.0,
}


# ═══════════════════════════════════════════════════════════════════════════
# LLM client
# ═══════════════════════════════════════════════════════════════════════════


def _load_env(start_dir: Path) -> None:
    for p in [
        start_dir / ".env",
        start_dir.parent / ".env",
        start_dir.parent.parent / ".env",
    ]:
        if p.exists():
            for line in p.read_text(encoding="utf-8").splitlines():
                s = line.strip()
                if not s or s.startswith("#") or "=" not in s:
                    continue
                k, v = s.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip("'\""))
            return


def get_client() -> OpenAI:
    _load_env(Path(__file__).parent)
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    base_url = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    if not api_key:
        sys.exit("[ERROR] OPENROUTER_API_KEY not set.")
    return OpenAI(api_key=api_key, base_url=base_url)


def call_llm(
    client: OpenAI,
    model: str,
    system: str,
    user: str,
    max_tokens: int = 2048,
    temperature: float = 0.0,
    retry_max: int = 3,
    retry_backoff_s: float = 2.0,
) -> str:
    last_err = None
    for attempt in range(retry_max):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return resp.choices[0].message.content or ""
        except Exception as e:
            last_err = e
            if attempt < retry_max - 1:
                time.sleep(retry_backoff_s * (attempt + 1))
    raise RuntimeError(f"LLM call failed after {retry_max} retries: {last_err}")


# ═══════════════════════════════════════════════════════════════════════════
# Data classes
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class Extraction:
    """One extraction attempt on one (model_answer, entity) pair."""

    extractor_model: str  # the LLM that did the extraction
    value: Any  # structured value, or None
    not_mentioned: bool
    supporting_span: str | None
    confidence: str  # "high" | "medium" | "low"
    grounding: str = "pending"  # grounded | grounded_fuzzy | hallucinated | n/a
    raw_llm_output: str = ""


@dataclass
class EntityResult:
    """Final result for one (target model, entity) pair."""

    entity_id: str
    entity_name: str
    primary: Extraction | None = None
    secondary: Extraction | None = None
    phase4: list[Extraction] = field(default_factory=list)
    analyzer_verdict: dict | None = None
    resolution: str = "pending"  # "confirmed" | "needs_review"
    reason: str | None = None  # sub-label if needs_review
    canonical_value: Any = None
    canonical_span: str | None = None


# ═══════════════════════════════════════════════════════════════════════════
# JSON extraction helpers
# ═══════════════════════════════════════════════════════════════════════════


_CJK_RE = r"一-鿿　-〿＀-￯—…·"
# "Inside-a-string" context: alphanumeric, underscore, or CJK / CJK punct.
_INSIDE_RE = rf"[\w{_CJK_RE}]"


def _repair_unescaped_quotes(text: str) -> str:
    """Escape straight `"` that appears between two word/CJK chars — common
    when LLMs embed quoted phrases like `"do"` inside JSON string values.
    Legitimate JSON delimiters are surrounded by whitespace / `:` / `,` / `{}`
    / `[]` / escapes, so won't match."""
    return re.sub(
        rf'(?<={_INSIDE_RE})"(?={_INSIDE_RE})',
        r'\\"',
        text,
    )


def _extract_json(text: str) -> dict | None:
    """Extract a JSON object from LLM text output.

    Robust to:
      - Code fences (```json ... ```)
      - LLM "self-correction" patterns where two JSON objects are written
        (e.g. an initial wrong answer followed by a corrected second JSON);
        in that case we prefer the **last** parseable object.
      - Trailing text after the final JSON.
    """
    if not text:
        return None
    t = text.strip()
    # strip leading code fence
    t = re.sub(r"^```(?:json)?\s*", "", t)
    # strip trailing code fence
    t = re.sub(r"\s*```$", "", t)

    # 1) Try direct full-text parse
    for candidate in (t, _repair_unescaped_quotes(t)):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    # 2) Walk from RIGHT to LEFT to find the LAST balanced {...} block that
    #    parses successfully. This handles "first wrong, then corrected"
    #    LLM outputs where the final JSON is the one we want.
    open_positions: list[int] = [i for i, ch in enumerate(t) if ch == "{"]
    close_positions: list[int] = [i for i, ch in enumerate(t) if ch == "}"]
    # Try pairings of opens/closes from rightmost close downward
    for ci in reversed(close_positions):
        for oi in reversed([p for p in open_positions if p < ci]):
            snippet = t[oi : ci + 1]
            for candidate in (snippet, _repair_unescaped_quotes(snippet)):
                try:
                    parsed = json.loads(candidate)
                    if isinstance(parsed, dict):
                        return parsed
                except json.JSONDecodeError:
                    pass
    return None


# ═══════════════════════════════════════════════════════════════════════════
# Prompt templates
# ═══════════════════════════════════════════════════════════════════════════

SYSTEM_EXTRACT = "你是事实提取专家。严格按照 JSON schema 输出，不要有任何其他文字。"

PRIMARY_PROMPT_TEMPLATE = """从下面给定模型的回答中，**只**抽取关于指定实体的声明。

## 输入
- 用户原始问题: {query_text}
- 目标实体: {entity_name}
- 实体说明: {entity_description}
- 被评模型名: {target_model_name}
- 被评模型回答全文:
---
{response}
---

## 任务
严格按以下 JSON schema 输出：

{schema}

## 字段规则
- `value`: 模型明确给出的值；未提及则 null
- `not_mentioned`: 若模型回答中完全没提到该实体，true；否则 false
- `supporting_span`: 从模型原文**逐字拷贝**30–200 字直接支撑 value。**不得改写**。value 为 null 时也为 null。
- `confidence`: "high" / "medium" / "low"

## 禁止
- 不从常识补全模型没说的内容
- 不在抽取时做单位换算
- 不改写 supporting_span 使其更通顺
"""

# 副 prompt 侧重"核实模型是否真谈到此实体"，避免与主 prompt 同构导致相关误差
SECONDARY_PROMPT_TEMPLATE = """核实任务：判断下面给定模型在回答里是否**真正**谈到了指定实体。

## 输入
- 用户原始问题: {query_text}
- 要核实的实体: {entity_name}
- 实体说明: {entity_description}
- 被核实模型名: {target_model_name}
- 被核实模型回答全文:
---
{response}
---

## 任务（严格按顺序判断）
1. 在回答里**通读一遍**是否真的提到了这个实体（注意不要把相邻/近似实体混为同一个）
2. 若提到，它声称的值是什么？
3. 把原文中最能支撑你判断的一段（30–200 字）作为证据

按下列 JSON schema 输出：

{schema}

## 特别注意
- 宁可 `not_mentioned=true` 也不要硬抽一个相似但非目标实体的值
- 如果模型用了隐喻 / 间接描述，confidence 设 low
- value 和 supporting_span 要对得上；value 有、span 却证不出来，视为 low 置信
"""

ANALYZER_SYSTEM = (
    "你是仲裁者。输入是同一个 (模型, 实体) 的 5 次独立抽取 + 原文。"
    "你的任务是：综合五次抽取的多数意见与原文证据，裁决这个模型的最终抽取值。"
    "只输出 JSON，不要多余文字。"
)

ANALYZER_PROMPT_TEMPLATE = """## 背景
- 用户原始问题: {query_text}
- 目标实体: {entity_name}
- 实体说明: {entity_description}
- 被评模型名: {target_model_name}

## 被评模型的原始回答
---
{response}
---

## 五次独立抽取
{extractions_block}

## 你的任务
1. 对比五次抽取，找出多数意见 / 分歧点
2. 回到原文，自己判断"被评模型到底说了什么"
3. 给出最终 canonical 值

输出 JSON：
{{
  "final_value": <与 schema 同结构，或 null>,
  "final_not_mentioned": true|false,
  "final_span": "<原文中最可靠的一段>",
  "rationale": "<80 字以内，为什么这样裁决>",
  "final_confidence": "high|medium|low"
}}

## 原则
- 优先多数意见，但若少数派有原文精确 span 支撑，可以采少数
- 原文完全未提及 → final_not_mentioned=true, final_value=null
- 不做单位换算，保留原文表述
"""


# ═══════════════════════════════════════════════════════════════════════════
# Span grounding
# ═══════════════════════════════════════════════════════════════════════════

_WS = re.compile(r"\s+")
_PUNCT = re.compile(r"[\s　，,。.、；;：:！!？?\"'“”‘’`~]+")


def _normalize(s: str) -> str:
    s = s.lower().strip()
    s = _WS.sub(" ", s)
    return s


def _strip_punct(s: str) -> str:
    return _PUNCT.sub("", s)


def span_grounding(span: str | None, response: str) -> str:
    if not span:
        return "n/a"
    s = _normalize(span)
    r = _normalize(response)
    if s in r:
        return "grounded"
    # punctuation-insensitive
    if _strip_punct(s) in _strip_punct(r):
        return "grounded"
    # fuzzy ratio
    if (
        SequenceMatcher(None, s, r).ratio() >= 0.6
        or _best_fuzzy_substring_ratio(s, r) >= 0.8
    ):
        return "grounded_fuzzy"
    return "hallucinated"


def _best_fuzzy_substring_ratio(needle: str, haystack: str, step: int = 50) -> float:
    """Approximate best local fuzzy match of `needle` within `haystack`."""
    if not needle or not haystack or len(needle) > len(haystack):
        return 0.0
    best = 0.0
    n = len(needle)
    for i in range(0, max(1, len(haystack) - n + 1), max(1, n // 4)):
        window = haystack[i : i + int(n * 1.5)]
        if not window:
            continue
        r = SequenceMatcher(None, needle, window).ratio()
        if r > best:
            best = r
        if best >= 0.95:
            return best
    return best


# ═══════════════════════════════════════════════════════════════════════════
# Extraction primitives
# ═══════════════════════════════════════════════════════════════════════════


def _make_extract_prompt(
    template: str,
    query_text: str,
    entity: dict,
    target_model_name: str,
    response: str,
    schema: str,
) -> str:
    return template.format(
        query_text=query_text,
        entity_name=entity["name"],
        entity_description=entity.get("description", entity["name"]),
        target_model_name=target_model_name,
        response=response[:20000],  # truncate extremely long responses
        schema=schema,
    )


def _run_single_extraction(
    client: OpenAI,
    extractor_model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    retry_max: int,
    retry_backoff_s: float,
) -> Extraction:
    raw = call_llm(
        client,
        extractor_model,
        SYSTEM_EXTRACT,
        prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        retry_max=retry_max,
        retry_backoff_s=retry_backoff_s,
    )
    parsed = _extract_json(raw)
    if parsed is None:
        return Extraction(
            extractor_model=extractor_model,
            value=None,
            not_mentioned=True,
            supporting_span=None,
            confidence="low",
            grounding="n/a",
            raw_llm_output=raw,
        )
    return Extraction(
        extractor_model=extractor_model,
        value=parsed.get("value"),
        not_mentioned=bool(parsed.get("not_mentioned", parsed.get("value") is None)),
        supporting_span=parsed.get("supporting_span"),
        confidence=str(parsed.get("confidence", "medium")).lower(),
        grounding="pending",
        raw_llm_output=raw,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Step 4 resolution
# ═══════════════════════════════════════════════════════════════════════════


def _values_equivalent(a: Any, b: Any) -> bool:
    """Loose equality for extracted values (strings / numbers / None)."""
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    sa, sb = str(a).strip().lower(), str(b).strip().lower()
    if sa == sb:
        return True
    # strip common units / whitespace / punctuation
    na = re.sub(r"[\s%¥$,]", "", sa)
    nb = re.sub(r"[\s%¥$,]", "", sb)
    return na == nb


def _classify_pair(
    primary: Extraction, secondary: Extraction
) -> tuple[str, str | None]:
    """Return (resolution, reason_or_none) based on primary vs secondary."""
    p_absent = primary.not_mentioned or primary.value is None
    s_absent = secondary.not_mentioned or secondary.value is None

    p_ungrounded = (not p_absent) and primary.grounding == "hallucinated"
    s_ungrounded = (not s_absent) and secondary.grounding == "hallucinated"

    if p_ungrounded and s_ungrounded:
        return "needs_review", "both_ungrounded"
    if p_ungrounded:
        return "needs_review", "primary_ungrounded"
    if s_ungrounded:
        return "needs_review", "secondary_ungrounded"

    if p_absent and s_absent:
        return "confirmed", None  # confirmed_absent at canonical layer
    if p_absent and not s_absent:
        return "needs_review", "possibly_missed"
    if (not p_absent) and s_absent:
        return "needs_review", "possibly_hallucinated"

    if _values_equivalent(primary.value, secondary.value):
        return "confirmed", None
    return "needs_review", "disagreement"


# ═══════════════════════════════════════════════════════════════════════════
# Phase 4: analyzer
# ═══════════════════════════════════════════════════════════════════════════


def _format_extraction_for_analyzer(ex: Extraction, label: str) -> str:
    return (
        f"### {label}（{ex.extractor_model}）\n"
        f"- value: {json.dumps(ex.value, ensure_ascii=False)}\n"
        f"- not_mentioned: {ex.not_mentioned}\n"
        f"- supporting_span: {ex.supporting_span!r}\n"
        f"- confidence: {ex.confidence}\n"
        f"- grounding: {ex.grounding}\n"
    )


def _run_analyzer(
    client: OpenAI,
    analyzer_model: str,
    query_text: str,
    entity: dict,
    target_model_name: str,
    response: str,
    extractions: list[Extraction],
    max_tokens: int,
    temperature: float,
    retry_max: int,
    retry_backoff_s: float,
) -> dict:
    labels = ["主抽取", "副抽取", "并行1", "并行2", "并行3"]
    blocks = [
        _format_extraction_for_analyzer(ex, labels[i])
        for i, ex in enumerate(extractions)
    ]
    prompt = ANALYZER_PROMPT_TEMPLATE.format(
        query_text=query_text,
        entity_name=entity["name"],
        entity_description=entity.get("description", entity["name"]),
        target_model_name=target_model_name,
        response=response[:20000],
        extractions_block="\n\n".join(blocks),
    )
    raw = call_llm(
        client,
        analyzer_model,
        ANALYZER_SYSTEM,
        prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        retry_max=retry_max,
        retry_backoff_s=retry_backoff_s,
    )
    parsed = _extract_json(raw) or {}
    parsed["_raw"] = raw
    return parsed


# ═══════════════════════════════════════════════════════════════════════════
# Per-model orchestrator
# ═══════════════════════════════════════════════════════════════════════════


def _extract_for_entity(
    client: OpenAI,
    target_model_name: str,
    response: str,
    entity: dict,
    query_text: str,
    schema: str,
    primary_prompt_tpl: str,
    secondary_prompt_tpl: str,
    primary_model: str,
    secondary_model: str,
    max_tokens: int,
    temperature: float,
    retry_max: int,
    retry_backoff_s: float,
) -> tuple[Extraction, Extraction]:
    """Run primary + secondary extraction for one (target_model, entity)."""
    p_prompt = _make_extract_prompt(
        primary_prompt_tpl, query_text, entity, target_model_name, response, schema
    )
    s_prompt = _make_extract_prompt(
        secondary_prompt_tpl, query_text, entity, target_model_name, response, schema
    )
    primary = _run_single_extraction(
        client,
        primary_model,
        p_prompt,
        max_tokens,
        temperature,
        retry_max,
        retry_backoff_s,
    )
    secondary = _run_single_extraction(
        client,
        secondary_model,
        s_prompt,
        max_tokens,
        temperature,
        retry_max,
        retry_backoff_s,
    )
    primary.grounding = span_grounding(primary.supporting_span, response)
    secondary.grounding = span_grounding(secondary.supporting_span, response)
    return primary, secondary


def _phase4_for_entity(
    client: OpenAI,
    target_model_name: str,
    response: str,
    entity: dict,
    query_text: str,
    schema: str,
    primary_prompt_tpl: str,
    parallel_models: list[str],
    analyzer_model: str,
    primary: Extraction,
    secondary: Extraction,
    max_tokens_extract: int,
    max_tokens_analyzer: int,
    temperature: float,
    retry_max: int,
    retry_backoff_s: float,
    concurrency: int,
) -> tuple[list[Extraction], dict]:
    """Run 3 parallel re-extractions + analyzer. Returns (phase4_list, verdict)."""
    prompt = _make_extract_prompt(
        primary_prompt_tpl, query_text, entity, target_model_name, response, schema
    )
    phase4: list[Extraction] = []
    with ThreadPoolExecutor(max_workers=min(concurrency, len(parallel_models))) as pool:
        futures = [
            pool.submit(
                _run_single_extraction,
                client,
                m,
                prompt,
                max_tokens_extract,
                temperature,
                retry_max,
                retry_backoff_s,
            )
            for m in parallel_models
        ]
        for fut in as_completed(futures):
            ex = fut.result()
            ex.grounding = span_grounding(ex.supporting_span, response)
            phase4.append(ex)
    # stable order matching parallel_models
    phase4.sort(key=lambda e: parallel_models.index(e.extractor_model))
    all5 = [primary, secondary, *phase4]
    verdict = _run_analyzer(
        client,
        analyzer_model,
        query_text,
        entity,
        target_model_name,
        response,
        all5,
        max_tokens_analyzer,
        temperature,
        retry_max,
        retry_backoff_s,
    )
    return phase4, verdict


def _resolve_entity(
    client: OpenAI,
    target_model_name: str,
    response: str,
    entity: dict,
    query_text: str,
    schema: str,
    primary_prompt_tpl: str,
    secondary_prompt_tpl: str,
    cfg: dict,
) -> EntityResult:
    result = EntityResult(entity_id=entity["id"], entity_name=entity["name"])

    # empty response shortcut
    if not response or not response.strip():
        result.resolution = "confirmed"
        result.canonical_value = None
        result.canonical_span = None
        return result

    # Step 1-3
    primary, secondary = _extract_for_entity(
        client,
        target_model_name,
        response,
        entity,
        query_text,
        schema,
        primary_prompt_tpl,
        secondary_prompt_tpl,
        cfg["primary_model"],
        cfg["secondary_model"],
        cfg["max_tokens_extract"],
        cfg["temperature"],
        cfg["retry_max"],
        cfg["retry_backoff_s"],
    )
    result.primary, result.secondary = primary, secondary

    resolution, reason = _classify_pair(primary, secondary)
    if resolution == "confirmed":
        # Guard against agreed false-negatives on non-trivial responses:
        # if both attempts say "not mentioned" but the response is long (>500
        # chars), force phase-4 re-extraction. Common for list-type schemas
        # where extractors semantically mis-parse the entity framing.
        if (
            primary.not_mentioned
            and secondary.not_mentioned
            and len(response.strip()) > 500
        ):
            pass  # fall through to phase 4
        else:
            result.resolution = "confirmed"
            if primary.not_mentioned and secondary.not_mentioned:
                result.canonical_value = None
                result.canonical_span = None
            else:
                # primary's value wins tie (both agreed)
                result.canonical_value = primary.value
                result.canonical_span = (
                    primary.supporting_span or secondary.supporting_span
                )
            return result

    # Step 4 — parallel 3 + analyzer
    phase4, verdict = _phase4_for_entity(
        client,
        target_model_name,
        response,
        entity,
        query_text,
        schema,
        primary_prompt_tpl,
        cfg["parallel_models"],
        cfg["analyzer_model"],
        primary,
        secondary,
        cfg["max_tokens_extract"],
        cfg["max_tokens_analyzer"],
        cfg["temperature"],
        cfg["retry_max"],
        cfg["retry_backoff_s"],
        cfg["concurrency"],
    )
    result.phase4 = phase4
    result.analyzer_verdict = verdict

    # Interpret analyzer verdict. If the analyzer returned an empty /
    # unparseable JSON (common when it hedges on list-type schemas), the
    # result would silently become `confirmed absent`. Guard against that:
    # when a majority of the 5 attempts produced a non-null value, use the
    # longest value among agreeing attempts as canonical.
    all_attempts = [primary, secondary, *list(phase4)]
    non_null_attempts = [
        a for a in all_attempts if a.value is not None and not a.not_mentioned
    ]
    analyzer_effectively_empty = (
        (not verdict.get("rationale"))
        and verdict.get("final_value") is None
        and verdict.get("final_confidence") is None
    )

    if analyzer_effectively_empty and len(non_null_attempts) >= 3:
        # majority rescue: pick attempt by length AND field fill-rate.
        # Problem this solves: several extractors may return the same list
        # length but with most fields null (output token truncation). We
        # want the one with most populated detail fields, not just longest.
        def _attempt_quality(a: Extraction) -> tuple[int, int]:
            v = a.value
            if not isinstance(v, list) or not v:
                return (0, 0)
            populated_fields = sum(
                sum(1 for val in item.values() if val)
                for item in v
                if isinstance(item, dict)
            )
            return (len(v), populated_fields)

        best = max(non_null_attempts, key=_attempt_quality)
        result.resolution = "confirmed"
        result.reason = "majority_rescue"
        result.canonical_value = best.value
        result.canonical_span = best.supporting_span
    else:
        fv = verdict.get("final_value")
        fna = bool(verdict.get("final_not_mentioned", fv is None))
        fc = str(verdict.get("final_confidence", "medium")).lower()

        if fc in {"high", "medium"}:
            result.resolution = "confirmed"
            result.reason = None
            result.canonical_value = None if fna else fv
            result.canonical_span = verdict.get("final_span")
        else:
            result.resolution = "needs_review"
            result.reason = "analyzer_uncertain"
            result.canonical_value = None if fna else fv
            result.canonical_span = verdict.get("final_span")

    # Preserve original reason for debuggability
    if reason and not result.reason:
        result.reason = reason

    return result


# ═══════════════════════════════════════════════════════════════════════════
# Public entry point
# ═══════════════════════════════════════════════════════════════════════════


def load_models_input(models_input: list[str], query_id: int) -> dict[str, str]:
    """Parse CLI-style `name=path` inputs; load each model's response text.

    Each path may be:
      - a JSON list `[{"id": N, "response": "...", "report_content": "..."}, ...]`
        — the entry with `id == query_id` is chosen; prefer `report_content` else
        `response`.
      - a single JSON object `{"response": "...", "report_content": "..."}`.
      - a plain text file (treated as the response verbatim).
    """
    out: dict[str, str] = {}
    for spec in models_input:
        if "=" not in spec:
            raise ValueError(f"Expected `name=path`, got: {spec}")
        name, path = spec.split("=", 1)
        name, path = name.strip(), path.strip()
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(path)
        text = p.read_text(encoding="utf-8")
        resp = ""
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            resp = text
            out[name] = resp
            continue
        if isinstance(data, list):
            match = next((d for d in data if d.get("id") == query_id), None)
            if match is None:
                resp = ""
            else:
                resp = match.get("report_content") or match.get("response") or ""
        elif isinstance(data, dict):
            resp = data.get("report_content") or data.get("response") or ""
        else:
            resp = text
        out[name] = resp or ""
    return out


def apply_prompt_hints(
    entities: list[dict], prompt_hints: dict[str, str] | None
) -> list[dict]:
    """Inject `prompt_hints[entity_id]` as entity['description'] for use in
    prompt templates. Non-destructive: returns new list with enriched dicts."""
    if not prompt_hints:
        return entities
    enriched = []
    for e in entities:
        d = dict(e)
        hint = prompt_hints.get(e["id"])
        if hint and "description" not in d:
            d["description"] = hint
        enriched.append(d)
    return enriched


def run_pipeline(
    *,
    query_id: int,
    query_text: str,
    entities: list[dict],
    schema: str,
    models_responses: dict[str, str] | None = None,
    models_input: list[str] | None = None,
    prompt_hints: dict[str, str] | None = None,
    output_dir: Path | str,
    primary_prompt_template: str = PRIMARY_PROMPT_TEMPLATE,
    secondary_prompt_template: str = SECONDARY_PROMPT_TEMPLATE,
    primary: str | None = None,
    secondary: str | None = None,
    parallel: list[str] | None = None,
    analyzer: str | None = None,
    concurrency: int | None = None,
    overrides: dict | None = None,
    verbose: bool = True,
) -> dict[str, dict]:
    """Run the full extraction pipeline for all (model, entity) pairs.

    Input styles (one required):
      - `models_responses`: {model_name: raw_text} — direct.
      - `models_input`: ["name=path", ...] — load via load_models_input().

    Model kwargs (all optional, fall back to DEFAULTS):
      primary / secondary / parallel / analyzer / concurrency

    Returns {model_name: payload} and writes auto_scores/{model}/extraction.json.
    """
    if models_responses is None and models_input is None:
        raise ValueError("Provide models_responses or models_input.")
    if models_responses is None:
        models_responses = load_models_input(models_input or [], query_id)
    entities = apply_prompt_hints(entities, prompt_hints)

    cfg = {**DEFAULTS, **(overrides or {})}
    if primary is not None:
        cfg["primary_model"] = primary
    if secondary is not None:
        cfg["secondary_model"] = secondary
    if parallel is not None:
        cfg["parallel_models"] = parallel
    if analyzer is not None:
        cfg["analyzer_model"] = analyzer
    if concurrency is not None:
        cfg["concurrency"] = concurrency
    client = get_client()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results: dict[str, dict] = {}
    started = datetime.now(UTC).isoformat()

    def _run_one_model(
        model_name: str, response: str
    ) -> tuple[str, list[EntityResult]]:
        if verbose:
            print(
                f"\n── target model: {model_name} (resp len={len(response or '')}) ──"
            )
        entity_results: list[EntityResult] = []
        with ThreadPoolExecutor(max_workers=cfg["concurrency"]) as pool:
            futures = {
                pool.submit(
                    _resolve_entity,
                    client,
                    model_name,
                    response or "",
                    ent,
                    query_text,
                    schema,
                    primary_prompt_template,
                    secondary_prompt_template,
                    cfg,
                ): ent
                for ent in entities
            }
            for fut in as_completed(futures):
                ent = futures[fut]
                try:
                    er = fut.result()
                except Exception as e:
                    er = EntityResult(
                        entity_id=ent["id"],
                        entity_name=ent["name"],
                        resolution="needs_review",
                        reason=f"exception:{type(e).__name__}",
                    )
                if verbose:
                    print(
                        f"  [{model_name}|{er.entity_id}] {er.resolution}"
                        f"{f' ({er.reason})' if er.reason else ''}"
                    )
                entity_results.append(er)
        entity_results.sort(
            key=lambda r: [e["id"] for e in entities].index(r.entity_id)
        )
        return model_name, entity_results

    def _persist_model(model_name: str, entity_results: list[EntityResult]) -> dict:
        summary = _summarize(entity_results)
        payload = {
            "query_id": query_id,
            "model": model_name,
            "extraction_version": "v2",
            "extracted_at": started,
            "models_used": {
                "primary": cfg["primary_model"],
                "secondary": cfg["secondary_model"],
                "parallel": cfg["parallel_models"],
                "analyzer": cfg["analyzer_model"],
            },
            "entities": [_entity_to_jsonable(er) for er in entity_results],
            "summary": summary,
        }
        model_dir = output_dir / model_name
        model_dir.mkdir(parents=True, exist_ok=True)
        (model_dir / "extraction.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        nr = [e for e in payload["entities"] if e["resolution"] == "needs_review"]
        (model_dir / "needs_review.json").write_text(
            json.dumps(nr, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return payload

    # Model-level parallelism. Write results as each model completes so that
    # incremental progress is visible on disk (useful for long runs / debug).
    outer = max(1, min(len(models_responses), max(2, cfg["concurrency"] // 2)))
    with ThreadPoolExecutor(max_workers=outer) as outer_pool:
        futures = {
            outer_pool.submit(_run_one_model, name, resp): name
            for name, resp in models_responses.items()
        }
        for fut in as_completed(futures):
            name, results = fut.result()
            all_results[name] = _persist_model(name, results)

    return all_results


def _summarize(entities: list[EntityResult]) -> dict:
    total = len(entities)
    confirmed = sum(1 for e in entities if e.resolution == "confirmed")
    absent = sum(
        1 for e in entities if e.resolution == "confirmed" and e.canonical_value is None
    )
    needs_review = sum(1 for e in entities if e.resolution == "needs_review")
    reason_counts: dict[str, int] = {}
    for e in entities:
        if e.resolution == "needs_review" and e.reason:
            reason_counts[e.reason] = reason_counts.get(e.reason, 0) + 1
    return {
        "total_entities": total,
        "confirmed": confirmed,
        "confirmed_absent": absent,
        "needs_review": needs_review,
        "needs_review_reasons": reason_counts,
    }


def _entity_to_jsonable(er: EntityResult) -> dict:
    """Output schema:
    canonical {value, span, rationale?} + flat all_attempts list."""

    def attempt(ex: Extraction, role: str) -> dict:
        return {
            "model": ex.extractor_model,
            "role": role,
            "value": ex.value,
            "not_mentioned": ex.not_mentioned,
            "span": ex.supporting_span,
            "confidence": ex.confidence,
            "grounded": ex.grounding in {"grounded", "grounded_fuzzy"},
            "grounding_detail": ex.grounding,
        }

    attempts: list[dict] = []
    if er.primary is not None:
        attempts.append(attempt(er.primary, "primary"))
    if er.secondary is not None:
        attempts.append(attempt(er.secondary, "secondary"))
    for i, ex in enumerate(er.phase4, start=1):
        attempts.append(attempt(ex, f"parallel-{i}"))

    canonical: dict = {
        "value": er.canonical_value,
        "span": er.canonical_span,
    }
    if er.analyzer_verdict:
        canonical["rationale"] = er.analyzer_verdict.get("rationale")
        canonical["analyzer_confidence"] = er.analyzer_verdict.get("final_confidence")

    return {
        "id": er.entity_id,
        "name": er.entity_name,
        "resolution": er.resolution,
        "reason": er.reason,
        "canonical": canonical,
        "all_attempts": attempts,
    }
