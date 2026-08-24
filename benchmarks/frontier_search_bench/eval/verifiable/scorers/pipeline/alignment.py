"""
Canonical-ID alignment layer for verifiable evals.

Purpose
-------
Given a per-model list of extracted claims (already structured by
`extraction_pipeline.py`), align each claim to an entry in a curated
baseline table (e.g. Q06 D1-D43, Q10 BASELINE/PARTIAL films) or to null
when no entry fits.

Design rationale
----------------
Replaces brittle keyword/exact matching done inside scorers with a
multi-model LLM vote + programmatic field cross-check + kw sanity, with
a judge LLM disambiguating ambiguous cases.

Public API
----------
- `align_claims(client, *, claims_by_model, baselines, query_text, cfg,
                output_dir)` — top-level; writes `{model}/alignment.json`
  and returns the same structure.
- `BASELINE_ENTRY_SCHEMA` — the shape expected in `baselines`.

Each aligned claim has the shape::

    {
      "canonical_id": "D5" | null,
      "alignment_confidence": "high" | "medium" | "low" | "needs_review",
      "alignment_reasoning": str,
      "votes": [{model, canonical_id, confidence, reasoning}, ...],
      "field_check": {hits, misses, unknowns, detail},
      "kw_match_id": "D5" | null,
      "judge_invoked": bool,
      "judge_verdict": {final_canonical_id, final_confidence, rationale} | None,
      "raw": {...original claim dict...},
    }
"""

from __future__ import annotations

import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS.parent))

from pipeline.extraction_pipeline import _extract_json, call_llm  # noqa: E402

# ═══════════════════════════════════════════════════════════════════════════
# Defaults (mirrors extraction_pipeline.DEFAULTS for model availability)
# ═══════════════════════════════════════════════════════════════════════════

ALIGN_DEFAULTS = {
    "aligner_models": [
        "anthropic/claude-sonnet-4",
        "openai/gpt-5",
        "google/gemini-2.5-pro",
    ],
    "judge_model": "anthropic/claude-opus-4.6",
    "temperature": 0.0,
    "max_tokens_align": 1024,
    "max_tokens_judge": 4096,
    "concurrency": 8,
    "retry_max": 3,
    "retry_backoff_s": 2.0,
}


# ═══════════════════════════════════════════════════════════════════════════
# Baseline entry schema (what callers must provide in `baselines`)
# ═══════════════════════════════════════════════════════════════════════════

BASELINE_ENTRY_SCHEMA = """\
Each baseline entry is a dict with keys:
  id (str)              : canonical id, e.g. "D5" or "相爱相亲"
  description (str)     : one-line human summary used in LLM prompts
  match_fields (dict)   : {field_name: str | list[str]} for programmatic
                          cross-check against the claim; each value is
                          substring-matched (case-insensitive, lists treated
                          as ANY-match)
  kw (list[str])        : optional keyword list for kw sanity check;
                          substring match (case-insensitive)
"""


# ═══════════════════════════════════════════════════════════════════════════
# Prompts
# ═══════════════════════════════════════════════════════════════════════════

ALIGN_SYSTEM = "你是对齐专家。只输出合法 JSON，不要任何解释文字。"

ALIGN_TEMPLATE = """## 用户原始查询
{query_text}

## 基准表（共 {n_baselines} 条）
{baseline_block}

## 任务
将下面这条"被评模型声明"对齐到基准表中的某一条。

## 待对齐的声明
```json
{claim_json}
```

## 判定规则
1. 按**实体身份（人物/作品名）+ 核心主题**综合判断，允许译名/表述差异
2. 若基准表**任何一条都不对应**（无论是基准外的新条目，还是模型编造/错误），canonical_id=null
3. 对齐要保守：宁愿 null 也不要硬配到看似相近但含义冲突的条目（例如基准 D40 明确要求"引力波论文，非获奖论文"，若声明说的是光电效应就应判 null）
4. reasoning 要直接引用或概括声明中的决定性字段（如人名、主题、期刊、年份），不要泛泛而谈

## 严格实体约束（**必须遵守**）
- **canonical_id 必须指向与 claim 核心实体（人名/片名/作品名）语义相同的基准条目**。允许语种、拼写、别名变体（繁简体、中英文、缩写、中间名）
- **禁止**用基准表里**不同实体**的 id "替模型纠错"。举例：
   - 如果 claim.film_name 是"叔·叔"，即使其他字段（导演=杨曜恺）与某基准片（也姓杨曜恺）匹配，也不能把 canonical_id 改成那个基准片。叔·叔如果在基准表就对齐叔·叔，不在就 null。
   - 如果 claim.name 是"Helmut Ernst"（不存在的人），即使姓 Ernst 的基准项是 "Richard Ernst"，也不能对齐——不同的人。
- 如果 claim 核心实体不在基准表（允许变体后），canonical_id=null

## Q06 专项规则（诺奖论文被拒查询）
若 claim.rejecting_journal 缺失/空，且 claim.note 或 rejection_reason 语义否认"期刊拒稿"（如国家审查、政治压制、出版商退稿、会议摘要、审稿人反对但发表等），无论 claim.name 是否匹配某 D 基准人物，canonical_id 必须为 null。任务限定**期刊**拒稿，非期刊场景不在评分范畴。

## 输出 JSON
```json
{{
  "canonical_id": "<Dxx 或基准 id>" 或 null,
  "confidence": "high|medium|low",
  "reasoning": "<80 字以内>"
}}
```
"""


JUDGE_SYSTEM = (
    "你是对齐仲裁者。综合 N 个对齐器的判断、客观字段对照、关键词匹配信号，"
    "给出最终 canonical_id。只输出合法 JSON。"
)

JUDGE_TEMPLATE = """## 用户原始查询
{query_text}

## 待对齐的声明
```json
{claim_json}
```

## 对齐器的判断
{votes_block}

## 客观字段对照（cross-check）
{cross_check_block}

## Keyword sanity match
{kw_block}

## 候选基准条目详情
{candidates_block}

## 任务
综合以上证据给出最终 canonical_id。决策优先级参考：
1. 对齐器多数意见 + 字段无冲突 + kw 一致 → 信多数
2. 多数一致但字段明显冲突（misses > 0）→ 认真核查，可能 null
3. 对齐器分歧 → 选 reasoning 更贴原文、字段更对得上的一方
4. kw 与多数对齐器矛盾 → 警惕 mode collapse，倾向 null 或重选
5. 证据不足 → null

## 严格实体约束（**必须遵守**）

1. **final_canonical_id 只能指向与 claim 核心实体（人名 / 片名 / 作品名）语义相同的基准条目**。允许语种、拼写、别名变体——例如：
   - "淪落人"（繁） ≡ "沦落人"（简）：允许对齐到基准 id "沦落人"
   - "Love Education" ≡ "相爱相亲"：允许对齐
   - "Richard R. Ernst" ≡ "Richard Ernst"（同一人，中间名差异）：允许
   - "Katalin Karikó" / "卡里科" / "考里科"：同一人，允许
2. **禁止**用基准表里**不同实体**的 id "帮模型纠错"，即使其他字段（导演、年份、合作者、主题）匹配也不允许。例如：
   - claim.film_name="叔·叔"（导演杨曜恺 2019）≠ 基准 id "从今以后"（也是杨曜恺，但是不同电影）→ 必须 null 或对齐到 "叔·叔"（如果在基准表里）
   - claim.name="赫尔穆特·恩斯特（Helmut Ernst）" ≠ 基准 D10 "Richard Ernst"（姓相同但名不同、是不同人）→ 必须 null
3. **如果 claim 的核心实体不在基准表的任何一条里（允许变体后），final_canonical_id=null**。不许"退而求其次"去选一个其他字段接近的基准条目。

## Q06 专项规则（诺奖论文被拒查询）
如果 claim.rejecting_journal 为 null/空 / "N/A" / 空字符串，且 claim.note 或 rejection_reason 的语义**否认期刊拒稿**（例如"国家审查"、"政治压制"、"被政府查禁"、"出版商退稿"、"会议摘要被拒"、"审稿人反对但编辑发表"、"state suppression"、"censored"、"not a journal rejection"等），则无论 claim.name 是否匹配某 D 基准的人物，final_canonical_id 都强制为 null。因为任务限定"**期刊**拒稿"，非期刊场景不在评分范畴。

**触发此规则时，rationale 必须以标签 `[Q06_FORCE_NULL:non_journal_rejection]` 开头**（例如："[Q06_FORCE_NULL:non_journal_rejection] claim 的 rejecting_journal 为 null，note 明确说'国家审查'，非期刊拒稿"）。下游 Stage C 验证层会识别此标签并禁止把这条 null 回填到 ✅ 基准。

## Q14 专项规则（任期内持双重国籍的国家元首查询）
**仅当 query 是关于"双重国籍国家元首"时（query_text 含'国家元首'+'国籍'/'双重国籍'）触发**。三条强制 null 规则按下列顺序逐项检查：

1. **任前已放弃外籍**（`[Q14_FORCE_NULL:renounced_before_term]`）：claim.self_caveat 字段或 claim.evidence_note 中明确表明该人在**就职/当选前**已放弃外国国籍（典型表述："任前放弃美籍"、"放弃外国国籍后才参选"、"renounced before taking office"、"在就职前两天放弃法国籍"等）。即便人物身份对、其他字段对，仍强制 null。典型案例：Valdas Adamkus（立陶宛任前放弃美籍）、Pedro Pablo Kuczynski（秘鲁 2015 年放弃美籍）、Benjamin Netanyahu（以色列任前放弃美籍）、Michaëlle Jean（加拿大总督任前放弃法籍）、Guy Scott（早期放弃英籍）。

2. **非国家元首**（`[Q14_FORCE_NULL:not_head_of_state]`）：claim.office 字段是**政府首脑/部长/州长/议长/议员/总督/地方首脑**等非 head of state 角色。严格定义：议会制国家的 PM 不是国家元首（如英国/加拿大/澳大利亚/日本/印度/德国的 PM）；总统制国家的总统是元首；议会制国家的礼仪性总统/君主/教宗/酋长是元首。若 claim.office 含 "总理"/"首相"/"prime minister"/"PM"/"州长"/"governor"/"minister"/"议长"/"speaker" 等关键词且不属总统制元首，强制 null。

3. **实际只持单一国籍**（`[Q14_FORCE_NULL:single_citizenship]`）：claim.citizenships 数组虽列出 ≥2 个国籍，但 claim.self_caveat 或 evidence_note 表明实际只持单一国籍（典型："实际上是单一国籍"、"co-prince 只是兼任职务非真国籍"、"已放弃所以只剩本国籍"）。注意：法国总统兼任安道尔大公属于"兼任他国职务"而非双重国籍，但若模型仍将其作为肯定例子列出，应**对齐到 D7（基准 ❌）**而不是触发 FORCE_NULL —— D7 的存在就是为了扣分这种错答；只有当 claim 自身明确说"co-prince 不是国籍"等自我否定语时才走 FORCE_NULL。

**触发时 rationale 必须以对应标签开头**，例如：
- `[Q14_FORCE_NULL:renounced_before_term] self_caveat='就职前已放弃美籍'`
- `[Q14_FORCE_NULL:not_head_of_state] office='加拿大总理'，PM 非 head of state`
- `[Q14_FORCE_NULL:single_citizenship] self_caveat='实际只持本国籍'`

下游 Stage C 验证层识别此类标签后，禁止把 null 回填到 ✅ 基准。

## Q39 专项规则（气候最宜居首都查询）
**仅当 query 是关于"全球首都气候最宜居筛选"时（query_text 含'首都'+'年均气温'+'年降水量'）触发**。三条强制 null 规则按下列顺序逐项检查（基于模型 claim 自身字段，无须外部知识）：

1. **非首都城市**（`[Q39_FORCE_NULL:non_capital]`）：claim.capital_name 字段是**非国家首都**的城市/地区。常见误列：'纽约'/'New York'、'香港'/'Hong Kong'、'墨尔本'/'Melbourne'、'上海'/'Shanghai'、'孟买'/'Mumbai'、'巴塞罗那'/'Barcelona'、'伊斯坦布尔'/'Istanbul'、'圣保罗'/'São Paulo'、'多伦多'/'Toronto'、'悉尼'/'Sydney'、'米兰'/'Milan'、'里约热内卢'/'Rio de Janeiro'、'广州'、'北京以外的中国城市' 等。注意：**Pretoria 是南非的行政首都**（Cape Town 是立法首都，Bloemfontein 是司法首都），均算合法首都；**伯尔尼/Bern 是瑞士首都**；**渥太华/Ottawa 是加拿大首都**等。**Washington D.C.**、**Bogotá** 等是首都不触发。

2. **气温超出 13-17°C 区间**（`[Q39_FORCE_NULL:temp_out_of_range]`）：claim.annual_temp_c 字段已填写且数值明显**不在 [13, 17] 区间内**（典型：23°C、19.5°C、10°C、6°C 等），但模型仍将其作为合规答案列出。即使温度名义边界值（如 12.9 / 17.1）在测量不确定性内可放过，明显超界（< 12.5 或 > 17.5）必须强制 null。注意：claim.annual_temp_c=null（模型未给数值）则**不触发**此规则，正常走对齐。

3. **降水超出 800-1200mm 区间**（`[Q39_FORCE_NULL:precip_out_of_range]`）：claim.annual_precip_mm 字段已填写且数值明显**不在 [800, 1200] 区间内**（典型：< 750 或 > 1300），但模型仍将其作为合规答案列出。同 Rule 2 处理边界容差（700-800 / 1200-1300 视具体情况）。claim.annual_precip_mm=null 不触发。

**触发时 rationale 必须以对应标签开头**，例如：
- `[Q39_FORCE_NULL:non_capital] capital_name='纽约'，纽约不是美国首都（Washington D.C. 才是）`
- `[Q39_FORCE_NULL:temp_out_of_range] annual_temp_c=22.5°C，超出 13-17°C 上限`
- `[Q39_FORCE_NULL:precip_out_of_range] annual_precip_mm=2300mm，超出 800-1200mm 上限`

下游 Stage C 验证层识别此类标签后，禁止把 null 回填到 ✅ 基准（但允许 baseline_add 到 ⚠️/❌——例如 Mexico City 模型基于 normals 误纳即可对齐到 ⚠️ D6）。

## 输出 JSON 与字符串转义规则（**必须严格遵守**）
- 输出**纯 JSON**，可选用 ```json``` 围栏包裹。
- `rationale` 字段是 JSON 字符串，**禁止包含未转义的双引号 `"`**。需引用术语/区域名/英文片名时：
  - 优先用中文角标「」或单引号 `'…'` 包裹（例：`region_name 「毁林弧」 与 D1 'Arc of Deforestation' 一致`）
  - 必须保留双引号字面量时**必须转义为 `\"`**（例：`region=\"Pará\"`）
- 因为下游 JSON 解析器对 `rationale` 中混入的未转义闭合括号 + 引号组合（如 `Rondônia)"语义`）无法可靠修复，违反此规则会让你的判定丢失。

```json
{{
  "final_canonical_id": "<id>" 或 null,
  "final_confidence": "high|medium|low",
  "rationale": "<100 字以内，指出你采信的关键证据；若触发 Q06/Q14/Q39 专项规则必须以对应 [QXX_FORCE_NULL:...] 标签开头；若触发实体约束必须明确说明。注意 rationale 内禁用未转义双引号——见上节规则。>"
}}
```
"""


# ═══════════════════════════════════════════════════════════════════════════
# Programmatic checks (no LLM)
# ═══════════════════════════════════════════════════════════════════════════


def _claim_text_for_matching(claim: dict) -> str:
    """Concat all string values in a claim into a single lowercase text blob,
    used as haystack for kw substring matching."""
    parts = [str(v) for v in claim.values() if isinstance(v, str)]
    return " ".join(parts).lower()


def kw_sanity_match(claim: dict, baselines: list[dict]) -> str | None:
    """Return the baseline id with the highest kw hit count, or None."""
    haystack = _claim_text_for_matching(claim)
    best_id, best_n = None, 0
    for b in baselines:
        kws = b.get("kw") or []
        n = sum(1 for k in kws if str(k).lower() in haystack)
        if n > best_n:
            best_id, best_n = b["id"], n
    return best_id if best_n > 0 else None


def cross_check_fields(claim: dict, baseline_entry: dict) -> dict:
    """Compare claim fields against baseline_entry.match_fields.

    Returns {hits, misses, unknowns, detail}:
      - hits: field was populated in claim AND matched expected
      - misses: field was populated in claim AND did NOT match expected
      - unknowns: field was missing/empty in claim
    """
    mf = baseline_entry.get("match_fields") or {}
    hits, misses, unknowns = 0, 0, 0
    detail: dict[str, str] = {}
    for field_name, expected in mf.items():
        got = str(claim.get(field_name, "") or "").strip().lower()
        if not got:
            unknowns += 1
            detail[field_name] = "unknown"
            continue
        exp_list = expected if isinstance(expected, list) else [expected]
        matched = any(str(e).lower() in got for e in exp_list)
        if matched:
            hits += 1
            detail[field_name] = "match"
        else:
            misses += 1
            detail[field_name] = f"mismatch(expected={expected!r}, got={got!r})"
    return {"hits": hits, "misses": misses, "unknowns": unknowns, "detail": detail}


# ═══════════════════════════════════════════════════════════════════════════
# LLM calls
# ═══════════════════════════════════════════════════════════════════════════


def _build_baseline_block(baselines: list[dict]) -> str:
    """Render baselines as a numbered text block for prompts."""
    lines = []
    for b in baselines:
        lines.append(f"- {b['id']}: {b['description']}")
    return "\n".join(lines)


def _run_one_aligner(
    client,
    model: str,
    claim: dict,
    baselines: list[dict],
    query_text: str,
    cfg: dict,
) -> dict:
    """Single alignment LLM call. Returns {model, canonical_id, confidence,
    reasoning, raw}."""
    prompt = ALIGN_TEMPLATE.format(
        query_text=query_text,
        n_baselines=len(baselines),
        baseline_block=_build_baseline_block(baselines),
        claim_json=json.dumps(claim, ensure_ascii=False, indent=2),
    )
    try:
        raw = call_llm(
            client,
            model,
            ALIGN_SYSTEM,
            prompt,
            max_tokens=cfg["max_tokens_align"],
            temperature=cfg["temperature"],
            retry_max=cfg["retry_max"],
            retry_backoff_s=cfg["retry_backoff_s"],
        )
    except Exception as e:
        return {
            "model": model,
            "canonical_id": None,
            "confidence": "low",
            "reasoning": f"aligner_error: {type(e).__name__}",
            "raw": "",
        }
    parsed = _extract_json(raw) or {}
    cid = parsed.get("canonical_id")
    # Coerce "null"/"none" strings to None
    if isinstance(cid, str) and cid.strip().lower() in {"null", "none", ""}:
        cid = None
    return {
        "model": model,
        "canonical_id": cid,
        "confidence": str(parsed.get("confidence", "medium")).lower(),
        "reasoning": parsed.get("reasoning", "") or "",
        "raw": raw,
    }


def _run_judge(
    client,
    model: str,
    claim: dict,
    votes: list[dict],
    field_check: dict,
    kw_match_id: str | None,
    baselines_by_id: dict[str, dict],
    query_text: str,
    cfg: dict,
) -> dict:
    votes_block = "\n\n".join(
        f"### Aligner {i + 1} ({v['model']})\n"
        f"- canonical_id: {v['canonical_id']}\n"
        f"- confidence: {v['confidence']}\n"
        f"- reasoning: {v['reasoning']}"
        for i, v in enumerate(votes)
    )

    if field_check:
        cross_check_block = (
            f"hits: {field_check.get('hits', 0)}, "
            f"misses: {field_check.get('misses', 0)}, "
            f"unknowns: {field_check.get('unknowns', 0)}\n"
            f"detail: {json.dumps(field_check.get('detail', {}), ensure_ascii=False)}"
        )
    else:
        cross_check_block = "n/a (无 consensus id 可对照)"

    kw_block = f"kw_match_id: {kw_match_id}" if kw_match_id else "kw_match_id: null"

    # Provide full descriptions for the candidates that appear in votes/kw
    candidate_ids = set()
    for v in votes:
        if v["canonical_id"]:
            candidate_ids.add(v["canonical_id"])
    if kw_match_id:
        candidate_ids.add(kw_match_id)
    if not candidate_ids:
        candidates_block = "无候选（所有 aligner 都给 null）"
    else:
        lines = []
        for cid in sorted(candidate_ids):
            b = baselines_by_id.get(cid)
            if b is None:
                lines.append(f"- {cid}: (未知 id，不在基准表中)")
            else:
                lines.append(f"- {cid}: {b['description']}")
        candidates_block = "\n".join(lines)

    prompt = JUDGE_TEMPLATE.format(
        query_text=query_text,
        claim_json=json.dumps(claim, ensure_ascii=False, indent=2),
        votes_block=votes_block,
        cross_check_block=cross_check_block,
        kw_block=kw_block,
        candidates_block=candidates_block,
    )
    try:
        raw = call_llm(
            client,
            model,
            JUDGE_SYSTEM,
            prompt,
            max_tokens=cfg["max_tokens_judge"],
            temperature=cfg["temperature"],
            retry_max=cfg["retry_max"],
            retry_backoff_s=cfg["retry_backoff_s"],
        )
    except Exception as e:
        return {
            "final_canonical_id": None,
            "final_confidence": "low",
            "rationale": f"judge_error: {type(e).__name__}",
            "raw": "",
        }
    parsed = _extract_json(raw) or {}
    fid = parsed.get("final_canonical_id")
    if isinstance(fid, str) and fid.strip().lower() in {"null", "none", ""}:
        fid = None
    return {
        "final_canonical_id": fid,
        "final_confidence": str(parsed.get("final_confidence", "medium")).lower(),
        "rationale": parsed.get("rationale", "") or "",
        "raw": raw,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Decision logic
# ═══════════════════════════════════════════════════════════════════════════


def _decide_path(votes: list[dict], field_check: dict, kw_match_id: str | None) -> str:
    """Return 'fast' (accept consensus) or 'slow' (invoke judge).

    Fast path requires ALL of:
      - unanimous canonical_id across aligners
      - no aligner reports confidence=low
      - field_check.misses == 0 (if consensus is not null; skipped otherwise)
      - kw_match_id is None OR equals consensus id
    """
    ids = [v.get("canonical_id") for v in votes]
    if not ids:
        return "slow"
    unanimous = len(set(ids)) == 1
    if not unanimous:
        return "slow"
    if any(v.get("confidence") == "low" for v in votes):
        return "slow"

    consensus_id = ids[0]
    if consensus_id is None:
        # Unanimous null: if kw found a match, kw says "wait, maybe in baseline"
        # → escalate to judge so it can decide
        if kw_match_id is not None:
            return "slow"
        return "fast"  # unanimous null, kw also null → accept

    # Unanimous non-null
    if field_check.get("misses", 0) > 0:
        return "slow"
    if kw_match_id is not None and kw_match_id != consensus_id:
        return "slow"
    return "fast"


# ═══════════════════════════════════════════════════════════════════════════
# Per-claim orchestrator
# ═══════════════════════════════════════════════════════════════════════════


def align_one_claim(
    client,
    *,
    claim: dict,
    baselines: list[dict],
    baselines_by_id: dict[str, dict],
    query_text: str,
    cfg: dict,
) -> dict:
    """Align a single claim: 3 aligner votes + field + kw + optional judge."""
    # Step 1: parallel aligner votes
    aligner_models = cfg["aligner_models"]
    votes: list[dict] = []
    with ThreadPoolExecutor(max_workers=len(aligner_models)) as pool:
        futures = [
            pool.submit(_run_one_aligner, client, m, claim, baselines, query_text, cfg)
            for m in aligner_models
        ]
        for fut in as_completed(futures):
            votes.append(fut.result())
    # stable order by aligner_models index
    votes.sort(key=lambda v: aligner_models.index(v["model"]))

    # Step 2: programmatic checks
    kw_match_id = kw_sanity_match(claim, baselines)
    ids = [v.get("canonical_id") for v in votes]
    unanimous_id = ids[0] if len(set(ids)) == 1 else None
    field_check: dict = {}
    if unanimous_id and unanimous_id in baselines_by_id:
        field_check = cross_check_fields(claim, baselines_by_id[unanimous_id])

    # Step 3: fast/slow routing
    path = _decide_path(votes, field_check, kw_match_id)

    judge_verdict: dict | None = None
    if path == "slow":
        judge_verdict = _run_judge(
            client,
            cfg["judge_model"],
            claim,
            votes,
            field_check,
            kw_match_id,
            baselines_by_id,
            query_text,
            cfg,
        )

    # Step 4: consolidate final answer
    if judge_verdict is not None:
        final_id = judge_verdict.get("final_canonical_id")
        final_conf = judge_verdict.get("final_confidence", "medium")
        final_reason = judge_verdict.get("rationale", "")
        # Judge low confidence → mark needs_review so scorer can isolate
        if final_conf == "low":
            final_conf = "needs_review"
    else:
        final_id = ids[0]  # unanimous by fast-path definition
        high_votes = sum(1 for v in votes if v.get("confidence") == "high")
        final_conf = "high" if high_votes >= 2 else "medium"
        final_reason = votes[0].get("reasoning", "")

    # If final id doesn't appear in baseline table, strip to null (guard
    # against LLM hallucinating ids like "D99")
    if final_id is not None and final_id not in baselines_by_id:
        final_id = None
        final_conf = "needs_review"
        final_reason = f"final id not in baseline table: {final_reason}"

    return {
        "canonical_id": final_id,
        "alignment_confidence": final_conf,
        "alignment_reasoning": final_reason,
        "votes": votes,
        "field_check": field_check,
        "kw_match_id": kw_match_id,
        "judge_invoked": judge_verdict is not None,
        "judge_verdict": judge_verdict,
        "raw": claim,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Top-level batch API
# ═══════════════════════════════════════════════════════════════════════════


def align_claims(
    client,
    *,
    claims_by_model: dict[str, list[dict]],
    baselines: list[dict],
    query_text: str,
    output_dir: Path | str | None = None,
    overrides: dict | None = None,
    verbose: bool = True,
) -> dict[str, list[dict]]:
    """Align every (model, claim) pair. Optionally write per-model
    `alignment.json` into {output_dir}/{model}/alignment.json.

    Returns {model_name: [aligned_claim, ...]}.
    """
    cfg = {**ALIGN_DEFAULTS, **(overrides or {})}
    baselines_by_id = {b["id"]: b for b in baselines}
    aligned_by_model: dict[str, list[dict]] = {}

    for model_name, claims in claims_by_model.items():
        if verbose:
            print(f"\n── aligning target model: {model_name} (claims={len(claims)}) ──")
        if not claims:
            aligned_by_model[model_name] = []
            if output_dir is not None:
                _write_alignment(output_dir, model_name, [], cfg)
            continue

        aligned: list[dict] = []
        with ThreadPoolExecutor(max_workers=cfg["concurrency"]) as pool:
            futures = {
                pool.submit(
                    align_one_claim,
                    client,
                    claim=c,
                    baselines=baselines,
                    baselines_by_id=baselines_by_id,
                    query_text=query_text,
                    cfg=cfg,
                ): i
                for i, c in enumerate(claims)
            }
            results: list[tuple[int, dict]] = []
            for fut in as_completed(futures):
                idx = futures[fut]
                try:
                    res = fut.result()
                except Exception as e:
                    res = {
                        "canonical_id": None,
                        "alignment_confidence": "needs_review",
                        "alignment_reasoning": f"exception: {type(e).__name__}",
                        "votes": [],
                        "field_check": {},
                        "kw_match_id": None,
                        "judge_invoked": False,
                        "judge_verdict": None,
                        "raw": claims[idx] if idx < len(claims) else {},
                    }
                results.append((idx, res))
                if verbose:
                    cid = res.get("canonical_id")
                    conf = res.get("alignment_confidence")
                    judge = "+judge" if res.get("judge_invoked") else ""
                    print(f"  [{model_name}|claim#{idx}] id={cid} conf={conf}{judge}")
        results.sort(key=lambda x: x[0])
        aligned = [r for _, r in results]

        aligned_by_model[model_name] = aligned
        if output_dir is not None:
            _write_alignment(output_dir, model_name, aligned, cfg)

    return aligned_by_model


def _write_alignment(
    output_dir: Path | str, model_name: str, aligned: list[dict], cfg: dict
) -> None:
    model_dir = Path(output_dir) / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model_name,
        "aligned_claims": aligned,
        "summary": _summarize(aligned),
        "models_used": {
            "aligners": cfg["aligner_models"],
            "judge": cfg["judge_model"],
        },
    }
    (model_dir / "alignment.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _summarize(aligned: list[dict]) -> dict:
    total = len(aligned)
    by_conf: dict[str, int] = {}
    judge_count = 0
    matched = 0
    null_count = 0
    for a in aligned:
        c = a.get("alignment_confidence", "medium")
        by_conf[c] = by_conf.get(c, 0) + 1
        if a.get("judge_invoked"):
            judge_count += 1
        if a.get("canonical_id"):
            matched += 1
        else:
            null_count += 1
    return {
        "total_claims": total,
        "matched_to_baseline": matched,
        "null_canonical": null_count,
        "judge_invoked": judge_count,
        "by_confidence": by_conf,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Null export — collect claims whose canonical_id is null into a single
# file for external verification (web search agent).
# ═══════════════════════════════════════════════════════════════════════════


def _extract_verification_hints(raw_claim: dict, query_id: int | None) -> dict:
    """Pick the most useful fields from raw_claim for an external verifier.
    Query-agnostic best-effort."""
    # Common hint fields across queries
    candidate_keys = [
        "name",
        "laureate_name",
        "film_name",
        "nobel_year",
        "nobel_field",
        "release_year",
        "director",
        "rejecting_journal",
        "rejected_paper_topic",
        "rejection_reason",
        "final_publication",
        "award_nomination",
        "box_office",
        "douban_rating",
        "note",
    ]
    return {k: raw_claim.get(k) for k in candidate_keys if raw_claim.get(k)}


def _extract_context_window(
    response_text: str,
    claim: dict,
    window: int = 400,
    max_occurrences: int = 3,
) -> str:
    """Find occurrences of the claim's identifier in raw model response
    and return concatenated windows around each (up to max_occurrences).

    Windows are deduped if they overlap. Returns empty string if no match.

    Useful for verification agents that need to see original model context —
    especially when extraction drops parenthetical clarifications that only
    appear at one of several occurrences (e.g. a name in an index vs. its
    detailed case section later).
    """
    if not response_text:
        return ""
    # Build a list of search terms from claim fields, most specific first
    candidates = []
    for key in ("name", "laureate_name", "film_name"):
        v = claim.get(key)
        if v and isinstance(v, str):
            candidates.append(v.strip())
            for sep in ["（", "(", "／", "/", "，", ","]:
                if sep in v:
                    first = v.split(sep)[0].strip()
                    if first and first not in candidates:
                        candidates.append(first)
                    break
    # Find all occurrences of the first matching term
    matched_term = None
    positions: list[int] = []
    for term in candidates:
        if not term or len(term) < 2:
            continue
        found = []
        start = 0
        while True:
            pos = response_text.find(term, start)
            if pos < 0:
                break
            found.append(pos)
            start = pos + len(term)
        if found:
            matched_term = term
            positions = found
            break
    if not positions or matched_term is None:
        return ""

    # Take up to `max_occurrences` positions (prefer the most spread-out set)
    if len(positions) > max_occurrences:
        step = len(positions) // max_occurrences
        positions = positions[::step][:max_occurrences]

    # Build non-overlapping windows
    windows: list[tuple[int, int]] = []
    for p in positions:
        start = max(0, p - window)
        end = min(len(response_text), p + len(matched_term) + window)
        # Merge with previous window if overlapping
        if windows and start <= windows[-1][1]:
            windows[-1] = (windows[-1][0], max(windows[-1][1], end))
        else:
            windows.append((start, end))
    # Concat with separators
    parts = []
    for _i, (s, e) in enumerate(windows):
        prefix = "…" if s > 0 else ""
        suffix = "…" if e < len(response_text) else ""
        parts.append(prefix + response_text[s:e] + suffix)
    return "\n\n---\n\n".join(parts)


def _load_model_responses(
    models_input: list[str] | None, query_id: int | None
) -> dict[str, str]:
    """Re-parse `name=path` model inputs to get raw response text per model.
    Used to enrich null_review items with context_span. Returns {} on
    absent or unparseable inputs."""
    if not models_input or query_id is None:
        return {}
    out: dict[str, str] = {}
    for spec in models_input:
        if "=" not in spec:
            continue
        name, path = spec.split("=", 1)
        p = Path(path.strip())
        if not p.exists():
            continue
        try:
            text = p.read_text(encoding="utf-8")
            data = json.loads(text)
        except (json.JSONDecodeError, OSError):
            continue
        resp = ""
        if isinstance(data, list):
            match = next((d for d in data if d.get("id") == query_id), None)
            if match:
                resp = match.get("report_content") or match.get("response") or ""
        elif isinstance(data, dict):
            resp = data.get("report_content") or data.get("response") or ""
        if resp:
            out[name.strip()] = resp
    return out


def export_null_claims_for_review(
    aligned_by_model: dict[str, list[dict]],
    output_path: Path | str,
    *,
    query_id: int | None = None,
    query_text: str = "",
    models_input: list[str] | None = None,
    models_responses: dict[str, str] | None = None,
    context_window: int = 400,
) -> list[dict]:
    """Gather all claims with canonical_id=null (or needs_review) across all
    models into a single JSON file suitable for an external web-search
    verification agent. Returns the list.

    When `models_input` or `models_responses` is provided, each item is
    enriched with a `context_span` — a window of the raw model response
    around the claim's first mention. This helps verification agents
    recover from extraction drops (e.g. parenthetical clarifications).

    Each entry::

      {
        "query_id": int,
        "query_text": str,
        "model": str,
        "claim_idx": int,
        "raw_claim": {...},
        "alignment_confidence": "high|medium|low|needs_review",
        "judge_invoked": bool,
        "judge_reasoning": str,
        "verification_hints": {k: v, ...},   # compact summary
        "kw_match_id": str | null,            # kw tripped something? useful hint
        "context_span": str,                  # 400-char window from raw response
      }
    """
    # Load raw responses for context_span (optional)
    responses_by_model: dict[str, str] = dict(models_responses or {})
    if not responses_by_model and models_input:
        responses_by_model = _load_model_responses(models_input, query_id)

    out_items: list[dict] = []
    for model_name, claims in aligned_by_model.items():
        for idx, c in enumerate(claims):
            cid = c.get("canonical_id")
            conf = c.get("alignment_confidence", "medium")
            if cid is not None and conf != "needs_review":
                continue  # scored normally; nothing to verify
            raw = c.get("raw", {}) or {}
            response_text = responses_by_model.get(model_name, "")
            context_span = _extract_context_window(
                response_text, raw, window=context_window
            )
            out_items.append(
                {
                    "query_id": query_id,
                    "query_text": query_text,
                    "model": model_name,
                    "claim_idx": idx,
                    "raw_claim": raw,
                    "alignment_confidence": conf,
                    "judge_invoked": c.get("judge_invoked", False),
                    "judge_reasoning": c.get("alignment_reasoning", ""),
                    "verification_hints": _extract_verification_hints(raw, query_id),
                    "kw_match_id": c.get("kw_match_id"),
                    "context_span": context_span,
                }
            )
    payload = {
        "query_id": query_id,
        "query_text": query_text,
        "total_nulls": len(out_items),
        "items": out_items,
        "resolution_schema": (
            "For each item, append {'resolution': 'baseline_add'|'hallucination'"
            "|'unresolved', 'verification_notes': str, 'evidence_urls': [...], "
            "'new_baseline_id': str?, 'new_judgment': '✅|⚠️|❌'?, "
            "'new_description': str?, 'new_match_fields': dict?} "
            "into a file 'null_resolutions.json' alongside this file."
        ),
        "handling_rules": [
            # Rule 1 — Q06 non-journal-rejection gate
            "If item['judge_reasoning'] starts with "
            "'[Q06_FORCE_NULL:non_journal_rejection]', the alignment judge "
            "deterministically null'd the claim because the model itself "
            "described a non-journal rejection scenario (state censorship, "
            "publisher reject, conference abstract, reviewer-opposed-but-"
            "editor-accepted, etc.). Verification layer MUST respect this: "
            "resolution must be either 'unresolved' (keep null, 0 points) "
            "or 'hallucination' (-1 points). Do NOT resolve to 'baseline_add' "
            "with a ✅ judgment. A claim describing state suppression does "
            "not match a baseline about journal rejection, even when the "
            "person name is identical.",
            # Rule 2 — co-laureate reuse
            "If a claim's entity appears in an existing baseline's kw list "
            "(e.g. 'weissman' is in D5.kw because Karikó and Weissman share "
            "the same Nobel and rejection event), resolve to 'baseline_add' "
            "with new_baseline_id set to that existing baseline id. This is "
            "correct even when alignment layer's strict entity rule null'd "
            "the claim: the baseline explicitly covers co-laureates.",
            # Rule 3 — encourage context reading
            "Each item includes 'context_span' (when available): a window "
            "from the original model response around the claim. Prefer it "
            "over 'raw_claim' fields when they disagree — extraction layer "
            "may have dropped parenthetical clarifications (e.g. claim.name "
            "might be a mistranslation while context_span has the correct "
            "English name in parentheses).",
            # Rule 4 — Q13 cold-start 6-condition checklist
            "If query_id == 13 (tech-stock earnings ~5% drop → "
            "upstream/downstream max single-day drop): the baseline starts "
            "EMPTY (cold-start), so every claim is trivially null'd by "
            "alignment. Stage C agent MUST run a 6-condition fact-check on "
            "each claim using public web data (SEC EDGAR 8-K Item 2.02, "
            "Yahoo Finance adjusted close, company 10-K Item 1, IR pages, "
            "Stooq for cross-check, Tushare/Eastmoney for A-shares, HKEx "
            "for HK-listed). For EACH claim, fill a 'condition_check' "
            "field with verdicts for: "
            "C1 earnings_date ∈ [2016-01-01, 2026-01-01]; "
            "C2 parent_company is a tech stock (GICS Information Technology "
            "OR META/GOOGL/AMZN/NFLX/TSLA — relaxed scope); "
            "C3 parent_company adjusted-close return on the next trading "
            "day after earnings_date is in [-6.0%, -4.0%] (±0.3% tolerance); "
            "C4 answer_ticker has publicly disclosed upstream/downstream "
            "OR competitor/同业 relationship with parent_company (10-K "
            "Item 1 customer/supplier disclosure, IR materials, or "
            "industry research); "
            "C5 answer_ticker's adjusted-close return on answer_drop_date "
            "matches the model's claimed answer_drop_pct (±0.3% tolerance); "
            "C6 answer_drop_date ∈ [earnings_date, earnings_date + 2 "
            "trading days]. Resolution decision: ALL 6 conditions yes → "
            "resolution='baseline_add', new_judgment='✅', score=+1.0, "
            "auto-assign new_baseline_id (D1, D2, …); ANY condition no "
            "with sufficient evidence → resolution='hallucination' (-1.0); "
            "ANY condition unknown after reasonable web search "
            "→ resolution='unresolved' (0.0, awaits human review). "
            "Each condition_check entry must include {'verdict': "
            "'yes'|'no'|'unknown', 'evidence': '<concise finding + URL>'}. "
            "Even for 'unresolved', dump every search attempt and partial "
            "evidence — the human reviewer needs all data to adjudicate.",
            # Rule 5 — Q14 dual-citizenship heads of state
            "If query_id == 14 (heads of state who held dual or "
            "multiple citizenships during their term in office): three "
            "FORCE_NULL tags are honored deterministically by the alignment "
            "judge. If item['judge_reasoning'] starts with "
            "'[Q14_FORCE_NULL:renounced_before_term]', "
            "'[Q14_FORCE_NULL:not_head_of_state]', or "
            "'[Q14_FORCE_NULL:single_citizenship]', the resolution must be "
            "either 'unresolved' (keep null, 0 points) or 'hallucination' "
            "(-1.0). Do NOT 'baseline_add' to a ✅ entry. For non-FORCE_NULL "
            "nulls, perform web verification on each claim using these "
            "criteria: "
            "(a) Is the person a 'head of state' in the strict sense? "
            "Acceptable: presidents in presidential systems (US, France, "
            "Russia, Brazil, Peru, Somalia, …); ceremonial presidents in "
            "parliamentary systems (Germany, India, Italy, …); reigning "
            "monarchs (kings, emperors, sultans); the Pope (Sovereign of "
            "Vatican City); Andorra's Co-Princes ARE heads of state but "
            "the role is a position not a citizenship. Reject: PMs in "
            "parliamentary systems (UK, Canada, Australia, Japan, India, "
            "Germany — these are heads of government, not state); state/"
            "provincial governors; ministers; legislators; viceroys/"
            "governors-general (they REPRESENT the head of state but are "
            "not themselves the head of state — Michaëlle Jean is the "
            "canonical example). "
            "(b) Did the person hold ≥2 distinct nationalities (citizenships, "
            "not ceremonial offices) at any point during their term? "
            "Verify with Wikipedia infobox + national archives + news. "
            "Crucially: if the person renounced their other nationality "
            "BEFORE taking office, they FAIL this test (Adamkus, Kuczynski, "
            "Netanyahu, Michaëlle Jean, Guy Scott, etc.). "
            "(c) For Pope-related claims, default ✅ — popes legally retain "
            "original citizenship under Vatican law (e.g. Francis renewed "
            "his Argentine passport in 2014; Leo XIV holds US/Peru/Vatican). "
            "Resolution decision: passes (a)+(b) with ≥2 citizenships during "
            "term → resolution='baseline_add' new_judgment='✅' score=+1.0; "
            "fails (a) or (b) with strong evidence → 'hallucination' (-1.0); "
            "fails (a)/(b) at semantic edge (e.g. Andorra co-prince claim, "
            "PM-but-arguably-elected-as-head-of-state) → 'baseline_add' "
            "new_judgment='❌' score=-1.0 (reuse existing D7 if it's the "
            "Andorra case); ambiguous after web search → 'unresolved' (0.0). "
            "Auto-assign new_baseline_id (D8, D9, …) for new entries. "
            "Trusted sources: Wikipedia, NobelPrize.org-style biographies, "
            "national archives, official biographies, citizenship-related "
            "news coverage, Vatican.va for popes.",
            # Rule 6 — Q39 climate-livable capitals
            "If query_id == 39 (global capitals with 2024 annual mean "
            "temp 13-17°C and annual precip 800-1200mm, plus average distance "
            "from equator): three FORCE_NULL tags are honored deterministically "
            "by the alignment judge. If item['judge_reasoning'] starts with "
            "'[Q39_FORCE_NULL:non_capital]', '[Q39_FORCE_NULL:temp_out_of_range]', "
            "or '[Q39_FORCE_NULL:precip_out_of_range]', the resolution must NOT "
            "be 'baseline_add' to a ✅ entry. Allowed: 'hallucination' (-1.0), "
            "'unresolved' (0.0), or 'baseline_add' to an existing ⚠️ partial "
            "(e.g. D6 Mexico City for normals-based misclassifications). "
            "For non-FORCE_NULL nulls, perform web verification using THE "
            "FOLLOWING CRITERIA, all based on **2024 single-year actual "
            "measurements** (NOT 1991-2020 climate normals): "
            "(a) Is `capital_name` a national capital? (Pretoria/Bloemfontein/"
            "Cape Town for South Africa all qualify; Washington D.C. is the "
            "US capital, not New York.) "
            "(b) Does the city have **2024 actual** annual mean temperature "
            "in [13, 17]°C? Use NOAA NCEI / DHMZ / NIWA / INUMET / IDEAM-RMCAB / "
            "JMA / Met Office / equivalent national meteorological agency 2024 "
            "annual reports. Tutiempo NOAA-ISD 2024 monthly aggregates are "
            "acceptable cross-source. "
            "(c) Does the city have **2024 actual** annual precipitation in "
            "[800, 1200] mm from the same authoritative sources? "
            "Resolution decision: passes (a)+(b)+(c) → 'baseline_add' "
            "new_judgment='✅' score=+1.0, auto-assign new_baseline_id "
            "(D15, D16, …); passes (a) but fails (b) or (c) by significant "
            "margin → 'baseline_add' new_judgment='⚠️' score=0.0 (real "
            "capital but wrong climate, like Tirana/Antananarivo); fails (a) "
            "or city is fabricated → 'hallucination' (-1.0); ambiguous data "
            "after web search → 'unresolved' (0.0). The 5 known core baselines "
            "are Bogotá, Montevideo, Washington D.C., Zagreb, Wellington — "
            "if a model lists one of these but it was null'd by alignment, "
            "verify and likely 'baseline_add' to the existing D1-D5. "
            "Trusted sources: NOAA NCEI (US), NIWA Annual Climate Summary "
            "(NZ), DHMZ (Croatia), INUMET (Uruguay), IDEAM/RMCAB Bogotá "
            "(Colombia), CONAGUA SMN (Mexico), JMA (Japan), Met Office (UK), "
            "Climate-Data.org & climatestotravel.com as cross-source, "
            "Wikipedia city climate boxes for normals only (NOT for 2024 "
            "single-year data).",
        ],
    }
    Path(output_path).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return out_items


# ═══════════════════════════════════════════════════════════════════════════
# Null resolution application — feeds web-search-verification results back
# into the aligned_by_model structure so scoring picks them up.
# ═══════════════════════════════════════════════════════════════════════════


def apply_null_resolutions(
    aligned_by_model: dict[str, list[dict]],
    resolutions_path: Path | str,
    *,
    dims_ref: list | None = None,
) -> list[dict]:
    """Read null_resolutions.json and patch aligned_by_model in place.

    Resolution semantics::

      - 'hallucination' → set canonical_id to special marker "__HALLUCINATION__"
        so scorer treats it as -1 (scored as ❌ penalty). Does NOT extend baseline.
      - 'baseline_add'  → set canonical_id to the new_baseline_id provided,
        and return a list of new baseline entries for downstream persistence.
      - 'unresolved'    → leave claim as-is (canonical_id=null, 0 points).

    Returns a list of {id, judgment, score, description, match_fields, kw}
    entries that should be persisted into DIMS (Q06) or BASELINE_FILMS /
    PARTIAL_FILMS (Q12) for future runs.
    """
    data = json.loads(Path(resolutions_path).read_text(encoding="utf-8"))
    # Support {items: [...]}, {resolutions: [...]}, and a flat list
    if isinstance(data, dict):
        resolutions = data.get("items") or data.get("resolutions") or []
    else:
        resolutions = data
    if not isinstance(resolutions, list):
        return []

    # Index existing DIMS ids for collision avoidance (Q06)
    existing_ids: set[str] = set()
    if dims_ref:
        for d in dims_ref:
            if isinstance(d, (list, tuple)) and len(d) >= 1:
                existing_ids.add(str(d[0]))
            elif isinstance(d, dict) and "id" in d:
                existing_ids.add(str(d["id"]))

    new_baselines: list[dict] = []
    seen_new_ids: set[str] = set()

    for r in resolutions:
        model = r.get("model")
        idx = r.get("claim_idx")
        resolution = r.get("resolution")
        if model is None or idx is None or resolution is None:
            continue
        claims = aligned_by_model.get(model, [])
        if idx >= len(claims):
            continue
        claim = claims[idx]

        if resolution == "hallucination":
            claim["canonical_id"] = "__HALLUCINATION__"
            claim["alignment_confidence"] = "high"
            claim["alignment_reasoning"] = (
                f"[null_resolution=hallucination] {r.get('verification_notes', '')}"
            )
            claim["resolution_evidence"] = r.get("evidence_urls", [])
        elif resolution == "baseline_add":
            new_id = r.get("new_baseline_id")
            if not new_id:
                continue  # skip malformed
            if new_id in existing_ids or new_id in seen_new_ids:
                # ID collision; reuse existing entry (update claim only)
                claim["canonical_id"] = new_id
            else:
                seen_new_ids.add(new_id)
                new_baselines.append(
                    {
                        "id": new_id,
                        "description": r.get("new_description", ""),
                        "judgment": r.get("new_judgment", "⚠️"),
                        "score": {
                            "✅": 1.0,
                            "⚠️": 0.0,
                            "❌": -1.0,
                        }.get(r.get("new_judgment", "⚠️"), 0.0),
                        "match_fields": r.get("new_match_fields", {}),
                        "kw": r.get("new_kw", []),
                        "evidence_urls": r.get("evidence_urls", []),
                        "verification_notes": r.get("verification_notes", ""),
                    }
                )
                claim["canonical_id"] = new_id
            claim["alignment_confidence"] = "high"
            claim["alignment_reasoning"] = (
                f"[null_resolution=baseline_add→{new_id}] "
                f"{r.get('verification_notes', '')}"
            )
            claim["resolution_evidence"] = r.get("evidence_urls", [])
        elif resolution == "unresolved":
            claim["alignment_reasoning"] = (
                f"[null_resolution=unresolved] {r.get('verification_notes', '')}"
            )
            # canonical_id stays null → 0 points

    return new_baselines


def persist_new_baseline_entries(
    new_entries: list[dict], auto_scorer_path: str | Path
) -> None:
    """Append new baseline entries to the auto_scorer.py DIMS constant (Q06)
    or BASELINE_FILMS / PARTIAL_FILMS (Q12, based on entry shape).

    Writes a timestamped comment block + new tuples / dict entries before the
    closing ']' or '}' of the target constant. If `auto_scorer_path` already
    contains all entries (by id match), skip.
    """
    if not new_entries:
        return
    from datetime import datetime

    path = Path(auto_scorer_path)
    src = path.read_text(encoding="utf-8")
    today = datetime.now().strftime("%Y-%m-%d")
    added_any = False
    for entry in new_entries:
        eid = entry["id"]
        # Skip if id already in file text (simple guard)
        if f'"{eid}"' in src or f"'{eid}'" in src:
            continue

        # Detect whether this is a Q06 DIMS (tuple form) or Q10 dict form
        if "DIMS = [" in src and eid.startswith("D"):
            # Q06: append tuple before closing ']' of DIMS
            tuple_block = (
                f"    # web-search-verified {today} — added from null_resolutions.json\n"
                f"    (\n"
                f"        {eid!r},\n"
                f"        {entry.get('description', '')!r},\n"
                f"        {entry.get('judgment', '⚠️')!r},\n"
                f"        {entry.get('score', 0.0)!r},\n"
                f"        {entry.get('kw', [])!r},\n"
                f"    ),\n"
            )
            # Insert before the line `]` that terminates DIMS
            marker = "\n]\n\nDIM_MAP"
            if marker in src:
                src = src.replace(marker, f"\n{tuple_block}]\n\nDIM_MAP", 1)
                added_any = True
        elif "BASELINE_FILMS" in src:
            # Q12 path: append as dict entry in BASELINE_FILMS or PARTIAL_FILMS
            # (choose based on judgment: ✅ → BASELINE, ⚠️ → PARTIAL, ❌ → skip)
            judgment = entry.get("judgment", "⚠️")
            mf = entry.get("match_fields", {})
            film_block = (
                f"    # web-search-verified {today}\n"
                f"    {eid!r}: {{\n"
                f"        'director': {mf.get('director', '')!r},\n"
                f"        'release_year': {mf.get('release_year', [])!r},\n"
                f"        'conditions_met': {3 if judgment == '✅' else 2},\n"
                f"        'note': {entry.get('verification_notes', '')!r},\n"
                f"    }},\n"
            )
            target_const = "BASELINE_FILMS" if judgment == "✅" else "PARTIAL_FILMS"
            # Insert before the closing '}' of the target constant
            import re as _re

            pattern = _re.compile(
                rf"({target_const}: dict = \{{.*?)\n\}}\n", _re.DOTALL
            )
            m = pattern.search(src)
            if m:
                src = (
                    src[: m.start()]
                    + m.group(1)
                    + f"\n{film_block}}}\n"
                    + src[m.end() :]
                )
                added_any = True
    if added_any:
        # write with a backup for safety
        backup = path.with_suffix(path.suffix + ".bak")
        backup.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
        path.write_text(src, encoding="utf-8")
        print(
            f"[*] Persisted {sum(1 for e in new_entries)} new baseline entries "
            f"to {path.name} (backup: {backup.name})"
        )
