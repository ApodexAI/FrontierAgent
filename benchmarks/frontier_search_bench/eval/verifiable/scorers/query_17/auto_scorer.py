"""Query 17 — Paper→stock 48h transmission events auto-scorer.

Reference whitelist (3 qualified events, locked 2026-05-08, per Q20_evaluation_rules.md):
  Item 3: DeepSeek NSA            — arXiv:2502.11089 — 2025-02-18 → 华虹/中芯/上海复旦 等
  Item 4: Qwen3-Omni              — arXiv:2509.17765 — 阿里巴巴 9988.HK +9.30%
  Item 5: ERNIE 5.0               — arXiv:2602.04705 — 百度 BIDU +5.78%

Three strict criteria (per spec, all required for an event to qualify):
  1. 必须是论文 / 技术报告（非模型发布 / 产品 / 投研博客 / 非同行评审报告）
  2. 股价反应必须在论文公开 48h 内
  3. 至少一支具体上市公司 48h 内绝对波动 > 5%

Scoring rule (max 3, min 0, no penalty):
  +1 per distinct whitelist event the model hit. Referencing unqualified
  events (DeepSeek R1 / TurboQuant / Citrini / MIT NANDA / 模型/产品发布) is
  neither rewarded nor penalised — just logged for inspection.

Matching layers (deterministic rules + LLM judge for borderline):
  Tier 1 — arXiv id substring match → auto YES
  Tier 2 — paper-name keyword combo match → auto YES
  Tier 3 — company + topic keyword combo match → auto YES
  Tier 4 — partial signal (e.g. paper name without company) → defer to LLM
  Otherwise → no match
"""

from __future__ import annotations

import argparse
import contextlib
import json
import re
import sys
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS.parent))


from extract import (  # noqa: E402
    ENTITIES,
    PROMPT_HINTS,
    QUERY_ID,
    QUERY_TEXT,
    VALUE_SCHEMA,
)
from pipeline.extraction_pipeline import (  # noqa: E402
    _extract_json,
    call_llm,
    get_client,
    run_pipeline,
)

# ═══════════════════════════════════════════════════════════════════════════
# Benchmark whitelist
# ═══════════════════════════════════════════════════════════════════════════

BENCHMARK: dict = {
    "version": "1.0",
    "locked_at": "2026-05-08",
    "whitelist": {
        "nsa": {
            "canonical_title": "DeepSeek NSA — Native Sparse Attention",
            "arxiv_ids": ["2502.11089"],
            "paper_name_keys": [
                "native sparse attention",
                "nativesparseattention",
                "deepseek nsa",
                "原生稀疏注意力",
                "原生稀疏",
            ],
            "company_keys": [
                "华虹",
                "hua hong",
                "huahong",
                "晶门",
                "上海复旦",
                "中芯国际",
                "smic",
                "0981.hk",
                "688981",
                "shanghai fudan",
            ],
            "topic_keys": [
                "稀疏注意力",
                "sparse attention",
                "稀疏",
                "long context",
                "长上下文",
            ],
            "subfield": "稀疏注意力 / 长上下文推理",
        },
        "qwen3_omni": {
            "canonical_title": "Qwen3-Omni Technical Report",
            "arxiv_ids": ["2509.17765"],
            "paper_name_keys": [
                "qwen3-omni",
                "qwen3 omni",
                "qwen3omni",
                "qwen 3 omni",
                "qwen-3-omni",
                "通义千问3 omni",
                "千问 omni",
            ],
            "company_keys": [
                "阿里巴巴",
                "alibaba",
                "9988.hk",
                "9988",
                "baba",
                "阿里",
            ],
            "topic_keys": [
                "多模态",
                "multimodal",
                "全模态",
                "omni",
                "大模型",
            ],
            "subfield": "大模型 / 多模态生成",
        },
        "ernie_5": {
            "canonical_title": "ERNIE 5.0 Technical Report",
            "arxiv_ids": ["2602.04705"],
            "paper_name_keys": [
                "ernie 5.0",
                "ernie5.0",
                "ernie-5.0",
                "ernie5",
                "ernie 5",
                "文心 5.0",
                "文心一言 5.0",
                "文心5",
                "文心5.0",
            ],
            "company_keys": [
                "百度",
                "baidu",
                "bidu",
                "9888.hk",
            ],
            "topic_keys": [
                "多模态",
                "multimodal",
                "大模型",
                "全模态",
            ],
            "subfield": "大模型 / 多模态生成",
        },
    },
    # Items the spec explicitly rejects — for inspection only (no penalty).
    "known_disqualified": {
        "deepseek_r1": [
            "deepseek r1",
            "deepseek-r1",
            "r1 paper",
        ],
        "turboquant": [
            "turboquant",
            "turbo quant",
        ],
        "qwq_32b": [
            "qwq-32b",
            "qwq 32b",
            "qwq32b",
        ],
        "claude_cowork": [
            "claude cowork",
            "claude for cowork",
            "anthropic cowork",
        ],
        "citrini": [
            "citrini",
            "citrini research",
            "2028 全球智能危机",
        ],
        "mit_nanda": [
            "the genai divide",
            "genai divide",
            "mit nanda",
            "state of ai in business",
        ],
    },
}

# LLM judge fallback (for Tier-4 deferrals).
LLM_JUDGE_MODEL = "anthropic/claude-sonnet-4"
LLM_JUDGE_SYSTEM = "你是严格的论文事件匹配判别器。只输出 JSON，无多余文字。"


# ═══════════════════════════════════════════════════════════════════════════
# Normalization
# ═══════════════════════════════════════════════════════════════════════════


def _norm(value) -> str:
    """NFKC + lower + strip surrounding punct + collapse whitespace."""
    if value is None:
        return ""
    v = unicodedata.normalize("NFKC", str(value)).lower().strip()
    v = v.replace("–", "-").replace("—", "-").replace("‑", "-")
    v = re.sub(r"\s+", " ", v)
    return v


def _compact(value) -> str:
    """_norm + drop punctuation/hyphens/spaces."""
    v = _norm(value)
    v = re.sub(r"[`\"'“”‘’,，。；：:（）()、/\-]+", "", v)
    v = v.replace(" ", "")
    return v


def _has_any(text: str, keys: list[str]) -> str | None:
    for k in keys:
        if k and k in text:
            return k
    return None


def _has_any_compact(compact_text: str, keys: list[str]) -> str | None:
    for k in keys:
        if k and _compact(k) in compact_text:
            return k
    return None


def _arxiv_extract(text: str) -> set[str]:
    """Pull arXiv-like IDs out of a free text field (e.g. 'arxiv:2502.11089',
    '2502.11089', 'arXiv 2502.11089v2')."""
    if not text:
        return set()
    ids = set(re.findall(r"\b(\d{4}\.\d{4,5})\b", text))
    return ids


# ═══════════════════════════════════════════════════════════════════════════
# Tiered matching
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class MatchResult:
    verdict: str  # "yes" | "no" | "defer"
    tier: str  # tier1 | tier2 | tier3 | tier4 | tier_none | disqualified
    matched_key: str | None = None
    reason: str = ""
    llm_used: bool = False
    llm_log: dict | None = None


def _check_disqualified(item: dict) -> tuple[str | None, str]:
    """If the model's claim clearly maps to a known-disqualified event,
    return (id, alias_hit). Used only for diagnostics — does not affect
    score (per spec: unqualified events don't cost points)."""
    blob = " ".join(
        str(item.get(k) or "")
        for k in ("paper_title", "paper_id", "company", "subfield")
    )
    n = _norm(blob)
    nc = _compact(blob)
    for dq_id, keys in BENCHMARK["known_disqualified"].items():
        for k in keys:
            kn = _norm(k)
            if kn and (kn in n or _compact(k) in nc):
                return (dq_id, k)
    return (None, "")


def _rule_match(item: dict) -> MatchResult:
    title = str(item.get("paper_title") or "")
    paper_id = str(item.get("paper_id") or "")
    company = str(item.get("company") or "")
    others = item.get("other_companies") or []
    others_text = " ".join(str(o) for o in others if isinstance(o, str))
    subfield = str(item.get("subfield") or "")

    # Tier 1 — arXiv id substring match (very strong signal).
    candidate_ids = _arxiv_extract(paper_id) | _arxiv_extract(title)
    for key, info in BENCHMARK["whitelist"].items():
        if candidate_ids & set(info["arxiv_ids"]):
            return MatchResult(
                "yes",
                "tier1",
                key,
                reason=f"arXiv id match → {key}",
            )

    # Pre-normalize the searchable blob for tiers 2-4.
    blob_norm = _norm(" ".join([title, paper_id, company, others_text, subfield]))
    blob_compact = _compact(" ".join([title, paper_id, company, others_text, subfield]))

    title_norm = _norm(title)
    title_compact = _compact(title)
    company_blob_norm = _norm(" ".join([company, others_text]))

    # Tier 2 — paper-name keyword match (compact-form, tolerant to hyphens/case).
    for key, info in BENCHMARK["whitelist"].items():
        title_hit = _has_any_compact(title_compact, info["paper_name_keys"]) or _has_any(
            title_norm, [_norm(k) for k in info["paper_name_keys"]]
        )
        if title_hit:
            return MatchResult(
                "yes",
                "tier2",
                key,
                reason=f"paper-name match '{title_hit}' → {key}",
            )

    # Tier 3 — company + topic combo (e.g. 阿里 + 多模态 → Qwen3-Omni).
    for key, info in BENCHMARK["whitelist"].items():
        ch = _has_any(company_blob_norm, [_norm(k) for k in info["company_keys"]])
        th = _has_any(blob_norm, [_norm(k) for k in info["topic_keys"]])
        if ch and th:
            return MatchResult(
                "yes",
                "tier3",
                key,
                reason=f"company '{ch}' + topic '{th}' combo → {key}",
            )

    # Tier 4 — single weak signal (paper-name OR company but not both). Defer.
    for key, info in BENCHMARK["whitelist"].items():
        title_signal = _has_any_compact(blob_compact, info["paper_name_keys"])
        company_signal = _has_any(blob_norm, [_norm(k) for k in info["company_keys"]])
        if title_signal or company_signal:
            return MatchResult(
                "defer",
                "tier4",
                key,
                reason=(
                    f"weak signal: title_hit={title_signal!r} "
                    f"company_hit={company_signal!r}, defer to LLM"
                ),
            )

    return MatchResult("no", "tier_none", None, reason="no whitelist signal matched")


# ═══════════════════════════════════════════════════════════════════════════
# LLM judge (Tier 4)
# ═══════════════════════════════════════════════════════════════════════════


def _llm_judge(client, item: dict, suggested_key: str | None) -> dict:
    wl_block = []
    for key, info in BENCHMARK["whitelist"].items():
        wl_block.append(
            f"- [{key}] {info['canonical_title']} (arXiv {'/'.join(info['arxiv_ids'])}) — "
            f"公司: {', '.join(info['company_keys'][:3])} — 子领域: {info['subfield']}"
        )

    prompt = f"""题目背景：判断一条 \"论文→股价 48h 内 >5%\" 候选事件是否**实质上**对应白名单的某一项。

白名单（合格事件，每项 1 分）：
{chr(10).join(wl_block)}

候选事件（模型抽取）：
- paper_title: {item.get("paper_title")!r}
- paper_id   : {item.get("paper_id")!r}
- company    : {item.get("company")!r}
- other_companies: {item.get("other_companies")!r}
- stock_change_pct: {item.get("stock_change_pct")!r}
- direction  : {item.get("direction")!r}
- subfield   : {item.get("subfield")!r}

判断原则：
- 候选是否在指代白名单某一项（考虑论文别名、缩写、翻译、公司股票简称差异等）。
- 候选明显是 R1 / TurboQuant / QwQ-32B / Citrini / MIT NANDA / 产品发布 等 → 'none'。
- 信息严重不全（如只有公司没论文，或只有泛泛行业讨论）→ 'none'。

输出 JSON：
{{
  "match_to": "<nsa | qwen3_omni | ernie_5 | none>",
  "confidence": "<high|medium|low>",
  "rationale": "<≤60 字>"
}}
"""
    raw = call_llm(
        client,
        LLM_JUDGE_MODEL,
        LLM_JUDGE_SYSTEM,
        prompt,
        max_tokens=512,
        temperature=0,
    )
    parsed = _extract_json(raw) or {}
    return {
        "prompt": prompt,
        "raw_output": raw,
        "match_to": parsed.get("match_to", "none"),
        "confidence": str(parsed.get("confidence", "low")).lower(),
        "rationale": parsed.get("rationale", ""),
    }


def _candidate_match(item: dict, client=None) -> MatchResult:
    result = _rule_match(item)
    if result.verdict != "defer":
        return result
    if client is None:
        client = get_client()
    judge = _llm_judge(client, item, result.matched_key)
    mt = judge["match_to"]
    if mt and mt != "none" and mt in BENCHMARK["whitelist"]:
        verdict = "yes" if judge["confidence"] in {"high", "medium"} else "no"
    else:
        verdict = "no"
    return MatchResult(
        verdict=verdict,
        tier=result.tier,
        matched_key=mt if verdict == "yes" else None,
        reason=f"LLM judge ({judge['confidence']}): {judge['rationale']}",
        llm_used=True,
        llm_log={
            "candidate": {k: item.get(k) for k in (
                "paper_title", "paper_id", "company", "subfield"
            )},
            "tier": result.tier,
            "rule_reason": result.reason,
            "judge_match_to": judge["match_to"],
            "judge_confidence": judge["confidence"],
            "judge_rationale": judge["rationale"],
            "final_verdict": verdict,
        },
    )


# ═══════════════════════════════════════════════════════════════════════════
# Per-model scoring
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class ScoreResult:
    model: str
    score: int
    matched: list[dict] = field(default_factory=list)
    unmatched: list[dict] = field(default_factory=list)
    disqualified_refs: list[dict] = field(default_factory=list)
    llm_judge_log: list[dict] = field(default_factory=list)
    resolution: str = "unknown"


def _score_one_model(model_name: str, payload: dict, client) -> ScoreResult:
    entities = payload.get("entities", [])
    if not entities:
        return ScoreResult(model_name, 0, resolution="empty")
    ent = entities[0]
    canonical = (ent.get("canonical") or {}).get("value") or {}
    if not isinstance(canonical, dict):
        canonical = {}
    items = canonical.get("items")
    if not isinstance(items, list):
        items = []

    matched: list[dict] = []
    unmatched: list[dict] = []
    disqualified: list[dict] = []
    llm_logs: list[dict] = []
    matched_keys: set[str] = set()

    for idx, raw_item in enumerate(items):
        if not isinstance(raw_item, dict):
            continue

        entry = {
            "item_index": idx,
            "paper_title": raw_item.get("paper_title"),
            "paper_id": raw_item.get("paper_id"),
            "company": raw_item.get("company"),
            "stock_change_pct": raw_item.get("stock_change_pct"),
            "direction": raw_item.get("direction"),
            "subfield": raw_item.get("subfield"),
            "claimed_within_48h": raw_item.get("claimed_within_48h"),
        }

        dq_id, dq_alias = _check_disqualified(raw_item)
        if dq_id:
            disqualified.append({**entry, "disqualified_id": dq_id, "alias_hit": dq_alias})
            unmatched.append(
                {
                    **entry,
                    "tier": "disqualified",
                    "reason": f"matches known-disqualified '{dq_id}' (alias '{dq_alias}')",
                }
            )
            continue

        r = _candidate_match(raw_item, client=client)
        if r.llm_used and r.llm_log:
            llm_logs.append(r.llm_log)

        if r.verdict == "yes" and r.matched_key:
            if r.matched_key in matched_keys:
                # already credited — log as duplicate, not new score
                unmatched.append(
                    {
                        **entry,
                        "tier": r.tier,
                        "reason": f"duplicate match to '{r.matched_key}' (already counted)",
                    }
                )
                continue
            matched_keys.add(r.matched_key)
            matched.append(
                {
                    **entry,
                    "tier": r.tier,
                    "matched_key": r.matched_key,
                    "reason": r.reason,
                }
            )
        else:
            unmatched.append({**entry, "tier": r.tier, "reason": r.reason})

    return ScoreResult(
        model=model_name,
        score=len(matched),
        matched=matched,
        unmatched=unmatched,
        disqualified_refs=disqualified,
        llm_judge_log=llm_logs,
        resolution=ent.get("resolution", "unknown"),
    )


# ═══════════════════════════════════════════════════════════════════════════
# I/O
# ═══════════════════════════════════════════════════════════════════════════


def _write_model_score(out_dir: Path, r: ScoreResult) -> None:
    path = out_dir / r.model / "score.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "query_id": QUERY_ID,
                "model": r.model,
                "score": r.score,
                "max_score": 3,
                "min_score": 0,
                "total_rate": r.score / 3.0,
                "scoring_rule": (
                    "+1 per distinct whitelist event matched (NSA / Qwen3-Omni / ERNIE 5.0); "
                    "unqualified or duplicate references: 0 (no penalty)."
                ),
                "matched_whitelist_keys": sorted(
                    {m["matched_key"] for m in r.matched if m.get("matched_key")}
                ),
                "matched": r.matched,
                "unmatched": r.unmatched,
                "disqualified_references": r.disqualified_refs,
                "llm_judge_log": r.llm_judge_log,
                "benchmark_version": BENCHMARK["version"],
                "extraction_resolution": r.resolution,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def _write_scores_json(out_dir: Path, results: list[ScoreResult]) -> None:
    ranked = sorted(results, key=lambda r: (-r.score, r.model))
    (out_dir / "scores.json").write_text(
        json.dumps(
            {
                "query_id": QUERY_ID,
                "max_score": 3,
                "min_score": 0,
                "scoring_rule": (
                    "+1 per distinct whitelist event matched (NSA / Qwen3-Omni / "
                    "ERNIE 5.0); unqualified or duplicate references: 0 (no penalty)."
                ),
                "benchmark_version": BENCHMARK["version"],
                "whitelist": list(BENCHMARK["whitelist"].keys()),
                "results": {
                    r.model: {
                        "total_score": r.score,
                        "total_rate": r.score / 3.0,
                        "matched_whitelist_keys": sorted(
                            {m["matched_key"] for m in r.matched if m.get("matched_key")}
                        ),
                        "unmatched_count": len(r.unmatched),
                        "disqualified_refs": len({d["disqualified_id"] for d in r.disqualified_refs}),
                        "llm_judge_calls": len(r.llm_judge_log),
                        "extraction_resolution": r.resolution,
                    }
                    for r in results
                },
                "ranking": [
                    {
                        "rank": i + 1,
                        "model": r.model,
                        "score": r.score,
                        "matched": sorted(
                            {m["matched_key"] for m in r.matched if m.get("matched_key")}
                        ),
                    }
                    for i, r in enumerate(ranked)
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def _write_ranking_report(out_dir: Path, results: list[ScoreResult]) -> None:
    ranked = sorted(results, key=lambda r: (-r.score, r.model))
    wl = BENCHMARK["whitelist"]

    lines: list[str] = [
        f"# Query {QUERY_ID} Ranking Report",
        "",
        f"> Benchmark v{BENCHMARK['version']} (locked {BENCHMARK['locked_at']})  ",
        "> Scoring: +1 per whitelist hit; unqualified events 0 (no penalty); max 3.  ",
        "> Whitelist: **DeepSeek NSA** / **Qwen3-Omni** / **ERNIE 5.0**.",
        "",
        "| Rank | Model | Score | NSA | Qwen3-Omni | ERNIE 5.0 | Unmatched | LLM calls |",
        "|---:|---|---:|:---:|:---:|:---:|---:|---:|",
    ]
    for i, r in enumerate(ranked, start=1):
        hits = {m.get("matched_key") for m in r.matched if m.get("matched_key")}
        nsa = "✓" if "nsa" in hits else "✗"
        qwn = "✓" if "qwen3_omni" in hits else "✗"
        ern = "✓" if "ernie_5" in hits else "✗"
        lines.append(
            f"| {i} | {r.model} | {r.score}/3 | {nsa} | {qwn} | {ern} "
            f"| {len(r.unmatched)} | {len(r.llm_judge_log)} |"
        )
    lines.append("")
    lines.append("## 每模型 matched 明细")
    lines.append("")
    for r in ranked:
        lines.append(f"### {r.model} (score={r.score}/3)")
        if r.matched:
            for m in r.matched:
                lines.append(
                    f"- ✅ **{wl[m['matched_key']]['canonical_title']}** "
                    f"[{m.get('tier')}] — paper: `{m.get('paper_title') or '—'}` / "
                    f"company: `{m.get('company') or '—'}` / "
                    f"Δ: `{m.get('stock_change_pct') or '—'}` — {m.get('reason')}"
                )
        else:
            lines.append("- （无 matched）")
        if r.disqualified_refs:
            lines.append(
                "- ⚠️  引用了已知不合格事件 ("
                + ", ".join({d["disqualified_id"] for d in r.disqualified_refs})
                + ")，按规则不扣分"
            )
        if r.unmatched:
            unmatched_brief = [
                f"{u.get('paper_title') or '?'}[{u.get('tier')}]"
                for u in r.unmatched[:6]
            ]
            lines.append(
                f"- unmatched ({len(r.unmatched)}): "
                + "; ".join(unmatched_brief)
                + ("…" if len(r.unmatched) > 6 else "")
            )
        lines.append("")
    (out_dir / "ranking_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


# ═══════════════════════════════════════════════════════════════════════════
# Entry
# ═══════════════════════════════════════════════════════════════════════════


def main() -> None:
    global LLM_JUDGE_MODEL
    ap = argparse.ArgumentParser(description=f"Query {QUERY_ID} auto-scorer")
    ap.add_argument("--models", nargs="+", required=True, help="name=path list")
    ap.add_argument("--output-dir", default=str(THIS / "auto_scores"))
    ap.add_argument("--primary-model", default=None)
    ap.add_argument("--secondary-model", default=None)
    ap.add_argument("--parallel-models", nargs="+", default=None)
    ap.add_argument("--analyzer-model", default=None)
    ap.add_argument("--concurrency", type=int, default=None)
    ap.add_argument("--skip-extract", action="store_true")
    ap.add_argument("--judge-model", default=LLM_JUDGE_MODEL)
    args = ap.parse_args()
    LLM_JUDGE_MODEL = args.judge_model

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.skip_extract:
        all_results: dict[str, dict] = {}
        for spec in args.models:
            name = spec.split("=", 1)[0].strip()
            ep = out_dir / name / "extraction.json"
            if not ep.exists():
                sys.exit(f"[ERROR] missing extraction: {ep}")
            all_results[name] = json.loads(ep.read_text(encoding="utf-8"))
    else:
        all_results = run_pipeline(
            query_id=QUERY_ID,
            query_text=QUERY_TEXT,
            entities=ENTITIES,
            prompt_hints=PROMPT_HINTS,
            schema=VALUE_SCHEMA,
            models_input=args.models,
            output_dir=out_dir,
            primary=args.primary_model,
            secondary=args.secondary_model,
            parallel=args.parallel_models,
            analyzer=args.analyzer_model,
            concurrency=args.concurrency,
        )

    # Lazy LLM client.
    client = None
    with contextlib.suppress(SystemExit):
        client = get_client()

    results: list[ScoreResult] = []
    for model_name, payload in all_results.items():
        r = _score_one_model(model_name, payload, client=client)
        _write_model_score(out_dir, r)
        results.append(r)

    _write_ranking_report(out_dir, results)
    _write_scores_json(out_dir, results)

    total_llm = sum(len(r.llm_judge_log) for r in results)
    print("\n" + "─" * 64)
    print(f"Query {QUERY_ID} scoring done. LLM judge calls: {total_llm}")
    for r in sorted(results, key=lambda x: (-x.score, x.model)):
        mark = "✅" if r.score == 3 else ("🟡" if r.score > 0 else "❌")
        hits = sorted({m.get("matched_key") for m in r.matched if m.get("matched_key")})
        print(
            f"  {mark} {r.model:28s} {r.score}/3  matched={hits}  "
            f"unmatched={len(r.unmatched)}  llm={len(r.llm_judge_log)}"
        )


if __name__ == "__main__":
    main()
