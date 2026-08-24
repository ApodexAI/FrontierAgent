"""
Query 10 — 2016-2025 Chinese film filter auto-scorer (v2.1 extraction
+ 2026-04-24 alignment layer).

Scoring rule:
  维度 A（条件符合度）:
    命中 BASELINE_FILMS（3/3 条件）       → +3
    命中 PARTIAL_FILMS（1-2/3 条件）      →  0（部分匹配，不加不扣）
    都不在（未知/编造 / 对齐返回 null）      → -1（扣分）
  维度 B（导演，仅 A=3 时评分）: 对 +1, 错 0（不扣）
  维度 C（上映年份，仅 A=3 时评分）: 对 +1, 错 0（不扣）
  单片小计 = A + B + C，可负
  MAX_SCORE = N_film × 5 = 30（6 基准片 × (3+1+1)）

Pipeline:
  1. Extraction — v2 pipeline produces per-model canonical film list.
  2. Alignment — 3-model vote + field cross-check + kw sanity + optional
     judge LLM maps each extracted film to a BASELINE / PARTIAL id (or null).
  3. Scoring — A/B/C based on canonical_id lookup.
"""

from __future__ import annotations

import argparse
import json
import sys
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
from pipeline.alignment import (  # noqa: E402
    align_claims,
    apply_null_resolutions,
    export_null_claims_for_review,
    persist_new_baseline_entries,
)
from pipeline.extraction_pipeline import get_client, run_pipeline  # noqa: E402

# ═══════════════════════════════════════════════════════════════════════════
# Benchmark — curated baseline/partial film sets (2026-04-24 revision).
# ═══════════════════════════════════════════════════════════════════════════

BASELINE_FILMS: dict = {
    "相爱相亲": {
        "director": "张艾嘉",
        "release_year": ["2017"],
        "douban_rating": 8.4,
        "box_office_lt_1y": True,
        "award": "第37届金像奖最佳电影提名（未获奖）",
        "conditions_met": 3,
    },
    "沦落人": {
        "director": "陈小娟",
        "release_year": ["2018", "2019", "2018/2019"],
        "douban_rating": 8.2,
        "box_office_lt_1y": True,
        "award": "第38届金像奖最佳电影提名（未获奖）",
        "conditions_met": 3,
    },
    "地久天长": {
        "director": "王小帅",
        "release_year": ["2019"],
        "douban_rating": 8.0,
        "box_office_lt_1y": True,
        "award": "第32届金鸡奖最佳故事片提名（未获奖）",
        "conditions_met": 3,
    },
    "年少日记": {
        "director": "卓亦谦",
        "release_year": ["2023", "2024"],
        "douban_rating": 8.4,
        "box_office_lt_1y": True,
        "award": "第42届金像奖最佳电影提名（未获奖）",
        "conditions_met": 3,
    },
    "从今以后": {
        "director": "杨曜恺",
        "release_year": ["2024"],
        "douban_rating": 8.1,
        "box_office_lt_1y": True,
        "award": "第43届金像奖最佳电影提名（未获奖）",
        "conditions_met": 3,
    },
    "看我今天怎么说": {
        "director": "黄修平",
        "release_year": ["2024", "2025"],
        "douban_rating": 8.3,
        "box_office_lt_1y": True,
        "award": "第43届金像奖最佳电影提名（未获奖）",
        "conditions_met": 3,
    },
}

# Source-verified updates:
#   + 新增《白日之下》(简君晋 2023/2024) — 金像奖第42届最佳电影提名 + 内地
#     票房 1220 万 ✓，仅豆瓣 7.8 < 8.0；来源：百度百科、维基百科。
#   * 《浊水漂流》fail_reason 7.7 → 7.6（源头核实）。
#   * 《正义回廊》fail_reason 文本注明豆瓣波动（开画 8.3+ 后回落至 7.5-7.9）。
PARTIAL_FILMS: dict = {
    "叔·叔": {"conditions_met": 2, "fail_reason": "豆瓣约7.6 < 8.0"},
    "浊水漂流": {"conditions_met": 2, "fail_reason": "豆瓣约7.6 < 8.0"},
    "爱情神话": {"conditions_met": 2, "fail_reason": "票房约2.6亿 > 1亿"},
    "正义回廊": {
        "conditions_met": 2,
        "fail_reason": "豆瓣约7.5~7.9 < 8.0（开画 8.3+ 后回落）",
    },
    "白日之下": {"conditions_met": 2, "fail_reason": "豆瓣约7.8 < 8.0"},
    # web-search-verified 2026-05-01
    "智齿": {
        "director": "郑保瑞",
        "release_year": ["2021"],
        "conditions_met": 2,
        "note": "智齿 (郑保瑞 2021): 第40届香港电影金像奖最佳电影提名 YES（败给《怒火·重案》），未获奖。豆瓣评分 7.2（百度百科+维基均为 7.2），claim 声称 8.1 不实，条件1不满足。内地未公映，box office=0 < 1亿，条件2满足。条件3满足（提名未获奖）。2/3 → PARTIAL ⚠️。",
    },
    # web-search-verified 2026-05-01
    "窄路微尘": {
        "director": "林森",
        "release_year": ["2022"],
        "conditions_met": 2,
        "note": "窄路微尘 (林森 2022): 第41届香港电影金像奖最佳电影提名 YES（败给《给十九岁的我》），未获奖。豆瓣评分 7.3（搜索结果 7.3），claim 声称 8.2 不实，条件1不满足。内地未公映，box office=0 < 1亿，条件2满足。条件3满足。2/3 → PARTIAL ⚠️。",
    },
}

N_FILM = len(BASELINE_FILMS)  # 6
MAX_SCORE = N_FILM * 5  # 30


# ═══════════════════════════════════════════════════════════════════════════
# Baseline spec for alignment layer (pipeline/alignment.py)
# ═══════════════════════════════════════════════════════════════════════════


def build_baselines() -> list[dict]:
    """Convert BASELINE_FILMS + PARTIAL_FILMS into alignment baseline spec.

    Canonical ids are the film names themselves (e.g. "相爱相亲")."""
    out = []
    for name, info in BASELINE_FILMS.items():
        out.append(
            {
                "id": name,
                "description": (
                    f"《{name}》— 导演: {info['director']}；"
                    f"上映: {'/'.join(info['release_year'])}；"
                    f"豆瓣 {info['douban_rating']}；"
                    f"{info['award']} [BASELINE 3/3 条件满足]"
                ),
                "match_fields": {
                    "director": info["director"],
                    "release_year": info["release_year"],
                },
                "kw": [name, info["director"]],
                "film_class": "baseline",
            }
        )
    for name, info in PARTIAL_FILMS.items():
        # Stage C-added entries use `note` instead of `fail_reason`; fall back gracefully.
        reason = info.get("fail_reason") or info.get("note") or "partial match"
        out.append(
            {
                "id": name,
                "description": (
                    f"《{name}》 [PARTIAL {info['conditions_met']}/3: {reason}]"
                ),
                "match_fields": {},
                "kw": [name],
                "film_class": "partial",
            }
        )
    return out


# ═══════════════════════════════════════════════════════════════════════════
# Scoring from aligned claims (A=3/0/-1, B/C correct=+1 wrong=0)
# ═══════════════════════════════════════════════════════════════════════════


def score_aligned_film(aligned_claim: dict) -> tuple[int, int, int, dict]:
    """Score a single aligned claim. `aligned_claim` has shape:
    {canonical_id, alignment_confidence, raw: {film_name, director, ...}, ...}
    """
    raw = aligned_claim.get("raw", {}) or {}
    cid = aligned_claim.get("canonical_id")
    conf = aligned_claim.get("alignment_confidence", "medium")
    director = raw.get("director", "") or ""
    year = str(raw.get("release_year", "") or "").replace("年", "").strip()
    claimed_name = str(raw.get("film_name", "") or "").strip()

    # needs_review → treat as "unknown" (no scoring), but do NOT apply the -1
    # penalty reserved for confident false claims; isolate into a separate path.
    if conf == "needs_review":
        a_score = 0
        cond_met = 0
        a_status = "needs_review"
    elif cid == "__HALLUCINATION__":
        # null_resolution marked this as verified hallucination → -1 penalty
        a_score = -1
        cond_met = 0
        a_status = "hallucination"
    elif cid in BASELINE_FILMS:
        a_score = 3
        cond_met = 3
        a_status = "baseline"
    elif cid in PARTIAL_FILMS:
        a_score = 0
        cond_met = PARTIAL_FILMS[cid]["conditions_met"]
        a_status = "partial"
    else:
        # canonical_id is None or unknown → unknown/fabricated
        a_score = -1
        cond_met = 0
        a_status = "unknown"

    b_score = 0
    c_score = 0
    director_correct: bool | None = None
    year_correct: bool | None = None
    if a_status == "baseline":
        bl = BASELINE_FILMS[cid]
        director_correct = director == bl["director"]
        b_score = 1 if director_correct else 0
        year_correct = year in bl["release_year"]
        c_score = 1 if year_correct else 0

    return (
        a_score,
        b_score,
        c_score,
        {
            "claimed_name": claimed_name,
            "canonical_id": cid,
            "a_status": a_status,
            "conditions_met": cond_met,
            "A_score": a_score,
            "director_claimed": director,
            "director_correct": director_correct,
            "B_score": b_score,
            "year_claimed": year,
            "year_correct": year_correct,
            "C_score": c_score,
            "subtotal": a_score + b_score + c_score,
            "alignment_confidence": conf,
            "alignment_reasoning": aligned_claim.get("alignment_reasoning", ""),
            "judge_invoked": aligned_claim.get("judge_invoked", False),
        },
    )


def score_model(model_name: str, aligned_claims: list[dict]) -> dict:
    details = []
    total = 0
    for ac in aligned_claims:
        a, b, c, detail = score_aligned_film(ac)
        details.append(detail)
        total += a + b + c
    hit_baseline = sum(1 for d in details if d["a_status"] == "baseline")
    partial = sum(1 for d in details if d["a_status"] == "partial")
    miss = sum(1 for d in details if d["a_status"] == "unknown")
    needs_review = sum(1 for d in details if d["a_status"] == "needs_review")
    return {
        "model": model_name,
        "total_score": total,
        "max_score": MAX_SCORE,
        # canonical 0-1 fraction（run_all.extract_total_rates 读这个）；
        # score_rate 同步改为 0-1（之前误存为 0-100 百分比，导致 Q10 被放大 100×
        # 进跨题平均，污染 ranking.md）。下方 ranking_report / stdout 显示位
        # 已改为 `score_rate * 100` 渲染成百分比。
        "total_rate": round(total / MAX_SCORE, 4),
        "score_rate": round(total / MAX_SCORE, 4),
        "films_listed": len(aligned_claims),
        "hit_baseline": hit_baseline,
        "partial_match": partial,
        "no_match": miss,
        "needs_review": needs_review,
        "per_film": details,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Input adapters — raw extraction claims and reloading alignment results
# ═══════════════════════════════════════════════════════════════════════════


def load_raw_claims(extraction_dir: Path) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for d in Path(extraction_dir).iterdir():
        if not d.is_dir():
            continue
        ext_file = d / "extraction.json"
        if not ext_file.exists():
            continue
        payload = json.loads(ext_file.read_text(encoding="utf-8"))
        entities = payload.get("entities", [])
        if not entities:
            out[d.name] = []
            continue
        val = (entities[0].get("canonical") or {}).get("value") or []
        if not isinstance(val, list):
            val = []
        out[d.name] = val
    return out


def _load_alignment(out_dir: Path) -> dict[str, list[dict]]:
    aligned: dict[str, list[dict]] = {}
    for d in Path(out_dir).iterdir():
        if not d.is_dir():
            continue
        af = d / "alignment.json"
        if not af.exists():
            continue
        payload = json.loads(af.read_text(encoding="utf-8"))
        aligned[d.name] = payload.get("aligned_claims", [])
    return aligned


# ═══════════════════════════════════════════════════════════════════════════
# Outputs
# ═══════════════════════════════════════════════════════════════════════════


def build_scores_json(all_results: list[dict]) -> dict:
    sorted_r = sorted(all_results, key=lambda x: -x["total_score"])
    return {
        "query_id": QUERY_ID,
        "scoring_mode": "per_film",
        "max_score": MAX_SCORE,
        "n_baseline_films": N_FILM,
        "baseline_films": list(BASELINE_FILMS.keys()),
        "extraction_pipeline": "v2.1 (primary=claude-sonnet-4, secondary=gpt-5, "
        "analyzer=claude-opus-4.6)",
        "results": {r["model"]: r for r in all_results},
        "ranking": [
            {
                "rank": i + 1,
                "model": r["model"],
                "score": r["total_score"],
                "rate": f"{r['score_rate'] * 100:.1f}%",
            }
            for i, r in enumerate(sorted_r)
        ],
    }


def build_ranking_md(all_results: list[dict]) -> str:
    sorted_r = sorted(all_results, key=lambda x: -x["total_score"])
    lines = [
        "# Query 10 排名报告（v2.1 抽取）",
        "",
        "> 抽取流水线：primary=claude-sonnet-4, secondary=gpt-5, "
        "analyzer=claude-opus-4.6",
        f"> 评分：逐片 A（命中基准 +3 / 部分匹配 0 / 未知编造 -1）"
        f" + B 导演 (对 +1, 错 0) + C 年份 (对 +1, 错 0)；B/C 仅 A=3 时计分。"
        f"6 基准片 × 单片满分 5 → 理论满分 {MAX_SCORE}，总分可为负。",
        "",
        "| Rank | Model | Score | Rate | Hit | Partial | Miss | Listed |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for i, r in enumerate(sorted_r, 1):
        lines.append(
            f"| {i} | {r['model']} | {r['total_score']}/{MAX_SCORE} "
            f"| {r['score_rate'] * 100:.1f}% | {r['hit_baseline']}/{N_FILM} "
            f"| {r['partial_match']} | {r['no_match']} | {r['films_listed']} |"
        )
    return "\n".join(lines) + "\n"


# ═══════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════


def main() -> None:
    ap = argparse.ArgumentParser(description="Query 10 auto-scorer (v2.1+align)")
    ap.add_argument("--models", nargs="+")
    ap.add_argument("--output-dir", default=str(THIS / "auto_scores"))
    ap.add_argument("--primary-model", default=None)
    ap.add_argument("--secondary-model", default=None)
    ap.add_argument("--parallel-models", nargs="+", default=None)
    ap.add_argument("--analyzer-model", default=None)
    ap.add_argument("--aligner-models", nargs="+", default=None)
    ap.add_argument("--judge-model", default=None)
    ap.add_argument("--concurrency", type=int, default=None)
    ap.add_argument("--skip-extract", action="store_true")
    ap.add_argument("--skip-align", action="store_true")
    args = ap.parse_args()

    if not args.models:
        sys.exit("[ERROR] --models required.")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Stage 1: extraction
    if not args.skip_extract:
        run_pipeline(
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

    # Stage 2: alignment
    baselines = build_baselines()
    if args.skip_align:
        aligned = _load_alignment(out_dir)
    else:
        client = get_client()
        overrides: dict = {}
        if args.aligner_models:
            overrides["aligner_models"] = args.aligner_models
        if args.judge_model:
            overrides["judge_model"] = args.judge_model
        if args.concurrency:
            overrides["concurrency"] = args.concurrency
        raw_claims = load_raw_claims(out_dir)
        aligned = align_claims(
            client,
            claims_by_model=raw_claims,
            baselines=baselines,
            query_text=QUERY_TEXT,
            output_dir=out_dir,
            overrides=overrides,
        )

    # Stage 3a: export null claims for external verification (with context_span
    # from raw model responses so the verifier can recover from extraction drops)
    null_items = export_null_claims_for_review(
        aligned,
        out_dir / "null_review.json",
        query_id=QUERY_ID,
        query_text=QUERY_TEXT,
        models_input=args.models,
    )
    print(
        f"\n[*] Exported {len(null_items)} null/needs_review claims → null_review.json"
    )

    # Stage 3b: apply null_resolutions.json if present (verification feedback)
    resolutions_path = out_dir / "null_resolutions.json"
    if resolutions_path.exists():
        print("[*] Found null_resolutions.json, applying …")
        new_baseline_entries = apply_null_resolutions(aligned, resolutions_path)
        if new_baseline_entries:
            print(
                f"[*] {len(new_baseline_entries)} new baseline entries from resolutions"
            )
            # merge into runtime BASELINE_FILMS / PARTIAL_FILMS so score_aligned_film
            # recognizes the new ids on this run
            for e in new_baseline_entries:
                judgment = e.get("judgment", "⚠️")
                mf = e.get("match_fields", {})
                target = BASELINE_FILMS if judgment == "✅" else PARTIAL_FILMS
                entry = {
                    "director": mf.get("director", ""),
                    "release_year": mf.get("release_year", []),
                    "conditions_met": 3 if judgment == "✅" else 2,
                    "fail_reason": e.get("verification_notes", ""),
                }
                target[e["id"]] = entry
            # persist for future runs
            persist_new_baseline_entries(new_baseline_entries, __file__)
    else:
        print("[*] No null_resolutions.json (run web-search agent to produce one).")

    # Stage 4: scoring (with resolutions applied, if any)
    all_results = [score_model(m, aligned.get(m, [])) for m in aligned]

    (out_dir / "scores.json").write_text(
        json.dumps(build_scores_json(all_results), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (out_dir / "ranking_report.md").write_text(
        build_ranking_md(all_results),
        encoding="utf-8",
    )

    print("\n" + "─" * 60)
    print("Query 10 scoring done.")
    for i, r in enumerate(sorted(all_results, key=lambda x: -x["total_score"]), 1):
        print(
            f"  #{i}  {r['model']:28s}  {r['total_score']:>3}/{MAX_SCORE}"
            f"  ({r['score_rate'] * 100:.1f}%)  hit={r['hit_baseline']}/{N_FILM}"
            f"  review={r.get('needs_review', 0)}"
        )


if __name__ == "__main__":
    main()
