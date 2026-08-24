"""
Query 18 — 建筑师谜题：摩天大楼现名识别 — auto-scorer.

Scoring rule (Q12-strict + 未答惩罚):
  ✅ 答对 OUE Downtown 1（原 DBS Building Tower 1）         → +1.0
  ❌ 答错（任何已知错答 D2..Dn）                              → -1.0
  __HALLUCINATION__（Stage C 验证为虚构建筑）                → -1.0
  null + Stage C unresolved（基准外、无法验证）              → -1.0
  alignment_confidence = "needs_review"                     → -1.0
  模型完全未给答案（extraction.value=[]）                    → -1.0
  MAX_SCORE = 1.0（仅 D1 贡献正分）
  total_score 取值范围 {-1.0, +1.0}（每模型只有 1 条 claim）

Pipeline (单实体 + 闭合集合骨架):
  1. Extraction — v2 pipeline (5-LLM voting + analyzer; see pipeline/extraction_pipeline.py)
  2. Alignment — 3-aligner vote + field cross-check + judge LLM
  3. Null verification — Stage C agent (WebSearch); produce null_resolutions.json
  4. Scoring — deterministic lookup from canonical_id to benchmark score

Usage:
  # 第一次全量跑（抽取 + 对齐 + 导出 null_review.json）
  python3 query_18/auto_scorer.py --models \\
    chatglm=<crawl-output>/chatglm.json \\
    claude=<crawl-output>/claude.json \\
    ...

  # Stage C agent 产出 null_resolutions.json 后，应用 + 重打分
  python3 query_18/auto_scorer.py --skip-extract --skip-align --models \\
    chatglm=<crawl-output>/chatglm.json ...
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
# Benchmark — 1 ✅ + several ❌（预填已知错答以便 alignment 直接命中扣分）
#   D1     : ✅ 唯一正解 OUE Downtown 1（原 DBS Building Tower 1）
#   D2-D6  : ❌ 已知错答（历史名 / 错认建筑 / 错认建筑师）
# Cross-verified 2026-05-02 against:
#   - en.wikipedia.org/wiki/OUE_Downtown
#   - en.wikipedia.org/wiki/Lim_Chong_Keat
#   - bfi.org/2015/06/02/pioneer-architect-lim-chong-keat
#   - mplus.org.hk Architects Team 3 / Lim Chong Keat archive
# ═══════════════════════════════════════════════════════════════════════════

SNAPSHOT_DATE = "2026-05-02"
SCORING_MODE = "C-strict"  # null+unresolved → -1; 未答 → -1（按 user 要求）

DIMS = [
    # ─── ✅ 唯一正解 (+1) ───
    (
        "D1",
        "OUE Downtown 1 / OUE Downtown Tower 1 / 大华城市中心一号 — 新加坡 6 Shenton Way，"
        "1975 年完工时名为 DBS Building Tower 1，2010 年 OUE 收购后改名 OUE Downtown 1。"
        "建筑师 Lim Chong Keat（林苍吉），MIT 校友，与 Buckminster Fuller 终生好友，"
        "两人 1975-1983 年在巴厘岛/槟城共同组织 Campuan Meetings。"
        "地基为 4 个直径 6.8 米的圆形 bored caissons（曾入吉尼斯纪录）",
        "✅",
        1.0,
        [
            "oue downtown",
            "oue downtown 1",
            "oue downtown tower 1",
            "大华城市中心",
            "大华城市中心一号",
            "lim chong keat",
            "林苍吉",
            "林仲杰",
            "林昌杰",
            "campuan",
            "campuhan",
            "architects team 3",
        ],
    ),
    # ─── ❌ 已知错答 (-1)：历史名（违反"现在的名字"硬限定）───
    (
        "D2",
        "DBS Building / DBS Tower / 星展大厦 / Development Bank of Singapore Building —"
        " 该楼的历史名（1975-2010）。题目硬限定'现在的名字'，只答历史名违反限定。"
        "实体识别正确（同一栋楼）但语义违限定 → ❌",
        "❌",
        -1.0,
        [
            "dbs building",
            "dbs tower",
            "dbs headquarters",
            "星展大厦",
            "星展银行大厦",
            "星展银行总部",
            "development bank of singapore",
        ],
    ),
    # ─── ❌ 已知错答 (-1)：错认建筑（KOMTAR 是 Lim Chong Keat 另一项目）───
    (
        "D3",
        "KOMTAR / Komtar Tower / 光大广场 — 槟城地标，林苍吉与富勒确有合作"
        "（KOMTAR 圆顶项目），但本题描述的'巨型圆形沉箱地基的摩天大楼'非 KOMTAR。"
        "建筑师识别正确但建筑识别错 → ❌",
        "❌",
        -1.0,
        ["komtar", "komtar tower", "光大广场", "槟城光大"],
    ),
    # ─── ❌ 已知错答 (-1)：错认建筑师为贝聿铭 ───
    (
        "D4",
        "Bank of China Tower (Hong Kong) / 香港中银大厦 — 贝聿铭设计，1990 年完工。"
        "贝聿铭虽 MIT 出身但与富勒非'终生好友'级关系，且中银大厦地基非'4 个圆形大沉箱'"
        "（用了 110 个密集小沉箱）。建筑师/建筑双重错认 → ❌",
        "❌",
        -1.0,
        [
            "bank of china tower",
            "中银大厦",
            "香港中银",
            "boc tower",
            "贝聿铭",
            "i.m. pei",
            "ieoh ming pei",
        ],
    ),
    # ─── ❌ 已知错答 (-1)：错认建筑师为 Edward Durell Stone ───
    (
        "D5",
        "Aon Center (Chicago) / Amoco Building / Standard Oil Building — 芝加哥 1973 年"
        "完工时名 Standard Oil，1985 改 Amoco，1999 改 Aon Center。Edward Durell Stone "
        "(MIT) + Perkins & Will 设计，地基用大量 caissons。但 Stone 的'岛屿研讨会'指向 "
        "Doxiadis 的 Delos Symposia 而非本题描述。建筑师错认 → ❌",
        "❌",
        -1.0,
        [
            "aon center",
            "aon center chicago",
            "amoco building",
            "standard oil building",
            "edward durell stone",
            "stone",
        ],
    ),
    # ─── ❌ 已知错答 (-1)：错认 Doxiadis ───
    (
        "D6",
        "Doxiadis 相关建筑（如伊斯兰堡规划等）— Constantinos Doxiadis 是希腊建筑师，"
        "组织 Delos Symposia 时富勒为常客，但 Doxiadis 毕业于雅典/柏林而非美国理工大学，"
        "且其代表作非'圆形沉箱地基的摩天楼'。整体错认 → ❌",
        "❌",
        -1.0,
        [
            "doxiadis",
            "constantinos doxiadis",
            "多西阿迪斯",
            "delos",
            "ekistics",
            "islamabad",
            "伊斯兰堡",
        ],
    ),
]

DIM_MAP = {
    d[0]: {"id": d[0], "name": d[1], "judgment": d[2], "score": d[3], "kw": d[4]}
    for d in DIMS
}
MAX_SCORE = sum(d[3] for d in DIMS if d[3] > 0)  # 仅累加 ✅ 正分基准 = 1.0


# ═══════════════════════════════════════════════════════════════════════════
# DIMS → alignment baseline spec
# ═══════════════════════════════════════════════════════════════════════════


def _derive_match_fields(description: str) -> dict:
    """从 DIMS description 提取结构化字段供 alignment cross-check。
    本题字段较少（建筑名是核心），主要靠 kw 匹配；这里仅尝试解析 architect。"""
    fields: dict = {}
    # 尝试从 description 抽出 architect 名（用于和 claim.architect_claimed 对照）
    for arch_kw in [
        "Lim Chong Keat",
        "贝聿铭",
        "I.M. Pei",
        "Edward Durell Stone",
        "Doxiadis",
    ]:
        if arch_kw.lower() in description.lower():
            fields["architect_claimed"] = arch_kw
            break
    return fields


def build_baselines() -> list[dict]:
    """Convert DIMS into the baseline spec consumed by pipeline.alignment."""
    out = []
    for d_id, desc, judgment, score, kw in DIMS:
        out.append(
            {
                "id": d_id,
                "description": f"{desc} [基准: {judgment}]",
                "match_fields": _derive_match_fields(desc),
                "kw": kw,
                "judgment": judgment,
                "score": score,
            }
        )
    return out


# ═══════════════════════════════════════════════════════════════════════════
# Load raw claims from v2 extraction output
# ═══════════════════════════════════════════════════════════════════════════


def load_raw_claims(extraction_dir: Path) -> dict[str, list[dict]]:
    """Read each model's `{dir}/{model}/extraction.json` and return the raw
    `entities[0].canonical.value` list."""
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
        canonical_value = (entities[0].get("canonical") or {}).get("value") or []
        if not isinstance(canonical_value, list):
            canonical_value = []
        out[d.name] = canonical_value
    return out


# ═══════════════════════════════════════════════════════════════════════════
# Scoring from aligned claims
# ═══════════════════════════════════════════════════════════════════════════


def score_aligned(aligned_by_model: dict[str, list[dict]]) -> dict[str, dict]:
    """Q18 scoring (单实体单维 + Q12-strict 闭合扣分):

    - claims = []（模型完全未给答案）          → -1.0
    - canonical_id ∈ DIM_MAP                  → DIM_MAP[cid].score
    - canonical_id == "__HALLUCINATION__"     → -1.0
    - canonical_id is None (null+unresolved)  → -1.0
    - alignment_confidence == "needs_review"  → -1.0（按 user 要求）
    - 同模型若多条 claim：本题应该至多 1 条；如出现多条只评第一条
    """
    all_scores: dict[str, dict] = {}
    for model_name, claims in aligned_by_model.items():
        scored: list[dict] = []
        unverified: list[dict] = []
        total = 0.0

        # 模型完全未给答案 → -1
        if not claims:
            scored.append(
                {
                    "id": "__NO_ANSWER__",
                    "name": "(model gave no definitive answer)",
                    "claimed_building": "",
                    "score": -1.0,
                    "judgment": "❌",
                    "reason": "模型完全未给出建筑名（extraction.value=[] 或 not_mentioned=true）",
                }
            )
            total = -1.0
            all_scores[model_name] = {
                "total_score": total,
                "max_score": MAX_SCORE,
                "score_rate": -1.0,
                "total_rate": round(total / MAX_SCORE, 4) if MAX_SCORE else 0.0,
                "dimensions_answered": 0,
                "per_dimension": scored,
                "unverified_claims": unverified,
            }
            continue

        # 仅评第一条 claim（本题单实体单答案）
        c = claims[0]
        raw = c.get("raw", {}) or {}
        cid = c.get("canonical_id")
        conf = c.get("alignment_confidence", "medium")
        reason = c.get("alignment_reasoning", "")
        judge_invoked = c.get("judge_invoked", False)
        claimed_building = (
            raw.get("building_name_now")
            or raw.get("building_name_historical")
            or raw.get("name", "")
        )

        if conf == "needs_review":
            scored.append(
                {
                    "id": "__NEEDS_REVIEW__",
                    "name": "(needs_review, treated as wrong per user rule)",
                    "claimed_building": claimed_building,
                    "canonical_id_tentative": cid,
                    "score": -1.0,
                    "judgment": "❌",
                    "confidence": conf,
                    "judge_invoked": judge_invoked,
                    "reason": f"needs_review: {reason}",
                }
            )
            total = -1.0
        elif cid == "__HALLUCINATION__":
            scored.append(
                {
                    "id": "__HALLUCINATION__",
                    "name": "(verified hallucination)",
                    "claimed_building": claimed_building,
                    "score": -1.0,
                    "judgment": "❌",
                    "confidence": conf,
                    "judge_invoked": judge_invoked,
                    "reason": reason,
                }
            )
            total = -1.0
        elif cid and cid in DIM_MAP:
            d = DIM_MAP[cid]
            scored.append(
                {
                    "id": cid,
                    "name": d["name"],
                    "claimed_building": claimed_building,
                    "score": d["score"],
                    "judgment": d["judgment"],
                    "confidence": conf,
                    "judge_invoked": judge_invoked,
                    "reason": reason or f"对齐到 {cid}（基准 {d['judgment']}）",
                }
            )
            total = d["score"]
        else:
            # null + Stage C unresolved → -1（Q12 strict 规则）
            scored.append(
                {
                    "id": "__UNRESOLVED_NULL__",
                    "name": "(null + unresolved, treated as wrong)",
                    "claimed_building": claimed_building,
                    "score": -1.0,
                    "judgment": "❌",
                    "confidence": conf,
                    "judge_invoked": judge_invoked,
                    "reason": reason or "未对齐到任何基准条目，且 Stage C 未能验证",
                }
            )
            total = -1.0

        # 如有多条 claim，把额外的记到 unverified（不再计分）
        if len(claims) > 1:
            for extra in claims[1:]:
                er = extra.get("raw", {}) or {}
                unverified.append(
                    {
                        "claimed_building": er.get("building_name_now")
                        or er.get("name", ""),
                        "canonical_id_tentative": extra.get("canonical_id"),
                        "reason": "extra claim beyond first (single-answer query)",
                        "score": 0.0,
                    }
                )

        all_scores[model_name] = {
            "total_score": total,
            "max_score": MAX_SCORE,
            "score_rate": total,  # 单维 score_rate = total_score
            "total_rate": round(total / MAX_SCORE, 4) if MAX_SCORE else 0.0,
            "dimensions_answered": 1,
            "per_dimension": scored,
            "unverified_claims": unverified,
        }
    return all_scores


# ═══════════════════════════════════════════════════════════════════════════
# Outputs
# ═══════════════════════════════════════════════════════════════════════════


def build_scores_json(all_scores: dict[str, dict]) -> dict:
    ranking = sorted(all_scores.items(), key=lambda x: -x[1]["total_score"])
    unverified_all = []
    for m, s in all_scores.items():
        for uv in s.get("unverified_claims", []):
            unverified_all.append({"model": m, **uv, "action_needed": "需人工核实"})
    return {
        "query_id": QUERY_ID,
        "query_text": QUERY_TEXT,
        "scoring_mode": SCORING_MODE,
        "snapshot_date": SNAPSHOT_DATE,
        "max_score": MAX_SCORE,
        "min_score": -1.0,
        "correct_answer": "OUE Downtown 1（原 DBS Building Tower 1）",
        "extraction_pipeline": "v2 (primary=claude-sonnet-4, secondary=gpt-5, analyzer=claude-opus-4.6)",
        "results": all_scores,
        "ranking": [
            {
                "rank": i + 1,
                "model": m,
                "score": s["total_score"],
                "rate": f"{s['total_rate'] * 100:+.0f}%",
            }
            for i, (m, s) in enumerate(ranking)
        ],
        "unverified_items": unverified_all,
    }


def build_ranking_md(all_scores: dict[str, dict]) -> str:
    ranked = sorted(
        all_scores.items(),
        key=lambda x: (-x[1]["total_score"], x[0]),
    )
    lines = [
        "# Query 18 排名报告（建筑师谜题：摩天大楼现名）",
        "",
        f"> Snapshot: {SNAPSHOT_DATE}  Mode: {SCORING_MODE}  Max: +{MAX_SCORE} / Min: -1",
        "> 标准答案: **OUE Downtown 1**（原 DBS Building Tower 1，2010 改名）",
        "> 抽取流水线：Primary=claude-sonnet-4, Secondary=gpt-5, Analyzer=claude-opus-4.6",
        "> 评分：✅+1（D1） / ❌-1（D2..D6 已知错答 / __HALLUCINATION__ / null+unresolved / "
        "needs_review / 模型未答）",
        "",
        "| Rank | Model | Claimed Building | Canonical | Score | Judge |",
        "|---:|---|---|---|---:|:---:|",
    ]
    for i, (m, s) in enumerate(ranked, 1):
        d = s["per_dimension"][0] if s["per_dimension"] else {}
        claimed = d.get("claimed_building", "")
        cid = d.get("id", "")
        score = d.get("score", 0)
        judg = d.get("judgment", "")
        lines.append(
            f"| {i} | {m} | {claimed or '（未给出）'} | {cid} | {score:+.0f} | {judg} |"
        )

    correct = sum(1 for s in all_scores.values() if s["total_score"] > 0)
    wrong = sum(1 for s in all_scores.values() if s["total_score"] < 0)
    lines.append("")
    lines.append(f"**统计**：{correct} 对 / {wrong} 错（共 {len(all_scores)} 模型）")
    return "\n".join(lines) + "\n"


# ═══════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════


def _load_alignment(out_dir: Path) -> dict[str, list[dict]]:
    """Reload alignment.json from disk when --skip-align is set."""
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


def main() -> None:
    ap = argparse.ArgumentParser(description="Query 18 auto-scorer")
    ap.add_argument(
        "--models", nargs="+", help="name=path list of model answer JSON files."
    )
    ap.add_argument("--output-dir", default=str(THIS / "auto_scores"))
    ap.add_argument("--primary-model", default=None)
    ap.add_argument("--secondary-model", default=None)
    ap.add_argument("--parallel-models", nargs="+", default=None)
    ap.add_argument("--analyzer-model", default=None)
    ap.add_argument("--aligner-models", nargs="+", default=None)
    ap.add_argument("--judge-model", default=None)
    ap.add_argument("--concurrency", type=int, default=None)
    ap.add_argument(
        "--skip-extract",
        action="store_true",
        help="Skip extraction; reuse existing extraction.json.",
    )
    ap.add_argument(
        "--skip-align",
        action="store_true",
        help="Skip alignment; reuse existing alignment.json.",
    )
    args = ap.parse_args()

    if not args.models:
        sys.exit("[ERROR] --models is required.")

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

    # Stage 3a: export null claims for Stage C verification
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

    # Stage 3b: apply null_resolutions.json if present
    resolutions_path = out_dir / "null_resolutions.json"
    if resolutions_path.exists():
        print("[*] Found null_resolutions.json, applying …")
        new_baseline_entries = apply_null_resolutions(
            aligned, resolutions_path, dims_ref=DIMS
        )
        if new_baseline_entries:
            print(
                f"[*] {len(new_baseline_entries)} new baseline entries from resolutions"
            )
            for e in new_baseline_entries:
                DIM_MAP[e["id"]] = {
                    "id": e["id"],
                    "name": e.get("description", e["id"]),
                    "judgment": e.get("judgment", "❌"),
                    "score": e.get("score", -1.0),
                    "kw": e.get("kw", []),
                }
            persist_new_baseline_entries(new_baseline_entries, __file__)
    else:
        print(
            f"[*] No null_resolutions.json yet."
            f"\n    → 用 Claude Code 读 {resolutions_path.parent}/null_review.json"
            f"\n      逐条 web 验证后产出 {resolutions_path.name}"
            f"\n      然后重跑：python3 {Path(__file__).name} "
            f"--skip-extract --skip-align --models …"
        )

    # Stage 4: scoring
    # 重要：score_aligned 需要"模型未答"的占位 → 显式补齐 aligned dict
    # raw_claims 的 keys 包含所有跑过 extraction 的模型；aligned dict 只含有 claim 的模型。
    raw_claims_for_keys = load_raw_claims(out_dir)
    for m in raw_claims_for_keys:
        if m not in aligned:
            aligned[m] = []
    scores = score_aligned(aligned)

    (out_dir / "scores.json").write_text(
        json.dumps(build_scores_json(scores), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (out_dir / "ranking_report.md").write_text(
        build_ranking_md(scores),
        encoding="utf-8",
    )

    print("\n" + "─" * 62)
    print("Query 18 scoring done.")
    for i, (m, s) in enumerate(
        sorted(scores.items(), key=lambda x: -x[1]["total_score"]), 1
    ):
        d = s["per_dimension"][0] if s["per_dimension"] else {}
        cid = d.get("id", "")
        print(
            f"  {i}. {m:28s} {s['total_score']:+5.1f}/{s['max_score']:+.1f}  cid={cid}"
        )


if __name__ == "__main__":
    main()
