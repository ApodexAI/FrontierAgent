"""Query 09 — Chinese Olympians ≥3 medals no-gold auto-scorer.

Reference answer (v1.1, locked 2026-05-05):
  Correct (19): 黄雪辰、孙文雁、李佳军、肖若腾、庞佳颖、谭良德、李敬、张博恒、
                唐钱婷、张文秀、贾宗洋、叶乔波、安玉龙、刘鸥、盛泽田、杨阳、
                郭爽、韩天宇、王春露
  Known-incorrect (5):
    - 呙俐 (奖牌数 <3)
    - 陈晓君 (奖牌数 <3)
    - 杨扬 / Yang Yang A (有 2002 冬奥短道金牌)
    - 王皓 (有奥运团体金牌，乒乓球)
    - 李静 (字形与李敬相近，但是不同人，不符合条件)

Scoring rule (max 19, can go negative):
  +1 per correct athlete mentioned
  -1 per known-incorrect athlete mentioned
   0 for unknowns (flagged for GT review)
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
from pipeline.alignment import align_claims  # noqa: E402
from pipeline.extraction_pipeline import get_client, run_pipeline  # noqa: E402

# ═══════════════════════════════════════════════════════════════════════════
# Benchmark
# ═══════════════════════════════════════════════════════════════════════════

SNAPSHOT_DATE = "2026-05-05"
BENCHMARK_VERSION = "1.1"

# Correct: 19 athletes — each (canonical_id, name, score, kw)
CORRECT = [
    ("C1",  "黄雪辰",   1.0, ["黄雪辰", "huang xuechen"]),
    ("C2",  "孙文雁",   1.0, ["孙文雁", "sun wenyan"]),
    ("C3",  "李佳军",   1.0, ["李佳军", "li jiajun"]),
    ("C4",  "肖若腾",   1.0, ["肖若腾", "xiao ruoteng"]),
    ("C5",  "庞佳颖",   1.0, ["庞佳颖", "pang jiaying"]),
    ("C6",  "谭良德",   1.0, ["谭良德", "tan liangde"]),
    ("C7",  "李敬",     1.0, ["李敬", "li jing"]),
    ("C8",  "张博恒",   1.0, ["张博恒", "zhang boheng"]),
    ("C9",  "唐钱婷",   1.0, ["唐钱婷", "tang qianting"]),
    ("C10", "张文秀",   1.0, ["张文秀", "zhang wenxiu"]),
    ("C11", "贾宗洋",   1.0, ["贾宗洋", "jia zongyang"]),
    ("C12", "叶乔波",   1.0, ["叶乔波", "ye qiaobo"]),
    ("C13", "安玉龙",   1.0, ["安玉龙", "an yulong"]),
    ("C14", "刘鸥",     1.0, ["刘鸥", "liu ou"]),
    ("C15", "盛泽田",   1.0, ["盛泽田", "sheng zetian"]),
    ("C16", "杨阳",     1.0, ["杨阳"]),  # NOT 杨扬
    ("C17", "郭爽",     1.0, ["郭爽", "guo shuang"]),
    ("C18", "韩天宇",   1.0, ["韩天宇", "han tianyu"]),
    ("C19", "王春露",   1.0, ["王春露", "wang chunlu"]),
]

# Incorrect: 5 traps — naming a trap costs -1
INCORRECT = [
    ("X1", "呙俐",  -1.0, ["呙俐", "guo li"], "奖牌数不满3枚"),
    ("X2", "陈晓君", -1.0, ["陈晓君", "chen xiaojun"], "奖牌数不满3枚"),
    ("X3", "杨扬",   -1.0, ["杨扬", "yang yang a"], "获 2002 冬奥短道速滑金牌"),
    ("X4", "王皓",   -1.0, ["王皓", "wang hao"], "获奥运团体金牌（乒乓球）"),
    ("X5", "李静",   -1.0, ["李静", "li jing"], "与李敬不同人，不符合条件"),
]

DIM_MAP = {}
for cid, name, score, kw in CORRECT:
    DIM_MAP[cid] = {"id": cid, "name": name, "score": score, "kw": kw, "judgment": "✅"}
for cid, name, score, kw, why in INCORRECT:
    DIM_MAP[cid] = {"id": cid, "name": name, "score": score, "kw": kw,
                    "judgment": "❌", "reason": why}

MAX_SCORE = sum(d["score"] for d in DIM_MAP.values() if d["score"] > 0)


# ═══════════════════════════════════════════════════════════════════════════
# DIMS → alignment baseline spec
# ═══════════════════════════════════════════════════════════════════════════


def build_baselines() -> list[dict]:
    out = []
    for cid, name, score, kw in CORRECT:
        out.append({
            "id": cid,
            "description": f"{name} — ≥3 奥运奖牌，无金牌（正确） [基准: ✅]",
            "match_fields": {"name": name},
            "kw": kw,
            "judgment": "✅",
            "score": score,
        })
    for cid, name, score, kw, why in INCORRECT:
        out.append({
            "id": cid,
            "description": f"{name} — {why}（已知错误） [基准: ❌]",
            "match_fields": {"name": name},
            "kw": kw,
            "judgment": "❌",
            "score": score,
        })
    return out


# ═══════════════════════════════════════════════════════════════════════════
# Load raw claims
# ═══════════════════════════════════════════════════════════════════════════


def load_raw_claims(out_dir: Path) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for d in Path(out_dir).iterdir():
        if not d.is_dir():
            continue
        ext = d / "extraction.json"
        if not ext.exists():
            continue
        payload = json.loads(ext.read_text(encoding="utf-8"))
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
# Score from aligned claims
# ═══════════════════════════════════════════════════════════════════════════


def score_aligned(aligned_by_model: dict[str, list[dict]]) -> dict[str, dict]:
    all_scores: dict[str, dict] = {}
    for model_name, claims in aligned_by_model.items():
        seen: set[str] = set()
        scored: list[dict] = []
        unverified: list[dict] = []
        total = 0.0
        for c in claims:
            raw = c.get("raw", {}) or {}
            cid = c.get("canonical_id")
            conf = c.get("alignment_confidence", "medium")
            reason = c.get("alignment_reasoning", "")

            if conf == "needs_review":
                unverified.append({
                    "name": raw.get("name", ""),
                    "canonical_id_tentative": cid,
                    "reason": f"needs_review: {reason}",
                })
                continue

            if cid and cid in DIM_MAP:
                if cid in seen:
                    continue
                seen.add(cid)
                d = DIM_MAP[cid]
                scored.append({
                    "id": cid,
                    "name": d["name"],
                    "score": d["score"],
                    "judgment": d["judgment"],
                    "model_answer_name": raw.get("name", ""),
                    "confidence": conf,
                })
                total += d["score"]
            else:
                unverified.append({
                    "name": raw.get("name", ""),
                    "canonical_id_tentative": cid,
                    "reason": reason or "not in baseline",
                })

        n_correct = sum(1 for s in scored if s["score"] > 0)
        n_wrong = sum(1 for s in scored if s["score"] < 0)
        all_scores[model_name] = {
            "total_score": total,
            "max_score": MAX_SCORE,
            "n_correct": n_correct,
            "n_known_incorrect_hit": n_wrong,
            "per_dimension": sorted(scored, key=lambda x: x["id"]),
            "unverified_claims": unverified,
        }
    return all_scores


# ═══════════════════════════════════════════════════════════════════════════
# Output writers
# ═══════════════════════════════════════════════════════════════════════════


def build_scores_json(all_scores: dict[str, dict]) -> dict:
    ranking = sorted(all_scores.items(), key=lambda x: -x[1]["total_score"])
    return {
        "query_id": QUERY_ID,
        "snapshot_date": SNAPSHOT_DATE,
        "benchmark_version": BENCHMARK_VERSION,
        "max_score": MAX_SCORE,
        "scoring_rule": (
            "+1 per correct athlete mentioned · -1 per known-incorrect "
            "(杨扬/李静/王皓/呙俐/陈晓君) · 0 for unknowns"
        ),
        "results": all_scores,
        "ranking": [
            {
                "rank": i + 1,
                "model": m,
                "score": s["total_score"],
                "n_correct": s["n_correct"],
                "n_wrong": s["n_known_incorrect_hit"],
            }
            for i, (m, s) in enumerate(ranking)
        ],
    }


def build_ranking_md(all_scores: dict[str, dict]) -> str:
    ranked = sorted(all_scores.items(), key=lambda x: -x[1]["total_score"])
    lines = [
        f"# Query {QUERY_ID} 排名报告",
        "",
        f"> 基准 v{BENCHMARK_VERSION}（locked {SNAPSHOT_DATE}）  Max: {MAX_SCORE}",
        "> +1 per correct · -1 per known-incorrect · 0 unknown",
        "",
        "| Rank | Model | Score | Correct | Wrong | Unverified |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for i, (m, s) in enumerate(ranked, 1):
        lines.append(
            f"| {i} | {m} | {s['total_score']}/{s['max_score']} "
            f"| {s['n_correct']} | {s['n_known_incorrect_hit']} "
            f"| {len(s['unverified_claims'])} |"
        )
    lines.append("")
    lines.append("## 各模型主张明细")
    lines.append("")
    for m, s in ranked:
        lines.append(f"### {m}  (total={s['total_score']}/{s['max_score']})")
        for d in s["per_dimension"]:
            mark = "✓" if d["score"] > 0 else "✗"
            lines.append(
                f"- {mark} {d['model_answer_name']} → {d['name']} "
                f"({d['judgment']}) → {d['score']:+}"
            )
        for uv in s["unverified_claims"]:
            lines.append(f"- ? {uv['name']} → {uv['reason']}")
        lines.append("")
    return "\n".join(lines) + "\n"


def _load_alignment(out_dir: Path) -> dict[str, list[dict]]:
    aligned: dict[str, list[dict]] = {}
    for d in Path(out_dir).iterdir():
        if not d.is_dir():
            continue
        af = d / "alignment.json"
        if not af.exists():
            continue
        aligned[d.name] = json.loads(af.read_text(encoding="utf-8")).get(
            "aligned_claims", []
        )
    return aligned


# ═══════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════


def main() -> None:
    ap = argparse.ArgumentParser(description=f"Query {QUERY_ID} auto-scorer")
    ap.add_argument("--models", nargs="+", required=True, help="name=path list")
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

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

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

    scores = score_aligned(aligned)
    (out_dir / "scores.json").write_text(
        json.dumps(build_scores_json(scores), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (out_dir / "ranking_report.md").write_text(
        build_ranking_md(scores), encoding="utf-8"
    )

    print("\n" + "─" * 64)
    print(f"Query {QUERY_ID} scoring done.")
    for m, s in sorted(scores.items(), key=lambda x: -x[1]["total_score"]):
        mark = "✅" if s["total_score"] >= MAX_SCORE * 0.7 else (
            "🟡" if s["total_score"] > 0 else "❌"
        )
        print(
            f"  {mark} {m:30s} {s['total_score']:+}/{s['max_score']}  "
            f"correct={s['n_correct']} wrong={s['n_known_incorrect_hit']}"
        )


if __name__ == "__main__":
    main()
