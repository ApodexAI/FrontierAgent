"""Query 01 — Singapore 2019-TOP condo top-3 resale-return auto-scorer.

Reference answer (URA public transactions, locked 2026-05-05):
  Top-3 (each +1 if mentioned as final answer):
    1. High Park Residences  — 6.10%
    2. Coco Palms            — 4.80%
    3. Botanique At Bartley  — 4.64%
  Rank 4-10 (no credit): Eon Shenton, Poiz, Commonwealth Towers, Panorama,
    Thomson Impressions, Principal Garden, Wisteria.
  Known-incorrect (-1 if mentioned as final answer):
    • Hundred Palms Residences  (TOP year is 2018, not 2019)

Scoring rule (max 3, min unbounded negative):
  +1 per top-3 GT project mentioned as final answer
  -1 per known-incorrect project mentioned as final answer
   0 for rank 4-10 GT projects (recognised but not top-3) or unknowns

Pipeline: Stage 1 extract → Stage 2 align → Stage 3 score (no null-review
since baseline list is small and scoring is per-canonical_id).
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
# Benchmark — 10 ranked GT + 1 known-incorrect
# ═══════════════════════════════════════════════════════════════════════════

SNAPSHOT_DATE = "2026-05-05"
TOP_K = 3

# (canonical_id, name, rank_or_None, score, kw)
DIMS = [
    ("D1",  "High Park Residences",   1,   1.0,
     ["high park", "high park residences"]),
    ("D2",  "Coco Palms",              2,   1.0,
     ["coco palms"]),
    ("D3",  "Botanique At Bartley",    3,   1.0,
     ["botanique", "botanique at bartley", "bartley"]),
    ("D4",  "Eon Shenton",             4,   0.0, ["eon shenton"]),
    ("D5",  "The Poiz Residences",     5,   0.0, ["poiz"]),
    ("D6",  "Commonwealth Towers",     6,   0.0, ["commonwealth towers"]),
    ("D7",  "The Panorama",            7,   0.0, ["panorama"]),
    ("D8",  "Thomson Impressions",     8,   0.0, ["thomson impressions"]),
    ("D9",  "Principal Garden",        9,   0.0, ["principal garden"]),
    ("D10", "The Wisteria",            10,  0.0, ["wisteria"]),
    ("BAD1", "Hundred Palms Residences (TOP=2018, wrong year)",
     None, -1.0, ["hundred palms"]),
]

DIM_MAP = {
    d[0]: {"id": d[0], "name": d[1], "rank": d[2], "score": d[3], "kw": d[4]}
    for d in DIMS
}
MAX_SCORE = TOP_K * 1.0


# ═══════════════════════════════════════════════════════════════════════════
# DIMS → alignment baseline spec
# ═══════════════════════════════════════════════════════════════════════════


def build_baselines() -> list[dict]:
    out = []
    for d_id, name, rank, score, kw in DIMS:
        rank_label = f"rank #{rank}" if rank else "known-incorrect (TOP=2018)"
        desc = f"{name} — {rank_label}"
        out.append(
            {
                "id": d_id,
                "description": desc,
                "match_fields": {"name": name},
                "kw": kw,
                "judgment": "✅" if score > 0 else ("❌" if score < 0 else "⚠️"),
                "score": score,
                "rank": rank,
            }
        )
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
                unverified.append(
                    {
                        "name": raw.get("name", ""),
                        "canonical_id_tentative": cid,
                        "reason": f"needs_review: {reason}",
                    }
                )
                continue

            if cid and cid in DIM_MAP:
                if cid in seen:
                    continue
                seen.add(cid)
                d = DIM_MAP[cid]
                scored.append(
                    {
                        "id": cid,
                        "name": d["name"],
                        "rank": d["rank"],
                        "score": d["score"],
                        "model_answer_name": raw.get("name", ""),
                        "model_answer_rate": raw.get("return_rate", ""),
                        "confidence": conf,
                        "reason": reason or f"aligned to {cid}",
                    }
                )
                total += d["score"]
            else:
                unverified.append(
                    {
                        "name": raw.get("name", ""),
                        "canonical_id_tentative": cid,
                        "reason": reason or "not in baseline (unknown project)",
                    }
                )

        n_top3 = sum(1 for s in scored if s["score"] > 0)
        n_wrong = sum(1 for s in scored if s["score"] < 0)
        n_other_gt = sum(1 for s in scored if s["score"] == 0)
        all_scores[model_name] = {
            "total_score": total,
            "max_score": MAX_SCORE,
            "top_k": TOP_K,
            "n_top3_hit": n_top3,
            "n_known_incorrect_hit": n_wrong,
            "n_other_gt": n_other_gt,
            "per_dimension": sorted(
                scored,
                key=lambda x: (x["rank"] if x["rank"] is not None else 99),
            ),
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
        "max_score": MAX_SCORE,
        "scoring_rule": (
            f"+1 per top-{TOP_K} GT project mentioned as final answer; "
            "-1 per known-incorrect (e.g. Hundred Palms Residences); "
            "0 for rank 4-10 GT or unknowns"
        ),
        "results": all_scores,
        "ranking": [
            {
                "rank": i + 1,
                "model": m,
                "score": s["total_score"],
                "top3_hit": s["n_top3_hit"],
                "wrong": s["n_known_incorrect_hit"],
            }
            for i, (m, s) in enumerate(ranking)
        ],
    }


def build_ranking_md(all_scores: dict[str, dict]) -> str:
    ranked = sorted(all_scores.items(), key=lambda x: -x[1]["total_score"])
    lines = [
        f"# Query {QUERY_ID} 排名报告",
        "",
        f"> Snapshot: {SNAPSHOT_DATE}  Max: {MAX_SCORE} (top-{TOP_K})",
        f"> +1 per top-{TOP_K} hit · -1 per known-incorrect · 0 for rank 4-10 / unknown",
        "",
        "| Rank | Model | Score | Top-3 hit | Known-incorrect | Other-GT | Unverified |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for i, (m, s) in enumerate(ranked, 1):
        lines.append(
            f"| {i} | {m} | {s['total_score']}/{s['max_score']} "
            f"| {s['n_top3_hit']} | {s['n_known_incorrect_hit']} "
            f"| {s['n_other_gt']} | {len(s['unverified_claims'])} |"
        )
    lines.append("")
    lines.append("## 各模型主张明细")
    lines.append("")
    for m, s in ranked:
        lines.append(f"### {m}  (total={s['total_score']}/{s['max_score']})")
        for d in s["per_dimension"]:
            mark = "✓" if d["score"] > 0 else ("✗" if d["score"] < 0 else "~")
            rate = d.get("model_answer_rate") or "—"
            rank_info = f"GT#{d['rank']}" if d["rank"] is not None else "incorrect"
            lines.append(
                f"- {mark} {d['model_answer_name']} → {d['name']} "
                f"({rank_info}, rate: {rate}) → {d['score']:+}"
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

    # Stage 3: scoring
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
        mark = "✅" if s["total_score"] >= 2 else (
            "🟡" if s["total_score"] > 0 else "❌"
        )
        print(
            f"  {mark} {m:30s} {s['total_score']:+}/{s['max_score']}  "
            f"top3={s['n_top3_hit']} wrong={s['n_known_incorrect_hit']}"
        )


if __name__ == "__main__":
    main()
