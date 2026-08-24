"""
Query 11 — As of 2026-01-01, papers submitted to ICLR/NeurIPS/ICML,
rejected, never accepted again, citation > 10000 — auto-scorer.

Scoring rule (Plan A, binary ✅/❌):
  ✅ 正确论文（投三大会被拒、之后未再中稿、citation>10000）→ +1.0
  __HALLUCINATION__（Stage C 验证为虚构）              → -1.0
  null + Stage C unresolved（无证据/闭合集合外）       → -1.0
  alignment_confidence = "needs_review"               →  0.0（不计分，特殊路径）
  MAX_SCORE = 所有 ✅ 基准分之和（仅累加正分）
  total_score 可为负

Pipeline:
  1. Extraction — v2 pipeline (primary Claude-Sonnet-4 + secondary GPT-5
     + analyzer Claude-Opus-4.6).
  2. Alignment — 3-model vote + field cross-check + kw sanity + judge LLM
     (see pipeline/alignment.py).
  3. Null verification — manual + Claude-Code-assisted; produce
     null_resolutions.json (Stage C, see pipeline/alignment.py).
  4. Scoring — deterministic lookup from canonical_id to benchmark score.
"""

from __future__ import annotations

import argparse
import json
import re as _re
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
# Benchmark — 7 curated ✅ dimensions (D1-D7).
# Cross-verified 2026-05-05 against arXiv / OpenReview / Google Scholar.
# ═══════════════════════════════════════════════════════════════════════════

SNAPSHOT_DATE = "2026-05-05"
SCORING_MODE = "A2-binary"  # ✅/❌ 二档（无 ⚠️），null+unresolved → -1

DIMS = [
    (
        "D1",
        "Word2Vec / Mikolov et al. 2013 / "
        "Efficient Estimation of Word Representations in Vector Space / "
        "ICLR 2013 拒 / arXiv only（后续未在三大会中稿）",
        "✅",
        1.0,
        [
            "word2vec",
            "mikolov",
            "efficient estimation of word representations",
            "word representations in vector space",
            "skip-gram",
            "cbow",
        ],
    ),
    (
        "D2",
        "Knowledge Distillation / Hinton et al. 2015 / "
        "Distilling the Knowledge in a Neural Network / "
        "NeurIPS 拒 / NIPS 2014 Deep Learning Workshop（后续未在三大会主会中稿）",
        "✅",
        1.0,
        [
            "knowledge distillation",
            "distilling the knowledge",
            "hinton distill",
            "soft targets",
            "model distillation",
        ],
    ),
    (
        "D3",
        "RoBERTa / Liu et al. 2019 / "
        "RoBERTa: A Robustly Optimized BERT Pretraining Approach / "
        "ICLR 2020 拒 / arXiv only（后续未在三大会中稿）",
        "✅",
        1.0,
        [
            "roberta",
            "robustly optimized bert",
            "robustly optimized bert pretraining",
        ],
    ),
    (
        "D4",
        "ELMo / Peters et al. 2018 / "
        "Deep Contextualized Word Representations / "
        "三大会拒 / NAACL 2018 Best Paper（题面要求三大会，未在三大会中稿）",
        "✅",
        1.0,
        [
            "elmo",
            "deep contextualized word representations",
            "peters et al",
            "contextualized word representation",
        ],
    ),
    (
        "D5",
        "Scaling Laws for Neural Language Models / Kaplan et al. 2020 / "
        "三大会拒 / arXiv only（后续未在三大会中稿）",
        "✅",
        1.0,
        [
            "scaling laws",
            "kaplan",
            "scaling laws for neural language models",
            "scaling law for language models",
        ],
    ),
    (
        "D6",
        "YOLO / Redmon et al. 2016 / "
        "You Only Look Once: Unified, Real-Time Object Detection / "
        "三大会拒 / CVPR 2016（题面要求三大会，未在三大会中稿）",
        "✅",
        1.0,
        [
            "yolo",
            "you only look once",
            "redmon",
            "real-time object detection",
        ],
    ),
    (
        "D7",
        "SqueezeNet / Iandola et al. 2016 / "
        "SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and "
        "<0.5MB model size / ICLR 2017 拒 / arXiv only（后续未在三大会中稿）",
        "✅",
        1.0,
        [
            "squeezenet",
            "iandola",
            "alexnet-level accuracy",
            "50x fewer parameters",
        ],
    ),
]

DIM_MAP = {
    d[0]: {"id": d[0], "name": d[1], "judgment": d[2], "score": d[3], "kw": d[4]}
    for d in DIMS
}
MAX_SCORE = sum(d[3] for d in DIMS if d[3] > 0)  # 仅累加 ✅ 正分基准 = 7.0


# ═══════════════════════════════════════════════════════════════════════════
# DIMS → alignment baseline spec
# ═══════════════════════════════════════════════════════════════════════════


def _derive_match_fields(description: str) -> dict:
    """Parse year and first_author surname out of the DIMS description string.

    Format expected: "<paper name> / <Surname> et al. <year> / ...". Returns
    an empty dict if regex doesn't match (alignment cross-check degrades to
    LLM-only).
    """
    fields: dict = {}
    m_year = _re.search(r"\b(19|20)\d{2}\b", description)
    if m_year:
        fields["year"] = m_year.group(0)
    m_author = _re.search(r"/\s*([A-Z][A-Za-z\-]+)\s+et al\.", description)
    if m_author:
        fields["first_author"] = m_author.group(1)
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
    """Aggregate scores from alignment output. Each aligned claim carries a
    `canonical_id` (or None) plus `alignment_confidence`. Scoring rules
    (Q11 binary):

      - canonical_id ∈ DIM_MAP (first occurrence)            → DIM_MAP[cid].score (+1)
      - canonical_id == "__HALLUCINATION__"                  → -1.0
      - canonical_id is None and confidence != needs_review  → -1.0
      - alignment_confidence == "needs_review"               →  0.0 (special path)
      - duplicates dropped (only first occurrence per cid counts)

    Totals can be negative.
    """
    all_scores: dict[str, dict] = {}
    for model_name, claims in aligned_by_model.items():
        seen_dims: set[str] = set()
        scored: list[dict] = []
        unverified: list[dict] = []
        total = 0.0
        for c in claims:
            raw = c.get("raw", {}) or {}
            cid = c.get("canonical_id")
            conf = c.get("alignment_confidence", "medium")
            reason = c.get("alignment_reasoning", "")
            judge_invoked = c.get("judge_invoked", False)
            paper = raw.get("name", "")

            if conf == "needs_review":
                unverified.append(
                    {
                        "name": paper,
                        "first_author": raw.get("first_author", ""),
                        "year": raw.get("year", ""),
                        "canonical_id_tentative": cid,
                        "reason": f"needs_review: {reason}",
                        "judge_invoked": judge_invoked,
                        "score": 0.0,
                    }
                )
                continue

            if cid == "__HALLUCINATION__":
                scored.append(
                    {
                        "id": "__HALLUCINATION__",
                        "name": "(verified hallucination)",
                        "paper": paper,
                        "score": -1.0,
                        "judgment": "❌",
                        "confidence": conf,
                        "judge_invoked": judge_invoked,
                        "reason": reason,
                    }
                )
                total += -1.0
                continue

            if cid and cid in DIM_MAP:
                if cid in seen_dims:
                    continue  # dedup
                seen_dims.add(cid)
                d = DIM_MAP[cid]
                scored.append(
                    {
                        "id": cid,
                        "name": d["name"],
                        "paper": paper,
                        "score": d["score"],
                        "judgment": d["judgment"],
                        "confidence": conf,
                        "judge_invoked": judge_invoked,
                        "reason": reason or f"对齐到 {cid}（基准 {d['judgment']}）",
                    }
                )
                total += d["score"]
            else:
                # null + Stage C unresolved → -1（Q11 binary 规则，与 Q12 一致）
                scored.append(
                    {
                        "id": "__UNRESOLVED_NULL__",
                        "name": "(null + unresolved, treated as wrong)",
                        "paper": paper,
                        "score": -1.0,
                        "judgment": "❌",
                        "confidence": conf,
                        "judge_invoked": judge_invoked,
                        "reason": reason or "未对齐到任何基准条目，且 Stage C 未能验证",
                    }
                )
                total += -1.0

        n_answered = len(scored)
        all_scores[model_name] = {
            "total_score": total,
            "max_score": MAX_SCORE,
            "score_rate": round(total / n_answered, 4) if n_answered else 0.0,
            "total_rate": round(total / MAX_SCORE, 4) if MAX_SCORE else 0.0,
            "dimensions_answered": n_answered,
            "per_dimension": sorted(scored, key=lambda x: x["id"]),
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
        "scoring_mode": SCORING_MODE,
        "snapshot_date": SNAPSHOT_DATE,
        "max_score": MAX_SCORE,
        "extraction_pipeline": "v2 (primary=claude-sonnet-4, secondary=gpt-5, analyzer=claude-opus-4.6)",
        "results": all_scores,
        "ranking": [
            {
                "rank": i + 1,
                "model": m,
                "score": s["total_score"],
                "rate": f"{s['total_rate'] * 100:.1f}%",
            }
            for i, (m, s) in enumerate(ranking)
        ],
        "unverified_items": unverified_all,
    }


def build_ranking_md(all_scores: dict[str, dict]) -> str:
    ranked = sorted(
        all_scores.items(),
        key=lambda x: (-x[1]["total_score"], -x[1].get("score_rate", 0)),
    )
    lines = [
        "# Query 11 排名报告（投三大会被拒、不再中稿且 citation>10000 的论文）",
        "",
        f"> Snapshot: {SNAPSHOT_DATE}  Mode: {SCORING_MODE}  Max: {MAX_SCORE}",
        "> 抽取流水线：Primary=claude-sonnet-4, Secondary=gpt-5, Analyzer=claude-opus-4.6",
        "> 评分：✅+1 / ❌-1 / null+unresolved -1 / __HALLUCINATION__ -1",
        "",
        "| Rank | Model | Score | Rate | Answered | Unverified |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for i, (m, s) in enumerate(ranked, 1):
        lines.append(
            f"| {i} | {m} | {s['total_score']}/{s['max_score']} "
            f"| {s['total_rate'] * 100:.1f}% "
            f"| {s['dimensions_answered']} "
            f"| {len(s['unverified_claims'])} |"
        )
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
    ap = argparse.ArgumentParser(description="Query 11 auto-scorer (v2+align)")
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

    # Stage 3a: export null claims for manual + Claude-Code-assisted verification
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
                    "judgment": e.get("judgment", "⚠️"),
                    "score": e.get("score", 0.0),
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
    print("Query 11 scoring done.")
    for i, (m, s) in enumerate(
        sorted(scores.items(), key=lambda x: -x[1]["total_score"]), 1
    ):
        print(
            f"  {i}. {m:28s} {s['total_score']:>5.1f}/{s['max_score']:.1f}"
            f"  answered={s['dimensions_answered']}"
            f"  unverified={len(s['unverified_claims'])}"
        )


if __name__ == "__main__":
    main()
