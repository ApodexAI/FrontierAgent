"""Query 32 — DOCBENCH GT answer average length auto-scorer.

Reference answer (DOCBENCH, Zou et al. 2025):
  Average character length = 62.18 (over 1102 samples,
  total 68524 characters). See `standard_answer.json`.

Scoring rule (binary, max 1):
  +1 if the model's main claim is:
     - final_claim_type ∈ {numeric, range}
     - metric is character length (not token / word / unknown)
     - |judged_value − 62.18| ≤ 5.0
        · numeric: judged_value = single_value
        · range  : judged_value = (range_min + range_max) / 2
   0 otherwise (unavailable / ambiguous / wrong metric / out of tolerance / missing).
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
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
from pipeline.extraction_pipeline import run_pipeline  # noqa: E402

DEFAULT_STANDARD_PATH = THIS / "standard_answer.json"


# ═══════════════════════════════════════════════════════════════════════════
# Scoring
# ═══════════════════════════════════════════════════════════════════════════


def _metric_is_character(value: dict) -> bool:
    metric = str(value.get("normalized_metric") or "").strip().lower()
    if metric == "character":
        return True
    unit_text = str(value.get("unit_text") or "").lower()
    return any(t in unit_text for t in ["字符", "character", "characters", "char"])


def _to_float(v) -> float | None:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _score_claim(value: dict, target: float, tolerance: float) -> dict:
    """Return {score, judged_value, metric_ok, reason}."""
    claim_type = str(value.get("final_claim_type") or "").strip().lower()
    metric_ok = _metric_is_character(value)
    judged: float | None = None
    reason = ""

    if claim_type == "numeric":
        judged = _to_float(value.get("single_value"))
        if not metric_ok:
            reason = "主答案单位不是字符长度"
        elif judged is None:
            reason = "numeric 但缺 single_value"
        else:
            diff = abs(judged - target)
            if diff <= tolerance:
                return {
                    "score": 1,
                    "judged_value": judged,
                    "metric_ok": True,
                    "reason": (
                        f"single_value={judged:.2f} 与标准 {target:.2f} 相差 "
                        f"{diff:.2f}（≤ 容差 {tolerance:.2f}）"
                    ),
                }
            reason = (
                f"single_value={judged:.2f} 与标准 {target:.2f} 相差 "
                f"{diff:.2f}（> 容差 {tolerance:.2f}）"
            )
    elif claim_type == "range":
        lo = _to_float(value.get("range_min"))
        hi = _to_float(value.get("range_max"))
        if not metric_ok:
            reason = "主答案单位不是字符长度"
        elif lo is None or hi is None:
            reason = "range 但缺 range_min / range_max"
        else:
            judged = (lo + hi) / 2.0
            diff = abs(judged - target)
            if diff <= tolerance:
                return {
                    "score": 1,
                    "judged_value": judged,
                    "metric_ok": True,
                    "reason": (
                        f"range 中点 {judged:.2f} 与标准 {target:.2f} 相差 "
                        f"{diff:.2f}（≤ 容差 {tolerance:.2f}）"
                    ),
                }
            reason = (
                f"range 中点 {judged:.2f} 与标准 {target:.2f} 相差 "
                f"{diff:.2f}（> 容差 {tolerance:.2f}）"
            )
    else:
        reason = f"final_claim_type='{claim_type}' 不是 numeric / range"

    return {"score": 0, "judged_value": judged, "metric_ok": metric_ok, "reason": reason}


@dataclass
class ModelScore:
    model: str
    score: int
    judged_value: float | None
    metric_ok: bool
    reason: str
    claim_type: str
    raw_value: dict
    resolution: str = "unknown"


def _score_model(
    model_name: str, payload: dict, target: float, tolerance: float
) -> ModelScore:
    entities = payload.get("entities", [])
    if not entities:
        return ModelScore(
            model=model_name,
            score=0,
            judged_value=None,
            metric_ok=False,
            reason="no entities in extraction",
            claim_type="",
            raw_value={},
            resolution="empty",
        )
    ent = entities[0]
    canonical = (ent.get("canonical") or {}).get("value") or {}
    if not isinstance(canonical, dict):
        canonical = {}

    judge = _score_claim(canonical, target, tolerance)
    return ModelScore(
        model=model_name,
        score=judge["score"],
        judged_value=judge["judged_value"],
        metric_ok=judge["metric_ok"],
        reason=judge["reason"],
        claim_type=str(canonical.get("final_claim_type") or ""),
        raw_value=canonical,
        resolution=ent.get("resolution", "unknown"),
    )


# ═══════════════════════════════════════════════════════════════════════════
# I/O
# ═══════════════════════════════════════════════════════════════════════════


def _write_model_score(out_dir: Path, r: ModelScore, standard: dict) -> None:
    path = out_dir / r.model / "score.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "query_id": QUERY_ID,
                "model": r.model,
                "score": r.score,
                "max_score": 1,
                "total_rate": float(r.score),
                "is_correct": bool(r.score),
                "judged_value": r.judged_value,
                "target_value": standard["standard_answer"]["标准值"],
                "absolute_tolerance": standard["scoring_rule"]["absolute_tolerance"],
                "metric_ok": r.metric_ok,
                "reason": r.reason,
                "claim_type": r.claim_type,
                "raw_value": r.raw_value,
                "scoring_rule": standard["scoring_rule"],
                "extraction_resolution": r.resolution,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def _write_scores_json(
    out_dir: Path, results: list[ModelScore], standard: dict
) -> None:
    ranked = sorted(results, key=lambda r: (-r.score, r.model))
    target = standard["standard_answer"]["标准值"]
    tolerance = standard["scoring_rule"]["absolute_tolerance"]
    (out_dir / "scores.json").write_text(
        json.dumps(
            {
                "query_id": QUERY_ID,
                "max_score": 1,
                "min_score": 0,
                "scoring_rule": (
                    f"binary — claim_type ∈ (numeric, range) + character metric + "
                    f"|value − {target}| ≤ {tolerance} → 1; otherwise 0."
                ),
                "target_value": target,
                "absolute_tolerance": tolerance,
                "results": {
                    r.model: {
                        "total_score": r.score,
                        "total_rate": float(r.score),
                        "is_correct": bool(r.score),
                        "judged_value": r.judged_value,
                        "metric_ok": r.metric_ok,
                        "claim_type": r.claim_type,
                        "reason": r.reason,
                        "extraction_resolution": r.resolution,
                    }
                    for r in results
                },
                "ranking": [
                    {
                        "rank": i + 1,
                        "model": r.model,
                        "score": r.score,
                        "judged_value": r.judged_value,
                        "claim_type": r.claim_type,
                    }
                    for i, r in enumerate(ranked)
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def _write_ranking_report(
    out_dir: Path, results: list[ModelScore], standard: dict
) -> None:
    target = standard["standard_answer"]["标准值"]
    tolerance = standard["scoring_rule"]["absolute_tolerance"]
    ranked = sorted(results, key=lambda r: (-r.score, r.model))
    lines: list[str] = [
        f"# Query {QUERY_ID} Ranking Report",
        "",
        f"> Reference: average character length = **{target}** "
        f"(tolerance ±{tolerance}, DOCBENCH Zou et al. 2025).  ",
        "> Scoring: binary — `numeric` / `range` claim with character metric and "
        f"|value − {target}| ≤ {tolerance} → 1; otherwise 0.",
        "",
        "| Rank | Model | Score | Claim type | Judged value | Metric OK | Reason |",
        "|---:|---|---:|---|---:|---|---|",
    ]
    for i, r in enumerate(ranked, start=1):
        jv = f"{r.judged_value:.2f}" if r.judged_value is not None else "—"
        mo = "✅" if r.metric_ok else "❌"
        lines.append(
            f"| {i} | {r.model} | {r.score} | {r.claim_type or '—'} | {jv} "
            f"| {mo} | {r.reason} |"
        )
    (out_dir / "ranking_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


# ═══════════════════════════════════════════════════════════════════════════
# Entry
# ═══════════════════════════════════════════════════════════════════════════


def main() -> None:
    ap = argparse.ArgumentParser(description=f"Query {QUERY_ID} auto-scorer")
    ap.add_argument("--models", nargs="+", required=True, help="name=path list")
    ap.add_argument("--output-dir", default=str(THIS / "auto_scores"))
    ap.add_argument("--standard", default=str(DEFAULT_STANDARD_PATH))
    ap.add_argument("--primary-model", default=None)
    ap.add_argument("--secondary-model", default=None)
    ap.add_argument("--parallel-models", nargs="+", default=None)
    ap.add_argument("--analyzer-model", default=None)
    ap.add_argument("--concurrency", type=int, default=None)
    ap.add_argument("--skip-extract", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    standard = json.loads(Path(args.standard).read_text(encoding="utf-8"))
    target = float(standard["standard_answer"]["标准值"])
    tolerance = float(standard["scoring_rule"]["absolute_tolerance"])

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

    results: list[ModelScore] = []
    for model_name, payload in all_results.items():
        r = _score_model(model_name, payload, target, tolerance)
        _write_model_score(out_dir, r, standard)
        results.append(r)

    _write_ranking_report(out_dir, results, standard)
    _write_scores_json(out_dir, results, standard)

    print("\n" + "─" * 64)
    print(
        f"Query {QUERY_ID} scoring done. (target={target}, tolerance=±{tolerance})"
    )
    for r in sorted(results, key=lambda x: (-x.score, x.model)):
        jv = f"{r.judged_value:.2f}" if r.judged_value is not None else "—"
        mark = "✅" if r.score else "❌"
        print(
            f"  {mark} {r.model:28s} score={r.score}  "
            f"claim={r.claim_type or '—':10s} judged={jv}  | {r.reason}"
        )


if __name__ == "__main__":
    main()
