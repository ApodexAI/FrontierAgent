"""
Query 36 — Auto scorer (binary, single-fact coordinate, multi-candidate)

Question: GeoGuessr NMPZ photo featuring Mickey/Minnie graffiti on a
retaining wall, narrow two-lane road with white dashed centerline,
wooden utility pole, dense vegetation, rolling green hills. Identify the
latitude/longitude.

Reference answer: 38.6449°N, 24.0229°E (Greece, near Evia).

Scoring rule:
  Single binary dimension —
    1 if ANY extracted coordinate (primary or alternative_candidate) is
      within ±10° in BOTH lat and lon of the reference.
    0 otherwise.
  Auxiliary fields (location_name / country / confidence_label) recorded
  for audit only.

Flow:
  1) Run shared extraction pipeline → auto_scores/{model}/extraction.json.
  2) Score each model deterministically; write
     auto_scores/{model}/score.json + ranking_report.md.
"""

from __future__ import annotations

import argparse
import json
import sys
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
from pipeline.extraction_pipeline import run_pipeline  # noqa: E402

# ═══════════════════════════════════════════════════════════════════════════
# Ground truth
# ═══════════════════════════════════════════════════════════════════════════

BENCHMARK: dict = {
    "version": "1.0",
    "locked_at": "2026-05-05",
    "correct_lat": 38.6449,
    "correct_lon": 24.0229,
    "tolerance_deg": 10.0,
    "reference": {
        "country": "Greece",
        "region": "Evia (Euboea) island region",
        "lat": 38.6449,
        "lon": 24.0229,
    },
}


# ═══════════════════════════════════════════════════════════════════════════
# Coordinate parsing & matching
# ═══════════════════════════════════════════════════════════════════════════


def _coerce_float(v) -> float | None:
    if v is None:
        return None
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip()
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        import re
        m = re.search(r"-?\d+(?:\.\d+)?", s)
        return float(m.group()) if m else None


def _within_tolerance(lat: float | None, lon: float | None) -> bool:
    if lat is None or lon is None:
        return False
    return (
        abs(lat - BENCHMARK["correct_lat"]) <= BENCHMARK["tolerance_deg"]
        and abs(lon - BENCHMARK["correct_lon"]) <= BENCHMARK["tolerance_deg"]
    )


def _collect_candidates(canonical_value) -> list[dict]:
    """Pool primary + alternative_candidates into a single list of
    {lat, lon, location_name, country, confidence_label, source}."""
    out: list[dict] = []
    if not isinstance(canonical_value, dict):
        return out
    primary_lat = _coerce_float(canonical_value.get("lat"))
    primary_lon = _coerce_float(canonical_value.get("lon"))
    if primary_lat is not None or primary_lon is not None:
        out.append(
            {
                "lat": primary_lat,
                "lon": primary_lon,
                "location_name": canonical_value.get("location_name"),
                "country": canonical_value.get("country"),
                "confidence_label": canonical_value.get("confidence_label"),
                "source": "primary",
            }
        )
    for alt in canonical_value.get("alternative_candidates") or []:
        if not isinstance(alt, dict):
            continue
        out.append(
            {
                "lat": _coerce_float(alt.get("lat")),
                "lon": _coerce_float(alt.get("lon")),
                "location_name": alt.get("location_name"),
                "country": alt.get("country"),
                "confidence_label": alt.get("confidence_label"),
                "source": "alternative",
            }
        )
    return out


# ═══════════════════════════════════════════════════════════════════════════
# Per-model scoring
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class ScoreResult:
    model: str
    score: int
    max_score: int
    candidates: list[dict]
    hit_candidate: dict | None
    refused_to_answer: bool
    resolution: str
    reference_snapshot: dict = field(default_factory=dict)


def _score_model(model_name: str, extraction_payload: dict) -> ScoreResult:
    entities = extraction_payload.get("entities", [])
    if not entities:
        return ScoreResult(
            model=model_name,
            score=0,
            max_score=1,
            candidates=[],
            hit_candidate=None,
            refused_to_answer=False,
            resolution="empty_extraction",
            reference_snapshot={},
        )
    ent = entities[0]
    canonical = (ent.get("canonical") or {}).get("value")
    candidates = _collect_candidates(canonical)

    hit: dict | None = None
    for c in candidates:
        if _within_tolerance(c.get("lat"), c.get("lon")):
            hit = c
            break

    score = 1 if hit else 0
    refused = False
    ref_snap: dict = {}
    if isinstance(canonical, dict):
        refused = bool(canonical.get("refused_to_answer"))
        for k in ("lat", "lon", "location_name", "country", "confidence_label"):
            ref_snap[k] = canonical.get(k)

    return ScoreResult(
        model=model_name,
        score=score,
        max_score=1,
        candidates=candidates,
        hit_candidate=hit,
        refused_to_answer=refused,
        resolution=ent.get("resolution", "unknown"),
        reference_snapshot=ref_snap,
    )


# ═══════════════════════════════════════════════════════════════════════════
# I/O
# ═══════════════════════════════════════════════════════════════════════════


def _write_model_score(output_dir: Path, r: ScoreResult) -> None:
    path = output_dir / r.model / "score.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "query_id": QUERY_ID,
                "model": r.model,
                "score": r.score,
                "max_score": r.max_score,
                "candidates": r.candidates,
                "hit_candidate": r.hit_candidate,
                "refused_to_answer": r.refused_to_answer,
                "extraction_resolution": r.resolution,
                "reference_snapshot": r.reference_snapshot,
                "benchmark_version": BENCHMARK["version"],
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )


def _fmt_coord(lat, lon) -> str:
    if lat is None or lon is None:
        return "—"
    return f"{lat:.4f}, {lon:.4f}"


def _write_ranking_report(output_dir: Path, results: list[ScoreResult]) -> None:
    lines: list[str] = []
    lines.append("# Query 36 Ranking Report (v1)")
    lines.append("")
    lines.append(
        f"> Benchmark version: {BENCHMARK['version']} (locked {BENCHMARK['locked_at']})"
    )
    lines.append(
        f"> Correct: {BENCHMARK['correct_lat']}°N, {BENCHMARK['correct_lon']}°E "
        f"({BENCHMARK['reference']['country']})  ·  Tolerance: ±{BENCHMARK['tolerance_deg']}°"
    )
    lines.append("")
    lines.append("## 排名")
    lines.append("")
    lines.append(
        "| Rank | Model | Score | Primary (lat,lon) | Hit | # Candidates | Refused | Resolution |"
    )
    lines.append("|---:|---|---:|---|:---:|---:|:---:|---|")
    ranked = sorted(results, key=lambda r: (-r.score, r.model))
    for i, r in enumerate(ranked, start=1):
        s = r.reference_snapshot
        primary = _fmt_coord(_coerce_float(s.get("lat")), _coerce_float(s.get("lon")))
        hit_mark = "✅" if r.hit_candidate else "❌"
        refused = "✅" if r.refused_to_answer else "—"
        lines.append(
            f"| {i} | {r.model} | {r.score}/{r.max_score} | {primary} | "
            f"{hit_mark} | {len(r.candidates)} | {refused} | {r.resolution} |"
        )
    lines.append("")
    lines.append("## 候选明细（含命中标记）")
    lines.append("")
    lines.append("| Model | Source | lat | lon | Δlat | Δlon | within ±10°? | Country | Place |")
    lines.append("|---|---|---:|---:|---:|---:|:---:|---|---|")
    for r in ranked:
        if not r.candidates:
            lines.append(f"| {r.model} | — | — | — | — | — | — | — | — |")
            continue
        for c in r.candidates:
            lat = c.get("lat")
            lon = c.get("lon")
            within = "✅" if _within_tolerance(lat, lon) else "❌"
            dlat = "—" if lat is None else f"{lat - BENCHMARK['correct_lat']:+.2f}"
            dlon = "—" if lon is None else f"{lon - BENCHMARK['correct_lon']:+.2f}"
            lines.append(
                f"| {r.model} | {c.get('source')} | "
                f"{lat if lat is not None else '—'} | "
                f"{lon if lon is not None else '—'} | "
                f"{dlat} | {dlon} | {within} | "
                f"{c.get('country') or '—'} | {c.get('location_name') or '—'} |"
            )
    (output_dir / "ranking_report.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


# ═══════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════


def main() -> None:
    ap = argparse.ArgumentParser(description="Query 36 auto-scorer (v1)")
    ap.add_argument(
        "--models",
        nargs="+",
        required=True,
        help="name=path list of model answer JSON files.",
    )
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--primary-model", default=None)
    ap.add_argument("--secondary-model", default=None)
    ap.add_argument("--parallel-models", nargs="+", default=None)
    ap.add_argument("--analyzer-model", default=None)
    ap.add_argument("--concurrency", type=int, default=None)
    ap.add_argument(
        "--skip-extract",
        action="store_true",
        help="Skip extraction; reuse existing extraction.json.",
    )
    args = ap.parse_args()

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

    results: list[ScoreResult] = []
    for model_name, payload in all_results.items():
        r = _score_model(model_name, payload)
        _write_model_score(out_dir, r)
        results.append(r)

    _write_ranking_report(out_dir, results)
    print("\n" + "─" * 60)
    print("Query 36 scoring done.")
    for r in sorted(results, key=lambda x: (-x.score, x.model)):
        mark = "✅" if r.score else "❌"
        primary_lat = _coerce_float(r.reference_snapshot.get("lat"))
        primary_lon = _coerce_float(r.reference_snapshot.get("lon"))
        primary = _fmt_coord(primary_lat, primary_lon)
        print(
            f"  {mark} {r.model}: {r.score}/{r.max_score}  "
            f"primary=({primary})  cand={len(r.candidates)}  "
            f"refused={'Y' if r.refused_to_answer else 'N'}"
        )


if __name__ == "__main__":
    main()
