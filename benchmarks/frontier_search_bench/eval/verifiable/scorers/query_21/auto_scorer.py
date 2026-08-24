"""
Query 21 — Auto scorer v2

Scoring rule (see scoring_framework.md):
  Single binary dimension — year ∈ {1922, 2025} → 1, else 0.
  Reference fields (award_name / proxy / winner / absence_reason) recorded
  for audit only, not included in total.

Flow:
  1) Run the shared extraction pipeline (primary + secondary; phase-4 on
     disagreements). Canonical extraction lands in
     auto_scores/{model}/extraction.json.
  2) Load each model's canonical value, judge year, write
     auto_scores/{model}/score.json and a top-level ranking_report.md.
"""

from __future__ import annotations

import argparse
import json
import re
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

# ═══════════════════════════════════════════════════════════════════════════
# Ground truth — see scoring_framework.md
# ═══════════════════════════════════════════════════════════════════════════

BENCHMARK: dict = {
    "version": "2.1",
    "locked_at": "2026-04-23",
    "accepted_years": {1922, 2025},
    # 2025 only credited if model writes a proxy_name containing the
    # required letters ("do" or "Machado").
    # with "Machado" so literal "do" is present (short form "Ana Corina Sosa"
    # does not contain "do"). 
    "reference": {
        "1922": {
            "award_name": "1921 Nobel Prize in Physics",
            "proxy_name": "Rudolf Nadolny",
            "winner_name": "Albert Einstein",
            "absence_reason": "Traveling to Japan (Far East lecture tour)",
        },
        "2025": {
            "award_name": "2025 Nobel Peace Prize",
            "proxy_name": "Ana Corina Sosa (Machado)",
            "winner_name": "María Corina Machado",
            "absence_reason": "Political persecution / restricted movement",
        },
    },
}


# ═══════════════════════════════════════════════════════════════════════════
# Year parsing
# ═══════════════════════════════════════════════════════════════════════════

_YEAR_RE = re.compile(r"(?:19|20)\d{2}")


def _coerce_years(v) -> list[int]:
    """Extract ALL year numbers from a value. A value may be:
    - int/float → single year
    - str → all 4-digit years found (e.g. "1921 (ceremony 1922)" → [1921, 1922])
    """
    if v is None:
        return []
    if isinstance(v, int):
        return [v] if 1900 <= v <= 2100 else []
    if isinstance(v, float):
        n = int(v)
        return [n] if 1900 <= n <= 2100 else []
    if isinstance(v, (list, tuple)):
        out: list[int] = []
        for item in v:
            for y in _coerce_years(item):
                if y not in out:
                    out.append(y)
        return out
    s = str(v)
    out: list[int] = []
    for m in _YEAR_RE.finditer(s):
        y = int(m.group())
        if y not in out:
            out.append(y)
    return out


def _collect_candidate_years(canonical_value) -> list[int]:
    """Pick years from canonical value + alternative_candidates list."""
    years: list[int] = []
    if not isinstance(canonical_value, dict):
        for y in _coerce_years(canonical_value):
            if y not in years:
                years.append(y)
        return years

    for y in _coerce_years(canonical_value.get("year")):
        if y not in years:
            years.append(y)

    for alt in canonical_value.get("alternative_candidates") or []:
        if isinstance(alt, dict):
            for ay in _coerce_years(alt.get("year")):
                if ay not in years:
                    years.append(ay)
    return years


# ═══════════════════════════════════════════════════════════════════════════
# Per-model scoring
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class ScoreResult:
    model: str
    score: int
    max_score: int
    claimed_years: list[int]
    matched_year: int | None
    resolution: str
    reference_snapshot: dict


def _proxy_contains_do(canonical) -> bool:
    """The proxy presenter's name must literally contain the substring 'do'
    (case-insensitive). For 2025 this requires the model to include "Machado"
    (mother's surname) since short form "Ana Corina Sosa" alone lacks "do".
    For 1922 the proxy "Rudolf Nadolny" always contains "do"."""
    if not isinstance(canonical, dict):
        return False
    # Check primary proxy_name + any alternative candidates' proxy_name
    names: list[str] = []
    p = canonical.get("proxy_name")
    if p:
        names.append(str(p))
    for alt in canonical.get("alternative_candidates") or []:
        if isinstance(alt, dict) and alt.get("proxy_name"):
            names.append(str(alt["proxy_name"]))
    return any("do" in n.lower() for n in names)


def _year_with_constraint(years: list[int], canonical) -> int | None:
    """Return the first GT-accepted year, applying 2025's Machado constraint."""
    for y in years:
        if y not in BENCHMARK["accepted_years"]:
            continue
        if y == 2025 and not _proxy_contains_do(canonical):
            continue  # 2025 requires proxy_name containing "do"
        return y
    return None


def _score_model(model_name: str, extraction_payload: dict) -> ScoreResult:
    entities = extraction_payload.get("entities", [])
    if not entities:
        return ScoreResult(
            model=model_name,
            score=0,
            max_score=1,
            claimed_years=[],
            matched_year=None,
            resolution="empty_extraction",
            reference_snapshot={},
        )
    ent = entities[0]
    canonical = (ent.get("canonical") or {}).get("value")
    years = _collect_candidate_years(canonical)

    matched = _year_with_constraint(years, canonical)
    score = 1 if matched is not None else 0

    ref_snap = {}
    if isinstance(canonical, dict):
        for k in ("award_name", "proxy_name", "winner_name", "absence_reason"):
            ref_snap[k] = canonical.get(k)

    return ScoreResult(
        model=model_name,
        score=score,
        max_score=1,
        claimed_years=years,
        matched_year=matched,
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
                "claimed_years": r.claimed_years,
                "matched_year": r.matched_year,
                "extraction_resolution": r.resolution,
                "reference_snapshot": r.reference_snapshot,
                "benchmark_version": BENCHMARK["version"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def _write_ranking_report(output_dir: Path, results: list[ScoreResult]) -> None:
    lines: list[str] = []
    lines.append("# Query 21 Ranking Report (v2)")
    lines.append("")
    lines.append(
        f"> Benchmark version: {BENCHMARK['version']} (locked {BENCHMARK['locked_at']})"
    )
    lines.append(f"> Accepted years: {sorted(BENCHMARK['accepted_years'])}")
    lines.append("")
    lines.append("## 排名")
    lines.append("")
    lines.append("| Rank | Model | Score | Claimed Year(s) | Matched | Resolution |")
    lines.append("|---:|---|---:|---|---|---|")
    ranked = sorted(results, key=lambda r: (-r.score, r.model))
    for i, r in enumerate(ranked, start=1):
        years = ", ".join(str(y) for y in r.claimed_years) or "—"
        matched = str(r.matched_year) if r.matched_year is not None else "—"
        lines.append(
            f"| {i} | {r.model} | {r.score}/{r.max_score} | "
            f"{years} | {matched} | {r.resolution} |"
        )
    lines.append("")
    lines.append("## 参考项（不计分）")
    lines.append("")
    lines.append("| Model | Award | Proxy | Winner | Absence Reason |")
    lines.append("|---|---|---|---|---|")
    for r in ranked:
        s = r.reference_snapshot
        lines.append(
            f"| {r.model} | {s.get('award_name') or '—'} | "
            f"{s.get('proxy_name') or '—'} | {s.get('winner_name') or '—'} | "
            f"{(s.get('absence_reason') or '—')[:80]} |"
        )
    (output_dir / "ranking_report.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


# ═══════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════


def main() -> None:
    ap = argparse.ArgumentParser(description="Query 21 auto-scorer v2")
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
    print("Query 21 scoring done.")
    for r in sorted(results, key=lambda x: (-x.score, x.model)):
        mark = "✅" if r.score else "❌"
        years = ", ".join(str(y) for y in r.claimed_years) or "—"
        print(f"  {mark} {r.model}: {r.score}/{r.max_score}  years={years}")


if __name__ == "__main__":
    main()
