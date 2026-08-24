"""
Query 24 — Auto scorer v2

Scoring rule (see scoring_framework.md):
  D1. Reporter name matches 薛涵仪 → 1, else 0
  D2. Either "no other documentaries" OR lists at least one real other work
      → 1, else 0 (D1 must pass first; chain dependency).
  Total max = 2.
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

# ═══════════════════════════════════════════════════════════════════════════
# Ground truth
# ═══════════════════════════════════════════════════════════════════════════

BENCHMARK: dict = {
    "version": "2.0",
    "locked_at": "2026-04-23",
    "reporter_name_variants": {
        "薛涵仪",
        "Xue Hanyi",
        "Xue Han-yi",
    },
    # Accepted "other work" titles (case-insensitive, substring match).
    # Participation in any of these → earn D2 (model said she has other works).
    "accepted_other_works": [
        "王暖暖",  # 《中国孕妇泰国坠崖案王暖暖专访》
        "胡同里的北京",
        "上露台",
        "杨一面对面",
        "常青藤之声",
    ],
    "ground_truth_has_other_docs": False,  # strictly documentaries: no.
}


# ═══════════════════════════════════════════════════════════════════════════
# Name matching
# ═══════════════════════════════════════════════════════════════════════════


def _norm(s: str) -> str:
    return "".join(c for c in (s or "").lower() if not c.isspace())


def _matches_reporter(claimed: str | None) -> bool:
    if not claimed:
        return False
    n = _norm(claimed)
    return any(
        _norm(v) in n or n in _norm(v) for v in BENCHMARK["reporter_name_variants"]
    )


def _matches_accepted_work(work: str | None) -> bool:
    if not work:
        return False
    w = _norm(work)
    return any(_norm(a) in w for a in BENCHMARK["accepted_other_works"])


# ═══════════════════════════════════════════════════════════════════════════
# Scoring
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class ScoreResult:
    model: str
    d1_score: int
    d2_score: int
    total: int
    max_total: int
    reporter_name: str | None
    no_other_works: bool | None
    other_works: list[str]
    resolution: str


def _score_model(model_name: str, payload: dict) -> ScoreResult:
    entities = payload.get("entities", [])
    if not entities:
        return ScoreResult(model_name, 0, 0, 0, 2, None, None, [], "empty")
    ent = entities[0]
    canonical = (ent.get("canonical") or {}).get("value")
    canonical = canonical if isinstance(canonical, dict) else {}

    reporter = canonical.get("reporter_name")
    no_other = canonical.get("no_other_works")
    other_works = canonical.get("other_works") or []
    if not isinstance(other_works, list):
        other_works = []

    # D1 — name
    d1 = 1 if _matches_reporter(reporter) else 0

    # D2 — other works (chain-dependent on D1)
    if d1 == 0:
        d2 = 0
    else:
        # Case A: model claims no other works → accept.
        # Case B: model lists ≥1 accepted work → accept.
        # Case C: model claims yes but lists no acceptable work → reject.
        d2 = 1 if no_other is True or any(_matches_accepted_work(w) for w in other_works) else 0

    return ScoreResult(
        model=model_name,
        d1_score=d1,
        d2_score=d2,
        total=d1 + d2,
        max_total=2,
        reporter_name=reporter,
        no_other_works=no_other,
        other_works=other_works,
        resolution=ent.get("resolution", "unknown"),
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
                "score": r.total,
                "max_score": r.max_total,
                "d1_name": r.d1_score,
                "d2_other_works": r.d2_score,
                "reporter_name": r.reporter_name,
                "no_other_works": r.no_other_works,
                "other_works": r.other_works,
                "extraction_resolution": r.resolution,
                "benchmark_version": BENCHMARK["version"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def _write_ranking_report(output_dir: Path, results: list[ScoreResult]) -> None:
    lines: list[str] = []
    lines.append("# Query 24 Ranking Report (v2)")
    lines.append("")
    lines.append(
        f"> Benchmark version {BENCHMARK['version']} (locked {BENCHMARK['locked_at']})"
    )
    lines.append("> Reporter: 薛涵仪 (Xue Hanyi); strictly: no other documentaries.")
    lines.append("")
    lines.append(
        "| Rank | Model | Total | D1(Name) | D2(Works) "
        "| Reporter | no_other | Other Works |"
    )
    lines.append("|---:|---|---:|---:|---:|---|---|---|")
    ranked = sorted(results, key=lambda r: (-r.total, r.model))
    for i, r in enumerate(ranked, start=1):
        works = "; ".join(r.other_works) if r.other_works else "—"
        lines.append(
            f"| {i} | {r.model} | {r.total}/{r.max_total} "
            f"| {r.d1_score} | {r.d2_score} "
            f"| {r.reporter_name or '—'} | {r.no_other_works} "
            f"| {works[:80]} |"
        )
    (output_dir / "ranking_report.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


# ═══════════════════════════════════════════════════════════════════════════
# Entry
# ═══════════════════════════════════════════════════════════════════════════


def main() -> None:
    ap = argparse.ArgumentParser(description="Query 24 auto-scorer v2")
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--primary-model", default=None)
    ap.add_argument("--secondary-model", default=None)
    ap.add_argument("--parallel-models", nargs="+", default=None)
    ap.add_argument("--analyzer-model", default=None)
    ap.add_argument("--concurrency", type=int, default=None)
    ap.add_argument("--skip-extract", action="store_true")
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

    results = []
    for model_name, payload in all_results.items():
        r = _score_model(model_name, payload)
        _write_model_score(out_dir, r)
        results.append(r)

    _write_ranking_report(out_dir, results)
    print("\n" + "─" * 60)
    print("Query 24 scoring done.")
    for r in sorted(results, key=lambda x: (-x.total, x.model)):
        print(
            f"  {r.model}: {r.total}/{r.max_total}  "
            f"(D1={r.d1_score} name={r.reporter_name!r}; "
            f"D2={r.d2_score} works={len(r.other_works)})"
        )


if __name__ == "__main__":
    main()
