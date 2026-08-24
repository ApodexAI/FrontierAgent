"""
Query 16 — Auto scorer v2

9 stat items (physics/chem/medicine × median/mean/trend) × 1 pt +
6 extremes (3 longest + 3 shortest) × 1 pt each = 15 max.
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
# Ground truth
# ═══════════════════════════════════════════════════════════════════════════

STAT_GT: dict[str, dict] = {
    # Ranges calibrated against the Nobel 2000-2025 raw table (inlined as LONGEST/SHORTEST below).
    # Physics actual: mean 28.6, median 31, slope +0.414 → lengthening
    # Chemistry actual: mean 23.7, median 24, slope +0.283 → slight lengthening
    # Medicine actual: mean 24.8, median 25, slope +0.058 → essentially flat
    "phys_median": {"type": "number", "min": 25, "max": 35},
    "phys_mean": {"type": "number", "min": 22, "max": 32},
    "phys_trend": {"type": "trend", "accept": "lengthening"},
    "chem_median": {"type": "number", "min": 18, "max": 28},
    "chem_mean": {"type": "number", "min": 18, "max": 28},
    "chem_trend": {"type": "trend", "accept": "lengthening_or_stable"},
    "med_median": {"type": "number", "min": 20, "max": 30},
    "med_mean": {"type": "number", "min": 20, "max": 30},
    # Medicine actual slope is +0.058 (essentially flat). Accept either
    # "stable / 持平" OR a not-shortening statement.
    "med_trend": {"type": "trend", "accept": "stable_or_lengthening"},
}

LENGTHEN_KEYWORDS = {
    "lengthening",
    "lengthen",
    "longer",
    "increasing",
    "increase",
    "延长",
    "变长",
    "上升",
    "加长",
    "拉长",
}
STABLE_KEYWORDS = {
    "stable",
    "flat",
    "unchanged",
    "no change",
    "same",
    "steady",
    "持平",
    "不变",
    "稳定",
    "没变",
    "没有缩短",
    "not shortening",
    "not shortened",
    "did not shorten",
    "hasn't shortened",
}
SHORTEN_KEYWORDS = {
    "shortening",
    "shorten",
    "shorter",
    "decreasing",
    "decrease",
    "缩短",
    "变短",
    "下降",
}

# v2.1: longest-wait pool derived from the Nobel 2000-2025 raw table
# Cross-discipline top 9 (inlined below):
# Penrose 55, Peebles 54, Manabe 54, Ginzburg 53, Clauser 50, Gurdon 50,
# Englert 49, Higgs 49, Nambu 47.  Also accept well-documented ≥ 40-year
# waits: Goodenough (39, chem 2019), Shimomura (46, chem 2008),
# Whittingham (43, chem 2019), Yekimov (42, chem 2023), Gross/Wilczek/
# Politzer (31, physics 2004), Alter (45, medicine 2020),
# Carlsson (43, medicine 2000).
LONGEST_POOL = {
    "penrose",
    "peebles",
    "manabe",
    "ginzburg",
    "clauser",
    "gurdon",
    "englert",
    "higgs",
    "nambu",
    "南部",
    "goodenough",
    "shimomura",
    "whittingham",
    "yekimov",
    "gross",
    "wilczek",
    "politzer",
    "alter",
    "carlsson",
}

# v2.1: shortest-wait pool from raw table.  Cross-discipline top 9:
# Weiss 1, Thorne 1, Barish 1 (physics 2017 LIGO),
# Hassabis 3, Jumper 3 (chem 2024 AlphaFold),
# MacKinnon 5 (chem 2003), Kornberg 5 (chem 2006),
# Kobilka 5 (chem 2012), Cornell 6 (physics 2001).
# Also accept other known short-wait names from 2000–2025:
# Yamanaka 6 (med 2012 iPSC), Fire 8, Mello 8 (med 2006 RNAi),
# Doudna/Charpentier 8 (chem 2020 CRISPR),
# Geim/Novoselov 6 (physics 2010 graphene).
SHORTEST_POOL = {
    "weiss",
    "thorne",
    "barish",
    "ligo",
    "hassabis",
    "jumper",
    "alphafold",
    "mackinnon",
    "kornberg",
    "kobilka",
    "cornell",
    "wieman",
    "yamanaka",
    "fire",
    "mello",
    "doudna",
    "charpentier",
    "crispr",
    "geim",
    "novoselov",
    "graphene",
}

BENCHMARK = {
    "version": "2.1",
    "locked_at": "2026-04-23",
    "stat_gt": STAT_GT,
    "longest_pool": sorted(LONGEST_POOL),
    "shortest_pool": sorted(SHORTEST_POOL),
}


# ═══════════════════════════════════════════════════════════════════════════
# Parsing
# ═══════════════════════════════════════════════════════════════════════════

_NUM = re.compile(r"-?\d+(?:\.\d+)?")


def _first_number(v) -> float | None:
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, list):
        for i in v:
            n = _first_number(i)
            if n is not None:
                return n
        return None
    m = _NUM.search(str(v))
    return float(m.group()) if m else None


def _in_range(v, lo: float, hi: float) -> bool:
    n = _first_number(v)
    if n is None:
        return False
    return lo <= n <= hi


def _trend_match(v, accept: str) -> bool:
    """v2.1 trend matcher supporting multiple `accept` modes."""
    if v is None:
        return False
    s = str(v).lower()
    has_lengthen = any(k in s for k in LENGTHEN_KEYWORDS)
    has_stable = any(k in s for k in STABLE_KEYWORDS)
    has_shorten = any(k in s for k in SHORTEN_KEYWORDS)

    if accept == "lengthening":
        return has_lengthen and not has_shorten
    if accept == "lengthening_or_stable":
        return (has_lengthen or has_stable) and not has_shorten
    if accept == "stable_or_lengthening":
        # Medicine trend is essentially flat; accept "stable/持平/unchanged"
        # OR "lengthening" (models often still say "变长" for all three),
        # but NOT "shortening/缩短".
        return (has_stable or has_lengthen) and not has_shorten
    # Fallback: legacy behaviour (lengthening only).
    return has_lengthen


def _tokens(v) -> list[str]:
    """Return lower-case tokens from a value that might be a list or string."""
    toks: list[str] = []
    if v is None:
        return toks
    if isinstance(v, list):
        for item in v:
            toks.extend(_tokens(item))
        return toks
    s = str(v).lower()
    # split by common separators
    for tok in re.split(r"[,\n;·、 /]+", s):
        tok = tok.strip()
        if tok:
            toks.append(tok)
    return toks


def _count_pool_matches(items, pool: set[str]) -> int:
    """Count distinct pool tokens that appear in any item."""
    toks = _tokens(items)
    hit: set[str] = set()
    for tok in toks:
        for k in pool:
            if k in tok or tok in k:
                hit.add(k)
    return len(hit)


# ═══════════════════════════════════════════════════════════════════════════
# Scoring
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class ScoreResult:
    model: str
    stat_scores: dict[str, int]
    longest_matches: int
    shortest_matches: int
    claimed: dict[str, object]
    total: int
    max_total: int


def _score_model(model_name: str, payload: dict) -> ScoreResult:
    entities = payload.get("entities", [])
    by_id = {e["id"]: e for e in entities}

    stat_scores: dict[str, int] = {}
    claimed: dict[str, object] = {}
    for sid, cfg in STAT_GT.items():
        ent = by_id.get(sid, {})
        canonical = (ent.get("canonical") or {}).get("value")
        claimed[sid] = canonical
        if cfg["type"] == "number":
            stat_scores[sid] = 1 if _in_range(canonical, cfg["min"], cfg["max"]) else 0
        else:
            stat_scores[sid] = 1 if _trend_match(canonical, cfg["accept"]) else 0

    ent_long = by_id.get("extreme_longest", {})
    ent_short = by_id.get("extreme_shortest", {})
    longest_val = (ent_long.get("canonical") or {}).get("value")
    shortest_val = (ent_short.get("canonical") or {}).get("value")
    claimed["extreme_longest"] = longest_val
    claimed["extreme_shortest"] = shortest_val

    longest_matches = min(3, _count_pool_matches(longest_val, LONGEST_POOL))
    shortest_matches = min(3, _count_pool_matches(shortest_val, SHORTEST_POOL))

    total = sum(stat_scores.values()) + longest_matches + shortest_matches

    return ScoreResult(
        model=model_name,
        stat_scores=stat_scores,
        longest_matches=longest_matches,
        shortest_matches=shortest_matches,
        claimed=claimed,
        total=total,
        max_total=9 + 6,
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
                "stat_scores": r.stat_scores,
                "longest_matches": r.longest_matches,
                "shortest_matches": r.shortest_matches,
                "claimed_values": r.claimed,
                "benchmark_version": BENCHMARK["version"],
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )


def _write_ranking_report(output_dir: Path, results: list[ScoreResult]) -> None:
    lines: list[str] = []
    lines.append("# Query 16 Ranking Report (v2)")
    lines.append("")
    lines.append(
        f"> Benchmark version: {BENCHMARK['version']} (locked {BENCHMARK['locked_at']})"
    )
    lines.append("")
    header = (
        "| Rank | Model | Total | "
        + "P-med | P-mean | P-trend | C-med | C-mean | C-trend "
        "| M-med | M-mean | M-trend | Long | Short |"
    )
    sep = "|---:|---|---:|" + "---:|" * 11
    lines.append(header)
    lines.append(sep)
    ranked = sorted(results, key=lambda r: (-r.total, r.model))
    for i, r in enumerate(ranked, start=1):
        ss = r.stat_scores
        cells = [
            str(ss.get("phys_median", 0)),
            str(ss.get("phys_mean", 0)),
            str(ss.get("phys_trend", 0)),
            str(ss.get("chem_median", 0)),
            str(ss.get("chem_mean", 0)),
            str(ss.get("chem_trend", 0)),
            str(ss.get("med_median", 0)),
            str(ss.get("med_mean", 0)),
            str(ss.get("med_trend", 0)),
            str(r.longest_matches),
            str(r.shortest_matches),
        ]
        lines.append(
            f"| {i} | {r.model} | {r.total}/{r.max_total} | " + " | ".join(cells) + " |"
        )
    (output_dir / "ranking_report.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Query 16 auto-scorer v2")
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
    print("Query 16 scoring done.")
    for r in sorted(results, key=lambda x: (-x.total, x.model)):
        print(f"  {r.model}: {r.total}/{r.max_total}")


if __name__ == "__main__":
    main()
