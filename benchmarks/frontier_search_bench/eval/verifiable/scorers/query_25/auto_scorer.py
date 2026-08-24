"""
Query 25 — Auto scorer v2

8 stages × 1 pt (ratio-only, ±3pp tolerance) + Ethiopian per-cup 0/1/2.
Total = 10.

Merging rule: if a stage's value note says "combined" (or list of stage ids),
the merged percent must lie within the sum of the merged stages' GT ranges
(also ±3pp loose). All merged stages then each earn 1 pt.
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

# base GT ranges (% of US $5.50 retail)
GT_RANGES: dict[str, tuple[float, float]] = {
    "S1_planting": (0.5, 1.5),
    "S2_picking": (0.3, 0.8),
    "S3_processing": (0.5, 1.5),
    "S4_transport": (1.0, 3.0),
    "S5_roasting": (3.0, 6.0),
    "S6_packaging": (2.0, 5.0),
    "S7_store_ops": (35.0, 50.0),
    "S8_taxes": (2.0, 10.0),
}
LOOSE_PP = 3.0  # pp tolerance on each side

ETH_GT = {
    "strict_min": 0.03,
    "strict_max": 0.08,  # 2 pts
    "loose_min": 0.01,
    "loose_max": 0.15,  # 1 pt
}

BENCHMARK: dict = {
    "version": "2.0",
    "locked_at": "2026-04-23",
    "loose_pp": LOOSE_PP,
    "gt_ranges": GT_RANGES,
    "ethiopian_cup": ETH_GT,
}


# ═══════════════════════════════════════════════════════════════════════════
# Parsing helpers
# ═══════════════════════════════════════════════════════════════════════════

_PCT = re.compile(r"(-?\d+(?:\.\d+)?)\s*%")
_NUM = re.compile(r"-?\d+(?:\.\d+)?")
_USD = re.compile(r"\$\s*(\d+(?:\.\d+)?)")
_CNY = re.compile(r"[¥￥]\s*(\d+(?:\.\d+)?)")


def _parse_percent(v) -> float | None:
    if v is None:
        return None
    if isinstance(v, (int, float)):
        n = float(v)
        # heuristic: >1 treat as already percent; 0-1 treat as decimal fraction
        return n if n > 1 else n * 100
    s = str(v)
    m = _PCT.search(s)
    if m:
        return float(m.group(1))
    m = _NUM.search(s)
    if m:
        return float(m.group())
    return None


def _parse_usd_per_cup(v) -> float | None:
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v)
    m = _USD.search(s)
    if m:
        return float(m.group(1))
    m = _CNY.search(s)
    if m:
        return float(m.group(1)) / 7.2  # crude ¥ → $ rate
    m = _NUM.search(s)
    if m:
        return float(m.group())
    return None


def _parse_note_merged(note) -> list[str]:
    """Return list of stage ids this stage is merged with, if any."""
    if not note:
        return []
    s = str(note).lower()
    if "combined" not in s and "merged" not in s and "合并" not in s:
        return []
    stages = []
    for sid in GT_RANGES:
        # match "S1", "s2" or a Chinese token for that stage
        if re.search(rf"\b{sid[:2].lower()}\b", s):
            stages.append(sid)
    # fallback Chinese keywords
    kw_map = {
        "种植": "S1_planting",
        "采摘": "S2_picking",
        "加工": "S3_processing",
        "运输": "S4_transport",
        "烘焙": "S5_roasting",
        "包装": "S6_packaging",
        "门店": "S7_store_ops",
        "税": "S8_taxes",
    }
    for kw, sid in kw_map.items():
        if kw in str(note) and sid not in stages:
            stages.append(sid)
    return stages


# ═══════════════════════════════════════════════════════════════════════════
# Scoring
# ═══════════════════════════════════════════════════════════════════════════


def _in_range_loose(pct: float, lo: float, hi: float) -> bool:
    return (lo - LOOSE_PP) <= pct <= (hi + LOOSE_PP)


@dataclass
class ScoreResult:
    model: str
    stage_scores: dict[str, int]
    eth_score: int
    total: int
    max_total: int
    claimed_pcts: dict[str, float | None]
    claimed_eth_amount: float | None


def _score_model(model_name: str, payload: dict) -> ScoreResult:
    entities = payload.get("entities", [])
    by_id = {e["id"]: e for e in entities}

    claimed_pcts: dict[str, float | None] = {}
    claimed_notes: dict[str, list[str]] = {}
    for sid in GT_RANGES:
        ent = by_id.get(sid, {})
        canonical = (ent.get("canonical") or {}).get("value") or {}
        if not isinstance(canonical, dict):
            canonical = {}
        pct = _parse_percent(canonical.get("percent"))
        note_stages = _parse_note_merged(canonical.get("note"))
        claimed_pcts[sid] = pct
        claimed_notes[sid] = note_stages

    stage_scores: dict[str, int] = {sid: 0 for sid in GT_RANGES}

    # First pass: non-merged entries
    for sid in GT_RANGES:
        if claimed_notes[sid]:
            continue
        pct = claimed_pcts[sid]
        if pct is None:
            continue
        lo, hi = GT_RANGES[sid]
        if _in_range_loose(pct, lo, hi):
            stage_scores[sid] = 1

    # Second pass: merged entries
    for sid in GT_RANGES:
        merged = claimed_notes[sid]
        if not merged:
            continue
        pct = claimed_pcts[sid]
        if pct is None:
            continue
        merged_set = set(merged) | {sid}
        lo_sum = sum(GT_RANGES[s][0] for s in merged_set)
        hi_sum = sum(GT_RANGES[s][1] for s in merged_set)
        if _in_range_loose(pct, lo_sum, hi_sum):
            for s in merged_set:
                stage_scores[s] = 1

    # Ethiopia cup
    ent_e = by_id.get("E_ethiopia_cup", {})
    canonical_e = (ent_e.get("canonical") or {}).get("value") or {}
    if not isinstance(canonical_e, dict):
        canonical_e = {}
    eth_amount = _parse_usd_per_cup(
        canonical_e.get("amount_per_cup") or canonical_e.get("percent")
    )
    if eth_amount is None:
        eth_score = 0
    elif ETH_GT["strict_min"] <= eth_amount <= ETH_GT["strict_max"]:
        eth_score = 2
    elif ETH_GT["loose_min"] <= eth_amount <= ETH_GT["loose_max"]:
        eth_score = 1
    else:
        eth_score = 0

    total = sum(stage_scores.values()) + eth_score
    return ScoreResult(
        model=model_name,
        stage_scores=stage_scores,
        eth_score=eth_score,
        total=total,
        max_total=len(GT_RANGES) + 2,
        claimed_pcts=claimed_pcts,
        claimed_eth_amount=eth_amount,
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
                "stage_scores": r.stage_scores,
                "eth_score": r.eth_score,
                "claimed_percentages": r.claimed_pcts,
                "claimed_eth_usd_per_cup": r.claimed_eth_amount,
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
    lines.append("# Query 25 Ranking Report (v2)")
    lines.append("")
    lines.append(
        f"> Benchmark version: {BENCHMARK['version']} (locked {BENCHMARK['locked_at']})"
    )
    lines.append("")
    hdr = (
        "| Rank | Model | Total | "
        + " | ".join([s[:6] for s in GT_RANGES])
        + " | EthCup |"
    )
    sep = "|---:|---|---:|" + "---:|" * (len(GT_RANGES) + 1)
    lines.append(hdr)
    lines.append(sep)
    ranked = sorted(results, key=lambda r: (-r.total, r.model))
    for i, r in enumerate(ranked, start=1):
        cells = [f"{r.stage_scores[s]}" for s in GT_RANGES]
        lines.append(
            f"| {i} | {r.model} | {r.total}/{r.max_total} | "
            + " | ".join(cells)
            + f" | {r.eth_score} |"
        )
    lines.append("")
    lines.append("## 声称比例 / 金额")
    lines.append("")
    hdr2 = (
        "| Model | "
        + " | ".join([s.split("_", 1)[1] for s in GT_RANGES])
        + " | Eth $/cup |"
    )
    sep2 = "|---|" + "---|" * (len(GT_RANGES) + 1)
    lines.append(hdr2)
    lines.append(sep2)
    for r in ranked:
        cells = [
            ("—" if r.claimed_pcts[s] is None else f"{r.claimed_pcts[s]:.1f}%")
            for s in GT_RANGES
        ]
        eth = "—" if r.claimed_eth_amount is None else f"${r.claimed_eth_amount:.3f}"
        lines.append(f"| {r.model} | " + " | ".join(cells) + f" | {eth} |")
    (output_dir / "ranking_report.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Query 25 auto-scorer v2")
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
    print("Query 25 scoring done.")
    for r in sorted(results, key=lambda x: (-x.total, x.model)):
        print(
            f"  {r.model}: {r.total}/{r.max_total}  "
            f"stages={r.stage_scores} eth={r.eth_score}"
        )


if __name__ == "__main__":
    main()
