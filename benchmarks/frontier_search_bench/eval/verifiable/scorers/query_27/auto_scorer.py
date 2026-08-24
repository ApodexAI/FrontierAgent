"""
Query 27 — High-order Mobius strip (540° / 3 half-twists) cut at 0.5 cm
from the edge — auto-scorer.

5 binary sub-dimensions × 1 point each, max = 5:
  D1: strip_count == 2                                                    (1 pt)
  D2: ∃ strip with width ≈ 0.5 cm AND length ≈ 60 cm                       (1 pt)
  D3: ∃ strip with width ≈ 1.0 cm AND length ≈ 30 cm                       (1 pt)
  D4: widest_half_twists == 3                                              (1 pt)
  D5: interlocking == True                                                 (1 pt)

Numeric tolerance: |claim - benchmark| < 0.05 cm.

Flow:
  1) Run shared extraction pipeline (primary + secondary; phase-4 on
     disagreements). Canonical extraction lands in
     auto_scores/{model}/extraction.json.
  2) Score each model's canonical value against 5 sub-dimensions
     deterministically. Write auto_scores/{model}/score.json + top-level
     ranking_report.md.
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
    "tolerance_cm": 0.05,
    "expected_strip_count": 2,
    "expected_strips": [
        {"width": 0.5, "length": 60.0},  # narrow / edge strip (60 cm)
        {"width": 1.0, "length": 30.0},  # wide / middle strip (30 cm)
    ],
    "expected_widest_half_twists": 3,
    "expected_interlocking": True,
}

DIM_SPECS = [
    ("D1", "strip_count == 2", 1),
    ("D2", "∃ strip ≈ {width 0.5 cm, length 60 cm}", 1),
    ("D3", "∃ strip ≈ {width 1.0 cm, length 30 cm}", 1),
    ("D4", "widest strip has 3 half-twists", 1),
    ("D5", "widest strip interlocks with another strip", 1),
]
MAX_SCORE = sum(s[2] for s in DIM_SPECS)  # 5


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════


def _approx_eq(a, b, tol: float = BENCHMARK["tolerance_cm"]) -> bool:
    if a is None or b is None:
        return False
    try:
        return abs(float(a) - float(b)) < tol
    except (TypeError, ValueError):
        return False


def _coerce_int(v) -> int | None:
    if v is None:
        return None
    if isinstance(v, bool):
        return None
    if isinstance(v, int):
        return v
    if isinstance(v, float):
        return int(v) if abs(v - int(v)) < 1e-6 else None
    s = str(v).strip()
    if not s:
        return None
    try:
        f = float(s)
        return int(f) if abs(f - int(f)) < 1e-6 else None
    except ValueError:
        # try first integer in string
        import re
        m = re.search(r"-?\d+", s)
        return int(m.group()) if m else None


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


def _normalize_strips(raw_strips) -> list[dict]:
    """Coerce strips list → [{'width': float|None, 'length': float|None}, ...]."""
    out: list[dict] = []
    if not isinstance(raw_strips, list):
        return out
    for s in raw_strips:
        if not isinstance(s, dict):
            continue
        out.append(
            {
                "width": _coerce_float(s.get("width")),
                "length": _coerce_float(s.get("length")),
            }
        )
    return out


def _has_strip(strips: list[dict], target: dict) -> bool:
    """True if `strips` contains a strip approximately matching target dims."""
    tw, tl = target.get("width"), target.get("length")
    return any(_approx_eq(s.get("width"), tw) and _approx_eq(s.get("length"), tl) for s in strips)


# ═══════════════════════════════════════════════════════════════════════════
# Per-dimension scoring
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class DimScore:
    id: str
    score: int
    max_score: int
    claimed: object
    benchmark: object
    reason: str


def _score_dimensions(canonical_value) -> list[DimScore]:
    val = canonical_value if isinstance(canonical_value, dict) else {}

    strip_count = _coerce_int(val.get("strip_count"))
    strips = _normalize_strips(val.get("strips"))
    widest_ht = _coerce_int(val.get("widest_half_twists"))
    interlocking = val.get("interlocking")

    # D1
    d1_ok = (strip_count == BENCHMARK["expected_strip_count"])
    d1 = DimScore(
        id="D1",
        score=1 if d1_ok else 0,
        max_score=1,
        claimed=strip_count,
        benchmark=BENCHMARK["expected_strip_count"],
        reason=("命中" if d1_ok else f"期望 {BENCHMARK['expected_strip_count']}，实际 {strip_count}"),
    )

    # D2
    target_narrow = BENCHMARK["expected_strips"][0]
    d2_ok = _has_strip(strips, target_narrow)
    d2 = DimScore(
        id="D2",
        score=1 if d2_ok else 0,
        max_score=1,
        claimed=strips,
        benchmark=target_narrow,
        reason=("命中（找到 width≈0.5, length≈60 的纸带）" if d2_ok else "未找到 width≈0.5 且 length≈60 的纸带"),
    )

    # D3
    target_wide = BENCHMARK["expected_strips"][1]
    d3_ok = _has_strip(strips, target_wide)
    d3 = DimScore(
        id="D3",
        score=1 if d3_ok else 0,
        max_score=1,
        claimed=strips,
        benchmark=target_wide,
        reason=("命中（找到 width≈1.0, length≈30 的纸带）" if d3_ok else "未找到 width≈1.0 且 length≈30 的纸带"),
    )

    # D4
    d4_ok = (widest_ht == BENCHMARK["expected_widest_half_twists"])
    d4 = DimScore(
        id="D4",
        score=1 if d4_ok else 0,
        max_score=1,
        claimed=widest_ht,
        benchmark=BENCHMARK["expected_widest_half_twists"],
        reason=("命中" if d4_ok else f"期望 {BENCHMARK['expected_widest_half_twists']}，实际 {widest_ht}"),
    )

    # D5
    d5_ok = (interlocking is True)
    d5 = DimScore(
        id="D5",
        score=1 if d5_ok else 0,
        max_score=1,
        claimed=interlocking,
        benchmark=BENCHMARK["expected_interlocking"],
        reason=("命中（互锁）" if d5_ok else f"期望互锁=True，实际 {interlocking}"),
    )

    return [d1, d2, d3, d4, d5]


# ═══════════════════════════════════════════════════════════════════════════
# Per-model scoring
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class ScoreResult:
    model: str
    total_score: int
    max_score: int
    dims: list[DimScore]
    resolution: str
    reasoning_summary: str | None = None
    extracted_value: dict = field(default_factory=dict)


def _score_model(model_name: str, extraction_payload: dict) -> ScoreResult:
    entities = extraction_payload.get("entities", [])
    if not entities:
        return ScoreResult(
            model=model_name,
            total_score=0,
            max_score=MAX_SCORE,
            dims=[
                DimScore(d_id, 0, max_s, None, None, "empty extraction")
                for d_id, _, max_s in DIM_SPECS
            ],
            resolution="empty_extraction",
            reasoning_summary=None,
            extracted_value={},
        )
    ent = entities[0]
    canonical = (ent.get("canonical") or {}).get("value")
    dims = _score_dimensions(canonical)
    total = sum(d.score for d in dims)

    summary = None
    extracted: dict = {}
    if isinstance(canonical, dict):
        summary = canonical.get("reasoning_summary")
        for k in ("strip_count", "strips", "widest_half_twists", "interlocking"):
            extracted[k] = canonical.get(k)

    return ScoreResult(
        model=model_name,
        total_score=total,
        max_score=MAX_SCORE,
        dims=dims,
        resolution=ent.get("resolution", "unknown"),
        reasoning_summary=summary,
        extracted_value=extracted,
    )


# ═══════════════════════════════════════════════════════════════════════════
# I/O
# ═══════════════════════════════════════════════════════════════════════════


def _dim_to_dict(d: DimScore) -> dict:
    return {
        "id": d.id,
        "score": d.score,
        "max_score": d.max_score,
        "claimed": d.claimed,
        "benchmark": d.benchmark,
        "reason": d.reason,
    }


def _write_model_score(output_dir: Path, r: ScoreResult) -> None:
    path = output_dir / r.model / "score.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "query_id": QUERY_ID,
                "model": r.model,
                "total_score": r.total_score,
                "max_score": r.max_score,
                "dims": [_dim_to_dict(d) for d in r.dims],
                "extraction_resolution": r.resolution,
                "extracted_value": r.extracted_value,
                "reasoning_summary": r.reasoning_summary,
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
    lines.append("# Query 27 Ranking Report (v1)")
    lines.append("")
    lines.append(
        f"> Benchmark version: {BENCHMARK['version']} (locked {BENCHMARK['locked_at']})"
    )
    lines.append(
        f"> Numeric tolerance: ±{BENCHMARK['tolerance_cm']} cm  ·  Max: {MAX_SCORE}"
    )
    lines.append(
        "> Dims: D1 strip_count · D2 narrow strip(0.5×60) · "
        "D3 wide strip(1×30) · D4 half-twists==3 · D5 interlocking"
    )
    lines.append("")
    lines.append("## 排名")
    lines.append("")
    lines.append(
        "| Rank | Model | Total | D1 | D2 | D3 | D4 | D5 | Resolution |"
    )
    lines.append("|---:|---|---:|:---:|:---:|:---:|:---:|:---:|---|")
    ranked = sorted(results, key=lambda r: (-r.total_score, r.model))
    for i, r in enumerate(ranked, start=1):
        cells = [("✅" if d.score else "❌") for d in r.dims]
        lines.append(
            f"| {i} | {r.model} | {r.total_score}/{r.max_score} | "
            + " | ".join(cells)
            + f" | {r.resolution} |"
        )
    lines.append("")
    lines.append("## 每模型抽取值（参考，不计分）")
    lines.append("")
    for r in ranked:
        lines.append(f"### {r.model} (total={r.total_score}/{r.max_score})")
        ev = r.extracted_value
        lines.append(f"- strip_count: `{ev.get('strip_count')}`")
        lines.append(f"- strips: `{ev.get('strips')}`")
        lines.append(f"- widest_half_twists: `{ev.get('widest_half_twists')}`")
        lines.append(f"- interlocking: `{ev.get('interlocking')}`")
        for d in r.dims:
            lines.append(f"  - **{d.id}** {d.score}/{d.max_score} — {d.reason}")
        lines.append("")
    (output_dir / "ranking_report.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


# ═══════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════


def main() -> None:
    ap = argparse.ArgumentParser(description="Query 27 auto-scorer (v1)")
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
    print("Query 27 scoring done.")
    for r in sorted(results, key=lambda x: (-x.total_score, x.model)):
        cells = "".join("✅" if d.score else "❌" for d in r.dims)
        print(f"  {r.model:28s} {r.total_score}/{r.max_score}  [{cells}]")


if __name__ == "__main__":
    main()
