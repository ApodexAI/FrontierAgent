"""
Query 26 — Auto scorer (binary, single-fact)

Question: 请在中国A股市场中识别一只股票：它在2025年12月附近曾跌至约5.50的阶段
低点，随后在2026年1月初快速上涨，并在短时间内触及约9.45的阶段高点；其后股价
有所回落，但并未完全回吐涨幅，而是在7元多的区间继续震荡。公开交易数据还显示，
该股某一活跃交易日的成交量约为184.78万手，换手率约为10%。请判断这只股票最
可能是哪只，并给出证据链。

Scoring rule:
  Single binary dimension —
    1 if extracted stock_name / supporting_span / canonical fields contains
      the literal Chinese name "天下秀"
    0 otherwise.
  Auxiliary fields (ticker / exchange / price-volume claims / evidence)
  recorded for audit only, not included in total. Ticker 600556 is recorded
  but not required by the judge (per original judge_q26 spec).

Flow:
  1) Run the shared extraction pipeline (primary + secondary; phase-4 on
     disagreements). Canonical extraction lands in
     auto_scores/{model}/extraction.json.
  2) Load each model's canonical value, judge presence of "天下秀",
     write auto_scores/{model}/score.json and a top-level ranking_report.md.
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
    "required_name_marker": "天下秀",  # literal substring; must be present
    "reference": {
        "stock_name": "天下秀",
        "ticker": "600556",
        "exchange": "上交所",
        "low_price": 5.50,
        "peak_price": 9.45,
        "volume_wanshou": 184.78,
        "turnover_rate": 10.0,
    },
}


# ═══════════════════════════════════════════════════════════════════════════
# Marker matching
# ═══════════════════════════════════════════════════════════════════════════


def _collect_text_fields(canonical_value, supporting_span: str | None) -> list[str]:
    """Pool every text field where the marker might appear."""
    fields: list[str] = []
    if supporting_span:
        fields.append(str(supporting_span))
    if isinstance(canonical_value, dict):
        for k in (
            "stock_name",
            "ticker",
            "exchange",
            "evidence_summary",
        ):
            v = canonical_value.get(k)
            if v:
                fields.append(str(v))
        for alt in canonical_value.get("alternative_candidates") or []:
            if isinstance(alt, dict):
                for k in ("stock_name", "ticker", "evidence_summary"):
                    v = alt.get(k)
                    if v:
                        fields.append(str(v))
    elif canonical_value:
        fields.append(str(canonical_value))
    return fields


def _has_name_marker(canonical_value, supporting_span: str | None) -> bool:
    marker = BENCHMARK["required_name_marker"]
    return any(marker in s for s in _collect_text_fields(canonical_value, supporting_span))


def _claimed_stock_names(canonical_value) -> list[str]:
    """All distinct stock names asserted (primary + alternatives)."""
    names: list[str] = []
    if isinstance(canonical_value, dict):
        n = canonical_value.get("stock_name")
        if n:
            names.append(str(n))
        for alt in canonical_value.get("alternative_candidates") or []:
            if isinstance(alt, dict):
                an = alt.get("stock_name")
                if an and str(an) not in names:
                    names.append(str(an))
    return names


# ═══════════════════════════════════════════════════════════════════════════
# Per-model scoring
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class ScoreResult:
    model: str
    score: int
    max_score: int
    claimed_stock_names: list[str]
    name_marker_present: bool
    resolution: str
    reference_snapshot: dict = field(default_factory=dict)


def _score_model(model_name: str, extraction_payload: dict) -> ScoreResult:
    entities = extraction_payload.get("entities", [])
    if not entities:
        return ScoreResult(
            model=model_name,
            score=0,
            max_score=1,
            claimed_stock_names=[],
            name_marker_present=False,
            resolution="empty_extraction",
            reference_snapshot={},
        )
    ent = entities[0]
    canonical = (ent.get("canonical") or {}).get("value")
    span = (ent.get("canonical") or {}).get("supporting_span")

    names = _claimed_stock_names(canonical)
    has_marker = _has_name_marker(canonical, span)
    score = 1 if has_marker else 0

    ref_snap: dict = {}
    if isinstance(canonical, dict):
        for k in (
            "stock_name",
            "ticker",
            "exchange",
            "low_price",
            "peak_price",
            "volume_wanshou",
            "turnover_rate",
            "evidence_summary",
        ):
            ref_snap[k] = canonical.get(k)

    return ScoreResult(
        model=model_name,
        score=score,
        max_score=1,
        claimed_stock_names=names,
        name_marker_present=has_marker,
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
                "claimed_stock_names": r.claimed_stock_names,
                "name_marker_present": r.name_marker_present,
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
    lines.append("# Query 26 Ranking Report (v1)")
    lines.append("")
    lines.append(
        f"> Benchmark version: {BENCHMARK['version']} (locked {BENCHMARK['locked_at']})"
    )
    lines.append(
        f"> Required name marker: '{BENCHMARK['required_name_marker']}' "
        f"(reference ticker {BENCHMARK['reference']['ticker']}, ticker not required)"
    )
    lines.append("")
    lines.append("## 排名")
    lines.append("")
    lines.append(
        "| Rank | Model | Score | Claimed Stock(s) | 天下秀? | Resolution |"
    )
    lines.append("|---:|---|---:|---|:---:|---|")
    ranked = sorted(results, key=lambda r: (-r.score, r.model))
    for i, r in enumerate(ranked, start=1):
        names = ", ".join(r.claimed_stock_names) or "—"
        marker = "✅" if r.name_marker_present else "❌"
        lines.append(
            f"| {i} | {r.model} | {r.score}/{r.max_score} | "
            f"{names} | {marker} | {r.resolution} |"
        )
    lines.append("")
    lines.append("## 参考项（不计分）")
    lines.append("")
    lines.append(
        "| Model | 股名 | 代码 | 交易所 | low | peak | 量(万手) | 换手率(%) |"
    )
    lines.append("|---|---|---|---|---:|---:|---:|---:|")
    for r in ranked:
        s = r.reference_snapshot
        def _f(x):
            return "—" if x is None else x
        lines.append(
            f"| {r.model} | {_f(s.get('stock_name'))} | {_f(s.get('ticker'))} | "
            f"{_f(s.get('exchange'))} | {_f(s.get('low_price'))} | "
            f"{_f(s.get('peak_price'))} | {_f(s.get('volume_wanshou'))} | "
            f"{_f(s.get('turnover_rate'))} |"
        )
    (output_dir / "ranking_report.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


# ═══════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════


def main() -> None:
    ap = argparse.ArgumentParser(description="Query 26 auto-scorer (v1)")
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
    print("Query 26 scoring done.")
    for r in sorted(results, key=lambda x: (-x.score, x.model)):
        mark = "✅" if r.score else "❌"
        names = ", ".join(r.claimed_stock_names) or "—"
        print(
            f"  {mark} {r.model}: {r.score}/{r.max_score}  "
            f"marker={'Y' if r.name_marker_present else 'N'}  "
            f"names={names}"
        )


if __name__ == "__main__":
    main()
