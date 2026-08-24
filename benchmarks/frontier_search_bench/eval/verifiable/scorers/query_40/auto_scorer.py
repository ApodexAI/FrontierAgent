"""
Query 40 — Route 66 elevation auto scorer.

Per-slot composite score 0/1/2/3; 9 slots × 3 = 27 max.

Slot scoring rules:
  3 = location matches a PRIMARY keyword set + elevation within ±tol_ft
  2 = location matches PRIMARY + elevation within (tol, 2×tol]; OR
      location matches SECONDARY (acceptable but not best) + elevation within tol
  1 = location matches PRIMARY but elevation > 2×tol off (or missing); OR
      location matches SECONDARY with weak elevation; OR
      elevation correct but location wrong/missing
  0 = not answered, completely wrong, or location ∈ PARTIAL (known
      misconceptions like "Sitgreaves Pass is highest")

HIGH slot has 3 PRIMARY answers (Brannigan/49 Hill, Glorieta Pass,
Continental Divide). Each has its own canonical elevation; matching
location selects the correct elevation reference.

Elevation unit:
  - Compare ft when extractor provides elevation_ft directly.
  - If only elevation_m given, convert (m × 3.28084) and use as ft.
  - If both given, prefer elevation_ft (avoids self-conversion compounding).
"""

from __future__ import annotations

import argparse
import json
import re
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
    SLOTS,
    VALUE_SCHEMA,
)
from pipeline.extraction_pipeline import run_pipeline  # noqa: E402

# ═══════════════════════════════════════════════════════════════════════════
# Ground truth — per-slot accepted locations + canonical elevation
# ═══════════════════════════════════════════════════════════════════════════
#
# Schema per slot:
#   primary:    list of (kw_list, elevation_ft, tol_ft) — any one is full credit
#   secondary:  list of (kw_list, elevation_ft, tol_ft) — partial credit (rank 2/1)
#   partial:    list of kw_list — known misconceptions; if claimed → 0
#
# Cross-verified 2026-05-02 against:
#   - Wikipedia (Glorieta Pass, Continental Divide NM, Hudspeth County, etc.)
#   - USGS NED 10m DEM via opentopodata.org
#   - open-elevation.com
#   - Wikipedia city/town infoboxes (cite GNIS)
# ═══════════════════════════════════════════════════════════════════════════

GT: dict[str, dict] = {
    "HIGH": {
        "primary": [
            (
                ["brannigan", "49 hill", "fortynine", "forty-nine", "forty nine"],
                7400,
                200,
            ),  # post-1937 main alignment high — sources cluster 7,300–7,500
            (
                ["glorieta", "santa fe loop", "格洛列塔", "格罗列塔"],
                7500,
                100,
            ),  # 1926-1937 original alignment — Wikipedia/GNIS = 7,500 ft
            (
                [
                    "continental divide",
                    "campbell pass",
                    "thoreau",
                    "大陆分水岭",
                    "坎贝尔山口",
                ],
                7263,
                75,
            ),  # 7,228-7,275 ft per multiple sources; commonly cited
        ],
        "secondary": [
            # Bellemont / Parks region without naming "Brannigan" or "49 Hill"
            (["bellemont", "parks az", "parks, az", "贝尔蒙特"], 7400, 250),
            # Mt Yampai / Yampai Divide (alternate names for Brannigan area)
            (["yampai"], 7400, 250),
            # "Arizona Divide" — used by GPT; same area as Brannigan/49 Hill
            (["arizona divide", "亚利桑那分水岭"], 7400, 250),
        ],
        "partial": [
            # Known misconceptions — answering these → 0
            ["sitgreaves", "oatman pass", "oatman summit", "西特格里夫斯"],  # ~3,586 ft
            ["flagstaff", "弗拉格斯塔夫"],  # ~6,910 ft, not the highest
        ],
    },
    "LOW": {
        "primary": [
            (
                [
                    "santa monica",
                    "pacific ocean",
                    "ocean ave",
                    "pier",
                    "终点",
                    "terminus",
                    "圣莫尼卡",
                    "太平洋",
                    "码头",
                    "西端",
                ],
                0,
                50,
            ),
        ],
        "secondary": [],
        "partial": [
            # Common wrong answers
            [
                "topock",
                "needles",
                "colorado river",
                "托波克",
                "尼德尔斯",
            ],  # ~456-499 ft, the W-of-Mississippi low but not overall
            [
                "chicago",
                "lake michigan",
                "grant park",
                "芝加哥",
                "密歇根湖",
            ],  # starting point ~595 ft
            ["mississippi", "密西西比"],
        ],
    },
    "BORDER_IL_MO": {
        "primary": [
            (
                [
                    "chain of rocks",
                    "mississippi",
                    "st louis",
                    "st. louis",
                    "圣路易斯",
                    "密西西比",
                    "岩链",
                    "锁链岩",
                ],
                420,
                75,  # river surface ~417-420; bridge deck ~437; allow either
            ),
        ],
        "secondary": [
            (["mckinley bridge", "eads bridge", "麦金利桥"], 420, 75),
        ],
        "partial": [],
    },
    "BORDER_MO_KS": {
        "primary": [
            (
                [
                    "joplin",
                    "galena",
                    "乔普林",
                    "贾普林",
                    "加利纳",
                    "加莱纳",
                ],
                950,
                125,  # WP infoboxes: Joplin 1004 / Galena 925; border ~940-977
            ),
        ],
        "secondary": [],
        "partial": [],
    },
    "BORDER_KS_OK": {
        "primary": [
            (
                [
                    "baxter springs",
                    "quapaw",
                    "巴克斯特斯普林斯",
                    "巴克斯特",
                    "夸保",
                    "夸帕",
                ],
                843,
                75,  # both town infoboxes: 843 ft
            ),
        ],
        "secondary": [],
        "partial": [],
    },
    "BORDER_OK_TX": {
        "primary": [
            (
                [
                    "texola",
                    "shamrock",
                    "特索拉",
                    "特克索拉",
                    "沙姆罗克",
                ],
                2140,
                100,  # NED border ~2130-2149; Texola WP 2152
            ),
        ],
        "secondary": [],
        "partial": [],
    },
    "BORDER_TX_NM": {
        "primary": [
            (
                [
                    "glenrio",
                    "格伦里奥",
                    "格兰里奥",
                ],
                3855,
                75,  # Glenrio WP infobox: 3,855 ft
            ),
        ],
        "secondary": [],
        "partial": [],
    },
    "BORDER_NM_AZ": {
        "primary": [
            (
                [
                    "lupton",
                    "卢普顿",
                    "勒普顿",
                    "鲁普顿",
                ],
                6188,
                75,  # Lupton WP infobox: 6,188 ft
            ),
        ],
        "secondary": [
            (["allentown az", "manuelito"], 6188, 150),
        ],
        "partial": [],
    },
    "BORDER_AZ_CA": {
        "primary": [
            (
                [
                    "topock",
                    "needles",
                    "colorado river",
                    "old trails bridge",
                    "托波克",
                    "尼德尔斯",
                    "科罗拉多河",
                ],
                495,
                100,  # Topock 515 / river ~456 / Needles 495 — wide tolerance
            ),
        ],
        "secondary": [],
        "partial": [],
    },
}


BENCHMARK = {
    "version": "1.1",
    "locked_at": "2026-05-04",
    "version_notes": "v1.1: added Chinese transliterations to primary kw lists (圣莫尼卡/卢普顿/特索拉/格伦里奥 etc.) so models writing Chinese place names hit the same kw match as English; added 'arizona divide' as secondary HIGH alias.",
    "gt": GT,
    "max_per_slot": 3,
    "n_slots": len(SLOTS),
    "max_total": len(SLOTS) * 3,
}


# ═══════════════════════════════════════════════════════════════════════════
# Matching helpers
# ═══════════════════════════════════════════════════════════════════════════


def _norm(s: str | None) -> str:
    if not s:
        return ""
    s = s.lower()
    s = re.sub(r"[^\w\s一-鿿]", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def _contains_any(hay: str, needles: list[str]) -> bool:
    return any(_norm(n) in hay for n in needles)


def _coerce_float(x) -> float | None:
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).strip().replace(",", "")
    m = re.search(r"-?\d+(?:\.\d+)?", s)
    return float(m.group()) if m else None


def _resolve_elevation_ft(canonical: dict) -> float | None:
    """Prefer elevation_ft; else convert elevation_m. Return None if neither."""
    ft = _coerce_float(canonical.get("elevation_ft"))
    if ft is not None:
        return ft
    m = _coerce_float(canonical.get("elevation_m"))
    if m is not None:
        return m * 3.28084
    return None


def _elev_error_ft(claimed_ft: float | None, gt_ft: float) -> float | None:
    if claimed_ft is None:
        return None
    return abs(claimed_ft - gt_ft)


# ═══════════════════════════════════════════════════════════════════════════
# Scoring
# ═══════════════════════════════════════════════════════════════════════════


def _score_slot(sid: str, canonical: dict | None) -> tuple[int, dict]:
    """Score one slot. Returns (score 0-3, detail_dict)."""
    gt = GT[sid]
    canonical = canonical if isinstance(canonical, dict) else {}
    location = _norm(canonical.get("location_name"))
    span = _norm(canonical.get("supporting_span", ""))
    haystack = location if location else span  # fallback to span if location empty
    claimed_ft = _resolve_elevation_ft(canonical)

    # 1. Not answered at all
    if not haystack and claimed_ft is None:
        return 0, {"rule": "not_answered"}

    # 2. PARTIAL list (known misconceptions) — only counts when location matches
    for kw_list in gt.get("partial", []):
        if location and _contains_any(haystack, kw_list):
            return 0, {
                "rule": "partial_misconception",
                "matched_partial_kw": kw_list,
                "claimed_location": canonical.get("location_name"),
                "claimed_ft": claimed_ft,
            }

    # 3. PRIMARY check
    best_primary: tuple[int, dict] | None = None
    for kw_list, gt_ft, tol_ft in gt["primary"]:
        if not _contains_any(haystack, kw_list):
            continue
        err = _elev_error_ft(claimed_ft, gt_ft)
        detail = {
            "rule": "primary_match",
            "matched_primary_kw": kw_list,
            "gt_ft": gt_ft,
            "tol_ft": tol_ft,
            "claimed_ft": claimed_ft,
            "elev_err_ft": err,
        }
        if err is None:
            score = 1  # location right, no elevation given
            detail["rule"] = "primary_no_elev"
        elif err <= tol_ft:
            score = 3
        elif err <= 2 * tol_ft:
            score = 2
        else:
            score = 1
        if best_primary is None or score > best_primary[0]:
            best_primary = (score, detail)
    if best_primary is not None:
        return best_primary

    # 4. SECONDARY check (no PRIMARY hit)
    best_secondary: tuple[int, dict] | None = None
    for kw_list, gt_ft, tol_ft in gt.get("secondary", []):
        if not _contains_any(haystack, kw_list):
            continue
        err = _elev_error_ft(claimed_ft, gt_ft)
        detail = {
            "rule": "secondary_match",
            "matched_secondary_kw": kw_list,
            "gt_ft": gt_ft,
            "tol_ft": tol_ft,
            "claimed_ft": claimed_ft,
            "elev_err_ft": err,
        }
        if err is None:
            score = 1
            detail["rule"] = "secondary_no_elev"
        elif err <= tol_ft:
            score = 2
        elif err <= 2 * tol_ft:
            score = 1
        else:
            score = 1
            detail["rule"] = "secondary_big_error"
        if best_secondary is None or score > best_secondary[0]:
            best_secondary = (score, detail)
    if best_secondary is not None:
        return best_secondary

    # 5. No location match — check elevation magnitude only against any primary GT
    if claimed_ft is not None:
        for _kw_list, gt_ft, tol_ft in gt["primary"]:
            err = abs(claimed_ft - gt_ft)
            if err <= tol_ft:
                return 1, {
                    "rule": "elev_only_match",
                    "gt_ft": gt_ft,
                    "tol_ft": tol_ft,
                    "claimed_ft": claimed_ft,
                    "elev_err_ft": err,
                    "claimed_location": canonical.get("location_name"),
                }

    # 6. Wrong location + wrong elevation
    return 0, {
        "rule": "wrong",
        "claimed_location": canonical.get("location_name"),
        "claimed_ft": claimed_ft,
    }


@dataclass
class ScoreResult:
    model: str
    slot_scores: dict[str, int] = field(default_factory=dict)
    slot_detail: dict[str, dict] = field(default_factory=dict)
    total: int = 0
    max_total: int = 0


def _score_model(model_name: str, payload: dict) -> ScoreResult:
    entities = payload.get("entities", [])
    by_id = {e["id"]: e for e in entities}
    slot_scores: dict[str, int] = {}
    slot_detail: dict[str, dict] = {}
    for sid, _label, _desc in SLOTS:
        ent = by_id.get(sid, {})
        canonical = (ent.get("canonical") or {}).get("value")
        # also pull supporting_span up into canonical for fallback matching
        if isinstance(canonical, dict):
            span = (ent.get("canonical") or {}).get("span")
            canonical = {**canonical, "supporting_span": span} if span else canonical
        s, d = _score_slot(sid, canonical)
        slot_scores[sid] = s
        d["claimed"] = canonical
        slot_detail[sid] = d
    return ScoreResult(
        model=model_name,
        slot_scores=slot_scores,
        slot_detail=slot_detail,
        total=sum(slot_scores.values()),
        max_total=BENCHMARK["max_total"],
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
                "slot_scores": r.slot_scores,
                "slot_detail": r.slot_detail,
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
    lines.append("# Query 40 Ranking Report — Route 66 Elevations")
    lines.append("")
    lines.append(
        f"> Benchmark version: {BENCHMARK['version']} "
        f"(locked {BENCHMARK['locked_at']})  "
        f"Max: {BENCHMARK['max_total']} = {len(SLOTS)} slots × 3 pts"
    )
    lines.append(
        "> Scoring per slot: 3=location+elev tight; 2=location+elev loose; "
        "1=partial (location-only or elev-only); 0=missing/wrong/known-misconception"
    )
    lines.append("")
    header_cells = ["Rank", "Model", "Total"] + [sid for sid, _, _ in SLOTS]
    lines.append("| " + " | ".join(header_cells) + " |")
    lines.append("|---:|---|---:|" + "---:|" * len(SLOTS))
    ranked = sorted(results, key=lambda r: (-r.total, r.model))
    for i, r in enumerate(ranked, 1):
        cells = [
            str(i),
            r.model,
            f"{r.total}/{r.max_total}",
        ] + [str(r.slot_scores[sid]) for sid, _, _ in SLOTS]
        lines.append("| " + " | ".join(cells) + " |")

    # Slot-by-slot legend
    lines.append("")
    lines.append("### Slot legend")
    lines.append("")
    lines.append("| Slot | Description | Canonical |")
    lines.append("|---|---|---|")
    for sid, label, _desc in SLOTS:
        gt_primary = GT[sid]["primary"]
        canonical_str = " / ".join(
            f"{kw[0]} ≈ {ft} ft (±{tol})" for kw, ft, tol in gt_primary
        )
        lines.append(f"| {sid} | {label} | {canonical_str} |")

    (output_dir / "ranking_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


# ═══════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Query 40 auto-scorer (Route 66 elevations)"
    )
    ap.add_argument("--models", nargs="+", required=True, help="name=path list")
    ap.add_argument("--output-dir", default=str(THIS / "auto_scores"))
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

    results: list[ScoreResult] = []
    for model_name, payload in all_results.items():
        r = _score_model(model_name, payload)
        _write_model_score(out_dir, r)
        results.append(r)

    _write_ranking_report(out_dir, results)
    print("\n" + "─" * 60)
    print("Query 40 scoring done.")
    for r in sorted(results, key=lambda x: (-x.total, x.model)):
        print(f"  {r.model:35s} {r.total}/{r.max_total}")


if __name__ == "__main__":
    main()
