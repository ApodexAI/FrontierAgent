"""
Query 37 — Amazon rainforest 2015-2024 net deforestation
+ top-3 most-deforested regions auto-scorer.

Scoring (per model, max 7):
  A_score (0/1) — loss_km2 落入 GT 任一接受区间：
                   - Brazilian Legal Amazon: 70,000-100,000 km^2
                   - Pan-Amazon biome:       130,000-230,000 km^2
                  中间间隙 (100k-130k) 不接受。
  B_score (0..3) — 模型 top-3 hotspots 中，命中 D1-D4 的国家正确数。
                   每个 DIM 仅累计一次（dedup）。
  C_score (0..3) — 模型 top-3 hotspots 中，坐标落入 D1-D4 任一锚点 ±4° 范围内的数。
                   每个 DIM 仅累计一次（dedup）。
  total = A + B + C (0..7)

Pipeline:
  Stage 1 抽取 — 标准 run_pipeline，提取一个 composite entity (loss + hotspots)。
  Stage 2 对齐 — 仅对 top3_hotspots 调 align_claims（A_loss 是数值，无 DIM 可对齐）。
  Stage 3a Null 导出 — 自动产出 null_review.json（保留契约）。
  Stage 3b Stage C — 跳过（按用户要求；GT 已锁定，不需 web 验证 baseline_add）。
  Stage 4 打分 — 自定义 score_amazon()：A_score + B_score + C_score。
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
# Ground Truth - Loss 数值（双口径接受）
# ═══════════════════════════════════════════════════════════════════════════

GT_LOSS = {
    "ranges_accepted": [
        {
            "label": "Brazilian Legal Amazon (PRODES)",
            "low_km2": 70_000,
            "high_km2": 100_000,
            "center_km2": 83_087,
            "scope_tag": "brazilian_legal_amazon",
        },
        {
            "label": "Pan-Amazon biome (9 countries; MapBiomas+RAISG+MAAP)",
            "low_km2": 130_000,
            "high_km2": 230_000,
            "center_km2": 180_000,
            "scope_tag": "pan_amazon",
        },
    ],
    "gap_zone_km2": (100_000, 130_000),  # 中间间隙：判 0
}


# ═══════════════════════════════════════════════════════════════════════════
# Ground Truth — Top-3 Hotspots (DIMS = 4 个 ✅ 候选区域，模型命中其中 3 个即满分)
# ═══════════════════════════════════════════════════════════════════════════

# (id, region_label, country, country_aliases, anchors_list[(lat,lon)], tol_deg, kw)
DIMS = [
    (
        "D1",
        "Brazil — Arc of Deforestation (Pará / Mato Grosso / Rondônia / Amacro)",
        "Brazil",
        ["brazil", "brasil", "巴西", "巴西联邦", "巴西法定亚马逊"],
        [(-9.0, -55.0), (-6.6, -52.0), (-12.0, -56.0), (-7.0, -60.0)],
        4.0,
        [
            "arc of deforestation",
            "deforestation arc",
            "毁林弧",
            "砍伐弧",
            "毁林前线",
            "pará",
            "para state",
            "帕拉",
            "mato grosso",
            "马托格罗索",
            "rondônia",
            "rondonia",
            "朗多尼亚",
            "br-163",
            "br-364",
            "br-319",
            "xingu",
            "辛古",
            "são félix",
            "sao felix",
            "apuí",
            "apui",
            "amacro",
            "amazonas state",
            "南亚马逊",
            "巴西亚马逊",
        ],
    ),
    (
        "D2",
        "Bolivia — Santa Cruz / Chiquitania",
        "Bolivia",
        ["bolivia", "玻利维亚", "玻利维亚多民族国"],
        [(-16.0, -62.0), (-15.5, -60.5)],
        4.0,
        [
            "santa cruz",
            "chiquitania",
            "chiquitanía",
            "beni",
            "bolivia",
            "玻利维亚",
            "圣克鲁斯",
            "奇基塔尼亚",
            "贝尼",
        ],
    ),
    (
        "D3",
        "Colombia — Northwest Amazon (Caquetá / Guaviare / Meta)",
        "Colombia",
        ["colombia", "哥伦比亚"],
        [(2.0, -73.0), (1.5, -73.5), (3.0, -74.0)],
        4.0,
        [
            "caquetá",
            "caqueta",
            "guaviare",
            "瓜维亚雷",
            "meta",
            "san josé del guaviare",
            "san jose del guaviare",
            "tinigua",
            "macarena",
            "chiribiquete",
            "colombia",
            "哥伦比亚",
            "卡克塔",
            "卡克塔省",
            "梅塔",
            "西北亚马逊",
        ],
    ),
    (
        "D4",
        "Peru — Madre de Dios (illegal gold-mining frontier)",
        "Peru",
        ["peru", "perú", "秘鲁"],
        [(-12.6, -69.2)],
        4.0,
        [
            "madre de dios",
            "puerto maldonado",
            "la pampa",
            "guacamayo",
            "peru",
            "perú",
            "秘鲁",
            "马德雷德迪奥斯",
            "马德雷-德迪奥斯",
            "马德雷·德·迪奥斯",
            "南秘鲁",
        ],
    ),
]

DIM_MAP = {
    d[0]: {
        "id": d[0],
        "label": d[1],
        "country": d[2],
        "country_aliases": d[3],
        "anchors": d[4],
        "tol_deg": d[5],
        "kw": d[6],
    }
    for d in DIMS
}

MAX_A = 1
MAX_B = 3
MAX_C = 3
MAX_SCORE = MAX_A + MAX_B + MAX_C  # 7


# ═══════════════════════════════════════════════════════════════════════════
# DIMS → alignment baseline spec
# ═══════════════════════════════════════════════════════════════════════════


def _derive_match_fields(dim: dict) -> dict:
    """Encode geographic anchor box into match_fields so cross_check can flag
    coord conflicts. The aligner LLM still does the semantic match via kw +
    region_name; match_fields adds programmatic signal."""
    return {
        "country": dim["country"],
        "country_aliases": dim["country_aliases"],
        # bounding box derived from anchor ± tol (used only as soft signal)
        "lat_min": min(a[0] for a in dim["anchors"]) - dim["tol_deg"],
        "lat_max": max(a[0] for a in dim["anchors"]) + dim["tol_deg"],
        "lon_min": min(a[1] for a in dim["anchors"]) - dim["tol_deg"],
        "lon_max": max(a[1] for a in dim["anchors"]) + dim["tol_deg"],
    }


def build_baselines() -> list[dict]:
    out = []
    for d in DIMS:
        dim = DIM_MAP[d[0]]
        out.append(
            {
                "id": dim["id"],
                "description": (
                    f"{dim['label']} — country: {dim['country']}; "
                    f"anchors(lat,lon): {dim['anchors']}; tol=±{dim['tol_deg']}°"
                ),
                "match_fields": _derive_match_fields(dim),
                "kw": dim["kw"],
                "judgment": "✅",
                "score": 1.0,
            }
        )
    return out


# ═══════════════════════════════════════════════════════════════════════════
# Load raw claims for alignment — flatten value.top3_hotspots per model
# ═══════════════════════════════════════════════════════════════════════════


def load_raw_claims(extraction_dir: Path) -> dict[str, list[dict]]:
    """Read each model's extraction.json and pull out value.top3_hotspots as
    the list of claims to be aligned. value.loss_km2 / scope are read in
    score_amazon() directly, not aligned."""
    out: dict[str, list[dict]] = {}
    for d in Path(extraction_dir).iterdir():
        if not d.is_dir():
            continue
        ext_file = d / "extraction.json"
        if not ext_file.exists():
            continue
        payload = json.loads(ext_file.read_text(encoding="utf-8"))
        entities = payload.get("entities", [])
        hotspots: list[dict] = []
        if entities:
            cval = (entities[0].get("canonical") or {}).get("value") or {}
            if isinstance(cval, dict):
                ts = cval.get("top3_hotspots") or []
                if isinstance(ts, list):
                    hotspots = [h for h in ts if isinstance(h, dict)]
        out[d.name] = hotspots
    return out


def load_loss_block(extraction_dir: Path) -> dict[str, dict]:
    """Read each model's value.loss_km2 / scope / cover figures."""
    out: dict[str, dict] = {}
    for d in Path(extraction_dir).iterdir():
        if not d.is_dir():
            continue
        ext_file = d / "extraction.json"
        if not ext_file.exists():
            continue
        payload = json.loads(ext_file.read_text(encoding="utf-8"))
        entities = payload.get("entities", [])
        block: dict = {}
        if entities:
            cval = (entities[0].get("canonical") or {}).get("value") or {}
            if isinstance(cval, dict):
                block = {
                    "loss_km2": cval.get("loss_km2"),
                    "scope": cval.get("scope"),
                    "year_2015_cover_km2": cval.get("year_2015_cover_km2"),
                    "year_2024_cover_km2": cval.get("year_2024_cover_km2"),
                }
        out[d.name] = block
    return out


def _load_alignment(out_dir: Path) -> dict[str, list[dict]]:
    aligned = {}
    for d in Path(out_dir).iterdir():
        if not d.is_dir():
            continue
        af = d / "alignment.json"
        if not af.exists():
            continue
        aligned[d.name] = json.loads(af.read_text(encoding="utf-8")).get(
            "aligned_claims", []
        )
    return aligned


# ═══════════════════════════════════════════════════════════════════════════
# Country normalization
# ═══════════════════════════════════════════════════════════════════════════


def _norm_country(raw: str | None) -> str:
    if not raw:
        return ""
    s = str(raw).lower().strip()
    s = _re.sub(r"[^\w\s一-鿿]", " ", s)
    s = _re.sub(r"\s+", " ", s).strip()
    return s


def _country_matches(claim_country: str | None, dim: dict) -> bool:
    n = _norm_country(claim_country)
    if not n:
        return False
    for alias in dim["country_aliases"]:
        if _norm_country(alias) in n or n in _norm_country(alias):
            return True
    return False


# ═══════════════════════════════════════════════════════════════════════════
# Coordinate matching: claim (lat,lon) within ±tol of any DIM anchor
# ═══════════════════════════════════════════════════════════════════════════


def _coord_matches(
    claim_lat: float | int | str | None,
    claim_lon: float | int | str | None,
    dim: dict,
) -> bool:
    try:
        lat = float(claim_lat) if claim_lat is not None else None
        lon = float(claim_lon) if claim_lon is not None else None
    except (TypeError, ValueError):
        return False
    if lat is None or lon is None:
        return False
    tol = dim["tol_deg"]
    for a_lat, a_lon in dim["anchors"]:
        if abs(lat - a_lat) <= tol and abs(lon - a_lon) <= tol:
            return True
    return False


# ═══════════════════════════════════════════════════════════════════════════
# A-score: numeric loss within either accepted range
# ═══════════════════════════════════════════════════════════════════════════


def _coerce_loss_km2(v) -> float | None:
    """Accept int/float/str (with units / commas / '万' / '千' / 'km²')."""
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip().replace(",", "").replace(" ", "")
    s = s.lower().replace("km²", "").replace("km2", "").replace("km", "")
    cn_mult = 1.0
    if "万" in s:
        s = s.replace("万", "")
        cn_mult = 10_000.0
    elif "千" in s:
        s = s.replace("千", "")
        cn_mult = 1_000.0
    m = _re.search(r"-?\d+(?:\.\d+)?", s)
    if not m:
        return None
    try:
        return float(m.group()) * cn_mult
    except ValueError:
        return None


def score_a(loss_block: dict) -> tuple[int, dict]:
    """A_score: 1 if loss_km2 in either accepted range, else 0."""
    raw_loss = loss_block.get("loss_km2")
    loss_km2 = _coerce_loss_km2(raw_loss)
    detail = {
        "raw_loss": raw_loss,
        "parsed_loss_km2": loss_km2,
        "scope": loss_block.get("scope"),
    }
    if loss_km2 is None:
        return 0, {**detail, "rule": "no_loss_value"}
    for r in GT_LOSS["ranges_accepted"]:
        if r["low_km2"] <= loss_km2 <= r["high_km2"]:
            return 1, {
                **detail,
                "rule": "in_range",
                "matched_caliber": r["label"],
                "matched_scope_tag": r["scope_tag"],
            }
    gap_lo, gap_hi = GT_LOSS["gap_zone_km2"]
    if gap_lo < loss_km2 < gap_hi:
        return 0, {**detail, "rule": "in_gap_zone"}
    return 0, {**detail, "rule": "out_of_range"}


# ═══════════════════════════════════════════════════════════════════════════
# B/C score from aligned hotspot claims
# ═══════════════════════════════════════════════════════════════════════════


def score_bc(aligned_hotspots: list[dict]) -> tuple[int, int, list[dict]]:
    """B = country match count (per DIM dedup), C = coord match count (per DIM dedup).

    For each of model's top-3 (sorted by rank, capped at 3), check the aligned
    canonical_id and verify (a) claim.country matches DIM.country, (b)
    claim.(lat,lon) within ±tol of any DIM anchor.
    """

    # sort by rank if available, else preserve original order; cap at 3
    def _rank_key(c):
        raw = c.get("raw", {}) or {}
        r = raw.get("rank")
        try:
            return int(r)
        except (TypeError, ValueError):
            return 999

    sorted_claims = sorted(aligned_hotspots, key=_rank_key)[:3]

    seen_b: set[str] = set()
    seen_c: set[str] = set()
    b_count = 0
    c_count = 0
    detail: list[dict] = []
    for c in sorted_claims:
        raw = c.get("raw", {}) or {}
        cid = c.get("canonical_id")
        conf = c.get("alignment_confidence", "medium")
        country_claim = raw.get("country")
        lat_claim = raw.get("lat")
        lon_claim = raw.get("lon")

        country_ok = False
        coord_ok = False
        if cid in DIM_MAP:
            dim = DIM_MAP[cid]
            if cid not in seen_b and _country_matches(country_claim, dim):
                seen_b.add(cid)
                country_ok = True
                b_count += 1
            if cid not in seen_c and _coord_matches(lat_claim, lon_claim, dim):
                seen_c.add(cid)
                coord_ok = True
                c_count += 1
        detail.append(
            {
                "rank": raw.get("rank"),
                "region_name": raw.get("region_name"),
                "country_claimed": country_claim,
                "lat_claimed": lat_claim,
                "lon_claimed": lon_claim,
                "canonical_id": cid,
                "alignment_confidence": conf,
                "country_match": country_ok,
                "coord_match": coord_ok,
            }
        )
    return b_count, c_count, detail


# ═══════════════════════════════════════════════════════════════════════════
# Per-model scoring
# ═══════════════════════════════════════════════════════════════════════════


def score_model(
    model_name: str, loss_block: dict, aligned_hotspots: list[dict]
) -> dict:
    a_score, a_detail = score_a(loss_block)
    b_score, c_score, hot_detail = score_bc(aligned_hotspots)
    total = a_score + b_score + c_score
    return {
        "model": model_name,
        "total_score": total,
        "max_score": MAX_SCORE,
        "A_score": a_score,
        "B_score": b_score,
        "C_score": c_score,
        "A_detail": a_detail,
        "hotspots_detail": hot_detail,
        "loss_block": loss_block,
        "n_hotspots_listed": len(aligned_hotspots),
    }


# ═══════════════════════════════════════════════════════════════════════════
# Outputs
# ═══════════════════════════════════════════════════════════════════════════


def build_scores_json(all_scores: dict[str, dict]) -> dict:
    ranked = sorted(
        all_scores.values(),
        key=lambda s: (
            -s["total_score"],
            -s["A_score"],
            -s["B_score"],
            -s["C_score"],
            s["model"],
        ),
    )
    return {
        "query_id": QUERY_ID,
        "max_score": MAX_SCORE,
        "scoring": {
            "A_max": MAX_A,
            "B_max": MAX_B,
            "C_max": MAX_C,
            "A_rule": (
                "loss_km2 ∈ [70k,100k] (Brazilian Legal) or [130k,230k] (Pan-Amazon) → 1; "
                "gap zone (100k,130k) → 0; out of range → 0"
            ),
            "B_rule": "model top-3 hotspots: count claims whose country matches DIM.country (dedup per DIM, max 3)",
            "C_rule": "model top-3 hotspots: count claims whose (lat,lon) within ±4° of any DIM anchor (dedup per DIM, max 3)",
        },
        "GT_LOSS": GT_LOSS,
        "DIMS_summary": [
            {
                "id": d[0],
                "label": d[1],
                "country": d[2],
                "anchors": d[4],
                "tol_deg": d[5],
            }
            for d in DIMS
        ],
        "results": all_scores,
        "ranking": [
            {
                "rank": i + 1,
                "model": s["model"],
                "total": s["total_score"],
                "A": s["A_score"],
                "B": s["B_score"],
                "C": s["C_score"],
            }
            for i, s in enumerate(ranked)
        ],
    }


def build_ranking_md(all_scores: dict[str, dict]) -> str:
    ranked = sorted(
        all_scores.values(),
        key=lambda s: (
            -s["total_score"],
            -s["A_score"],
            -s["B_score"],
            -s["C_score"],
            s["model"],
        ),
    )
    lines: list[str] = [
        f"# Query {QUERY_ID} (dir: query_37) Ranking Report",
        "",
        f"> 题目：{QUERY_TEXT}",
        "",
        f"> 满分 {MAX_SCORE} = A({MAX_A}) + B({MAX_B}) + C({MAX_C})",
        "",
        "**A** = 数值落入接受区间（巴西法定 70k-100k 或 泛亚马逊 130k-230k）",
        "**B** = top-3 hotspots 中国家匹配数（DIM dedup，max 3）",
        "**C** = top-3 hotspots 中坐标匹配数（±4° 锚点容差，DIM dedup，max 3）",
        "",
        "| Rank | Model | Total | A | B | C | loss_km² | scope |",
        "|---:|---|---:|---:|---:|---:|---:|---|",
    ]
    for i, s in enumerate(ranked, 1):
        loss_v = s["A_detail"].get("parsed_loss_km2")
        loss_s = "—" if loss_v is None else f"{int(loss_v):,}"
        scope_s = s["A_detail"].get("scope") or "—"
        lines.append(
            f"| {i} | {s['model']} | {s['total_score']}/{MAX_SCORE} | "
            f"{s['A_score']} | {s['B_score']} | {s['C_score']} | "
            f"{loss_s} | {scope_s} |"
        )
    lines.extend(["", "## Per-model hotspot detail", ""])
    for s in ranked:
        lines.append(f"### {s['model']} (total {s['total_score']}/{MAX_SCORE})")
        lines.append("")
        lines.append(
            "| rank | region_name | country | lat | lon | aligned_DIM | country_match | coord_match |"
        )
        lines.append("|---:|---|---|---:|---:|---|:---:|:---:|")
        for h in s["hotspots_detail"]:
            lines.append(
                f"| {h.get('rank') or '—'} | {h.get('region_name') or '—'} | "
                f"{h.get('country_claimed') or '—'} | "
                f"{h.get('lat_claimed') if h.get('lat_claimed') is not None else '—'} | "
                f"{h.get('lon_claimed') if h.get('lon_claimed') is not None else '—'} | "
                f"{h.get('canonical_id') or '—'} | "
                f"{'✓' if h.get('country_match') else '✗'} | "
                f"{'✓' if h.get('coord_match') else '✗'} |"
            )
        lines.append("")
    return "\n".join(lines) + "\n"


# ═══════════════════════════════════════════════════════════════════════════
# main()
# ═══════════════════════════════════════════════════════════════════════════


def main() -> None:
    ap = argparse.ArgumentParser(
        description=f"Query {QUERY_ID} (dir query_37) auto-scorer"
    )
    ap.add_argument("--models", nargs="+", required=True, help="name=path list")
    ap.add_argument("--output-dir", default=str(THIS / "auto_scores"))
    ap.add_argument("--skip-extract", action="store_true")
    ap.add_argument("--skip-align", action="store_true")
    ap.add_argument("--primary-model", default=None)
    ap.add_argument("--secondary-model", default=None)
    ap.add_argument("--parallel-models", nargs="+", default=None)
    ap.add_argument("--analyzer-model", default=None)
    ap.add_argument("--aligner-models", nargs="+", default=None)
    ap.add_argument("--judge-model", default=None)
    ap.add_argument("--concurrency", type=int, default=None)
    args = ap.parse_args()

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

    # Stage 2: alignment (hotspots only)
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
        raw_hotspots = load_raw_claims(out_dir)
        aligned = align_claims(
            client,
            claims_by_model=raw_hotspots,
            baselines=baselines,
            query_text=QUERY_TEXT,
            output_dir=out_dir,
            overrides=overrides,
        )

    # Stage 3a: export null hotspots for review (Stage C 跳过，仅保留契约)
    null_items = export_null_claims_for_review(
        aligned,
        out_dir / "null_review.json",
        query_id=QUERY_ID,
        query_text=QUERY_TEXT,
        models_input=args.models,
    )
    print(
        f"\n[*] Exported {len(null_items)} null hotspots → null_review.json (Stage C 跳过)"
    )

    # Stage 3b: skip (no Stage C web verification by user request)
    resolutions_path = out_dir / "null_resolutions.json"
    if resolutions_path.exists():
        print("[*] Found null_resolutions.json (manual override), applying …")
        new_baseline_entries = apply_null_resolutions(
            aligned, resolutions_path, dims_ref=DIMS
        )
        if new_baseline_entries:
            for e in new_baseline_entries:
                # add minimal entry; not used heavily since we score by alignment alone
                DIM_MAP[e["id"]] = {
                    "id": e["id"],
                    "label": e.get("description", e["id"]),
                    "country": e.get("country", ""),
                    "country_aliases": [],
                    "anchors": [],
                    "tol_deg": 4.0,
                    "kw": e.get("kw", []),
                }
            persist_new_baseline_entries(new_baseline_entries, __file__)

    # Stage 4: scoring
    loss_blocks = load_loss_block(out_dir)
    all_scores: dict[str, dict] = {}
    for model_name in sorted(set(loss_blocks.keys()) | set(aligned.keys())):
        all_scores[model_name] = score_model(
            model_name,
            loss_blocks.get(model_name, {}),
            aligned.get(model_name, []),
        )

    (out_dir / "scores.json").write_text(
        json.dumps(build_scores_json(all_scores), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (out_dir / "ranking_report.md").write_text(
        build_ranking_md(all_scores), encoding="utf-8"
    )

    print("\n" + "─" * 62)
    print(f"Query {QUERY_ID} (dir query_37) scoring done.")
    ranked = sorted(
        all_scores.values(),
        key=lambda s: (
            -s["total_score"],
            -s["A_score"],
            -s["B_score"],
            -s["C_score"],
            s["model"],
        ),
    )
    for i, s in enumerate(ranked, 1):
        print(
            f"  {i:2d}. {s['model']:30s} "
            f"{s['total_score']}/{MAX_SCORE}  "
            f"A={s['A_score']} B={s['B_score']} C={s['C_score']}"
        )


if __name__ == "__main__":
    main()
