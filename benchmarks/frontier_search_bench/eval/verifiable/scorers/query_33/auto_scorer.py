"""
Query 33 — Denver, CO 沿正南至南极点路径上 US Counties (按北→南顺序) auto-scorer。

GT (17 个 US Counties)：
  CO (9): Denver → Arapahoe → Douglas → El Paso → Teller → Fremont → Pueblo
          → Huerfano → Las Animas
  NM (7): Colfax → Mora → San Miguel → Guadalupe → Lincoln → Chaves → Otero
  TX (1): Hudspeth

GT 来源：FCC Census Block API + OSM Nominatim 双源验证（沿 -104.99°W 经线
21 个采样点，纬度 38.85°N → 31.45°N，2026-05-06 完成）。

打分规则（2026-05-06 拍板）：
  ✅ name 在 baseline + state 对 + 顺序对           → +1.0
  🅢 name 在 baseline + state 对 + 顺序错位          → +0.5
  ⚠️ name 在 baseline + state 未指定（unspecified）  → +0.5
  ❌ name 在 baseline 但 state 错（同名县混淆）     → 0.0
  ❌ name 不在 baseline（幻觉）                     → -1.0
  ❌ baseline 中某县未被模型提到（漏掉）            → -1.0
  alignment_confidence == "needs_review"          → 0.0（特殊路径，不计分）
  MAX_SCORE = 17 × 1.0 = 17.0；total 可负，下限 -17 起更负。

"顺序对" 定义：模型在 raw 列表里给出的"已命中且 state 正确"的所有县，
其 raw 顺序与 baseline 期望顺序（CO 1-9 → NM 10-16 → TX 17）一致。
更具体地，对于每个命中县 c：
  - 设 R(c) = 该县在模型 raw 列表里的位次（1-based，仅在已命中且 state 正
    确的县中重新编号）
  - 设 E(c) = 该县在 baseline 中的期望位次（1-17）
  - 若所有命中县按 R 排序 == 按 E 排序（即 R 与 E 单调一致），则**全部**
    命中县的"顺序对"成立（每个 +1.0）。
  - 否则只有那些"局部位置一致"（即 R-rank == E-rank）的县计 +1.0，其余
    +0.5。

Pipeline:
  1. Extraction (Stage 1) — v2 三模型 + analyzer，输出 extraction.json
  2. Alignment (Stage 2) — 多模型对齐到 17 个 county canonical_id 之一 / null
  3. Null verification (Stage 3) — 对齐 null 的 claim 走 Stage C web 验证
  4. Scoring (Stage 4) — 按上规则打分，输出 scores.json + ranking_report.md
"""

from __future__ import annotations

import argparse
import json
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
# Benchmark — 17 counties 沿 -104.99°W 经线从北到南
#   每条记录：(canonical_id, description, judgment, score, kw_aliases)
#   description 格式："<Name> / <State> / <说明>"
#     说明里包含 expected_position（北→南排序）和州代码
#   id 命名约定 "<STATE>_<NN>"，NN 从 01 起按 baseline 顺序
# ═══════════════════════════════════════════════════════════════════════════

SNAPSHOT_DATE = "2026-05-06"
SCORING_MODE = "ordered_sequence_strict_v1"
GT_SOURCE = (
    "FCC Census Block API (geo.fcc.gov/api/census/area) + "
    "OSM Nominatim reverse geocoding；沿 -104.99°W 经线 21 个采样点，"
    "2026-05-06 双源交叉验证"
)

DIMS = [
    # ─── Colorado (9 counties，期望位次 1-9) ───
    (
        "CO_01",
        "Denver County / Colorado / 起点 (City and County of Denver)，"
        "市中心约 39.7392°N, -104.99°W [expected_position=1]",
        "✅",
        1.0,
        ["denver", "denver county", "丹佛", "丹佛县", "city and county of denver"],
    ),
    (
        "CO_02",
        "Arapahoe County / Colorado / Denver 正南紧邻县 [expected_position=2]",
        "✅",
        1.0,
        ["arapahoe", "arapahoe county", "阿拉帕霍县", "阿拉帕霍"],
    ),
    (
        "CO_03",
        "Douglas County / Colorado / Castle Rock 县治 [expected_position=3]",
        "✅",
        1.0,
        ["douglas", "douglas county", "道格拉斯县", "道格拉斯"],
    ),
    (
        "CO_04",
        "El Paso County / Colorado / Colorado Springs 县治；"
        "**注意与 El Paso County, TX 区别** [expected_position=4]",
        "✅",
        1.0,
        ["el paso county, colorado", "el paso county, co", "埃尔帕索县（科罗拉多）"],
    ),
    (
        "CO_05",
        "Teller County / Colorado / Cripple Creek 县治，派克峰西麓 "
        "[expected_position=5]；经线 ~38.78°N → 38.68°N 段在 Teller 内",
        "✅",
        1.0,
        ["teller", "teller county", "特勒县", "cripple creek"],
    ),
    (
        "CO_06",
        "Fremont County / Colorado / Cañon City 县治 [expected_position=6]；"
        "经线 ~38.65°N → 38.27°N 段在 Fremont 内",
        "✅",
        1.0,
        [
            "fremont",
            "fremont county",
            "弗里蒙特县",
            "弗里蒙特",
            "cañon city",
            "canon city",
        ],
    ),
    (
        "CO_07",
        "Pueblo County / Colorado [expected_position=7]",
        "✅",
        1.0,
        ["pueblo", "pueblo county", "普韦布洛县", "普韦布洛"],
    ),
    (
        "CO_08",
        "Huerfano County / Colorado [expected_position=8]",
        "✅",
        1.0,
        ["huerfano", "huerfano county", "韦尔法诺县", "韦尔法诺", "韦尔法洛"],
    ),
    (
        "CO_09",
        "Las Animas County / Colorado / CO 最南端，与 NM 边界相接 "
        "[expected_position=9]",
        "✅",
        1.0,
        ["las animas", "las animas county", "拉斯阿尼马斯县", "拉斯阿尼马斯"],
    ),
    # ─── New Mexico (7 counties，期望位次 10-16) ───
    (
        "NM_01",
        "Colfax County / New Mexico / NM 北部第一个县 [expected_position=10]",
        "✅",
        1.0,
        ["colfax", "colfax county", "科尔法克斯县", "科尔法克斯"],
    ),
    (
        "NM_02",
        "Mora County / New Mexico [expected_position=11]",
        "✅",
        1.0,
        ["mora", "mora county", "莫拉县", "莫拉"],
    ),
    (
        "NM_03",
        "San Miguel County / New Mexico / "
        "**注意与 San Miguel County, CA 区别** [expected_position=12]",
        "✅",
        1.0,
        [
            "san miguel county, new mexico",
            "san miguel county, nm",
            "圣米格尔县（新墨西哥）",
        ],
    ),
    (
        "NM_04",
        "Guadalupe County / New Mexico / "
        "**注意与 Guadalupe County, TX 区别** [expected_position=13]",
        "✅",
        1.0,
        [
            "guadalupe county, new mexico",
            "guadalupe county, nm",
            "瓜达卢佩县（新墨西哥）",
        ],
    ),
    (
        "NM_05",
        "Lincoln County / New Mexico / "
        "**注意'林肯县'多州同名（NM/NV/MT/...），此处特指 NM** "
        "[expected_position=14]",
        "✅",
        1.0,
        ["lincoln county, new mexico", "lincoln county, nm", "林肯县（新墨西哥）"],
    ),
    (
        "NM_06",
        "Chaves County / New Mexico / Roswell 县治 [expected_position=15]",
        "✅",
        1.0,
        ["chaves", "chaves county", "查韦斯县", "查韦斯", "roswell"],
    ),
    (
        "NM_07",
        "Otero County / New Mexico / "
        "**注意与 Otero County, CO 区别** [expected_position=16]",
        "✅",
        1.0,
        ["otero county, new mexico", "otero county, nm", "奥特罗县（新墨西哥）"],
    ),
    # ─── Texas (1 county，期望位次 17) ───
    (
        "TX_01",
        "Hudspeth County / Texas / 最后一个 US 县；~31.4°N 跨 Rio Grande "
        "进墨西哥 Chihuahua 州。"
        "**注意：Culberson 在 Hudspeth 东边，El Paso (TX) 在西边，都不在线上** "
        "[expected_position=17]",
        "✅",
        1.0,
        ["hudspeth", "hudspeth county", "赫德斯佩斯县"],
    ),
]

DIM_MAP = {
    d[0]: {"id": d[0], "name": d[1], "judgment": d[2], "score": d[3], "kw": d[4]}
    for d in DIMS
}
EXPECTED_POSITION = {d[0]: i + 1 for i, d in enumerate(DIMS)}
EXPECTED_STATE = {d[0]: d[0].split("_")[0] for d in DIMS}  # "CO_01" → "CO"
MAX_SCORE = sum(d[3] for d in DIMS if d[3] > 0)  # 17.0


# ═══════════════════════════════════════════════════════════════════════════
# DIMS → alignment baseline spec
# ═══════════════════════════════════════════════════════════════════════════


def _derive_match_fields(d_id: str, description: str) -> dict:
    """Parse name + state out of the canonical-id and description.
    Format expected: '<Name> / <State> / <details>'."""
    fields: dict = {}
    parts = description.split(" / ")
    if len(parts) >= 1:
        fields["name"] = parts[0].strip()
    if len(parts) >= 2:
        fields["state_full"] = parts[1].strip()  # e.g. "Colorado"
    fields["state"] = EXPECTED_STATE.get(d_id, "")  # "CO" / "NM" / "TX"
    return fields


def build_baselines() -> list[dict]:
    """Convert DIMS into the baseline spec consumed by pipeline.alignment."""
    out = []
    for d_id, desc, judgment, score, kw in DIMS:
        out.append(
            {
                "id": d_id,
                "description": f"{desc} [基准: {judgment}]",
                "match_fields": _derive_match_fields(d_id, desc),
                "kw": kw,
                "judgment": judgment,
                "score": score,
            }
        )
    return out


# ═══════════════════════════════════════════════════════════════════════════
# Load raw claims from v2 extraction output
# ═══════════════════════════════════════════════════════════════════════════


def load_raw_claims(extraction_dir: Path) -> dict[str, list[dict]]:
    """Read each model's `{dir}/{model}/extraction.json` and return the raw
    `entities[0].canonical.value` list (preserving the model's claimed order)."""
    out: dict[str, list[dict]] = {}
    for d in Path(extraction_dir).iterdir():
        if not d.is_dir():
            continue
        ext_file = d / "extraction.json"
        if not ext_file.exists():
            continue
        payload = json.loads(ext_file.read_text(encoding="utf-8"))
        entities = payload.get("entities", [])
        if not entities:
            out[d.name] = []
            continue
        canonical_value = (entities[0].get("canonical") or {}).get("value") or []
        if not isinstance(canonical_value, list):
            canonical_value = []
        out[d.name] = canonical_value
    return out


# ═══════════════════════════════════════════════════════════════════════════
# Order-correctness helper
# ═══════════════════════════════════════════════════════════════════════════


def _compute_order_correctness(
    canonical_in_raw_order: list[str],
) -> dict[str, bool]:
    """For a list of canonical_ids in the model's raw output order, decide
    for each id whether its rank in the model order matches its rank in the
    expected baseline order.

    Strategy:
      - Restrict to ids that are in DIM_MAP (already valid hits) and dedup
        by first-occurrence.
      - rank_in_model[id] = its 1-based index in canonical_in_raw_order.
      - rank_in_expected[id] = its rank when sorting by EXPECTED_POSITION
        among the same restricted set.
      - "顺序对" iff rank_in_model[id] == rank_in_expected[id].

    This penalizes only the locally-out-of-place items. If the entire list
    is monotonic in EXPECTED_POSITION, every id gets True (顺序全对)."""
    seen: set[str] = set()
    valid_in_order = []
    for cid in canonical_in_raw_order:
        if cid in DIM_MAP and cid not in seen:
            seen.add(cid)
            valid_in_order.append(cid)

    rank_in_model = {cid: i + 1 for i, cid in enumerate(valid_in_order)}
    sorted_by_expected = sorted(valid_in_order, key=lambda c: EXPECTED_POSITION[c])
    rank_in_expected = {cid: i + 1 for i, cid in enumerate(sorted_by_expected)}

    return {
        cid: (rank_in_model[cid] == rank_in_expected[cid]) for cid in valid_in_order
    }


# ═══════════════════════════════════════════════════════════════════════════
# Scoring from aligned claims
# ═══════════════════════════════════════════════════════════════════════════


def _normalize_state(raw_state: str | None) -> str:
    """Return one of 'CO', 'NM', 'TX', 'unspecified', or 'OTHER'."""
    if not raw_state:
        return "unspecified"
    s = str(raw_state).strip().upper()
    if s in ("CO", "COLORADO"):
        return "CO"
    if s in ("NM", "NEW MEXICO", "N.M."):
        return "NM"
    if s in ("TX", "TEXAS"):
        return "TX"
    if s in ("UNSPECIFIED", "UNKNOWN", "N/A", "NULL", ""):
        return "unspecified"
    return "OTHER"


def score_aligned(aligned_by_model: dict[str, list[dict]]) -> dict[str, dict]:
    """Aggregate scores per Q33 ordered_sequence rules.

    Per claim:
      - canonical_id ∈ DIM_MAP and state correct and order correct  → +1.0
      - canonical_id ∈ DIM_MAP and state correct and order wrong    → +0.5
      - canonical_id ∈ DIM_MAP and state == 'unspecified'           → +0.5
      - canonical_id ∈ DIM_MAP and state wrong (e.g. CO_04 but model said TX)
                                                                     → 0.0
      - canonical_id == '__HALLUCINATION__'                          → -1.0
      - canonical_id is None and confidence != needs_review          → -1.0
      - alignment_confidence == 'needs_review'                       → 0.0
      - duplicates dropped (first-occurrence per cid only)

    Per missing canonical_id (in DIM_MAP but not mentioned by model): -1.0
    """
    all_scores: dict[str, dict] = {}
    for model_name, claims in aligned_by_model.items():
        # First pass — collect canonical_ids in raw order (skip dup, skip null)
        canonical_in_raw_order = [
            c.get("canonical_id") for c in claims if c.get("canonical_id") in DIM_MAP
        ]
        order_ok = _compute_order_correctness(canonical_in_raw_order)

        seen_dims: set[str] = set()
        scored: list[dict] = []
        unverified: list[dict] = []
        total = 0.0

        for c in claims:
            raw = c.get("raw", {}) or {}
            cid = c.get("canonical_id")
            conf = c.get("alignment_confidence", "medium")
            reason = c.get("alignment_reasoning", "")
            judge_invoked = c.get("judge_invoked", False)
            claimed_name = raw.get("name", "")
            claimed_state = _normalize_state(raw.get("state"))

            if conf == "needs_review":
                unverified.append(
                    {
                        "name": claimed_name,
                        "claimed_state": claimed_state,
                        "canonical_id_tentative": cid,
                        "reason": f"needs_review: {reason}",
                        "judge_invoked": judge_invoked,
                        "score": 0.0,
                    }
                )
                continue

            if cid == "__HALLUCINATION__":
                scored.append(
                    {
                        "id": "__HALLUCINATION__",
                        "name": "(verified hallucination)",
                        "claimed_county": claimed_name,
                        "claimed_state": claimed_state,
                        "score": -1.0,
                        "judgment": "❌(hallucination)",
                        "confidence": conf,
                        "judge_invoked": judge_invoked,
                        "reason": reason,
                    }
                )
                total += -1.0
                continue

            if cid and cid in DIM_MAP:
                if cid in seen_dims:
                    continue  # dedup
                seen_dims.add(cid)

                expected_state = EXPECTED_STATE[cid]
                if claimed_state == expected_state:
                    order_correct = order_ok.get(cid, False)
                    s = 1.0 if order_correct else 0.5
                    judgment = "✅(in_order)" if order_correct else "🅢(out_of_order)"
                    detail = (
                        f"对齐到 {cid}（{expected_state}），"
                        f"state 正确，顺序{'对' if order_correct else '错'}"
                    )
                elif claimed_state == "unspecified":
                    s = 0.5
                    judgment = "⚠️(state_unspecified)"
                    detail = (
                        f"对齐到 {cid}（{expected_state}），"
                        f"但模型未指明 state，按 partial credit"
                    )
                else:
                    # state explicitly wrong (e.g. claimed_state=='TX' but
                    # expected 'CO') — same-name confusion
                    s = 0.0
                    judgment = "❌(state_wrong)"
                    detail = (
                        f"对齐到 {cid}（期望 {expected_state}），"
                        f"模型却写 state={claimed_state}（同名县混淆）"
                    )

                scored.append(
                    {
                        "id": cid,
                        "name": DIM_MAP[cid]["name"],
                        "claimed_county": claimed_name,
                        "claimed_state": claimed_state,
                        "expected_state": expected_state,
                        "score": s,
                        "judgment": judgment,
                        "confidence": conf,
                        "judge_invoked": judge_invoked,
                        "reason": reason or detail,
                    }
                )
                total += s
            else:
                # null + Stage C unresolved → -1
                scored.append(
                    {
                        "id": "__UNRESOLVED_NULL__",
                        "name": "(null + unresolved, treated as wrong)",
                        "claimed_county": claimed_name,
                        "claimed_state": claimed_state,
                        "score": -1.0,
                        "judgment": "❌(unresolved_null)",
                        "confidence": conf,
                        "judge_invoked": judge_invoked,
                        "reason": reason or "未对齐到任何基准条目，且 Stage C 未能验证",
                    }
                )
                total += -1.0

        # Missed counties (in DIM_MAP but never mentioned) → -1.0 each
        missed_ids = [d_id for d_id in DIM_MAP if d_id not in seen_dims]
        for d_id in missed_ids:
            scored.append(
                {
                    "id": d_id,
                    "name": DIM_MAP[d_id]["name"],
                    "claimed_county": "(not mentioned)",
                    "claimed_state": "(not mentioned)",
                    "expected_state": EXPECTED_STATE[d_id],
                    "score": -1.0,
                    "judgment": "❌(missed)",
                    "confidence": "n/a",
                    "judge_invoked": False,
                    "reason": "模型未提及此 baseline 县（漏掉 → -1）",
                }
            )
            total += -1.0

        n_answered = len(seen_dims) + sum(
            1 for s in scored if s["id"] in ("__HALLUCINATION__", "__UNRESOLVED_NULL__")
        )
        all_scores[model_name] = {
            "total_score": round(total, 3),
            "max_score": MAX_SCORE,
            "score_rate": round(total / n_answered, 4) if n_answered else 0.0,
            "total_rate": round(total / MAX_SCORE, 4) if MAX_SCORE else 0.0,
            "dimensions_answered": n_answered,
            "dimensions_hit": len(seen_dims),
            "dimensions_missed": len(missed_ids),
            "per_dimension": sorted(
                scored,
                key=lambda x: x["id"] if x["id"] in DIM_MAP else f"ZZ_{x['id']}",
            ),
            "unverified_claims": unverified,
        }
    return all_scores


# ═══════════════════════════════════════════════════════════════════════════
# Outputs
# ═══════════════════════════════════════════════════════════════════════════


def build_scores_json(all_scores: dict[str, dict]) -> dict:
    ranking = sorted(all_scores.items(), key=lambda x: -x[1]["total_score"])
    unverified_all = []
    for m, s in all_scores.items():
        for uv in s.get("unverified_claims", []):
            unverified_all.append({"model": m, **uv, "action_needed": "需人工核实"})
    return {
        "query_id": QUERY_ID,
        "scoring_mode": SCORING_MODE,
        "snapshot_date": SNAPSHOT_DATE,
        "max_score": MAX_SCORE,
        "gt_source": GT_SOURCE,
        "extraction_pipeline": (
            "v2 (primary=claude-sonnet-4, secondary=gpt-5, analyzer=claude-opus-4.6)"
        ),
        "results": all_scores,
        "ranking": [
            {
                "rank": i + 1,
                "model": m,
                "score": s["total_score"],
                "rate": f"{s['total_rate'] * 100:.1f}%",
                "hit": s["dimensions_hit"],
                "missed": s["dimensions_missed"],
            }
            for i, (m, s) in enumerate(ranking)
        ],
        "unverified_items": unverified_all,
    }


def build_ranking_md(all_scores: dict[str, dict]) -> str:
    ranked = sorted(
        all_scores.items(),
        key=lambda x: (-x[1]["total_score"], -x[1].get("score_rate", 0)),
    )
    lines = [
        "# Query 33 排名报告（Denver 沿正南至南极路径上的 US Counties）",
        "",
        f"> Snapshot: {SNAPSHOT_DATE}  Mode: {SCORING_MODE}  Max: {MAX_SCORE}",
        "> 抽取流水线：Primary=claude-sonnet-4, Secondary=gpt-5, Analyzer=claude-opus-4.6",
        "> 评分：✅+1（顺序对） / 🅢+0.5（顺序错位） / ⚠️+0.5（state 未指定） / ",
        ">      ❌0（state 错） / ❌-1（幻觉） / ❌-1（漏掉 baseline 县）",
        f"> GT 来源：{GT_SOURCE}",
        "",
        "**Baseline 17 个 US Counties（北→南顺序）**：",
        "  - CO (9): Denver → Arapahoe → Douglas → El Paso → **Teller** → "
        "**Fremont** → Pueblo → Huerfano → Las Animas",
        "  - NM (7): Colfax → Mora → San Miguel → Guadalupe → Lincoln → Chaves → Otero",
        "  - TX (1): Hudspeth",
        "",
        "| Rank | Model | Score | Rate | Hit | Missed | Unverified |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for i, (m, s) in enumerate(ranked, 1):
        lines.append(
            f"| {i} | {m} | {s['total_score']}/{s['max_score']} "
            f"| {s['total_rate'] * 100:.1f}% "
            f"| {s['dimensions_hit']} "
            f"| {s['dimensions_missed']} "
            f"| {len(s['unverified_claims'])} |"
        )
    return "\n".join(lines) + "\n"


# ═══════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════


def _load_alignment(out_dir: Path) -> dict[str, list[dict]]:
    aligned: dict[str, list[dict]] = {}
    for d in Path(out_dir).iterdir():
        if not d.is_dir():
            continue
        af = d / "alignment.json"
        if not af.exists():
            continue
        payload = json.loads(af.read_text(encoding="utf-8"))
        aligned[d.name] = payload.get("aligned_claims", [])
    return aligned


def main() -> None:
    ap = argparse.ArgumentParser(description="Query 33 auto-scorer (v2+align)")
    ap.add_argument(
        "--models", nargs="+", help="name=path list of model answer JSON files."
    )
    ap.add_argument("--output-dir", default=str(THIS / "auto_scores"))
    ap.add_argument("--primary-model", default=None)
    ap.add_argument("--secondary-model", default=None)
    ap.add_argument("--parallel-models", nargs="+", default=None)
    ap.add_argument("--analyzer-model", default=None)
    ap.add_argument("--aligner-models", nargs="+", default=None)
    ap.add_argument("--judge-model", default=None)
    ap.add_argument("--concurrency", type=int, default=None)
    ap.add_argument(
        "--skip-extract",
        action="store_true",
        help="Skip extraction; reuse existing extraction.json.",
    )
    ap.add_argument(
        "--skip-align",
        action="store_true",
        help="Skip alignment; reuse existing alignment.json.",
    )
    args = ap.parse_args()

    if not args.models:
        sys.exit("[ERROR] --models is required.")

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

    # Stage 2: alignment
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
        raw_claims = load_raw_claims(out_dir)
        aligned = align_claims(
            client,
            claims_by_model=raw_claims,
            baselines=baselines,
            query_text=QUERY_TEXT,
            output_dir=out_dir,
            overrides=overrides,
        )

    # Stage 3a: export null claims for manual + Claude-Code-assisted verification
    null_items = export_null_claims_for_review(
        aligned,
        out_dir / "null_review.json",
        query_id=QUERY_ID,
        query_text=QUERY_TEXT,
        models_input=args.models,
    )
    print(
        f"\n[*] Exported {len(null_items)} null/needs_review claims → null_review.json"
    )

    # Stage 3b: apply null_resolutions.json if present
    resolutions_path = out_dir / "null_resolutions.json"
    if resolutions_path.exists():
        print("[*] Found null_resolutions.json, applying …")
        new_baseline_entries = apply_null_resolutions(
            aligned, resolutions_path, dims_ref=DIMS
        )
        if new_baseline_entries:
            print(
                f"[*] {len(new_baseline_entries)} new baseline entries from resolutions"
            )
            for e in new_baseline_entries:
                DIM_MAP[e["id"]] = {
                    "id": e["id"],
                    "name": e.get("description", e["id"]),
                    "judgment": e.get("judgment", "⚠️"),
                    "score": e.get("score", 0.0),
                    "kw": e.get("kw", []),
                }
            persist_new_baseline_entries(new_baseline_entries, __file__)
    else:
        print(
            f"[*] No null_resolutions.json yet."
            f"\n    → 用 Claude Code 读 {resolutions_path.parent}/null_review.json"
            f"\n      逐条 web 验证后产出 {resolutions_path.name}"
            f"\n      然后重跑：python3 {Path(__file__).name} "
            f"--skip-extract --skip-align --models …"
        )

    # Stage 4: scoring
    scores = score_aligned(aligned)

    (out_dir / "scores.json").write_text(
        json.dumps(build_scores_json(scores), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (out_dir / "ranking_report.md").write_text(
        build_ranking_md(scores),
        encoding="utf-8",
    )

    print("\n" + "─" * 62)
    print("Query 33 scoring done.")
    for i, (m, s) in enumerate(
        sorted(scores.items(), key=lambda x: -x[1]["total_score"]), 1
    ):
        print(
            f"  {i}. {m:28s} {s['total_score']:>6.1f}/{s['max_score']:.1f}"
            f"  hit={s['dimensions_hit']}/{len(DIMS)}"
            f"  missed={s['dimensions_missed']}"
            f"  unverified={len(s['unverified_claims'])}"
        )


if __name__ == "__main__":
    main()
