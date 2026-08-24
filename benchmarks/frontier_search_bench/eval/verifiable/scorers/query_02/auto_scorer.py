"""
Query 02 — 中国国家级新区 GDP 增速差值 / 由正转负判断 auto-scorer.

Scoring (5×3 matrix + out-of-scope row, per-area conflict 取最低分)：

  Tier 1 ✅ 干净转负 (7 个：滨海/两江/南沙/湘江/哈尔滨/西咸/长春)
    flipped+年份对 +2 / 年份错 +1 / not_flipped -1 / indeterminate 0
  Tier 2 ⚠️ 边缘转负 (3 个：舟山/西海岸/金普)
    flipped+年份对 +1 / 年份错 +0.5 / not_flipped -0.5 / indeterminate 0
  Tier 3 ❌ 干净未转负 (2 个：天府/江北)
    flipped -1 / not_flipped +1 / indeterminate 0
  Tier 4 🟡 部分可答 (5 个：兰州/福州/滇中/赣江/贵安)
    flipped 0 / not_flipped 0 / indeterminate +0.5
  Tier 5 ⚫ 不可答 (1 个：雄安)
    flipped -1 / not_flipped -0.5 / indeterminate +0.5
  Out of scope (浦东 1992 时间窗外 / 编造名字)
    flipped/not_flipped -1 / indeterminate 0

  MAX_SCORE = 7×2 + 3×1 + 2×1 + 5×0.5 + 1×0.5 = 22.0
  v2 修订 (2026-05-21)：哈尔滨/西咸/长春 由 Tier 4 升至 Tier 1，truth = 2021；依据 2 个独立 web research agent 核证地方统计公报。详见 build/q02_tier4_agent1_report.md
  Conflict 处理：同一新区多条 claim 不一致 → 取最低分（鼓励清晰最终判断）
  Dedup：同一 (新区, claim_type) 重复出现，由 min 自动收敛为单条

Pipeline:
  Stage 1 — Extraction (v2 pipeline)
  Stage 2 — Alignment (3-aligner vote + judge)
  Stage 3 — Null verification → null_resolutions.json
  Stage 4 — Scoring deterministic from canonical_id
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
# Benchmark — 18 国家级新区 + 浦东（out of scope）
# Cross-verified 2026-05-01 against 国家统计局 / 各市统计公报 / 新区管委会官网
# 数据来源详见 baseline 调研附件 (3 轮 web research)
# ═══════════════════════════════════════════════════════════════════════════

SNAPSHOT_DATE = "2026-05-21"
SCORING_MODE = "Q02-tier-matrix-v2"


BASELINE_AREAS: dict[str, dict] = {
    # ─────────── Tier 1 ✅ 干净转负（7 个）───────────
    "滨海新区": {
        "approval_year": 2006,
        "tier": 1,
        "first_flip_year": "2017",
        "judgment": "✅",
        "description": (
            "天津滨海新区（2006-09 国务院批复）。差值首次由正转负年份：2017。"
            "[Tier 1 ✅ 干净转负 — 注：2017 起改在地核算口径，与 2016 不可比；"
            "按字面定义首次新口径年即转负]"
        ),
        "kw": [
            "滨海新区",
            "天津滨海",
            "binhai",
            "tianjin binhai",
            "滨海",
        ],
    },
    "两江新区": {
        "approval_year": 2010,
        "tier": 1,
        "first_flip_year": "2021",
        "judgment": "✅",
        "description": (
            "重庆两江新区（2010-06 国务院批复）。差值首次由正转负年份：2021"
            "（差值 -0.9）。[Tier 1 ✅ 干净转负]"
        ),
        "kw": [
            "两江新区",
            "重庆两江",
            "liangjiang",
            "chongqing liangjiang",
        ],
    },
    "南沙新区": {
        "approval_year": 2012,
        "tier": 1,
        "first_flip_year": "2023",
        "judgment": "✅",
        "description": (
            "广州南沙新区（2012-09 国务院批复）。差值首次由正转负年份：2023"
            "（差值 -1.1）。[Tier 1 ✅ 干净转负]"
        ),
        "kw": [
            "南沙新区",
            "广州南沙",
            "nansha",
            "guangzhou nansha",
        ],
    },
    "湘江新区": {
        "approval_year": 2015,
        "tier": 1,
        "first_flip_year": "2021",
        "judgment": "✅",
        "description": (
            "湖南湘江新区（2015-04 国务院批复）。差值首次由正转负年份：2021"
            "（差值 -0.8）。[Tier 1 ✅ 干净转负]"
        ),
        "kw": [
            "湘江新区",
            "湖南湘江",
            "长沙湘江",
            "xiangjiang",
        ],
    },
    # ─── v2 promotions (2026-05-21): 3 个原 Tier 4 升至 Tier 1，truth = 2021 ───
    "哈尔滨新区": {
        "approval_year": 2015,
        "tier": 1,
        "first_flip_year": "2021",
        "judgment": "✅",
        "description": (
            "哈尔滨新区（2015-12 国务院批复，松北区/江北一体发展区口径）。"
            "差值首次由正转负年份：2021（新区 +7.6% vs 全国 +8.1% = 差值 -0.5）。"
            "[Tier 1 ✅ 干净转负 — v2 升级自 Tier 4，依据松北区 2020-2023 统计公报]"
        ),
        "kw": [
            "哈尔滨新区",
            "黑龙江哈尔滨",
            "harbin",
        ],
    },
    "西咸新区": {
        "approval_year": 2014,
        "tier": 1,
        "first_flip_year": "2021",
        "judgment": "✅",
        "description": (
            "西咸新区（2014-01 国务院批复，2017 起西安代管）。"
            "差值首次由正转负年份：2021（新区 +3.7% vs 全国 +8.1% = 差值 -4.4）。"
            "[Tier 1 ✅ 干净转负 — v2 升级自 Tier 4，依据 gotohui GDP index + 西安统计公报]"
        ),
        "kw": [
            "西咸新区",
            "陕西西咸",
            "xixian",
            "xi'an xianyang",
        ],
    },
    "长春新区": {
        "approval_year": 2016,
        "tier": 1,
        "first_flip_year": "2021",
        "judgment": "✅",
        "description": (
            "长春新区（2016-02 国务院批复）。"
            "差值首次由正转负年份：2021（新区 +6.5% vs 全国 +8.1% = 差值 -1.6）。"
            "[Tier 1 ✅ 干净转负 — v2 升级自 Tier 4，依据中国日报/吉林日报 2021 公报 + 长春市 2022 −4.5% 公报推算]"
        ),
        "kw": [
            "长春新区",
            "吉林长春",
            "changchun",
        ],
    },
    # ─────────── Tier 2 ⚠️ 边缘转负（3 个）───────────
    "舟山群岛新区": {
        "approval_year": 2011,
        "tier": 2,
        "first_flip_year": "2018",
        "judgment": "⚠️",
        "description": (
            "浙江舟山群岛新区（2011-06 国务院批复，舟山市口径）。"
            "差值首次由正转负年份：2018（差值仅 -0.1，边缘事件，次年立即反弹）。"
            "[Tier 2 ⚠️ 边缘转负]"
        ),
        "kw": [
            "舟山群岛新区",
            "舟山群岛",
            "舟山新区",
            "舟山",
            "zhoushan archipelago",
            "zhoushan",
        ],
    },
    "西海岸新区": {
        "approval_year": 2014,
        "tier": 2,
        "first_flip_year": "2021",
        "judgment": "⚠️",
        "description": (
            "青岛西海岸新区（2014-06 国务院批复）。差值首次由正转负年份：2021"
            "（差值约 -0.1，边缘）。[Tier 2 ⚠️ 边缘转负]"
        ),
        "kw": [
            "西海岸新区",
            "青岛西海岸",
            "qingdao xihaian",
            "xihaian",
        ],
    },
    "金普新区": {
        "approval_year": 2014,
        "tier": 2,
        "first_flip_year": "2015",
        "judgment": "⚠️",
        "description": (
            "大连金普新区（2014-06 国务院批复）。差值首次由正转负年份：2015"
            "（辽宁省 2014-2016 集中挤水分，非真实下滑）。"
            "[Tier 2 ⚠️ 边缘转负 — 受省级数据调整干扰]"
        ),
        "kw": [
            "金普新区",
            "大连金普",
            "jinpu",
            "dalian jinpu",
        ],
    },
    # ─────────── Tier 3 ❌ 干净未转负（2 个）───────────
    "天府新区": {
        "approval_year": 2014,
        "tier": 3,
        "first_flip_year": None,
        "judgment": "❌",
        "description": (
            "四川天府新区（2014-10 国务院批复，成都直管区为主口径）。"
            "观察期内年年正差值，未发生由正转负。"
            "[Tier 3 ❌ 干净未转负 — 模型若主张转负属错误]"
        ),
        "kw": [
            "天府新区",
            "四川天府",
            "成都天府",
            "tianfu",
            "sichuan tianfu",
        ],
    },
    "江北新区": {
        "approval_year": 2015,
        "tier": 3,
        "first_flip_year": None,
        "judgment": "❌",
        "description": (
            "南京江北新区（2015-06 国务院批复，江北直管区口径）。"
            "观察期内年年正差值，未发生由正转负。"
            "[Tier 3 ❌ 干净未转负 — 模型若主张转负属错误]"
        ),
        "kw": [
            "江北新区",
            "南京江北",
            "jiangbei",
            "nanjing jiangbei",
        ],
    },
    # ─────────── Tier 4 🟡 部分可答（5 个）───────────
    "兰州新区": {
        "approval_year": 2012,
        "tier": 4,
        "first_flip_year": None,
        "judgment": "⚠️",
        "description": (
            "兰州新区（2012-08 国务院批复）。逐年增速点值多以'连续 X 年超 15%'"
            "聚合表述，公开数据无法严谨判断首次转负。"
            "[Tier 4 🟡 部分可答 — indeterminate]"
        ),
        "kw": [
            "兰州新区",
            "甘肃兰州",
            "lanzhou",
        ],
    },
    "福州新区": {
        "approval_year": 2015,
        "tier": 4,
        "first_flip_year": None,
        "judgment": "⚠️",
        "description": (
            "福州新区（2015-08 国务院批复）。2024 年起统计口径改为单一长乐区，"
            "前后不可比；逐年增速大量缺失。"
            "[Tier 4 🟡 部分可答 — indeterminate]"
        ),
        "kw": [
            "福州新区",
            "福建福州",
            "fuzhou",
        ],
    },
    "滇中新区": {
        "approval_year": 2015,
        "tier": 4,
        "first_flip_year": None,
        "judgment": "⚠️",
        "description": (
            "云南滇中新区（2015-09 国务院批复）。2016-2020 有 5 年逐年增速，"
            "可推算 2020 年首次转负，但 2021-2022 数据缺失影响最终判断。"
            "[Tier 4 🟡 部分可答 — indeterminate]"
        ),
        "kw": [
            "滇中新区",
            "云南滇中",
            "dianzhong",
            "yunnan dianzhong",
        ],
    },
    "赣江新区": {
        "approval_year": 2016,
        "tier": 4,
        "first_flip_year": None,
        "judgment": "⚠️",
        "description": (
            "赣江新区（2016-06 国务院批复）。2018-2021 中间四年逐年增速完全缺失。"
            "[Tier 4 🟡 部分可答 — indeterminate]"
        ),
        "kw": [
            "赣江新区",
            "江西赣江",
            "ganjiang",
            "jiangxi ganjiang",
        ],
    },
    "贵安新区": {
        "approval_year": 2014,
        "tier": 4,
        "first_flip_year": None,
        "judgment": "⚠️",
        "description": (
            "贵安新区（2014-01 国务院批复）。2017 增速 32.8% 后续疑似挤水分；"
            "2021 起贵阳贵安融合，连续可比序列断裂。"
            "[Tier 4 🟡 部分可答 — indeterminate]"
        ),
        "kw": [
            "贵安新区",
            "贵州贵安",
            "guian",
            "guizhou guian",
        ],
    },
    # ─────────── Tier 5 ⚫ 不可答（1 个）───────────
    "雄安新区": {
        "approval_year": 2017,
        "tier": 5,
        "first_flip_year": None,
        "judgment": "⚫",
        "description": (
            "河北雄安新区（2017-04 设立）。至今未单独公布年度 GDP 增速，"
            "公开数据多为投资额或'十四五年均增长 17.1%'汇总口径。"
            "[Tier 5 ⚫ 不可答 — 主张转负即虚构]"
        ),
        "kw": [
            "雄安新区",
            "河北雄安",
            "xiongan",
        ],
    },
    # ─────────── Out of scope（时间窗外）───────────
    "浦东新区": {
        "approval_year": 1992,
        "tier": "out_of_scope",
        "first_flip_year": None,
        "judgment": "❌",
        "description": (
            "上海浦东新区（1992-04 国务院批复）。"
            "[基准: 时间窗外 — 1992 早于 2005-01-01，由打分层按 out_of_scope 规则处理]"
        ),
        "kw": [
            "浦东新区",
            "上海浦东",
            "pudong",
            "shanghai pudong",
        ],
    },
}


TIER_BY_AREA: dict[str, object] = {
    n: info["tier"] for n, info in BASELINE_AREAS.items()
}
FIRST_FLIP_YEAR: dict[str, str] = {
    n: info["first_flip_year"]
    for n, info in BASELINE_AREAS.items()
    if info.get("first_flip_year")
}


# ═══════════════════════════════════════════════════════════════════════════
# Baseline → alignment spec
# ═══════════════════════════════════════════════════════════════════════════


def _derive_match_fields(info: dict) -> dict:
    """Cross-check fields exposed to alignment layer."""
    fields: dict = {}
    if info.get("approval_year"):
        fields["approval_year"] = str(info["approval_year"])
    return fields


def build_baselines() -> list[dict]:
    """Convert BASELINE_AREAS into the spec consumed by pipeline.alignment."""
    out = []
    for name, info in BASELINE_AREAS.items():
        out.append(
            {
                "id": name,
                "description": info["description"],
                "match_fields": _derive_match_fields(info),
                "kw": info["kw"],
                "judgment": info["judgment"],
                "score": 0.0,  # tier matrix 决定实际得分，此字段仅占位
                "tier": info["tier"],
                "approval_year": info.get("approval_year"),
                "first_flip_year": info.get("first_flip_year"),
            }
        )
    return out


# ═══════════════════════════════════════════════════════════════════════════
# 5×3 scoring matrix
# ═══════════════════════════════════════════════════════════════════════════


def get_score(tier, claim_type: str, year_correct: bool | None) -> tuple[float, str]:
    """5×3 矩阵查表。Returns (score, reason)."""
    ct = (claim_type or "").lower().strip()

    if tier == 1:
        if ct == "flipped":
            return (
                (2.0, "Tier1+flipped+年份对 → +2")
                if year_correct
                else (1.0, "Tier1+flipped+年份错 → +1")
            )
        if ct == "not_flipped":
            return -1.0, "Tier1+not_flipped → -1（实际转负被否认）"
        if ct == "indeterminate":
            return 0.0, "Tier1+indeterminate → 0（过度保守不扣分）"

    elif tier == 2:
        if ct == "flipped":
            return (
                (1.0, "Tier2+flipped+年份对 → +1")
                if year_correct
                else (0.5, "Tier2+flipped+年份错 → +0.5")
            )
        if ct == "not_flipped":
            return -0.5, "Tier2+not_flipped → -0.5"
        if ct == "indeterminate":
            return 0.0, "Tier2+indeterminate → 0"

    elif tier == 3:
        if ct == "flipped":
            return -1.0, "Tier3+flipped → -1（实际未转负被错列）"
        if ct == "not_flipped":
            return 1.0, "Tier3+not_flipped → +1（正确识别未转负）"
        if ct == "indeterminate":
            return 0.0, "Tier3+indeterminate → 0（过度保守不扣分）"

    elif tier == 4:
        if ct == "flipped":
            return 0.0, "Tier4+flipped → 0（数据不完整无法验证）"
        if ct == "not_flipped":
            return 0.0, "Tier4+not_flipped → 0（数据不完整无法验证）"
        if ct == "indeterminate":
            return 0.5, "Tier4+indeterminate → +0.5（认知诚实）"

    elif tier == 5:
        if ct == "flipped":
            return -1.0, "Tier5+flipped → -1（雄安无 GDP 数据，主张转负=虚构）"
        if ct == "not_flipped":
            return -0.5, "Tier5+not_flipped → -0.5（无数据下断言未转负）"
        if ct == "indeterminate":
            return 0.5, "Tier5+indeterminate → +0.5（认知诚实）"

    elif tier == "out_of_scope":
        if ct == "flipped":
            return -1.0, "out_of_scope+flipped → -1（时间窗外或编造）"
        if ct == "not_flipped":
            return -1.0, "out_of_scope+not_flipped → -1（时间窗外或编造）"
        if ct == "indeterminate":
            return 0.0, "out_of_scope+indeterminate → 0"

    return 0.0, f"未知组合: tier={tier} claim_type={ct}"


def _year_eq(claim_year, baseline_year) -> bool:
    """严格年份匹配（去 '年' 后缀+ trim）。任一为 falsy → False。"""
    if not claim_year or not baseline_year:
        return False
    return (
        str(claim_year).strip().rstrip("年").rstrip(" ") == str(baseline_year).strip()
    )


# Theoretical max score: ideal model picks best path per area.
def _compute_max_score() -> float:
    total = 0.0
    for info in BASELINE_AREAS.values():
        tier = info["tier"]
        if tier == 1:
            total += 2.0
        elif tier == 2 or tier == 3:
            total += 1.0
        elif tier == 4 or tier == 5:
            total += 0.5
        # out_of_scope: best is to NOT mention → 0
    return total


MAX_SCORE = _compute_max_score()


# ═══════════════════════════════════════════════════════════════════════════
# Load extraction (canonical.value lists)
# ═══════════════════════════════════════════════════════════════════════════


def load_raw_claims(extraction_dir: Path) -> dict[str, list[dict]]:
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


# ═══════════════════════════════════════════════════════════════════════════
# Scoring (with conflict-取最低分)
# ═══════════════════════════════════════════════════════════════════════════


def score_aligned(aligned_by_model: dict[str, list[dict]]) -> dict[str, dict]:
    """Aggregate scores. Per-area conflict (multiple claim_types) → take MIN."""
    all_scores: dict[str, dict] = {}
    for model_name, claims in aligned_by_model.items():
        # Group by canonical_id
        by_area: dict[str, list[dict]] = {}
        unaligned: list[dict] = []  # null + Stage C unresolved
        hallucinations: list[dict] = []
        needs_review: list[dict] = []

        for c in claims:
            cid = c.get("canonical_id")
            conf = c.get("alignment_confidence", "medium")
            reason_align = c.get("alignment_reasoning", "")
            judge_invoked = c.get("judge_invoked", False)
            raw = c.get("raw", {}) or {}
            claim_type = raw.get("claim_type") or "flipped"
            claim_year = raw.get("first_flip_year")
            area_name = raw.get("area_name", "") or ""

            if conf == "needs_review":
                needs_review.append(
                    {
                        "area_name": area_name,
                        "claim_type": claim_type,
                        "first_flip_year": claim_year,
                        "canonical_id_tentative": cid,
                        "score": 0.0,
                        "reason": f"needs_review: {reason_align}",
                    }
                )
                continue

            if cid == "__HALLUCINATION__":
                # treat as out_of_scope
                score, reason_score = get_score("out_of_scope", claim_type, None)
                hallucinations.append(
                    {
                        "area_name": area_name,
                        "claim_type": claim_type,
                        "first_flip_year": claim_year,
                        "score": score,
                        "judgment": "❌",
                        "judge_invoked": judge_invoked,
                        "reason": f"__HALLUCINATION__ → {reason_score}; align: {reason_align}",
                    }
                )
                continue

            if cid is None or cid not in BASELINE_AREAS:
                # null + Stage C unresolved → treat as out_of_scope
                score, reason_score = get_score("out_of_scope", claim_type, None)
                unaligned.append(
                    {
                        "area_name": area_name,
                        "claim_type": claim_type,
                        "first_flip_year": claim_year,
                        "canonical_id": cid,
                        "score": score,
                        "judgment": "❌",
                        "judge_invoked": judge_invoked,
                        "reason": f"null/未知基准 → {reason_score}; align: {reason_align}",
                    }
                )
                continue

            tier = TIER_BY_AREA[cid]
            year_correct: bool | None = None
            if tier in (1, 2):
                year_correct = _year_eq(claim_year, FIRST_FLIP_YEAR.get(cid))
            score, reason_score = get_score(tier, claim_type, year_correct)

            by_area.setdefault(cid, []).append(
                {
                    "canonical_id": cid,
                    "tier": tier,
                    "claim_type": claim_type,
                    "first_flip_year_claimed": claim_year,
                    "first_flip_year_truth": FIRST_FLIP_YEAR.get(cid),
                    "year_correct": year_correct,
                    "score": score,
                    "reason": reason_score,
                    "raw_area_name": area_name,
                    "judge_invoked": judge_invoked,
                    "alignment_confidence": conf,
                }
            )

        # Conflict resolution: per (model, canonical_id), pick MIN score
        per_area: list[dict] = []
        for _cid, claims_list in by_area.items():
            min_claim = min(claims_list, key=lambda x: x["score"])
            entry = dict(min_claim)
            entry["n_claims"] = len(claims_list)
            entry["all_claim_types"] = [c["claim_type"] for c in claims_list]
            if len(claims_list) > 1:
                entry["reason"] = (
                    f"[CONFLICT n={len(claims_list)} 取最低分] " + entry["reason"]
                )
            per_area.append(entry)

        total = (
            sum(c["score"] for c in per_area)
            + sum(h["score"] for h in hallucinations)
            + sum(u["score"] for u in unaligned)
        )

        all_scores[model_name] = {
            "total_score": round(total, 3),
            "max_score": MAX_SCORE,
            "total_rate": round(total / MAX_SCORE, 4) if MAX_SCORE else 0.0,
            "areas_judged": len(per_area),
            "tier_breakdown": _tier_breakdown(per_area),
            "per_area": sorted(per_area, key=lambda x: x["canonical_id"]),
            "hallucinations": hallucinations,
            "unaligned": unaligned,
            "needs_review": needs_review,
        }
    return all_scores


def _tier_breakdown(per_area: list[dict]) -> dict:
    """Aggregate counts per tier for the ranking report."""
    out = {
        "tier_1": 0,
        "tier_2": 0,
        "tier_3": 0,
        "tier_4": 0,
        "tier_5": 0,
        "out_of_scope": 0,
    }
    for c in per_area:
        t = c.get("tier")
        key = f"tier_{t}" if isinstance(t, int) else "out_of_scope"
        out[key] = out.get(key, 0) + 1
    return out


# ═══════════════════════════════════════════════════════════════════════════
# Outputs
# ═══════════════════════════════════════════════════════════════════════════


def build_scores_json(all_scores: dict[str, dict]) -> dict:
    ranking = sorted(all_scores.items(), key=lambda x: -x[1]["total_score"])
    return {
        "query_id": QUERY_ID,
        "scoring_mode": SCORING_MODE,
        "snapshot_date": SNAPSHOT_DATE,
        "max_score": MAX_SCORE,
        "extraction_pipeline": (
            "v2 (primary=claude-sonnet-4, secondary=gpt-5, analyzer=claude-opus-4.6)"
        ),
        "scoring_rule": (
            "5×3 matrix: Tier1 ✅(4) flipped+yr=+2/yr错=+1/not_flipped=-1/indeterminate=0; "
            "Tier2 ⚠️(3) flipped+yr=+1/yr错=+0.5/not_flipped=-0.5/indeterminate=0; "
            "Tier3 ❌(2) flipped=-1/not_flipped=+1/indeterminate=0; "
            "Tier4 🟡(8) flipped=0/not_flipped=0/indeterminate=+0.5; "
            "Tier5 ⚫(1雄安) flipped=-1/not_flipped=-0.5/indeterminate=+0.5; "
            "out_of_scope flipped/not_flipped=-1/indeterminate=0. "
            "Per-area conflict (multiple claim_types) → MIN."
        ),
        "results": all_scores,
        "ranking": [
            {
                "rank": i + 1,
                "model": m,
                "score": s["total_score"],
                "rate": f"{s['total_rate'] * 100:.1f}%",
            }
            for i, (m, s) in enumerate(ranking)
        ],
    }


def build_ranking_md(all_scores: dict[str, dict]) -> str:
    ranked = sorted(
        all_scores.items(),
        key=lambda x: (-x[1]["total_score"], -x[1].get("total_rate", 0)),
    )
    lines = [
        f"# Query {QUERY_ID} 排名报告 — 国家级新区 GDP 差值由正转负",
        "",
        f"> Snapshot: {SNAPSHOT_DATE}  Mode: {SCORING_MODE}  Max: {MAX_SCORE}",
        "> Pipeline: Primary=claude-sonnet-4, Secondary=gpt-5, Analyzer=claude-opus-4.6",
        "> 评分矩阵：",
        ">   Tier1 ✅(4 个) flipped+yr=+2/yr错=+1/not_flipped=-1/indeterminate=0",
        ">   Tier2 ⚠️(3 个) flipped+yr=+1/yr错=+0.5/not_flipped=-0.5/indeterminate=0",
        ">   Tier3 ❌(2 个) flipped=-1/not_flipped=+1/indeterminate=0",
        ">   Tier4 🟡(8 个) flipped=0/not_flipped=0/indeterminate=+0.5",
        ">   Tier5 ⚫(1 个 雄安) flipped=-1/not_flipped=-0.5/indeterminate=+0.5",
        ">   out_of_scope flipped/not_flipped=-1/indeterminate=0",
        "> 冲突处理：同一新区多 claim_type → 取最低分",
        "",
        "## 总排名",
        "",
        "| Rank | Model | Score | Rate | Areas | T1 | T2 | T3 | T4 | T5 | OOS | Halluc | Unalign |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for i, (m, s) in enumerate(ranked, 1):
        tb = s.get("tier_breakdown", {})
        lines.append(
            f"| {i} | {m} "
            f"| {s['total_score']:.1f}/{s['max_score']:.1f} "
            f"| {s['total_rate'] * 100:.1f}% "
            f"| {s['areas_judged']} "
            f"| {tb.get('tier_1', 0)} "
            f"| {tb.get('tier_2', 0)} "
            f"| {tb.get('tier_3', 0)} "
            f"| {tb.get('tier_4', 0)} "
            f"| {tb.get('tier_5', 0)} "
            f"| {tb.get('out_of_scope', 0)} "
            f"| {len(s['hallucinations'])} "
            f"| {len(s['unaligned'])} |"
        )
    lines.append("")

    # Per-model detail
    lines.append("## 每模型逐条得分明细")
    lines.append("")
    for m, s in ranked:
        lines.append(f"### {m} (total={s['total_score']:.1f}/{s['max_score']:.1f})")
        lines.append("")
        lines.append(
            "| 新区 | Tier | claim_type | 模型年份 | 真值 | 年份对? | 分 | 备注 |"
        )
        lines.append("|---|---|---|---|---|---|---:|---|")
        for c in s["per_area"]:
            cid = c.get("canonical_id", "")
            tier = c.get("tier", "")
            ct = c.get("claim_type", "")
            cy = c.get("first_flip_year_claimed") or "-"
            ty = c.get("first_flip_year_truth") or "-"
            yc = c.get("year_correct")
            yc_str = "✓" if yc else ("✗" if yc is False else "-")
            sc = c.get("score", 0)
            rs = c.get("reason", "")[:60]
            lines.append(
                f"| {cid} | {tier} | {ct} | {cy} | {ty} | {yc_str} | {sc:+.1f} | {rs} |"
            )
        if s["hallucinations"]:
            lines.append("")
            lines.append("**Hallucinations:**")
            for h in s["hallucinations"]:
                lines.append(
                    f"- {h['area_name']!r} ({h['claim_type']}) → {h['score']:+.1f}"
                )
        if s["unaligned"]:
            lines.append("")
            lines.append("**Unaligned (null/未知基准):**")
            for u in s["unaligned"]:
                lines.append(
                    f"- {u['area_name']!r} ({u['claim_type']}) → {u['score']:+.1f}"
                )
        lines.append("")
    return "\n".join(lines) + "\n"


# ═══════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════


def main() -> None:
    ap = argparse.ArgumentParser(description=f"Query {QUERY_ID} auto-scorer")
    ap.add_argument(
        "--models",
        nargs="+",
        required=True,
        help="name=path list of model answer JSON files.",
    )
    ap.add_argument("--output-dir", default=str(THIS / "auto_scores"))
    ap.add_argument("--primary-model", default=None)
    ap.add_argument("--secondary-model", default=None)
    ap.add_argument("--parallel-models", nargs="+", default=None)
    ap.add_argument("--analyzer-model", default=None)
    ap.add_argument("--aligner-models", nargs="+", default=None)
    ap.add_argument("--judge-model", default=None)
    ap.add_argument("--concurrency", type=int, default=None)
    ap.add_argument("--skip-extract", action="store_true")
    ap.add_argument("--skip-align", action="store_true")
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

    # Stage 3a: export null claims for review
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
        # Build DIMS-equivalent list for apply_null_resolutions / persist
        dims_list = [
            (
                name,
                info["description"],
                info["judgment"],
                0.0,
                info["kw"],
            )
            for name, info in BASELINE_AREAS.items()
        ]
        new_baseline_entries = apply_null_resolutions(
            aligned, resolutions_path, dims_ref=dims_list
        )
        if new_baseline_entries:
            print(
                f"[*] {len(new_baseline_entries)} new baseline entries from resolutions"
            )
            for e in new_baseline_entries:
                # Inject into BASELINE_AREAS at runtime; tier defaults to out_of_scope
                # unless metadata supplied.
                BASELINE_AREAS[e["id"]] = {
                    "approval_year": e.get("approval_year"),
                    "tier": e.get("tier", "out_of_scope"),
                    "first_flip_year": e.get("first_flip_year"),
                    "judgment": e.get("judgment", "⚠️"),
                    "description": e.get("description", e["id"]),
                    "kw": e.get("kw", []),
                }
                TIER_BY_AREA[e["id"]] = e.get("tier", "out_of_scope")
                if e.get("first_flip_year"):
                    FIRST_FLIP_YEAR[e["id"]] = e["first_flip_year"]
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
    print(f"Query {QUERY_ID} scoring done.")
    for i, (m, s) in enumerate(
        sorted(scores.items(), key=lambda x: -x[1]["total_score"]), 1
    ):
        print(
            f"  {i}. {m:28s} {s['total_score']:>6.1f}/{s['max_score']:.1f}"
            f"  areas={s['areas_judged']}"
            f"  hall={len(s['hallucinations'])}"
            f"  unalign={len(s['unaligned'])}"
        )


if __name__ == "__main__":
    main()
