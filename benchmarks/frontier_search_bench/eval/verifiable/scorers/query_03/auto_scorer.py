"""Query 03 — SpaceX Starship IFT auto-scorer (v3, locked 2026-04-29).

Scoring:
  A_flight_count       ±1   (accept ∈ {10, 11}; missing → -1)
  Per IFT-N (N=1..11):
    progress sub-dim  ±1   (any wrong claim → -1; else any correct → +1; missing IFT → -1; mentioned but no kw match → 0)
    failures sub-dim  ±1   (same rule)
  Total ranges from -23 to +23.

GT is hard-coded in the PER_IFT_GT dict below.
Each fact has a `kw` array of 3-5 short matchers; any kw substring
(case-insensitive) appearing in a model claim counts as a hit.
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
    VALUE_SCHEMA,
)
from pipeline.extraction_pipeline import run_pipeline  # noqa: E402

# ═══════════════════════════════════════════════════════════════════════════
# Benchmark
# ═══════════════════════════════════════════════════════════════════════════

BENCHMARK = {
    "version": "3.0",
    "locked_at": "2026-04-29",
    "cutoff_date": "2026-01-01",
    "A_flight_count_accept": {10, 11},
}


# Per-IFT structured GT.
# Each fact: {"canonical": str, "kw": [str, ...]}
# kw matching: any kw substring (case-insensitive) appearing in a model
# claim string counts as a hit on that fact.
PER_IFT_GT: dict[str, dict] = {
    "IFT_1": {
        "date": "2023-04-20",
        "block": "Block 1",
        "progress": [
            # Mostly catastrophic; still a milestone — first integrated stack liftoff
            {
                "canonical": "首次完整栈 Super Heavy + Starship 发射并穿越 Max Q",
                "kw": [
                    "首次完整栈",
                    "完整栈发射",
                    "first integrated",
                    "first full stack",
                    "穿越max q",
                    "max-q",
                    "max q",
                ],
            },
            {
                "canonical": "推力 ~16M lbf (Block 1)",
                "kw": ["16m lbf", "16百万磅", "约16m", "16 million lbf"],
            },
        ],
        "failures": [
            {
                "canonical": "3 台 Raptor 未点火（仅 30/33 工作）",
                "kw": [
                    "3台raptor未点火",
                    "30/33",
                    "3 engines fail",
                    "3台未点火",
                    "30 of 33",
                ],
            },
            {
                "canonical": "TVC T+85s 失效翻滚 ~39km",
                "kw": ["tvc", "t+85", "翻滚", "39km", "tumble", "失控翻滚"],
            },
            # NOTE: dropped "t+4:01" kw because it conflicts with confirmed_wrong
            # IFT_1 "AFSS 在 T+4:01 触发" (model误把解体时间当触发时间).
            {
                "canonical": "FTS/AFTS 触发较晚 T+3:59 解体 ~29km",
                "kw": [
                    "fts",
                    "afts",
                    "解体",
                    "t+3:59",
                    "29km",
                    "flight termination",
                    "解体~29km",
                ],
            },
            {
                "canonical": "发射台被毁 385 英亩碎片",
                "kw": [
                    "385英亩",
                    "385 acres",
                    "发射台被毁",
                    "发射台毁",
                    "pad destroyed",
                    "concrete crater",
                ],
            },
        ],
        "confirmed_wrong": [
            {
                "canonical": "AFSS 在 T+4:01 触发（实际 T+3:20 触发，T+4:01 是解体时间）",
                "kw": ["afss在t+4:01触发", "afss t+4:01"],
            },
            {
                "canonical": "促使加装导流系统（实际 IFT-1 前已开始建造，IFT-1 加速完工）",
                "kw": ["ift-1后加装", "ift-1后开始建造导流", "ift-1 后才有导流"],
            },
        ],
    },
    "IFT_2": {
        "date": "2023-11-18",
        "block": "Block 1",
        "progress": [
            {
                "canonical": "33 台 Raptor 全部点火",
                "kw": [
                    "33台全部点火",
                    "33台正常",
                    "33 raptor",
                    "33 engines",
                    "33-engine",
                    "33台raptor",
                    "全部点火",
                    "all 33",
                ],
            },
            {
                "canonical": "首次成功完成热分离 (hot-staging)",
                "kw": ["热分离", "hot-stag", "hot stag"],
            },
            {
                "canonical": "首次入太空 ~148km / ~24000km/h",
                "kw": [
                    "148km",
                    "首次入太空",
                    "24000km",
                    "24,000km",
                    "150km高度",
                    "进入太空",
                ],
            },
        ],
        "failures": [
            {
                "canonical": "LOX 滤网堵塞 → 涡轮泵故障 → 助推器 ~90km 解体",
                "kw": [
                    "lox滤网",
                    "lox filter",
                    "涡轮泵",
                    "助推器爆炸",
                    "助推器解体",
                    "booster explod",
                ],
            },
            {
                "canonical": "飞船排气泄漏起火 AFSS 触发",
                "kw": ["飞船泄漏", "排气泄漏", "起火", "afss触发", "ship leak"],
            },
        ],
        "confirmed_wrong": [],
    },
    "IFT_3": {
        "date": "2024-03-14",
        "block": "Block 1",
        "progress": [
            {
                "canonical": "首次达近轨道速度 ~7.5km/s（亚轨道）",
                "kw": ["7.5km/s", "近轨道速度", "亚轨道速度", "轨道速度"],
            },
            {
                "canonical": "推进剂转移 demo (NASA Tipping Point LOX 转移)",
                "kw": ["推进剂转移", "tipping point", "lox转移", "propellant transfer"],
            },
            {"canonical": "Pez 舱门测试", "kw": ["pez", "pez舱门", "payload door"]},
        ],
        "failures": [
            {
                "canonical": "助推器 13 台中 6 台关机, 7 台中 2 台达主级 ~462m 失控",
                "kw": [
                    "6台关机",
                    "462m",
                    "助推器失控",
                    "boost-back failure",
                    "6 engines out",
                ],
            },
            {
                "canonical": "飞船滚转阀堵塞 ~65km 失联",
                "kw": [
                    "滚转阀",
                    "姿态阀",
                    "65km",
                    "失联",
                    "ship lost contact",
                    "roll control",
                ],
            },
        ],
        "confirmed_wrong": [],
    },
    "IFT_4": {
        "date": "2024-06-06",
        "block": "Block 1",
        "progress": [
            {
                "canonical": "双级首次受控溅落 (FAA 无需调查)",
                "kw": [
                    "首次受控溅落",
                    "受控溅落",
                    "soft splashdown",
                    "双级溅落",
                    "controlled splashdown",
                ],
            },
        ],
        "failures": [
            {
                "canonical": "1 台 Raptor 早期关机（1/33，不影响任务）",
                "kw": ["1台raptor", "1 engine out", "1台早期关机", "1/33关机"],
            },
            {
                "canonical": "热防护瓦/前襟翼严重受损",
                "kw": [
                    "热防护瓦",
                    "前襟翼",
                    "前鳍",
                    "瓦脱落",
                    "tile loss",
                    "flap damage",
                ],
            },
            {
                "canonical": "溅落偏差 ~6km (3.7 mi)",
                "kw": ["6km", "3.7mi", "splash偏差", "溅落偏差"],
            },
        ],
        "confirmed_wrong": [],
    },
    "IFT_5": {
        "date": "2024-10-13",
        "block": "Block 1",
        "progress": [
            {
                "canonical": "首次 Mechazilla 捕获 B12（航天史首次）",
                "kw": [
                    "mechazilla",
                    "捕获b12",
                    "捕获助推器",
                    "首次捕获",
                    "chopstick",
                    "航天史首次",
                    "mechazilla catch",
                ],
            },
            {"canonical": "飞船 212km 亚轨道", "kw": ["212km", "212 km亚轨道"]},
        ],
        "failures": [
            {
                "canonical": "飞船溅落后爆炸（计划内/属预期）",
                "kw": ["溅落后爆炸", "落水后侧翻", "splash然后爆炸", "ship explod"],
            },
        ],
        "confirmed_wrong": [],
    },
    "IFT_6": {
        "date": "2024-11-19",
        "block": "Block 1",
        "progress": [
            {
                "canonical": "Block 1 最后一飞 (S31/B13)",
                "kw": ["block 1最后", "s31", "b13", "block-1最后"],
            },
            {
                "canonical": "首次正近地点轨道 8×190km",
                "kw": ["8×190", "190km", "正近地点", "positive perigee"],
            },
            {
                "canonical": "首次在轨 Raptor 再点火 50×228km",
                "kw": [
                    "在轨重点火",
                    "在轨再点火",
                    "raptor restart",
                    "in-orbit reignition",
                    "228km",
                ],
            },
            {
                "canonical": "首次日照条件再入溅落",
                "kw": ["日照", "daylight reentry", "白天再入"],
            },
        ],
        "failures": [
            {
                "canonical": "天线受损 → 通信丢失 → B13 捕获中止",
                "kw": [
                    "天线受损",
                    "通信丢失",
                    "b13捕获中止",
                    "捕获中止",
                    "catch waved off",
                    "通信失败",
                ],
            },
        ],
        "confirmed_wrong": [
            {
                "canonical": "助推器未满足安全判定（实际是塔通信设施失败非助推器问题）",
                "kw": ["助推器未满足安全判定", "助推器未达安全"],
            },
        ],
    },
    "IFT_7": {
        "date": "2025-01-16",
        "block": "Block 2 / V2 (首飞)",
        "progress": [
            {
                "canonical": "Block 2 / V2 首飞 (B14 + S33)",
                "kw": [
                    "block 2首飞",
                    "v2首飞",
                    "b14+s33",
                    "b14 + s33",
                    "first block 2",
                ],
            },
            {
                "canonical": "第二次 Mechazilla 捕获助推器",
                "kw": [
                    "第二次捕获",
                    "second catch",
                    "二次捕获",
                    "捕获助推器",
                    "第二次mechazilla",
                ],
            },
            {
                "canonical": "10 颗 Starlink 模拟载荷",
                "kw": [
                    "10颗模拟",
                    "starlink simulator",
                    "10颗模拟载荷",
                    "10个starlink",
                    "10颗starlink",
                    "starlink模拟",
                ],
            },
        ],
        "failures": [
            {
                "canonical": "甲烷下降管谐振 → 应力 → 泄漏 → 燃烧 → 姿态丧失",
                "kw": [
                    "谐振",
                    "harmonic",
                    "甲烷下降管",
                    "下降管",
                    "downcomer",
                    "vibration",
                ],
            },
            {
                "canonical": "S33 在 Turks and Caicos 上空爆炸（AFSS 触发，碎片散落）",
                "kw": [
                    "turks and caicos",
                    "特克斯和凯科斯",
                    "s33爆炸",
                    "ship 33 explode",
                ],
            },
        ],
        "confirmed_wrong": [
            {
                "canonical": "甲烷下降管谐振 + 11 项纠正（11 项纠正未经核实）",
                "kw": ["11项纠正", "11项修复", "11 corrections"],
            },
        ],
    },
    "IFT_8": {
        "date": "2025-03-06",
        "block": "Block 2 / V2",
        "progress": [
            {
                "canonical": "B15 捕获（第三次 Mechazilla 捕获）",
                "kw": ["b15", "第三次捕获", "third catch", "三次捕获"],
            },
        ],
        "failures": [
            {
                "canonical": "S34 6 台 Raptor 中 4 台提前关机（中心 Raptor 闪光）",
                "kw": [
                    "s34",
                    "4/6",
                    "4台提前关机",
                    "中心raptor",
                    "center raptor",
                    "4 of 6",
                ],
            },
            {
                "canonical": "火炬点火器过热（GSE/ground side issue）",
                "kw": ["火炬点火器", "torch igniter", "点火器热"],
            },
        ],
        "confirmed_wrong": [
            {
                "canonical": "第二次 Block 2 因振动起火（实际根因是发动机硬件故障，非振动）",
                "kw": ["block 2因振动起火", "振动起火", "vibration ignite"],
            },
        ],
    },
    "IFT_9": {
        "date": "2025-05-27",
        "block": "Block 2 / V2",
        "progress": [
            {
                "canonical": "首次助推器复飞 B14（29/33 经飞行验证）",
                "kw": [
                    "首次复飞",
                    "助推器复飞",
                    "b14复飞",
                    "booster reuse",
                    "first reflown booster",
                    "29/33",
                ],
            },
            {
                "canonical": "S35 首次 V2 速度 + 33 台 Raptor 全部正常",
                "kw": ["s35", "v2速度", "33台正常", "33 raptor全正常"],
            },
            {
                "canonical": "首次可控分离方向（封堵通风口排气推动分离）",
                "kw": [
                    "可控分离",
                    "controlled separation",
                    "通风口排气",
                    "vent thrust",
                ],
            },
        ],
        "failures": [
            {
                "canonical": "助推器 1/13 关机 ~1km 解体",
                "kw": ["1/13关机", "1km解体", "boostback failure", "助推器解体"],
            },
            {
                "canonical": "飞船推进剂泄漏 → 姿态丧失 → 被动再入解体",
                "kw": [
                    "推进剂泄漏",
                    "姿态丧失",
                    "被动再入",
                    "passive reentry",
                    "loss of attitude",
                ],
            },
        ],
        "confirmed_wrong": [
            {
                "canonical": "极限应力测试解体（实际解体非预期）",
                "kw": ["极限应力测试", "intentional stress test", "stress test破坏"],
            },
        ],
    },
    "IFT_10": {
        "date": "2025-08-26",
        "block": "Block 2 / V2",
        "progress": [
            {
                "canonical": "8 颗 Starlink 模拟器首次部署",
                "kw": [
                    "8颗模拟",
                    "8颗部署",
                    "首次部署",
                    "first deployment",
                    "starlink deploy",
                ],
            },
            {
                "canonical": "第二次在轨 Raptor 重点火",
                "kw": [
                    "第二次在轨重点火",
                    "二次在轨重点火",
                    "second in-orbit reignition",
                    "在轨raptor重点火",
                    "在轨重点火",
                ],
            },
            {
                "canonical": "溅落精度 3 米",
                "kw": ["3米", "溅落精度", "3 meter", "splashdown accuracy"],
            },
        ],
        "failures": [
            {
                "canonical": "助推器 1 台 Raptor 关机墨西哥湾溅落",
                "kw": ["墨西哥湾", "gulf of mexico", "助推器溅落", "1台关机墨西哥"],
            },
            {
                "canonical": "飞船 ~90km 冷却堵塞 → 尾舱爆炸 → aft skirt 受损",
                "kw": [
                    "90km冷却",
                    "尾舱爆炸",
                    "aft skirt",
                    "aft受损",
                    "cooling blockage",
                ],
            },
        ],
        "confirmed_wrong": [
            {
                "canonical": "首次在轨重点火（实际 IFT-6 才是首次，IFT-10 为第二次）",
                "kw": [
                    "首次在轨重点火",
                    "first in-orbit reignition",
                    "first orbital relight",
                ],
            },
            {
                "canonical": "报告无失败/全部成功（遗漏发动机故障和尾舱爆炸）",
                "kw": ["无失败", "全部成功", "no failures", "完美成功"],
            },
        ],
    },
    "IFT_11": {
        "date": "2025-10-13",
        "block": "Block 2 / V2 (最后一飞)",
        "progress": [
            {
                "canonical": "Block 2 / V2 最后一飞 (B15-2 + S38)",
                "kw": ["b15-2", "s38", "block 2最后", "v2最后", "block-2最后"],
            },
            {
                "canonical": "第二次助推器复飞 B15（24/33 经验证）",
                "kw": ["第二次复飞", "second reflown", "二次复飞", "24/33"],
            },
            {
                "canonical": "所有主要目标达成",
                "kw": ["所有主要目标", "all primary objectives", "全部目标达成"],
            },
            {
                "canonical": "boost-back 12/13 台",
                "kw": ["12/13", "boost-back 12", "回推12台"],
            },
            {
                "canonical": "13 发动机着陆点火 + banking 机动",
                "kw": ["13发动机着陆", "banking机动", "13-engine landing"],
            },
        ],
        "failures": [
            {
                "canonical": "飞船溅落后爆炸（计划内/属预期）",
                "kw": ["溅落后爆炸", "落水后爆炸", "ship explod after splash"],
            },
        ],
        "confirmed_wrong": [],
    },
}


# ═══════════════════════════════════════════════════════════════════════════
# Value parsing helpers
# ═══════════════════════════════════════════════════════════════════════════

_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")


def _to_int(v) -> int | None:
    """Coerce v to a plausible IFT count integer."""
    if v is None:
        return None
    if isinstance(v, bool):
        return None
    if isinstance(v, int):
        return v
    if isinstance(v, float):
        return int(v)
    if isinstance(v, list):
        for item in v:
            n = _to_int(item)
            if n is not None:
                return n
        return None
    s = str(v)
    nums = [int(m.group()) for m in _NUM_RE.finditer(s) if "." not in m.group()]
    nums = [n for n in nums if 0 <= n <= 30]  # plausible IFT count range
    return nums[0] if nums else None


_HYPHEN_LIKE = "‐‑‒–—―−﹘﹣－"


def _normalize(s: str) -> str:
    """Normalize for kw matching: lowercase, NFKC (full-width → half-width),
    remove all whitespace, unify hyphen variants."""
    import unicodedata as _u

    s = _u.normalize("NFKC", str(s)).lower()
    for ch in _HYPHEN_LIKE:
        s = s.replace(ch, "-")
    s = re.sub(r"\s+", "", s)
    return s


def _kw_hit(claim: str, kws: list[str]) -> bool:
    """True if any normalized kw is a substring of normalized claim."""
    s = _normalize(claim)
    return any(_normalize(kw) in s for kw in kws)


# ═══════════════════════════════════════════════════════════════════════════
# Per-dimension scoring
# ═══════════════════════════════════════════════════════════════════════════


def score_A(value) -> tuple[int, str]:
    """A_flight_count: ±1 (accept ∈ {10, 11}; missing → -1)."""
    n = _to_int(value)
    if n is None:
        return -1, "未提供数字"
    if n in BENCHMARK["A_flight_count_accept"]:
        return 1, f"{n} ∈ {{10, 11}}"
    return -1, f"{n} ∉ {{10, 11}}"


def score_ift_subdim(
    facts_strings: list[str],
    correct_gt: list[dict],
    wrong_gt: list[dict],
) -> tuple[int, str]:
    """Score one (IFT × {progress | failures}) sub-dimension.

    Rules (strict):
      - any model claim matches a confirmed_wrong kw  → -1 (covers any correct)
      - else any model claim matches a correct GT kw  → +1
      - else                                          →  0
    """
    if not facts_strings:
        # called for a present-but-empty list (model mentioned IFT but didn't
        # give any claims under this category). Returns 0 here; the caller
        # is responsible for distinguishing this from "IFT not mentioned at all"
        # (which scores -1).
        return 0, "列表为空"

    wrong_hits: list[str] = []
    correct_hits: list[str] = []
    for s in facts_strings:
        for w in wrong_gt:
            if _kw_hit(s, w["kw"]):
                wrong_hits.append(f"{s!r}↔ wrong: {w['canonical']}")
                break
        for c in correct_gt:
            if _kw_hit(s, c["kw"]):
                correct_hits.append(f"{s!r}↔ ok: {c['canonical']}")
                break

    if wrong_hits:
        return -1, f"命中错误声明: {wrong_hits[0]}"
    if correct_hits:
        return 1, f"命中正确 GT: {correct_hits[0]}"
    return 0, "列了但都没匹配上 GT 也未触发错误声明"


# ═══════════════════════════════════════════════════════════════════════════
# Per-model scoring
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class ModelScore:
    model: str
    A_score: int
    A_reason: str
    ift_scores: dict[str, dict] = field(default_factory=dict)
    total: int = 0


def _ift_value_to_lists(value) -> tuple[list[str] | None, list[str] | None]:
    """From canonical.value extract (progress, failures) lists.
    Returns (None, None) if entity is missing / not_mentioned."""
    if value is None:
        return None, None
    if isinstance(value, dict):
        prog = value.get("progress")
        fail = value.get("failures")
        if prog is None and fail is None:
            return None, None
        return (
            [str(x) for x in prog] if isinstance(prog, list) else [],
            [str(x) for x in fail] if isinstance(fail, list) else [],
        )
    return None, None


def _score_one_model(model: str, payload: dict) -> ModelScore:
    entities = {e["id"]: e for e in payload.get("entities", [])}

    # A dimension
    a_ent = entities.get("A_flight_count", {})
    a_value = (a_ent.get("canonical") or {}).get("value")
    a_score, a_reason = score_A(a_value)

    ms = ModelScore(model=model, A_score=a_score, A_reason=a_reason)
    total = a_score

    # 11 IFT dimensions × {progress, failures}
    for n in range(1, 12):
        eid = f"IFT_{n}_facts"
        ift_key = f"IFT_{n}"
        gt = PER_IFT_GT[ift_key]

        ent = entities.get(eid, {})
        canonical = ent.get("canonical") or {}
        value = canonical.get("value")
        prog_list, fail_list = _ift_value_to_lists(value)

        # Decide "mentioned at all" — if both lists are None, model didn't mention this IFT
        if prog_list is None and fail_list is None:
            prog_score, prog_reason = -1, "模型未提及该 IFT (entity null/empty)"
            fail_score, fail_reason = -1, "模型未提及该 IFT (entity null/empty)"
        else:
            # progress sub-dim
            if not prog_list:
                # mentioned IFT but listed nothing under progress
                prog_score, prog_reason = -1, "提到 IFT 但未列任何 progress"
            else:
                prog_score, prog_reason = score_ift_subdim(
                    prog_list, gt["progress"], gt["confirmed_wrong"]
                )
            # failures sub-dim
            if not fail_list:
                fail_score, fail_reason = -1, "提到 IFT 但未列任何 failures"
            else:
                fail_score, fail_reason = score_ift_subdim(
                    fail_list, gt["failures"], gt["confirmed_wrong"]
                )

        ms.ift_scores[ift_key] = {
            "date": gt["date"],
            "progress_score": prog_score,
            "progress_reason": prog_reason,
            "progress_claims": prog_list or [],
            "failures_score": fail_score,
            "failures_reason": fail_reason,
            "failures_claims": fail_list or [],
        }
        total += prog_score + fail_score

    ms.total = total
    return ms


# ═══════════════════════════════════════════════════════════════════════════
# I/O
# ═══════════════════════════════════════════════════════════════════════════


def _write_model_score(out_dir: Path, ms: ModelScore) -> None:
    p = out_dir / ms.model / "score.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        json.dumps(
            {
                "query_id": QUERY_ID,
                "model": ms.model,
                "total_score": ms.total,
                "max_score": 23,
                "min_score": -23,
                "A_flight_count": {"score": ms.A_score, "reason": ms.A_reason},
                "ift_breakdown": ms.ift_scores,
                "benchmark_version": BENCHMARK["version"],
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )


def _write_outputs(out_dir: Path, results: list[ModelScore]) -> None:
    ranked = sorted(results, key=lambda r: (-r.total, r.model))

    # scores.json
    (out_dir / "scores.json").write_text(
        json.dumps(
            {
                "query_id": QUERY_ID,
                "max_score": 23,
                "min_score": -23,
                "benchmark_version": BENCHMARK["version"],
                "scoring_rule": (
                    "A flight count ±1, per-IFT progress ±1, per-IFT failures ±1; "
                    "strict rule: any wrong-claim → -1 overrides correct hits; "
                    "missing IFT → -1; mentioned-but-no-match → 0"
                ),
                "results": {
                    r.model: {
                        "total_score": r.total,
                        "A": r.A_score,
                        "per_ift": {
                            k: {
                                "prog": v["progress_score"],
                                "fail": v["failures_score"],
                            }
                            for k, v in r.ift_scores.items()
                        },
                    }
                    for r in results
                },
                "ranking": [
                    {"rank": i + 1, "model": r.model, "score": r.total}
                    for i, r in enumerate(ranked)
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    # ranking_report.md
    lines = [
        f"# Query {QUERY_ID} Ranking Report (v3, β2 strict)",
        "",
        f"> Benchmark v{BENCHMARK['version']} · Cutoff {BENCHMARK['cutoff_date']}  ",
        "> Scoring: A ±1 + 11 IFT × (progress ±1, failures ±1) → range [-23, +23]  ",
        "> Strict rule: any wrong-claim → -1; missing IFT → -1",
        "",
        "## 排名",
        "",
        "| Rank | Model | Total | A |"
        + "".join(f" IFT-{n}p IFT-{n}f |" for n in range(1, 12)),
        "|---:|---|---:|---:|" + "---:|---:|" * 11,
    ]
    for i, r in enumerate(ranked, 1):
        cells = [f"{r.A_score:+d}"]
        for n in range(1, 12):
            sc = r.ift_scores.get(f"IFT_{n}", {})
            cells.append(f"{sc.get('progress_score', 0):+d}")
            cells.append(f"{sc.get('failures_score', 0):+d}")
        lines.append(
            f"| {i} | {r.model} | {r.total:+d}/23 | " + " | ".join(cells) + " |"
        )
    lines.append("")

    # Per-model detail
    lines.append("## 每模型 IFT 明细")
    lines.append("")
    for r in ranked:
        lines.append(f"### {r.model} (total={r.total:+d})")
        lines.append(f"- **A flight count**: {r.A_score:+d} — {r.A_reason}")
        for n in range(1, 12):
            ift_key = f"IFT_{n}"
            sc = r.ift_scores.get(ift_key, {})
            ps = sc.get("progress_score", 0)
            fs = sc.get("failures_score", 0)
            pr = sc.get("progress_reason", "")
            fr = sc.get("failures_reason", "")
            lines.append(
                f"- **IFT-{n}** ({sc.get('date', '?')}): "
                f"progress {ps:+d} ({pr}); failures {fs:+d} ({fr})"
            )
        lines.append("")

    (out_dir / "ranking_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


# ═══════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════


def main() -> None:
    ap = argparse.ArgumentParser(description=f"Query {QUERY_ID} auto-scorer (v3)")
    ap.add_argument("--models", nargs="+", required=True)
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

    results = [_score_one_model(m, p) for m, p in all_results.items()]
    for r in results:
        _write_model_score(out_dir, r)
    _write_outputs(out_dir, results)

    print("\n" + "─" * 60)
    print(f"Query {QUERY_ID} scoring done.")
    for r in sorted(results, key=lambda x: (-x.total, x.model)):
        mark = "✅" if r.total >= 15 else ("🟡" if r.total > 0 else "❌")
        print(f"  {mark} {r.model:28s} {r.total:+d}/23")


if __name__ == "__main__":
    main()
