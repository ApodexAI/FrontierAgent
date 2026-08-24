"""
Query 14 — Heads of state who held citizenship of two or more countries
during their term in office.


QUERY_ID = 14 (matches folder name and queries/verifiable.json id).

Scoring rule (open-set, single dimension):
  ✅ baseline 命中 →  +1.0
  ⚠️ baseline 命中 →   0.0  (边界条目，如确认是双重国籍但属灰色定义)
  ❌ baseline 命中 →  -1.0  (基准里就是错答案，如法国总统兼安道尔大公)
  __HALLUCINATION__ → -1.0  (Stage C 验证为虚构)
  null + unresolved →  0.0  (开放集合外，未验证)

Pipeline:
  Stage 1  Extraction (v2 multi-LLM voting)
  Stage 2  Alignment (3 aligners + judge with Q14 FORCE_NULL rules)
  Stage 3a Export null_review.json
  Stage 3b Apply null_resolutions.json (after Stage C agent runs)
  Stage 4  Scoring → scores.json + ranking_report.md
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
# Baseline — seed entries (open-set; Stage C agent extends via baseline_add)
# ═══════════════════════════════════════════════════════════════════════════

SNAPSHOT_DATE = "2026-05-01"
SCORING_MODE = "Q14-open-set"

# Tuple shape: (id, description, judgment, score, [kw, ...])
DIMS = [
    (
        "D1",
        "Alberto Fujimori / 秘鲁总统 1990-2000 / 秘鲁+日本国籍（父母通过 koseki "
        "在驻秘鲁日本大使馆保留日本籍；日本政府 2000 年正式确认其日本国籍并据此拒绝引渡）",
        "✅",
        1.0,
        [
            "fujimori",
            "藤森",
            "alberto",
            "alberto fujimori",
            "藤森谦也",
            "秘鲁",
            "peru",
            "日本",
            "japan",
            "koseki",
            "戸籍",
            "户籍",
        ],
    ),
    (
        "D2",
        "Mohamed Abdullahi Mohamed (Farmaajo) / 索马里总统 2017-2022 / "
        "美国+索马里双国籍（任内持双国籍，2019-08 才放弃美国国籍）",
        "✅",
        1.0,
        [
            "farmaajo",
            "法马约",
            "mohamed abdullahi",
            "abdullahi mohamed",
            "索马里",
            "somalia",
            "美国",
            "us",
            "u.s.",
            "美籍",
            "buffalo",
            "renounce",
            "放弃美籍",
        ],
    ),
    (
        "D3",
        "Pope Francis (Jorge Mario Bergoglio) / 罗马教宗（梵蒂冈城国元首）"
        "2013-2025 / 阿根廷+梵蒂冈（任内续签阿根廷护照 2014）",
        "✅",
        1.0,
        [
            "francis",
            "方济各",
            "fransisco",
            "贝戈格里奥",
            "bergoglio",
            "阿根廷",
            "argentina",
            "梵蒂冈",
            "vatican",
            "holy see",
            "教宗",
            "教皇",
            "pope",
        ],
    ),
    (
        "D4",
        "Pope Leo XIV (Robert Prevost) / 罗马教宗（梵蒂冈城国元首） 2025- / "
        "美国+秘鲁+梵蒂冈（首位美籍教宗，亦持秘鲁籍）",
        "✅",
        1.0,
        [
            "leo xiv",
            "leo 14",
            "利奥十四世",
            "良十四世",
            "prevost",
            "robert prevost",
            "美国",
            "u.s.",
            "us",
            "美籍",
            "梵蒂冈",
            "vatican",
            "教宗",
            "教皇",
            "pope",
        ],
    ),
    (
        "D5",
        "Pope Benedict XVI (Joseph Ratzinger) / 罗马教宗（梵蒂冈城国元首） "
        "2005-2013 / 德国+梵蒂冈",
        "✅",
        1.0,
        [
            "benedict xvi",
            "benedict 16",
            "本笃十六世",
            "拉辛格",
            "ratzinger",
            "joseph ratzinger",
            "德国",
            "germany",
            "梵蒂冈",
            "vatican",
            "教宗",
            "教皇",
            "pope",
        ],
    ),
    (
        "D6",
        "Pope John Paul II (Karol Wojtyła) / 罗马教宗（梵蒂冈城国元首） "
        "1978-2005 / 波兰+梵蒂冈",
        "✅",
        1.0,
        [
            "john paul ii",
            "john paul 2",
            "若望保禄二世",
            "若望保祿二世",
            "约翰保罗二世",
            "wojtyla",
            "wojtyła",
            "karol wojtyla",
            "波兰",
            "poland",
            "梵蒂冈",
            "vatican",
            "教宗",
            "教皇",
            "pope",
        ],
    ),
    (
        "D7",
        "法国总统兼任安道尔大公（co-prince of Andorra） / 此为'兼任他国职务'而非"
        "'持双重国籍'：法国总统从未自动获得安道尔国籍，安道尔 co-prince 是中世纪"
        "继承的礼仪职务。模型若把此当成双重国籍的肯定案例 → 错答 ❌",
        "❌",
        -1.0,
        [
            "andorra",
            "安道尔",
            "co-prince",
            "coprince",
            "大公",
            "亲王",
            "macron",
            "马克龙",
            "hollande",
            "奥朗德",
            "sarkozy",
            "萨科齐",
            "chirac",
            "希拉克",
            "mitterrand",
            "密特朗",
            "法国总统",
            "president of france",
            "french president",
        ],
    ),
    # web-search-verified 2026-05-01 — added from null_resolutions.json
    # 2026-05-01 audit: 法国法院 1983 年否认其法籍主张 (UPI Archives)，任内法籍 disputed → 改 ⚠️ 与 Senghor/Wade 灰度一致
    (
        "D8",
        "Jean-Bédel Bokassa / 中非共和国总统(1966-1976)及中非帝国皇帝(1976-1979) / 中非+法国双国籍（在法国军队服役期间被授予法国国籍，任内主张持双国籍；但法国 1983 法院否认其法籍主张）",
        "⚠️",
        0.0,
        [
            "bokassa",
            "博卡萨",
            "jean-bedel",
            "jean-bédel",
            "中非",
            "central african",
            "中非帝国",
            "central african empire",
            "法国",
            "france",
        ],
    ),
    # web-search-verified 2026-05-01 — added from null_resolutions.json
    (
        "D9",
        "Andry Rajoelina / 马达加斯加总统 2019-2025 / 马达加斯加+法国双国籍（2014年秘密入籍法国，任内持双国籍至2025年被剥夺马籍）",
        "✅",
        1.0,
        [
            "rajoelina",
            "拉乔利纳",
            "andry",
            "马达加斯加",
            "madagascar",
            "法国",
            "france",
            "2014",
        ],
    ),
    # web-search-verified 2026-05-01 — added from null_resolutions.json
    (
        "D10",
        "Armen Sarkissian / 亚美尼亚总统 2018-2022 / 亚美尼亚+圣基茨和尼维斯（任内秘密持投资入籍取得的圣基茨籍，2022年曝光后辞职）",
        "✅",
        1.0,
        [
            "sarkissian",
            "sarkisian",
            "萨尔基相",
            "armen",
            "亚美尼亚",
            "armenia",
            "圣基茨",
            "saint kitts",
            "st. kitts",
            "nevis",
            "investment citizenship",
        ],
    ),
    # web-search-verified 2026-05-01 — added from null_resolutions.json
    (
        "D11",
        "Barham Salih / 伊拉克总统 2018-2022 / 伊拉克+英国（2018-10当选时仍持英籍，2018-12才完成放弃，任职初期约两个半月内有重叠双国籍）",
        "✅",
        1.0,
        [
            "barham salih",
            "巴尔哈姆",
            "萨利赫",
            "salih",
            "伊拉克",
            "iraq",
            "英国",
            "uk",
            "british",
            "renounce",
        ],
    ),
    # web-search-verified 2026-05-01 — added from null_resolutions.json
    (
        "D12",
        "Abdul Latif Rashid / 伊拉克总统 2022-2026 / 伊拉克+英国（维基百科明确双重国籍，长居英国、当选时仍持英籍，按宪法应放弃但任职初期存在重叠）",
        "✅",
        1.0,
        [
            "abdul latif rashid",
            "拉希德",
            "rashid",
            "伊拉克",
            "iraq",
            "英国",
            "uk",
            "british",
        ],
    ),
    # web-search-verified 2026-05-01 — added from null_resolutions.json
    (
        "D13",
        "Mauricio Macri / 阿根廷总统 2015-2019 / 阿根廷+意大利双国籍（通过意大利血统取得意籍，整个任期持续持有）",
        "✅",
        1.0,
        [
            "macri",
            "马克里",
            "mauricio",
            "阿根廷",
            "argentina",
            "意大利",
            "italy",
            "italian",
            "ius sanguinis",
        ],
    ),
    # web-search-verified 2026-05-01 — added from null_resolutions.json
    (
        "D14",
        "Javier Milei / 阿根廷总统 2023- / 阿根廷+意大利双国籍（2024年12月任内被意大利政府基于血统原则授予意大利国籍）",
        "✅",
        1.0,
        [
            "milei",
            "米莱",
            "javier",
            "阿根廷",
            "argentina",
            "意大利",
            "italy",
            "italian",
            "meloni",
            "ius sanguinis",
        ],
    ),
    # web-search-verified 2026-05-01 — added from null_resolutions.json
    (
        "D15",
        "Gustavo Petro / 哥伦比亚总统 2022- / 哥伦比亚+意大利双国籍（通过意大利血统取得，任内持续持有）",
        "✅",
        1.0,
        [
            "petro",
            "佩特罗",
            "gustavo",
            "哥伦比亚",
            "colombia",
            "意大利",
            "italy",
            "italian",
        ],
    ),
    # web-search-verified 2026-05-01 — added from null_resolutions.json
    (
        "D16",
        "Alejandro Giammattei / 危地马拉总统 2020-2024 / 危地马拉+意大利双国籍（通过意大利血统取得，任内持续持有）",
        "✅",
        1.0,
        [
            "giammattei",
            "贾马特",
            "贾马泰",
            "alejandro",
            "危地马拉",
            "guatemala",
            "意大利",
            "italy",
            "italian",
            "ius sanguinis",
        ],
    ),
    # web-search-verified 2026-05-01 — added from null_resolutions.json
    (
        "D17",
        "Daniel Noboa / 厄瓜多尔总统 2023- / 厄瓜多尔+美国双国籍（出生于迈阿密，自动获得美国出生地国籍，任内持续）",
        "✅",
        1.0,
        [
            "noboa",
            "诺沃亚",
            "daniel",
            "厄瓜多尔",
            "ecuador",
            "美国",
            "us",
            "u.s.",
            "miami",
            "迈阿密",
        ],
    ),
    # web-search-verified 2026-05-01 — added from null_resolutions.json
    (
        "D18",
        "Maia Sandu / 摩尔多瓦总统 2020- / 摩尔多瓦+罗马尼亚双国籍（任内持续公开持双国籍）",
        "✅",
        1.0,
        ["sandu", "桑杜", "maia", "摩尔多瓦", "moldova", "罗马尼亚", "romania"],
    ),
    # web-search-verified 2026-05-01 — added from null_resolutions.json
    (
        "D19",
        "Yevgeny Shevchuk / 德涅斯特河沿岸总统 2011-2016 / 德涅斯特河沿岸+俄罗斯（任内本人确认持双国籍，但德涅斯特河沿岸是未获普遍承认国家）",
        "⚠️",
        0.0,
        [
            "shevchuk",
            "舍夫丘克",
            "yevgeny",
            "transnistria",
            "德涅斯特河沿岸",
            "俄罗斯",
            "russia",
        ],
    ),
    # web-search-verified 2026-05-01 — added from null_resolutions.json
    (
        "D20",
        "Léopold Sédar Senghor / 塞内加尔首任总统 1960-1980 / 塞内加尔+法国双国籍（1933年获法籍未放弃，但塞内加尔1992年才禁止双重国籍，任期内法律状态有争议）",
        "⚠️",
        0.0,
        [
            "senghor",
            "桑戈尔",
            "léopold",
            "leopold",
            "塞内加尔",
            "senegal",
            "法国",
            "france",
            "1933",
        ],
    ),
    # web-search-verified 2026-05-01 — added from null_resolutions.json
    (
        "D21",
        "Abdoulaye Wade / 塞内加尔总统 2000-2012 / 塞内加尔+法国双国籍（留法期间获法籍并未放弃，任内持续持有）",
        "⚠️",
        0.0,
        ["wade", "瓦德", "abdoulaye", "塞内加尔", "senegal", "法国", "france"],
    ),
    # web-search-verified 2026-05-01 — added from null_resolutions.json
    (
        "D23",
        "Éamon de Valera / 爱尔兰总统 1959-1973 / 爱尔兰+美国（出生地） — 1882 年生于纽约市按美国宪法第十四修正案享有出生地国籍。维基 infobox 列其 Citizenship: Ireland + United States (birthright)。其美籍是否在任内被正式放弃或主动取消尚未确证 → ⚠️ 边界",
        "⚠️",
        0.0,
        [
            "de valera",
            "éamon de valera",
            "eamon de valera",
            "德瓦莱拉",
            "德·瓦莱拉",
            "埃蒙",
            "爱尔兰",
            "ireland",
            "美国",
            "usa",
            "birthright",
            "出生地国籍",
            "new york",
            "1882",
        ],
    ),
    # web-search-verified 2026-05-01 — added from null_resolutions.json
    (
        "D22",
        "Ricardo Martinelli / 巴拿马总统 2009-2014 / 巴拿马+意大利双国籍（通过意大利血统取得，任内持续持有）",
        "✅",
        1.0,
        [
            "martinelli",
            "马蒂内利",
            "ricardo",
            "巴拿马",
            "panama",
            "意大利",
            "italy",
            "italian",
        ],
    ),
]

DIM_MAP: dict = {
    d[0]: {"id": d[0], "name": d[1], "judgment": d[2], "score": d[3], "kw": d[4]}
    for d in DIMS
}
MAX_SCORE = sum(d[3] for d in DIMS if d[3] > 0)


# ═══════════════════════════════════════════════════════════════════════════
# DIMS → alignment baseline spec
# ═══════════════════════════════════════════════════════════════════════════


def _derive_match_fields(description: str) -> dict:
    """Parse term years and office country from description text.

    Returned keys must align with claim field names in extract.py
    VALUE_SCHEMA:  term / office_country.
    """
    fields: dict = {}
    # Extract a 4-digit year range like "1990-2000" or "2017-2022"
    m = _re.search(r"(\d{4})\s*-\s*(\d{4})", description)
    if m:
        fields["term"] = f"{m.group(1)}-{m.group(2)}"
    elif mm := _re.search(r"(\d{4})\s*-\s*$", description):
        fields["term"] = f"{mm.group(1)}-"
    # Office country / political entity hints
    for country, alias in [
        ("秘鲁", "Peru"),
        ("索马里", "Somalia"),
        ("梵蒂冈", "Vatican"),
        ("法国", "France"),
    ]:
        if country in description or alias.lower() in description.lower():
            fields["office_country"] = country
            break
    return fields


def build_baselines() -> list[dict]:
    return [
        {
            "id": d_id,
            "description": f"{desc} [基准: {judgment}]",
            "match_fields": _derive_match_fields(desc),
            "kw": kw,
            "judgment": judgment,
            "score": score,
        }
        for d_id, desc, judgment, score, kw in DIMS
    ]


# ═══════════════════════════════════════════════════════════════════════════
# Load raw claims from extraction.json
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


# ═══════════════════════════════════════════════════════════════════════════
# Scoring (open-set, single dimension)
# ═══════════════════════════════════════════════════════════════════════════


def score_aligned(aligned_by_model: dict[str, list[dict]]) -> dict[str, dict]:
    """Per-dim scoring; dedup; null + needs_review → 0; HALLUCINATION → -1."""
    all_scores: dict[str, dict] = {}
    for model_name, claims in aligned_by_model.items():
        seen_dims: set = set()
        scored: list[dict] = []
        unverified: list[dict] = []
        total = 0.0

        for c in claims:
            raw = c.get("raw", {}) or {}
            cid = c.get("canonical_id")
            conf = c.get("alignment_confidence", "medium")
            reason = c.get("alignment_reasoning", "")
            judge_invoked = c.get("judge_invoked", False)
            person_label = (
                f"{raw.get('name', '?')} ({raw.get('office_country', '?')} "
                f"{raw.get('term', '?')})"
            )

            if conf == "needs_review":
                unverified.append(
                    {
                        "person": person_label,
                        "raw_claim": raw,
                        "canonical_id_tentative": cid,
                        "reason": f"needs_review: {reason}",
                        "score": 0.0,
                    }
                )
                continue

            if cid == "__HALLUCINATION__":
                scored.append(
                    {
                        "id": "__HALLUCINATION__",
                        "name": "(Stage C 判定虚构)",
                        "person": person_label,
                        "score": -1.0,
                        "judgment": "❌",
                        "confidence": conf,
                        "judge_invoked": judge_invoked,
                        "reason": reason,
                    }
                )
                total += -1.0
                continue

            if cid and cid in DIM_MAP:
                if cid in seen_dims:
                    continue
                seen_dims.add(cid)
                d = DIM_MAP[cid]
                scored.append(
                    {
                        "id": cid,
                        "name": d["name"],
                        "person": person_label,
                        "score": d["score"],
                        "judgment": d["judgment"],
                        "confidence": conf,
                        "judge_invoked": judge_invoked,
                        "reason": reason or f"对齐到 {cid}（基准 {d['judgment']}）",
                    }
                )
                total += d["score"]
            else:
                unverified.append(
                    {
                        "person": person_label,
                        "raw_claim": raw,
                        "canonical_id_tentative": cid,
                        "reason": reason or "Stage C unresolved / 未走 Stage C",
                        "score": 0.0,
                    }
                )

        n_answered = len(scored)
        all_scores[model_name] = {
            "total_score": total,
            "max_score": MAX_SCORE,
            "score_rate": round(total / n_answered, 4) if n_answered else 0.0,
            "total_rate": round(total / MAX_SCORE, 4) if MAX_SCORE else 0.0,
            "dimensions_answered": n_answered,
            "per_dimension": sorted(scored, key=lambda x: x.get("id", "")),
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
            unverified_all.append(
                {"model": m, **uv, "action_needed": "需 Stage C / 人工核实"}
            )
    return {
        "query_id": QUERY_ID,
        "scoring_mode": SCORING_MODE,
        "snapshot_date": SNAPSHOT_DATE,
        "max_score": MAX_SCORE,
        "extraction_pipeline": "v2 (primary=claude-sonnet-4, secondary=gpt-5, analyzer=claude-opus-4.6)",
        "scoring_rule": (
            "✅ baseline 命中 +1.0；⚠️ baseline 命中 0；"
            "❌ baseline 命中 -1.0；__HALLUCINATION__ -1.0；"
            "null + unresolved 0"
        ),
        "results": all_scores,
        "ranking": [
            {
                "rank": i + 1,
                "model": m,
                "score": s["total_score"],
                "answered": s["dimensions_answered"],
                "unverified": len(s.get("unverified_claims", [])),
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
        "# Query 14 排名报告（任期内持双重国籍的国家元首）",
        "",
        f"> Snapshot: {SNAPSHOT_DATE}  Mode: {SCORING_MODE}",
        f"> 基准条目数：{len(DIMS)}（"
        f"✅ {sum(1 for d in DIMS if d[2] == '✅')} / "
        f"⚠️ {sum(1 for d in DIMS if d[2] == '⚠️')} / "
        f"❌ {sum(1 for d in DIMS if d[2] == '❌')}）",
        f"> MAX_SCORE = {MAX_SCORE}  评分：✅+1 / ⚠️ 0 / ❌-1 / null 0",
        "",
        "| Rank | Model | Score | Answered | Unverified |",
        "|---:|---|---:|---:|---:|",
    ]
    for i, (m, s) in enumerate(ranked, 1):
        lines.append(
            f"| {i} | {m} | {s['total_score']:+.1f}/{MAX_SCORE:.1f} "
            f"| {s['dimensions_answered']} "
            f"| {len(s.get('unverified_claims', []))} |"
        )
    lines.append("")
    lines.append("> Unverified 见 `null_review.json` → 派 Stage C agent 处理。")
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
    ap = argparse.ArgumentParser(description="Query 14 auto-scorer")
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

    # Stage 1: Extraction
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

    # Stage 2: Alignment
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

    # Stage 3a: Export null claims for Stage C web verification
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

    # Stage 3b: Apply null_resolutions if present
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
            f"\n    → Stage C agent: read {resolutions_path.parent}/null_review.json"
            f"\n      逐条 web 验证 → 产出 {resolutions_path.name}"
            f"\n      然后重跑：python3 {Path(__file__).name} --skip-extract --skip-align --models …"
        )

    # Stage 4: Scoring
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
            f"  {i}. {m:28s} {s['total_score']:+5.1f}/{MAX_SCORE:.1f}"
            f"  answered={s['dimensions_answered']}"
            f"  unverified={len(s.get('unverified_claims', []))}"
        )


if __name__ == "__main__":
    main()
