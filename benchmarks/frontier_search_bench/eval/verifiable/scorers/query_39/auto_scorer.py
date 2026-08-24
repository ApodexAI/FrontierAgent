"""Query 39 auto-scorer — 全球"气候最宜居"首都筛选 + 平均距赤道距离。

Composite scoring (closed set, multi-dim):
  Part A — 城市命中（每首都 +1 / partial 0 / 非命中 -1）；max = 5（5 个 baseline）
  Part B — 经纬度（每命中 baseline 城市，模型若给出**具体经纬度数值**则 +1）；max = 5
  Part C — 整体平均距赤道（在 ±10% 容差内 +1，否则 0）；max = 1

Total max = 11. Total score 可为负。

Pipeline:
  1. Extraction — two entities (qualifying_capitals 列表 + summary_avg_distance 单条).
  2. Alignment — only E1 (qualifying_capitals) goes through align_claims.
                 E2 (summary_avg_distance) is read directly from extraction.json.
  3. Null verification — Stage C agent produces:
       - null_resolutions.json  (E1 nulls; standard schema)
  4. Scoring — three independent score functions; aggregate per model.

Baseline 5 城（基于 2024 单年实测数据，多 agent 交叉验证 + review agent 复核）：
  - Bogotá       (4.71°N,  -74.07°W)  : 15.2°C / 885mm  (RMCAB 2024 城市平均)
  - Montevideo   (-34.90°, -56.16°W)  : 17.3°C / 938mm  (INUMET 2024)
  - Washington D.C. (38.91°N, -77.04°W): 16.6°C / 954mm  (NOAA NCEI 2024 DCA)
  - Zagreb       (45.81°N, 15.98°E)   : 14.3°C / 1092mm (Tutiempo NOAA-ISD 2024)
  - Wellington   (-41.29°, 174.78°E)  : 13.8°C / 974mm  (NIWA Annual 2024)
官方答案：5 城；平均距赤道 ≈ 3,687 km。
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
# Part A baseline — 5 个气候最宜居首都 (DIMS)
#
# Cross-verified 2026-05-04 against 2024 single-year actual measurements:
#   D1 Bogotá    : RMCAB 2024 informe anual (city avg) + tutiempo SKBO
#   D2 Montevideo: INUMET 2024 boletín climático anual
#   D3 Washington: NOAA NCEI 2024 (DCA station, dcatemps.pdf + dcaprecip.pdf)
#   D4 Zagreb    : tutiempo NOAA-ISD station 142400 (Maksimir) 2024 monthly
#   D5 Wellington: NIWA Annual Climate Summary 2024 (Kelburn / Airport)
#
# 边缘候选（PARTIAL_CAPITALS, ⚠️ 0 分不加不扣）：
#   Mexico City  : 2024 实测 19.5°C / 765mm — 双重不符；多数 LLM 用 normals 误纳 → ⚠️
#   Tirana       : normals 14.8-15.2°C / 1136-1262mm — 降水边缘
#   Antananarivo : normals 17.9°C / 1170-1456mm — 边缘
#   Maseru       : normals 14.6°C / 641-926mm — 降水来源分歧
#   Harare       : normals 17.95°C / 853-1230mm — 双重边缘
#   Sucre, Buenos Aires, Asmara, Mbabane, Pretoria : 各项指标紧贴边界
# ═══════════════════════════════════════════════════════════════════════════

SNAPSHOT_DATE = "2026-05-04"

DIMS = [
    # ─── ✅ 5 个核心 baseline 首都 (+1 each) ───
    (
        "D1",
        "Bogotá / Colombia / lat=4.71°N, lon=-74.07°W / 距赤道 ≈ 524 km / "
        "2024 实测 (RMCAB 城市平均) 年均气温 15.2°C，年降水 ~885 mm；"
        "El Dorado 站 14.7°C (tutiempo NOAA-ISD)；"
        "符合 13-17°C 且 800-1200mm 双重条件",
        "✅",
        1.0,
        ["bogota", "bogotá", "波哥大", "colombia", "哥伦比亚", "el dorado", "rmcab"],
    ),
    (
        "D2",
        "Montevideo / Uruguay / lat=-34.90°, lon=-56.16°W / 距赤道 ≈ 3,885 km / "
        "2024 实测 (INUMET Boletín Climático Anual) Carrasco 站 ~17.3°C "
        "(climatology 16.7 + anomaly +0.6), 年降水 937.9 mm；"
        "温度紧贴 17°C 上限但在测量不确定性内；降水稳居 800-1200mm 区间",
        "✅",
        1.0,
        [
            "montevideo",
            "蒙得维的亚",
            "蒙特维多",
            "uruguay",
            "乌拉圭",
            "carrasco",
            "inumet",
        ],
    ),
    (
        "D3",
        "Washington D.C. / United States / lat=38.91°N, lon=-77.04°W / 距赤道 ≈ 4,331 km / "
        "2024 实测 (NOAA NCEI / NWS LWX, DCA Reagan National Airport) "
        "年均 61.8°F = 16.6°C, 年降水 37.57 in = 954 mm；"
        "2024 是 DCA 自 1871 以来最暖年；完美命中 13-17°C 且 800-1200mm",
        "✅",
        1.0,
        [
            "washington",
            "washington d.c.",
            "washington dc",
            "华盛顿",
            "华盛顿特区",
            "united states",
            "美国",
            "usa",
            "noaa",
            "dca",
            "reagan national",
        ],
    ),
    (
        "D4",
        "Zagreb / Croatia / lat=45.81°N, lon=15.98°E / 距赤道 ≈ 5,099 km / "
        "2024 实测 (Tutiempo NOAA-ISD station 142400 Maksimir) "
        "年均 14.3°C, 年降水 ~1092 mm；"
        "2024 是历史性暖年（1991-2020 normal 仅 11.9°C，2022 纪录 13.9°C）；"
        "完美命中 13-17°C 且 800-1200mm",
        "✅",
        1.0,
        [
            "zagreb",
            "萨格勒布",
            "croatia",
            "克罗地亚",
            "maksimir",
            "dhmz",
            "grič",
        ],
    ),
    (
        "D5",
        "Wellington / New Zealand / lat=-41.29°, lon=174.78°E / 距赤道 ≈ 4,596 km / "
        "2024 实测 (NIWA Annual Climate Summary 2024) "
        "Kelburn 站 13.6°C / 1203 mm（降水仅微超 3mm，3 天数据缺失）；"
        "Wellington Airport 14.3°C / 898 mm（完美命中区间）；"
        "综合判定符合 13-17°C 且 800-1200mm",
        "✅",
        1.0,
        [
            "wellington",
            "惠灵顿",
            "new zealand",
            "新西兰",
            "kelburn",
            "wellington airport",
            "niwa",
        ],
    ),
    # ─── ⚠️ 边缘候选 (0 分不加不扣，避免误判 ❌) ───
    (
        "D6",
        "Mexico City / Mexico / lat=19.43°N, lon=-99.13°W / "
        "2024 实测 (SMN/CONAGUA 联邦实体公报) 年均 19.5°C, 年降水 764.7 mm — "
        "**双重不符**：温度 +2.5°C 超上限 (历史性热浪 + 多年干旱), 降水低于 800 下限 35 mm。"
        "若使用 1991-2020 climate normals (~16°C / ~850-1058mm) 则会被误纳为符合，"
        "故大量 LLM 错列。判 ⚠️ 不加不扣（区分'实测不符'与'幻觉'）",
        "⚠️",
        0.0,
        [
            "mexico city",
            "墨西哥城",
            "ciudad de méxico",
            "ciudad de mexico",
            "cdmx",
            "tacubaya",
            "smn",
            "conagua",
        ],
    ),
    (
        "D7",
        "Tirana / Albania / lat=41.33°N / 1961-1990 normal 15.2°C / 1262-1345 mm — "
        "温度命中但降水超 1200 上限 5-12%；2024 单年实测未公开，按 normals 排除。"
        "但因紧贴边界判 ⚠️",
        "⚠️",
        0.0,
        ["tirana", "地拉那", "albania", "阿尔巴尼亚"],
    ),
    (
        "D8",
        "Antananarivo / Madagascar / lat=-18.88° / 17.9-19.6°C / 1170-1456 mm — "
        "温度多源分歧，降水多源超上限；判 ⚠️",
        "⚠️",
        0.0,
        ["antananarivo", "塔那那利佛", "madagascar", "马达加斯加"],
    ),
    (
        "D9",
        "Maseru / Lesotho / lat=-29.31° / 14.6°C / 641-926 mm — "
        "温度命中；降水来源分歧（Wikipedia 1981-2010 给 641mm 不符；"
        "climate-data.org 给 896mm 符合）；判 ⚠️",
        "⚠️",
        0.0,
        ["maseru", "马塞卢", "lesotho", "莱索托"],
    ),
    (
        "D10",
        "Harare / Zimbabwe / lat=-17.83° / 17.9-18.6°C / 824-1230 mm — "
        "温度普遍超 17°C 上限 0.95-1.6°C；降水来源分歧；判 ⚠️",
        "⚠️",
        0.0,
        ["harare", "哈拉雷", "zimbabwe", "津巴布韦"],
    ),
    (
        "D11",
        "Sucre / Bolivia / lat=-19.05° / 12.6°C / 1086 mm — "
        "温度紧贴 13°C 下限（差 0.4°C），降水符合；判 ⚠️",
        "⚠️",
        0.0,
        ["sucre", "苏克雷", "bolivia", "玻利维亚"],
    ),
    (
        "D12",
        "Buenos Aires / Argentina / lat=-34.61° / 17.3°C / 1112-1260 mm — "
        "温度紧贴 17°C 上限，降水部分超上限；判 ⚠️",
        "⚠️",
        0.0,
        ["buenos aires", "布宜诺斯艾利斯", "argentina", "阿根廷"],
    ),
    (
        "D13",
        "Asmara / Eritrea / lat=15.32°N / 17.1°C / 500-859 mm — "
        "温度紧贴上限，降水多数 <800；判 ⚠️",
        "⚠️",
        0.0,
        ["asmara", "阿斯马拉", "eritrea", "厄立特里亚"],
    ),
    (
        "D14",
        "Mbabane / Eswatini / lat=-26.31° / 16.0°C / 1304-1350 mm — "
        "温度命中，降水超 1200 上限；判 ⚠️",
        "⚠️",
        0.0,
        ["mbabane", "姆巴巴内", "eswatini", "斯威士兰", "swaziland"],
    ),
    # web-search-verified 2026-05-04 — added from null_resolutions.json
    (
        "D15",
        "Addis Ababa / Ethiopia / lat=9.03°N, lon=38.75°E / 距赤道 ≈ 1,005 km / 1991-2020 normals 15.6°C / ~1100-1800mm; 2024 单年实测 NOAA-ISD station 634500 (Bole) 数据缺失无法计算年均; 模型声称 16.4°C/1168mm 与 climate-data.org 长期均值 (15.6°C/1089-1801mm) 大致相符但非 2024 单年；降水多源差异极大 (1089-1874mm)；高海拔 (~2355m) 热带高原确实接近 13-17°C 范围；判 ⚠️：实体真实但 2024 单年实测不可获取，长期均值降水可能超 1200mm 上限",
        "⚠️",
        0.0,
        [
            "addis ababa",
            "addis abeba",
            "亚的斯亚贝巴",
            "ethiopia",
            "ethiopian",
            "埃塞俄比亚",
            "bole",
        ],
    ),
    # web-search-verified 2026-05-04 — added from null_resolutions.json
    (
        "D16",
        "Bern / Switzerland / lat=46.95°N, lon=7.45°E / 距赤道 ≈ 5,229 km / 1991-2020 normals 8.8-9.4°C / 1025-1311mm; 2024 是瑞士第三暖年 (anomaly +3.3°C vs 1871-1900 pre-industrial baseline ≈ +1.5°C vs 1991-2020); 即使最暖年估算值 ~10.5-11°C, 远低于 13°C 下限; 模型声称 13.5°C 高估约 2.5-4°C; 判 ⚠️: 实体真实 (瑞士首都) 但温度严重不符 13-17°C 区间",
        "⚠️",
        0.0,
        ["bern", "berne", "伯尔尼", "switzerland", "瑞士", "belp", "meteoswiss"],
    ),
    # web-search-verified 2026-05-04 — added from null_resolutions.json
    (
        "D17",
        "Sarajevo / Bosnia and Herzegovina / lat=43.85°N, lon=18.36°E / 距赤道 ≈ 4,881 km / FHMZBIH 长期年均 9.5°C / 919mm (normals); 2024 夏季 (JJA) 异常 +3.2°C vs 1991-2020 baseline，但年均整体异常约 +1.5-2°C; 估算 2024 ~11-11.5°C, 低于 13°C 下限; 模型声称 13.2°C 高估约 1.7-2.2°C; 判 ⚠️: 实体真实 (波黑首都) 但温度不符 13-17°C 区间",
        "⚠️",
        0.0,
        [
            "sarajevo",
            "萨拉热窝",
            "bosnia",
            "波黑",
            "波斯尼亚",
            "herzegovina",
            "黑塞哥维那",
            "fhmzbih",
            "bjelave",
        ],
    ),
    # web-search-verified 2026-05-04 — added from null_resolutions.json
    (
        "D18",
        "Dushanbe / Tajikistan / lat=38.56°N, lon=68.78°E / 距赤道 ≈ 4,290 km / 长期均值 12.1°C / 620-926mm (来源差异大); 2024 单年实测 (Tutiempo NOAA-ISD station 388360) 16.6°C — 落入 13-17°C 区间但接近上限; 2024 年降水数据缺失 (tutiempo 显示 0); 长期均值降水 620-926mm 多数源 < 800mm 下限; 模型声称 15.5°C/937mm (基于 ERA5) 温度合理但降水可能不达标; 判 ⚠️: 实体真实 (塔吉克斯坦首都), 2024 温度合规但降水 2024 数据无法证实且长期值多 <800mm",
        "⚠️",
        0.0,
        ["dushanbe", "杜尚别", "tajikistan", "塔吉克斯坦", "tajikgidromet", "388360"],
    ),
    # web-search-verified 2026-05-04 — added from null_resolutions.json
    (
        "D19",
        "Quito / Ecuador / lat=-0.18°, lon=-78.47°W / 距赤道 ≈ 20 km / 长期均值 11.3-14.4°C / 1023-3216mm (来源差异大); 2850m 高海拔抑制赤道热; 即使 2024 暖年也难以达到 13°C 以上; 模型声称 15.5°C/1098mm 温度高估; 降水部分来源 (climate-data.org 1100mm) 与之相符, 但其他来源 (3216mm) 远超上限; 判 ⚠️: 实体真实 (厄瓜多尔首都) 但温度不符且降水多源严重分歧",
        "⚠️",
        0.0,
        ["quito", "基多", "ecuador", "厄瓜多尔", "inamhi", "mariscal sucre"],
    ),
    # web-search-verified 2026-05-04 — added from null_resolutions.json
    (
        "D20",
        "Rome / Italy / lat=41.90°N, lon=12.50°E / 距赤道 ≈ 4,665 km / 1991-2020 normals 15.7°C / 804mm (Wikipedia/ISPRA); 2024 是 1991 以来罗马最暖年, 城区年均 19.7°C (+2.5°C anomaly), 即使 Ciampino 站 (suburb) 也估 ~17.0°C (Italy national anomaly +1.33°C); **2024 单年实测温度超 17°C 上限**; 模型声称 15.8°C/878mm 实际是用 normals 而非 2024 单年; 降水量 878mm 与 normals 相符; 判 ⚠️: 实体真实 (意大利首都) 但 2024 单年实测温度超上限",
        "⚠️",
        0.0,
        [
            "rome",
            "roma",
            "罗马",
            "italy",
            "italia",
            "意大利",
            "ispra",
            "ciampino",
            "fiumicino",
        ],
    ),
]

DIM_MAP = {
    d[0]: {"id": d[0], "name": d[1], "judgment": d[2], "score": d[3], "kw": d[4]}
    for d in DIMS
}
PART_A_MAX = sum(d[3] for d in DIMS if d[3] > 0)  # = 5.0
PART_B_MAX = 5.0  # one +1 per baseline-hit capital with explicit lat/lon
PART_C_MAX = 1.0  # one +1 if avg distance within ±10% of 3687 km
MAX_SCORE = PART_A_MAX + PART_B_MAX + PART_C_MAX  # = 11.0

# C 维度参考值（5 城基准）+ 容差
C_REFERENCE_KM = 3687.0
C_TOLERANCE_PCT = 0.10  # ±10% → 容差区间 [3318, 4055]
C_LOWER = C_REFERENCE_KM * (1 - C_TOLERANCE_PCT)  # 3318.3
C_UPPER = C_REFERENCE_KM * (1 + C_TOLERANCE_PCT)  # 4055.7


# ═══════════════════════════════════════════════════════════════════════════
# DIMS → alignment baseline spec
# ═══════════════════════════════════════════════════════════════════════════


def _derive_match_fields(description: str) -> dict:
    """Parse country & latitude from DIMS description format:
    "<Capital> / <Country> / lat=<NN.NN°N|S>, lon=<...> / ..."
    Returns empty dict if regex doesn't match."""
    fields: dict = {}
    m_country = _re.match(r"^[^/]+/\s*([^/]+?)\s*/", description)
    if m_country:
        fields["country"] = m_country.group(1).strip()
    m_lat = _re.search(r"lat=(-?\d+\.?\d*)", description)
    if m_lat:
        fields["latitude"] = m_lat.group(1)
    return fields


def build_baselines() -> list[dict]:
    """Convert DIMS into baseline spec consumed by pipeline.alignment."""
    out = []
    for d_id, desc, judgment, score, kw in DIMS:
        out.append(
            {
                "id": d_id,
                "description": f"{desc} [基准: {judgment}]",
                "match_fields": _derive_match_fields(desc),
                "kw": kw,
                "judgment": judgment,
                "score": score,
            }
        )
    return out


# ═══════════════════════════════════════════════════════════════════════════
# Load extraction — separate E1 (capitals list) and E2 (avg_distance summary)
# ═══════════════════════════════════════════════════════════════════════════


def _load_entity_value(out_dir: Path, entity_id: str) -> dict[str, list[dict]]:
    """For each model subdir, read extraction.json and return canonical.value
    list for the named entity."""
    out: dict[str, list[dict]] = {}
    for d in Path(out_dir).iterdir():
        if not d.is_dir():
            continue
        ext_file = d / "extraction.json"
        if not ext_file.exists():
            continue
        payload = json.loads(ext_file.read_text(encoding="utf-8"))
        entities = payload.get("entities", [])
        match = next((e for e in entities if e.get("id") == entity_id), None)
        if match is None:
            out[d.name] = []
            continue
        cv = (match.get("canonical") or {}).get("value") or []
        if not isinstance(cv, list):
            cv = []
        out[d.name] = cv
    return out


def load_e1_claims(out_dir: Path) -> dict[str, list[dict]]:
    """Load qualifying_capitals list per model."""
    return _load_entity_value(out_dir, "qualifying_capitals")


def load_e2_summary(out_dir: Path) -> dict[str, dict]:
    """Load summary_avg_distance single record per model. Returns {} if model
    didn't produce a record."""
    raw = _load_entity_value(out_dir, "summary_avg_distance")
    out: dict[str, dict] = {}
    for model, items in raw.items():
        if items and isinstance(items[0], dict):
            out[model] = items[0]
        else:
            out[model] = {}
    return out


def _load_alignment(out_dir: Path) -> dict[str, list[dict]]:
    """Load existing alignment.json for each model (used with --skip-align)."""
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
# Part A scoring — 城市命中（Q12 strict 风格：null + unresolved → -1）
# ═══════════════════════════════════════════════════════════════════════════


def score_part_a_b(aligned_claims: list[dict]) -> tuple[float, float, dict]:
    """Score Part A (城市命中) + Part B (经纬度) together (per-claim).

    Part A rules (closed set, 5 baseline):
      - canonical_id ∈ DIM_MAP, judgment=✅, first occurrence → +1（baseline 命中）
      - canonical_id ∈ DIM_MAP, judgment=⚠️, first occurrence →  0（partial）
      - canonical_id == "__HALLUCINATION__"                  → -1
      - canonical_id is None and confidence != needs_review  → -1（闭合集合外视为错）
      - alignment_confidence == "needs_review"               →  0（特殊路径）
      - dup canonical_id → 0（dedup）

    Part B rules: 仅 baseline 命中 (judgment=✅) 时才查；
      若 raw['latitude'] 是数字 且 raw['longitude'] 是数字 → +1，否则 0。
    """
    seen: set = set()
    a_total = 0.0
    b_total = 0.0
    per_claim: list[dict] = []
    unverified: list[dict] = []

    for c in aligned_claims:
        cid = c.get("canonical_id")
        conf = c.get("alignment_confidence", "medium")
        raw = c.get("raw") or {}
        capital = raw.get("capital_name", "") or ""
        lat = raw.get("latitude")
        lon = raw.get("longitude")
        has_latlon = isinstance(lat, (int, float)) and isinstance(lon, (int, float))

        if conf == "needs_review":
            unverified.append(
                {
                    "capital_name": capital,
                    "canonical_id_tentative": cid,
                    "reason": "needs_review",
                }
            )
            continue

        if cid == "__HALLUCINATION__":
            per_claim.append(
                {
                    "id": cid,
                    "capital_name": capital,
                    "A_score": -1.0,
                    "B_score": 0.0,
                    "judgment": "❌",
                    "verdict": "hallucination",
                    "has_latlon": has_latlon,
                }
            )
            a_total += -1.0
            continue

        if cid and cid in DIM_MAP:
            if cid in seen:
                continue  # dedup
            seen.add(cid)
            d = DIM_MAP[cid]
            a = d["score"]  # +1 for ✅, 0 for ⚠️
            # B 只在 baseline 命中 (✅) 时才查
            b = 1.0 if (d["judgment"] == "✅" and has_latlon) else 0.0
            per_claim.append(
                {
                    "id": cid,
                    "capital_name": capital,
                    "A_score": a,
                    "B_score": b,
                    "judgment": d["judgment"],
                    "verdict": (
                        "baseline_hit" if d["judgment"] == "✅" else "partial_hit"
                    ),
                    "has_latlon": has_latlon,
                    "lat": lat,
                    "lon": lon,
                }
            )
            a_total += a
            b_total += b
        else:
            # null + Stage C unresolved（闭合集合外）→ -1
            per_claim.append(
                {
                    "id": "__UNRESOLVED_NULL__",
                    "capital_name": capital,
                    "A_score": -1.0,
                    "B_score": 0.0,
                    "judgment": "❌",
                    "verdict": "unresolved_null",
                    "has_latlon": has_latlon,
                }
            )
            a_total += -1.0

    return (
        a_total,
        b_total,
        {
            "per_claim": per_claim,
            "unverified": unverified,
            "n_listed": len(aligned_claims),
            "n_baseline_hits": sum(
                1 for x in per_claim if x["verdict"] == "baseline_hit"
            ),
            "n_partial_hits": sum(
                1 for x in per_claim if x["verdict"] == "partial_hit"
            ),
            "n_hallucinations": sum(
                1 for x in per_claim if x["verdict"] == "hallucination"
            ),
            "n_unresolved": sum(
                1 for x in per_claim if x["verdict"] == "unresolved_null"
            ),
        },
    )


# ═══════════════════════════════════════════════════════════════════════════
# Part C scoring — 整体平均距赤道距离
# ═══════════════════════════════════════════════════════════════════════════


def _parse_avg_distance(record: dict) -> float | None:
    """Extract the model's avg_distance_km value, normalizing units if needed.

    Handles common cases:
      - avg_distance_km is a number → use directly
      - unit_in_response indicates degrees → convert (1° ≈ 111.32 km)
      - avg_distance_km is "3687.4 km" string → parse
    """
    if not record:
        return None
    val = record.get("avg_distance_km")
    unit = (record.get("unit_in_response") or "").lower()

    if val is None:
        return None

    # Numeric or numeric string
    if isinstance(val, (int, float)):
        num = float(val)
    else:
        m = _re.search(r"(-?\d+(?:\.\d+)?)", str(val))
        if not m:
            return None
        num = float(m.group(1))

    # Unit normalization
    if "mile" in unit or "英里" in unit:
        num = num * 1.60934  # mi → km
    elif "度" in unit or "degree" in unit or unit in {"°"}:
        num = num * 111.32  # ° latitude → km

    return num


def score_part_c(e2_record: dict) -> tuple[float, dict]:
    """Score Part C — average distance from equator.

    +1 if model-claimed avg distance is in [C_LOWER, C_UPPER] (i.e., ±10% of 3687 km)
    0  otherwise (including no claim).
    """
    if not e2_record:
        return 0.0, {
            "score": 0.0,
            "raw_avg": None,
            "parsed_km": None,
            "verdict": "no_avg_distance_claim",
            "reference_km": C_REFERENCE_KM,
            "tolerance_pct": C_TOLERANCE_PCT,
            "tolerance_range_km": [C_LOWER, C_UPPER],
        }

    parsed = _parse_avg_distance(e2_record)
    if parsed is None:
        return 0.0, {
            "score": 0.0,
            "raw_avg": e2_record.get("avg_distance_km"),
            "parsed_km": None,
            "verdict": "unparsable",
            "reference_km": C_REFERENCE_KM,
            "tolerance_pct": C_TOLERANCE_PCT,
            "tolerance_range_km": [C_LOWER, C_UPPER],
        }

    in_range = C_LOWER <= parsed <= C_UPPER
    return (1.0 if in_range else 0.0), {
        "score": 1.0 if in_range else 0.0,
        "raw_avg": e2_record.get("avg_distance_km"),
        "unit_in_response": e2_record.get("unit_in_response"),
        "num_capitals": e2_record.get("num_capitals"),
        "parsed_km": parsed,
        "verdict": "in_range" if in_range else "out_of_range",
        "reference_km": C_REFERENCE_KM,
        "tolerance_pct": C_TOLERANCE_PCT,
        "tolerance_range_km": [C_LOWER, C_UPPER],
        "deviation_km": parsed - C_REFERENCE_KM,
        "deviation_pct": (parsed - C_REFERENCE_KM) / C_REFERENCE_KM,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Aggregate per-model score
# ═══════════════════════════════════════════════════════════════════════════


def score_model(
    model_name: str,
    aligned_e1: list[dict],
    e2_record: dict,
) -> dict:
    """Combine Part A + B + C for one model."""
    a_score, b_score, ab_detail = score_part_a_b(aligned_e1)
    c_score, c_detail = score_part_c(e2_record)
    total = a_score + b_score + c_score
    return {
        "model": model_name,
        "total_score": total,
        "max_score": MAX_SCORE,
        "score_rate": round(total / MAX_SCORE, 4) if MAX_SCORE else 0.0,
        "part_A_capital_hit": {
            "score": a_score,
            "max": PART_A_MAX,
            "n_baseline_hits": ab_detail["n_baseline_hits"],
            "n_partial_hits": ab_detail["n_partial_hits"],
            "n_hallucinations": ab_detail["n_hallucinations"],
            "n_unresolved": ab_detail["n_unresolved"],
        },
        "part_B_latlon": {
            "score": b_score,
            "max": PART_B_MAX,
            "n_with_explicit_latlon": int(b_score),
        },
        "part_C_avg_distance": c_detail,
        "claims_detail": ab_detail["per_claim"],
        "unverified": ab_detail["unverified"],
    }


# ═══════════════════════════════════════════════════════════════════════════
# Output writers
# ═══════════════════════════════════════════════════════════════════════════


def build_scores_json(all_scores: list[dict]) -> dict:
    ranked = sorted(all_scores, key=lambda x: -x["total_score"])
    return {
        "query_id": QUERY_ID,
        "snapshot_date": SNAPSHOT_DATE,
        "max_score": MAX_SCORE,
        "part_max": {
            "A_capital_hit": PART_A_MAX,
            "B_latlon": PART_B_MAX,
            "C_avg_distance": PART_C_MAX,
        },
        "c_reference": {
            "km": C_REFERENCE_KM,
            "tolerance_pct": C_TOLERANCE_PCT,
            "range_km": [C_LOWER, C_UPPER],
        },
        # dict-form: keyed by model name, matches Q22 reference schema so
        # run_all.extract_total_rates picks it up via the canonical path.
        "results": {s["model"]: s for s in all_scores},
        "ranking": [
            {
                "rank": i + 1,
                "model": r["model"],
                "score": r["total_score"],
                "rate": f"{r['score_rate'] * 100:.1f}%",
            }
            for i, r in enumerate(ranked)
        ],
    }


def build_ranking_md(all_scores: list[dict]) -> str:
    ranked = sorted(all_scores, key=lambda x: -x["total_score"])
    lines = [
        f"# Query {QUERY_ID} 排名报告（气候最宜居首都 + 平均距赤道）",
        "",
        f"> Snapshot: {SNAPSHOT_DATE}; MAX_SCORE = {MAX_SCORE:.0f}"
        f" (A={PART_A_MAX:.0f} 城市命中 + B={PART_B_MAX:.0f} 经纬度 + C={PART_C_MAX:.0f} 平均距赤道)",
        "> 5 个 baseline: Bogotá, Montevideo, Washington D.C., Zagreb, Wellington",
        f"> 平均距赤道参考值 {C_REFERENCE_KM:.0f} km, 容差 ±{C_TOLERANCE_PCT * 100:.0f}% → [{C_LOWER:.0f}, {C_UPPER:.0f}] km",
        "> 评分：✅ 城市 +1 / ⚠️ 0 / ❌ -1 / null+unresolved -1; 经纬度 +1 per baseline hit; 距赤道 +1 once",
        "",
        "| Rank | Model | Total | A (city) | B (latlon) | C (avg-eq) | hits/partial/halluc/null |",
        "|---:|---|---:|---:|---:|---:|---|",
    ]
    for i, r in enumerate(ranked, 1):
        a = r["part_A_capital_hit"]
        b = r["part_B_latlon"]
        c = r["part_C_avg_distance"]
        breakdown = (
            f"{a['n_baseline_hits']}/{a['n_partial_hits']}/"
            f"{a['n_hallucinations']}/{a['n_unresolved']}"
        )
        lines.append(
            f"| {i} | {r['model']} | {r['total_score']:.1f}/{MAX_SCORE:.0f} "
            f"| {a['score']:+.0f} | {b['score']:+.0f} | {c['score']:+.0f} "
            f"| {breakdown} |"
        )
    return "\n".join(lines) + "\n"


# ═══════════════════════════════════════════════════════════════════════════
# main()
# ═══════════════════════════════════════════════════════════════════════════


def main() -> None:
    ap = argparse.ArgumentParser(description=f"Query {QUERY_ID} auto-scorer")
    ap.add_argument("--models", nargs="+", required=True, help="name=path list")
    ap.add_argument("--output-dir", default=str(THIS / "auto_scores"))
    ap.add_argument("--skip-extract", action="store_true")
    ap.add_argument("--skip-align", action="store_true")
    ap.add_argument("--aligner-models", nargs="+", default=None)
    ap.add_argument("--judge-model", default=None)
    ap.add_argument("--concurrency", type=int, default=None)
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Stage 1: Extract (both entities) ─────────────────────────────────
    if not args.skip_extract:
        run_pipeline(
            query_id=QUERY_ID,
            query_text=QUERY_TEXT,
            entities=ENTITIES,
            prompt_hints=PROMPT_HINTS,
            schema=VALUE_SCHEMA,
            models_input=args.models,
            output_dir=out_dir,
            concurrency=args.concurrency,
        )

    # ── Stage 2: Align (only E1) ─────────────────────────────────────────
    baselines = build_baselines()
    if args.skip_align:
        aligned_e1 = _load_alignment(out_dir)
    else:
        client = get_client()
        overrides = {}
        if args.aligner_models:
            overrides["aligner_models"] = args.aligner_models
        if args.judge_model:
            overrides["judge_model"] = args.judge_model
        if args.concurrency:
            overrides["concurrency"] = args.concurrency
        e1_claims = load_e1_claims(out_dir)
        aligned_e1 = align_claims(
            client,
            claims_by_model=e1_claims,
            baselines=baselines,
            query_text=QUERY_TEXT,
            output_dir=out_dir,
            overrides=overrides,
        )

    # ── Stage 3a: Export E1 null_review for Stage C ──────────────────────
    null_items = export_null_claims_for_review(
        aligned_e1,
        out_dir / "null_review.json",
        query_id=QUERY_ID,
        query_text=QUERY_TEXT,
        models_input=args.models,
    )
    print(f"[*] Exported {len(null_items)} E1 null/needs_review → null_review.json")

    # ── Stage 3b: Apply E1 null_resolutions if present ───────────────────
    resolutions_path = out_dir / "null_resolutions.json"
    if resolutions_path.exists():
        print("[*] Found null_resolutions.json, applying …")
        new_baselines = apply_null_resolutions(
            aligned_e1, resolutions_path, dims_ref=DIMS
        )
        if new_baselines:
            print(f"[*] {len(new_baselines)} new E1 baseline entries from resolutions")
            for e in new_baselines:
                DIM_MAP[e["id"]] = {
                    "id": e["id"],
                    "name": e.get("description", e["id"]),
                    "judgment": e.get("judgment", "⚠️"),
                    "score": e.get("score", 0.0),
                    "kw": e.get("kw", []),
                }
            persist_new_baseline_entries(new_baselines, __file__)
    else:
        print(
            f"[*] No null_resolutions.json yet."
            f"\n    → 用 Claude Code 读 {resolutions_path.parent}/null_review.json"
            f"\n      逐条 web 验证后产出 {resolutions_path.name}"
            f"\n      然后重跑：python3 {Path(__file__).name} "
            f"--skip-extract --skip-align --models …"
        )

    # ── Stage 3c: Load E2 (no alignment, direct read) ────────────────────
    e2_summary = load_e2_summary(out_dir)
    print(f"[*] Loaded E2 summary_avg_distance for {len(e2_summary)} models")

    # ── Stage 4: Score ────────────────────────────────────────────────────
    all_scores: list[dict] = []
    all_models = sorted(set(list(aligned_e1.keys()) + list(e2_summary.keys())))
    for model in all_models:
        s = score_model(
            model,
            aligned_e1.get(model, []),
            e2_summary.get(model, {}),
        )
        all_scores.append(s)

    (out_dir / "scores.json").write_text(
        json.dumps(build_scores_json(all_scores), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (out_dir / "ranking_report.md").write_text(
        build_ranking_md(all_scores), encoding="utf-8"
    )

    print("\n" + "─" * 62)
    print(f"Query {QUERY_ID} scoring done.")
    for i, r in enumerate(sorted(all_scores, key=lambda x: -x["total_score"]), 1):
        a = r["part_A_capital_hit"]["score"]
        b = r["part_B_latlon"]["score"]
        c = r["part_C_avg_distance"]["score"]
        print(
            f"  {i}. {r['model']:30s} {r['total_score']:>5.1f}/{MAX_SCORE:.0f}"
            f" (A={a:+.0f} B={b:+.0f} C={c:+.0f})"
        )


if __name__ == "__main__":
    main()
