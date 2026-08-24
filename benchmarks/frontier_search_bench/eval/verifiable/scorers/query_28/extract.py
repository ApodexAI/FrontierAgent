"""Query 28 — NASA Exoplanet Archive count + JWST water-signal planets +
latest such planet (name / announce date / distance).

Two entities (consumed independently by auto_scorer):
  E1 `jwst_water_planets`  — list of planets the model claims have JWST-detected
                              atmospheric water signal. Goes through the full
                              alignment + Stage C verification path.
  E2 `summary_facts`        — single record with total exoplanet count, latest
                              planet name, latest announce date, latest distance.
                              Does NOT go through alignment; auto_scorer reads
                              the canonical.value directly and runs custom
                              scoring (Part A + Part C).
"""

from __future__ import annotations

QUERY_ID = 28
QUERY_TEXT = (
    "截至2026年3月，NASA系外行星档案库确认的系外行星总数是多少？"
    "其中被JWST观测到大气中含有水信号的有哪几颗？"
    "截至2026年3月，最新的一颗是什么时候公布的、距地球多远？"
)


ENTITIES = [
    {
        "id": "jwst_water_planets",
        "name": "模型主张为'被 JWST 观测到大气中含有水信号'的系外行星清单",
    },
    {
        "id": "summary_facts",
        "name": (
            "模型对该 query 给出的三项概要事实："
            "(1) NASA 系外行星档案库确认行星总数；"
            "(2) 截至 2026-03 最新一颗 JWST 探测到水信号的行星名；"
            "(3) 该最新行星的公布时间与距地球距离"
        ),
    },
]


PROMPT_HINTS = {
    "jwst_water_planets": (
        "请抽出模型回答里**明确主张**为'被 JWST（James Webb Space Telescope）"
        "观测到大气中含有水（H₂O / 水蒸气）信号'的系外行星。"
        "每颗行星一个字段齐全的 JSON 对象。\n\n"
        "**抽取范围约束（重要）：**\n"
        "- 只抽取模型以**肯定或接近肯定**口吻列为'JWST 探测到含水信号'的行星。"
        "  '据传'、'可能'、'有报告称'、'尚有争议但仍被列为含水'按肯定处理（保留）。\n"
        "- **不要**抽取模型自己在'排除'、'反例'、'非 JWST 而是 Hubble'、"
        "  '不是水世界'、'是岩浆世界而非水'、'信号其实来自恒星'等语境下"
        "  **明确否定**的行星。\n"
        "- 例如：模型说'L 98-59 d 最初以为含水，但 2026-03 paper 证实是岩浆世界，"
        "  故不是 JWST 含水行星' → **不抽取**；\n"
        "  模型说'L 98-59 d 是 JWST 探测到水的行星'（无否定）→ **抽取**。\n"
        "- 模型若把行星分组为'确认含水 vs 争议含水 vs 非 JWST/非含水'，"
        "  **抽取前两组**（确认 + 争议），不抽取第三组。\n\n"
        "**字段约定：**\n"
        "- `name` 必填：行星名（如 'WASP-96 b' / 'GJ 9827 d' / 'WASP-51 b'），"
        "  保留原文写法；若有别名（HAT-P-30 b / WASP-51 b 是同一颗）按模型原文。\n"
        "- `announce_date`：模型为该颗给出的水信号公布时间，如 '2022-07' / "
        "  '2025年12月' / '2024' / null。\n"
        "- `distance_ly`：距地球距离（光年，纯数字字符串）/ null。\n"
        "- `note`：模型对该颗的限定/争议说明（如'信号有争议'/"
        "  '可能来自恒星黑子'/'JWST 首张光谱'）/ null。\n"
        "- 字段必须**贴近原文**；模型未明确写出的信息填 null，**不要**用外部知识补全。\n\n"
        "**E2（summary_facts）entity 的内容此处不抽取**：\n"
        "- 总数（如 6,150）不在本 entity；\n"
        "- '最新一颗'仅作为本 entity 的一条记录抽取（给 note='模型称为最新'），"
        "  其结构化字段（latest_*）由 E2 单独抽取。"
    ),
    "summary_facts": (
        "请抽出模型对本 query 三项**概要事实**的回答。**仅输出一个对象**"
        "（即 value 列表只含 1 条记录）。\n\n"
        "**字段约定（此 entity 用以下字段；其他字段填 null）：**\n"
        "- `name` 固定填 'summary_facts'。\n"
        "- `total_count`：模型给出的'NASA 系外行星档案库确认的系外行星总数'"
        "  （纯数字字符串，如 '6150'；如模型给区间或多个数字，取主要数字）/ null。\n"
        "- `latest_planet_name`：模型主张为'截至 2026-03 最新一颗 JWST 观测到水信号'"
        "  的那颗行星名 / null（若模型未明确指认最新一颗，填 null）。\n"
        "- `latest_announce_date`：该最新行星的水信号公布时间，如 '2025-12' / "
        "  '2026年1月' / null。\n"
        "- `latest_distance_ly`：该最新行星距地球距离（光年，纯数字字符串）/ null。\n"
        "- `note`：模型原文中对该最新一颗的简短描述（保留模型自己的限定语，"
        "  如'存在争议'/'同行评审中'/'首次检测'）/ null。\n\n"
        "**抽取规则：**\n"
        "- 只抽取模型**明确主张**为'最新一颗'的那颗行星。如果模型给出多个候选"
        "  （如'最新可能是 X 或 Y'），**只抽 X**（即模型首选）；其它候选不抽。\n"
        "- 如果模型把'最新发现的系外行星'误解为'最新发现的非含水普通行星'，"
        "  按字面填模型回答；下游打分会判错。\n"
        "- 总数和最新一颗的字段**必须忠实于模型原文**，不要用外部知识纠错。\n"
        "- 列表中的其他字段（announce_date、distance_ly）此 entity 不填，留 null。"
    ),
}


VALUE_SCHEMA = """{
  "value": [
    {
      "name": "<E1: 行星名（如 'WASP-96 b' / 'HAT-P-30 b'）；E2: 固定 'summary_facts'>",
      "announce_date": "<E1: 该颗水信号公布时间字符串 / null；E2: 不填，填 null>",
      "distance_ly": "<E1: 该颗距地球距离（光年，纯数字字符串）/ null；E2: 不填，填 null>",
      "total_count": "<E2: NASA 档案库系外行星确认总数（纯数字字符串）/ null；E1: 不填，填 null>",
      "latest_planet_name": "<E2: 模型主张的最新一颗 JWST 含水行星名 / null；E1: 不填，填 null>",
      "latest_announce_date": "<E2: 最新一颗的公布时间 / null；E1: 不填，填 null>",
      "latest_distance_ly": "<E2: 最新一颗的距离（光年，纯数字字符串）/ null；E1: 不填，填 null>",
      "note": "<可选：模型原文中对该条目的简短描述/限定说明 / null>"
    }
  ],
  "not_mentioned": <true 仅当模型完全未提任何相关条目时为 true；否则 false>,
  "supporting_span": "<原文片段 30-200 字，能证明你抽出的 value 来自回答>",
  "confidence": "<high|medium|low>"
}"""
