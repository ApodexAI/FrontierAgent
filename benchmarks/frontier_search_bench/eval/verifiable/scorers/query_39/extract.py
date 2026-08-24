"""
Query 39 — 全球符合"气候最宜居"条件的首都筛选 + 平均距赤道距离。

题目原文：
    获取全球所有国家首都2024年的年均气温和年降水量。
    找出年均气温在15°C±2°C 且 年降水量在800-1200mm之间的"气候最宜居"首都，
    在地图上找到它们的位置，并计算它们与赤道的平均距离。

T1 granularity 设计（沿用 query_28 双 entity 模式）：
- 两个 entity，使用统一 schema（不相关字段填 null）：
    E1 `qualifying_capitals`  — 模型主张为"气候最宜居"的首都列表（走 alignment）
    E2 `summary_avg_distance` — 模型给出的整体平均距赤道距离（不走 alignment，scorer 直读）
- 评分维度（auto_scorer.py 实现）：
    A 城市命中：+1 / 0 / -1（每首都独立，闭合集合 5 城）
    B 经纬度：  +1 / 0（命中 baseline 城市需提供具体经纬度数值才计分）
    C 平均距赤道：+1 / 0（仅在模型给出整体平均距离 且 在 ±10% 容差内）
"""

from __future__ import annotations

QUERY_ID = 39
QUERY_TEXT = (
    "获取全球所有国家首都2024年的年均气温和年降水量。"
    '找出年均气温在15°C±2°C且年降水量在800-1200mm之间的"气候最宜居"首都，'
    "在地图上找到它们的位置，并计算它们与赤道的平均距离。"
)


ENTITIES = [
    {
        "id": "qualifying_capitals",
        "name": (
            "模型最终主张为同时满足'年均气温 13–17°C'与'年降水量 800–1200mm'两个条件、"
            "且为某国家首都的城市清单"
        ),
    },
    {
        "id": "summary_avg_distance",
        "name": (
            "模型给出的'符合条件的首都集合'与赤道的平均距离（整体汇总值，单位 km）"
        ),
    },
]


PROMPT_HINTS = {
    "qualifying_capitals": (
        "请抽出模型回答里**最终结论**主张为同时满足"
        "'年均气温 13–17°C'与'年降水量 800–1200mm'两个条件、"
        "且明确列为'某国首都'的城市。每个城市一个字段齐全的 JSON 对象。\n\n"
        "**抽取范围约束（重要）：**\n"
        "- 只抽取模型在**最终结论 / 最终列表 / 总结表**中明确列为'气候最宜居首都'的城市。\n"
        "- 模型在**中间分析、候选列表、初筛后被排除**的城市**不抽**。\n"
        "  （例：模型先列 20 个候选，最后筛剩 5 个，只抽这 5 个）\n"
        "- 模型用'排除/不符合/边缘/接近但不满足/差一点/数据存疑'等语境讨论的城市**不抽**。\n"
        "- 模型先肯定后又改口说'不计入'的，按最终结论抽（如最终改口排除则不抽）。\n\n"
        "**字段填写规则（E1 字段）：**\n"
        "- `capital_name` 必填：首都名称（中英文都行）；如模型只给国家名（'美国'）请把首都补上（'Washington D.C.'）。\n"
        "- `country`：所属国家。\n"
        "- `annual_temp_c`：模型给出的 2024 年均气温（°C 数字）；模型未给数值写 null。\n"
        "- `annual_precip_mm`：模型给出的 2024 年降水量（mm 数字）；模型未给数值写 null。\n"
        "- `latitude`：模型明确给出的纬度（十进制度数；南纬记为负数；'4°N'写 4.0；'34.9°S'写 -34.9）。\n"
        "  **如果模型只给城市名而完全未提及具体经纬度数值，写 null**（不要用外部知识填）。\n"
        "- `longitude`：同 latitude（西经为负，例 Washington 77.04°W → -77.04）。\n"
        "- `distance_from_equator_km`：模型给出的该首都距赤道的具体距离（km）；未给写 null。\n"
        "- `note`：模型对此条目的限定/说明（如'用 1991-2020 normals 估算'、'部分站点不符'等）。\n"
        "- E2 专用字段（`avg_distance_km`/`unit_in_response`/`num_capitals`）此 entity **全部留 null**。\n\n"
        "**关键判别：**\n"
        "- 模型若把'地区/区域'当首都列出（如'纽约'、'香港'、'墨尔本'）：仍抽出，下游会判它不是首都。\n"
        "- 模型给出多个候选层级（'核心 5 城 + 边缘 2 城'）：**只抽核心 5 城**，边缘候选不抽。\n"
        "- 字段事实即使有误（年份/温度/降水值），只要模型把它列为合规答案就要抽出（下游判分）。"
    ),
    "summary_avg_distance": (
        "请抽出模型在最终回答中给出的'**整体集合**与赤道的**平均距离**'数值。**仅输出一个对象**"
        "（即 value 列表只含 1 条记录）。\n\n"
        "**抽取范围：**\n"
        "- 只抽**整体平均**或**所有合格首都的平均距离**，不抽单一首都的距离。\n"
        "- 模型若给多个候选集合的不同平均（如'核心 4 城平均 X km'、'含边缘 5 城平均 Y km'）：\n"
        "  抽**核心/最终结论**对应的那一个值（与 E1 抽出的城市数对应）。\n"
        "- 模型完全未给整体平均距离（只列单首都距离）：not_mentioned=true，value=[]。\n\n"
        "**字段填写规则（E2 字段；E1 专用字段全留 null）：**\n"
        "- `capital_name` 固定填 'summary_avg_distance'。\n"
        "- `avg_distance_km`：模型给出的平均距离（km）数字。如模型用度数表示（如'33°'）请按 1°≈111.32 km 转换并在 note 里说明。\n"
        "- `unit_in_response`：模型原文使用的单位（'km' / 'mile' / '度纬度' / 等）。\n"
        "- `num_capitals`：模型该平均值对应的首都数量（4 / 5 / 6 / 等）；未明确写 null。\n"
        "- `note`：模型的限定（'四城核心'、'含边缘 5 城'、'仅 baseline 集合'等）。\n"
        "- E1 专用字段（`country`/`annual_temp_c`/`annual_precip_mm`/`latitude`/`longitude`/`distance_from_equator_km`）此 entity **全部留 null**。"
    ),
}


VALUE_SCHEMA = """{
  "value": [
    {
      "capital_name": "<E1: 首都名（中英文皆可，例 Bogotá / 波哥大）；E2: 固定 'summary_avg_distance'>",
      "country": "<E1: 国家（例 哥伦比亚 / Colombia）；E2: null>",
      "annual_temp_c": "<E1: 模型给出的年均气温 °C 数字；E2: null>",
      "annual_precip_mm": "<E1: 模型给出的年降水量 mm 数字；E2: null>",
      "latitude": "<E1: 十进制度数（南纬负），数字；E2: null>",
      "longitude": "<E1: 十进制度数（西经负），数字；E2: null>",
      "distance_from_equator_km": "<E1: 模型给出的该首都距赤道距离 km，数字；E2: null>",
      "avg_distance_km": "<E2: 模型给出的整体平均距赤道距离 km，数字；E1: null>",
      "unit_in_response": "<E2: 模型原文使用的单位 'km'/'mile'/'度纬度'；E1: null>",
      "num_capitals": "<E2: 该平均对应的首都数量整数；E1: null>",
      "note": "<可选：模型对该条目的额外说明 / 限定 / null>"
    }
  ],
  "not_mentioned": <true 仅当模型完全未提任何相关条目时为 true；否则 false>,
  "supporting_span": "<原文片段 30-200 字，能证明你抽出的 value 来自回答>",
  "confidence": "<high|medium|low>"
}"""
