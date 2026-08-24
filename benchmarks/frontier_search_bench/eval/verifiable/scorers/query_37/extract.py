"""
Query 37 — Amazon rainforest 2015-2024 net deforestation
+ top-3 most-deforested regions.

T1 granularity: ONE composite entity holding both:
  (a) the numerical net loss in km^2 with methodology scope, and
  (b) the model's top-3 hotspot regions (country + lat/lon).

Two methodology calibers are accepted as GT (locked in auto_scorer.py):
  - Brazilian Legal Amazon (PRODES, ~83,087 km^2)
  - Pan-Amazon biome / 9 countries (MapBiomas+RAISG+MAAP, ~170-190k km^2)

This module exports constants only. The runner is auto_scorer.py.
"""

from __future__ import annotations

QUERY_ID = 37
QUERY_TEXT = (
    "获取亚马逊雨林 2015 年和 2024 年的覆盖数据，"
    "估算这十年间雨林面积的净减少量（正数，单位：平方公里），"
    "并列出森林减少最严重的前三个区域的所在国家和近似地理坐标。"
)


ENTITIES = [
    {
        "id": "amazon_answer",
        "name": (
            "模型对'亚马逊雨林 2015-2024 净减少面积（km^2）+ "
            "森林减少最严重的前 3 个区域（国家+地理坐标）'的全部主张"
        ),
    },
]


PROMPT_HINTS = {
    "amazon_answer": (
        "请从模型回答中抽出**两类信息**，组装成一个 value 对象：\n\n"
        "**(A) 净减少面积**\n"
        "- `loss_km2`：模型给出的 2015-2024 期间亚马逊雨林**净减少面积**（正数，单位 km^2）。\n"
        "  - 如果模型给出多个数字（如巴西法定亚马逊 vs 泛亚马逊），优先取模型在结论/最终答案处明确给出的"
        "    那个总值；如果模型同时给出两个口径，**优先取泛亚马逊（9 国整体）**那个。\n"
        "  - 如果模型给区间（如 '170,000-190,000 km^2'），取**中点**（180,000）。\n"
        "  - 如果模型只给公顷（ha），换算为 km^2（1 ha = 0.01 km^2）。\n"
        "  - 如果模型未给数字，置 null。\n"
        "- `scope`：模型主张该数字的覆盖范围。从如下取值二选一或 null：\n"
        "  - `brazilian_legal_amazon`：巴西法定亚马逊（仅巴西部分；常见词：PRODES / 巴西法定亚马逊 / Legal Amazon / Brazilian Amazon）\n"
        "  - `pan_amazon`：泛亚马逊整个生物群落（9 国；常见词：泛亚马逊 / pan-Amazon / Amazon biome / 整个亚马逊雨林 / 9 countries）\n"
        "  - `unclear`：模型未明示口径\n"
        "  - `null`：模型未给数字\n"
        "- `year_2015_cover_km2` / `year_2024_cover_km2`：模型给出的 2015 / 2024 年覆盖面积（km^2）；未给则 null。\n\n"
        "**(B) Top-3 热点区域**（模型自己列出的前 3 个 '森林减少最严重' 的区域）\n"
        "- 抽出 `top3_hotspots` 数组，**最多 3 项**（按模型给出的排序，第 1/2/3 名）。\n"
        "- 每项包含：\n"
        "  - `region_name`：模型原文中的区域名（如 'Pará' / 'Arc of Deforestation' / 'Madre de Dios' / '砍伐弧'）\n"
        "  - `country`：所在国家（中文或英文，按模型表述）\n"
        "  - `lat`：纬度（**十进制度数**，南纬为负数；模型未给则 null）\n"
        "  - `lon`：经度（**十进制度数**，西经为负数；模型未给则 null）\n"
        "  - `rank`：排名 1/2/3\n"
        "- 坐标解析规则：\n"
        "  - '9°S, 55°W' → lat=-9, lon=-55\n"
        "  - '6.6°S 51.99°W' → lat=-6.6, lon=-51.99\n"
        "  - '2°N, 73°W' → lat=2, lon=-73\n"
        "  - '12.59 deg S, 69.19 deg W' → lat=-12.59, lon=-69.19\n"
        "- 若模型给的是范围（如 '6-12°S, 50-70°W'），取**中点**。\n"
        "- 若模型给一个区域但只列 city/state 名称没有坐标，置 lat=lon=null。\n"
        "- 若模型只列了 1 或 2 个区域，照实只抽出这 1-2 项即可（不要凑数）。\n\n"
        "**抽取范围约束（重要）：**\n"
        "- 只抽取模型作为**正面回答**给出的内容；模型在 '局限性 / 不确定性' 段落讨论的备选数字不抽。\n"
        "- 字段**必须贴近原文**；模型未明确写出的信息一律置 null，不要用外部知识补全。\n"
        "- 若模型完全未回答该题（response 为空或与亚马逊无关），置 not_mentioned=true。\n"
    ),
}


VALUE_SCHEMA = """{
  "value": {
    "loss_km2": <十年净减少面积（km^2，正数；未给则 null）>,
    "scope": "<brazilian_legal_amazon|pan_amazon|unclear|null>",
    "year_2015_cover_km2": <2015 年覆盖面积（km^2）/ null>,
    "year_2024_cover_km2": <2024 年覆盖面积（km^2）/ null>,
    "top3_hotspots": [
      {
        "region_name": "<区域名（模型原文）>",
        "country": "<国家名（模型原文，中文或英文）>",
        "lat": <十进制度数（南纬负） / null>,
        "lon": <十进制度数（西经负） / null>,
        "rank": <1|2|3>
      }
    ]
  },
  "not_mentioned": <true 仅当模型完全未回答时；否则 false>,
  "supporting_span": "<原文片段 30-200 字，能证明你抽出的信息来自回答>",
  "confidence": "<high|medium|low>"
}"""
