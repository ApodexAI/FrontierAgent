"""Query 01 — Singapore condo top-3 resale return rate.

Question (Chinese): of the condos that received TOP (Temporary Occupation
Permit) in 2019, which 3 projects had the highest 2025 resale return rate?

Reference answer (URA public transactions, locked 2026-05-05):
  1. High Park Residences   — 6.10%
  2. Coco Palms             — 4.80%
  3. Botanique At Bartley   — 4.64%
  Known incorrect: Hundred Palms Residences (TOP year is 2018, not 2019).

T-list granularity — the answer is a list of projects with names and rates.
"""

from __future__ import annotations

QUERY_ID = 1

QUERY_TEXT = (
    "我想投资新加坡房产。请帮我找出在2019年获得TOP的condo中，"
    "在2025年转售回报率最高的三个项目，并给出具体回报率。"
)

ENTITIES = [
    {
        "id": "condo_top_list",
        "name": "模型作为最终答案给出的'2019 TOP condo 转售回报率最高'项目列表",
    }
]

PROMPT_HINTS = {
    "condo_top_list": (
        "请抽取模型作为该题最终答案给出的项目列表（按模型给出的排名顺序）。\n\n"
        "**抽取规则：**\n"
        "- **只**抽取模型作为最终答案/结论列出的项目；不要抽取仅在分析过程中"
        "提到但未被作为答案的项目。\n"
        "- 项目名优先用英文全名（如 'High Park Residences' 而非 'High Park'）；"
        "原文若用中文译名，按原文保留。\n"
        "- **return_rate** 字段保留模型给出的数值字符串（如 '6.1%'、'5.8%'、"
        "'每年 4.5%'），不做单位换算。模型未给具体数值则留 null。\n"
        "- 模型给的项目数 **不一定是 3 个**，按模型实际列出的全部抽出。\n"
        "- 若模型明确表示无法回答 / 没找到具体项目，返回空数组，"
        "  not_mentioned 设为 true。\n"
        "- 模型在'排除'、'不符合'、'不算'语境下提到的项目不抽。"
    ),
}

VALUE_SCHEMA = """{
  "value": [
    {
      "name": "<项目名（优先英文全名）>",
      "return_rate": "<回报率字符串，如 '6.1%' 或 null>",
      "rank_in_model_answer": <模型答案中的排名（1=最高）或 null>
    }
  ],
  "not_mentioned": <true 仅当模型完全未给出任何项目；否则 false>,
  "supporting_span": "<原文片段 30-200 字，证明列表来自模型回答>",
  "confidence": "<high|medium|low>"
}"""
