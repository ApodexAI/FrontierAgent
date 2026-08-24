"""Query 22 — Drug fast-track first approval lookup

This module exports constants only. The runner is auto_scorer.py.

Question (Chinese): A country's drug regulator has a special "fast track"
program; the historically first drug approved through this fast track has a
brand name containing the letters "zy" and the applicant company name
contains "tech". What is the official approval number? What is the exact
approval date?

Reference answer: Gazyva (obinutuzumab) by Genentech, BLA 125486, FDA
approval 2013-11-01 (first drug approved under the FDA Breakthrough Therapy
Designation program).
"""

from __future__ import annotations

QUERY_ID = 22

QUERY_TEXT = (
    '某个国家的药品监管机构有一个特殊的"快速通道"项目，历史上第一个通过这个'
    "快速通道获批上市的药物，它的品牌名里有zy两个字母。我记得提交申请的"
    "公司名字里有tech。请问这个药的官方审批编号是多少？确切获批日期是哪天？"
)

# T1 granularity — the entire answer is one composite value (drug + company +
# approval number + date), not a list.
ENTITIES = [
    {
        "id": "drug_answer",
        "name": (
            "模型主张为题目答案的那个药物（含品牌名/通用名、申请公司、"
            "官方审批编号、确切获批日期、对应监管机构与快速通道项目）"
        ),
    }
]

PROMPT_HINTS = {
    "drug_answer": (
        "请抽取模型作为该题最终答案给出的那个药物的关键事实。\n\n"
        "**抽取规则：**\n"
        "- 模型可能讨论多个候选药物，但通常会明确选定一个作为答案——"
        "把那个被选定的填到主字段；其他候选放进 alternative_candidates 列表。\n"
        '- 如果模型明确说"无法确定"或"找不到"，把所有主字段留 null，'
        "  not_mentioned 设为 true。\n"
        "- **company** 字段：填模型主张为该药申请人/申报公司的那个名字（通常是 BLA/NDA 持有人）。\n"
        '  原文若同时提到母公司与子公司（例如 "Genentech, a member of Roche Group"），\n'
        "  以模型主张提交申请的那个为准（通常是 Genentech 这种子公司）；不要替模型脑补。\n"
        "- **approval_number** 字段：原样保留模型给出的编号字符串"
        '  （可能是 "BLA 125486"、"125486"、"125486Orig1s000"、"国药准字XXX" 等）。\n'
        "- **approval_date** 字段：原样保留模型给出的日期字符串"
        '  （可能是 "2013-11-01"、"November 1, 2013"、"2013年11月1日"、'
        '  "2013年11月" 等）；不要补全模型未明说的日。\n'
        "- **fast_track_program** 字段：模型说的快速通道项目名（如 "
        '  "Breakthrough Therapy Designation"、"突破性疗法"、"FDA Fast Track"、'
        '  "NMPA 优先审评"、"PMDA Sakigake" 等）。\n'
        "- **country_agency** 字段：模型主张的国家/监管机构（FDA/NMPA/PMDA/MHRA 等）。\n"
        "- 拼写、大小写、原文格式忠于模型；不要纠错或补全。\n"
        '- 模型在排除/否定/对照语境中提到的药物不抽（例如 "不是 Zytiga 因为..."）。'
    ),
}

VALUE_SCHEMA = """{
  "value": {
    "brand_name": "<品牌名字符串或 null>",
    "generic_name": "<通用名/INN 字符串或 null>",
    "company": "<申请公司字符串或 null>",
    "country_agency": "<国家/监管机构字符串或 null>",
    "fast_track_program": "<快速通道项目名字符串或 null>",
    "approval_number": "<官方审批编号字符串或 null>",
    "approval_date": "<获批日期字符串或 null>",
    "alternative_candidates": [
      {
        "brand_name": "...",
        "generic_name": "...",
        "company": "...",
        "approval_number": "...",
        "approval_date": "..."
      }
    ]
  },
  "not_mentioned": <true 仅当模型完全没给出任何药物候选；否则 false>,
  "supporting_span": "<原文片段 30-200 字，能证明上面字段确实来自模型回答>",
  "confidence": "<high|medium|low>"
}"""
