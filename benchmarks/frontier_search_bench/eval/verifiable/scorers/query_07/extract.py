"""Query 07 — IMO countries with most Fields Medal alumni.

Question (Chinese): Among all countries that have ever sent IMO teams, which
country has the most participants who later won the Fields Medal? How many?

Reference answer: Soviet Union (USSR) — 4 laureates (Margulis, Drinfeld,
Perelman, Smirnov). France is runner-up with 3.

T1 granularity — the answer is one composite value (country + count + names),
not a list of independent claims.
"""

from __future__ import annotations

QUERY_ID = 7

QUERY_TEXT = (
    "截至2026年1月1日，所有参加过国际数学奥林匹克（IMO）的国家中，"
    "本国参赛选手后来获得菲尔兹奖人数最多的是哪个国家？是多少人？"
)

ENTITIES = [
    {
        "id": "fields_country_answer",
        "name": (
            "模型主张为题目最终答案的那个国家"
            "（含国家名、人数、对应的菲尔兹奖得主姓名列表）"
        ),
    }
]

PROMPT_HINTS = {
    "fields_country_answer": (
        "请抽取模型作为该题最终答案给出的国家、人数和具体人名。\n\n"
        "**抽取规则：**\n"
        "- 模型可能讨论多国，但通常会明确选定一个作为答案——"
        "把那个被选定的填到主字段；其他候选放进 alternative_candidates。\n"
        '- 如果模型明确说"无法确定"或"找不到"，所有主字段留 null，'
        "  not_mentioned 设为 true。\n"
        "- **country** 字段：模型主张的最终答案国家名（中文或英文均可）。\n"
        '  常见正确答案为"苏联"系列（Soviet Union / USSR / 前苏联 / 苏联/俄罗斯 / Russia/USSR）。\n'
        '  注意"俄罗斯"如果模型明确说指含前苏联，也按主张处理；不要替模型脑补口径。\n"'
        "- **count** 字段：模型给出的人数（整数）。\n"
        "- **names** 字段：模型作为该国 IMO 参赛后获菲尔兹奖的具体人名列表"
        "（中英文皆可，原样保留模型表述）。\n"
        "- 模型在排除/否定语境提到的国家或人名不抽。"
    ),
}

VALUE_SCHEMA = """{
  "value": {
    "country": "<模型主张的最终答案国家名或 null>",
    "count": <模型给出的人数（整数）或 null>,
    "names": ["<人名字符串>", ...],
    "alternative_candidates": [
      {"country": "...", "count": <int>, "names": ["..."]}
    ]
  },
  "not_mentioned": <true 仅当模型完全没给出任何国家答案；否则 false>,
  "supporting_span": "<原文片段 30-200 字，证明上面字段确实来自模型回答>",
  "confidence": "<high|medium|low>"
}"""
