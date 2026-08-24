"""Query 08 — Kobe's 2nd-highest scoring game vs LeBron's Cavaliers + that
game's top Cavalier scorer.

This module exports constants only. The runner is auto_scorer.py.

Closed-set, multi-field answer:
  - The unique correct game: 2007-02-11 (Cavaliers 99, Lakers 90; Kobe 36 pts).
  - The unique top scorer on the Cavs that night: Sasha Pavlović (21 pts).
  - No tie at 2nd-highest (1st=38, 2nd=36, 3rd=35) and no tie for top scorer.

The schema is a list of answer-tuples to allow models that (mistakenly) claim
multiple tied games or multiple tied top scorers; downstream auto_scorer.py
picks the best-scoring tuple per model and penalises extras.
"""

from __future__ import annotations

QUERY_ID = 8
QUERY_TEXT = (
    "在勒布朗·詹姆斯效力骑士期间与科比的交手记录中，"
    "确定科比单场得分排名第二的所有比赛（若存在并列则全部选取），"
    "并在每场比赛中找出詹姆斯队中得分最高的球员（答案可能为多个）"
)

ENTITIES = [
    {
        "id": "answer_tuples",
        "name": (
            "模型最终主张为答案的（场次日期, 科比该场得分, 该场骑士队得分王, "
            "得分王该场得分）四元组列表"
        ),
    },
]

PROMPT_HINTS = {
    "answer_tuples": (
        "请抽出模型**最终给出的答案**：每条答案是一个四元组（场次日期 / 科比该场得分 / "
        "该场骑士队内得分最高的球员 / 该球员得分）。每条一个 JSON 对象。\n\n"
        "**抽取范围约束（重要）：**\n"
        "- 只抽取模型在'最终结论/答案/小结'中明确主张为'符合查询要求'的元组。\n"
        "- 模型在过程中讨论的中间候选、被自己排除的、或仅作为对照列出的比赛，**不抽取**。\n"
        '    - 例：模型说"2009-12-25 是第三高得分场次，不是答案" → **不抽取**。\n'
        '    - 例：模型在结论里说"答案为 2007-02-11，骑士得分王是 Pavlović" → **抽取**。\n\n'
        "**并列处理：**\n"
        "- 若模型主张科比第二高得分有多场（多场比赛并列），每场作为一个独立 entry，"
        "其中各自填入该场对应的骑士得分王和得分。\n"
        "- 若模型对**同一场比赛**主张多个骑士队员并列得分王（如 A 和 B 都得 21 分），"
        "每个球员作为独立 entry，game_date 和 kobe_points 字段相同，"
        "cavs_top_scorer_name / cavs_top_scorer_points 各自不同。\n\n"
        "**字段格式约束（下游 Python 会做归一化兜底，但请尽量按下列格式抽出）：**\n"
        "- `game_date`：YYYY-MM-DD 格式字符串。模型若写'2007年2月11日'/'Feb 11, 2007'，"
        "请你统一转换为 '2007-02-11'。\n"
        "- `kobe_points` / `cavs_top_scorer_points`：纯整数（例如 36 / 21），"
        "**不要**带'分'/'pts'/'points'等后缀。\n"
        "- `cavs_top_scorer_name`：球员姓名，中英文皆可，按模型原文写。"
        "若模型写 'Sasha Pavlović (萨沙·帕夫洛维奇)'，可以保留括号原样写出，"
        "下游归一化层会处理变体。\n"
        "- 任何字段模型未明确给出 → 写 null（不要瞎填）。\n\n"
        "**特别注意：**\n"
        "- 若模型回答自相矛盾（前后给出不同答案），以**最终结论部分**为准。\n"
        "- 若模型完全没给出答案（比如只复述题目、未作答），整个 value 设为空 list "
        "且 not_mentioned=true。"
    ),
}

VALUE_SCHEMA = """{
  "value": [
    {
      "game_date": "<YYYY-MM-DD 格式字符串，如 '2007-02-11'。模型未给则 null>",
      "kobe_points": <科比该场得分（整数），如 36。模型未给则 null>,
      "cavs_top_scorer_name": "<骑士队内该场得分王姓名（中英文皆可）。模型未给则 null>",
      "cavs_top_scorer_points": <得分王该场得分（整数），如 21。模型未给则 null>,
      "note": "<其他相关备注（如比分、比赛地点、引用来源等）或 null>"
    }
  ],
  "not_mentioned": <true 仅当模型完全未给出任何答案时为 true；否则 false>,
  "supporting_span": "<原文片段 30-200 字，能证明你抽出的答案来自模型最终结论>",
  "confidence": "<high|medium|low>"
}"""
