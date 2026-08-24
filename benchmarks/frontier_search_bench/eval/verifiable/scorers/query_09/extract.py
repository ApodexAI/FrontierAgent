"""Query 09 — Chinese Olympians with ≥3 medals but no Olympic gold.

Question (Chinese): up to and including the 2024 Paris Olympics, list all
Chinese athletes (active or retired) who have won 3 or more Olympic medals
but never an Olympic gold.

Reference answer: 19-name list (locked v1.1, 2026-05-05). Includes summer
& winter, individual & team, e.g. 黄雪辰 (synchronised swimming), 李佳军
(short-track), 叶乔波 (speed skating), 王春露 (short-track), etc.

Easy-to-confuse traps (in known-incorrect):
  - 杨扬 ≠ 杨阳 (杨扬 won gold in 2002 short-track)
  - 李静 ≠ 李敬 (different person, fails the medal-count condition)
  - 王皓 (table tennis) won team gold in 2008/2012
  - 呙俐, 陈晓君: <3 medals

T-list granularity — the answer is a flat list of athlete names.
"""

from __future__ import annotations

QUERY_ID = 9

QUERY_TEXT = (
    "截至2024年巴黎奥运会（含），在奥运会历史上，"
    "获得过3枚及以上奖牌但从未拿到过金牌的中国运动员（现役+退役）有哪些？"
)

ENTITIES = [
    {
        "id": "athletes_list",
        "name": "模型作为最终答案给出的'≥3 枚奥运奖牌但无金牌的中国运动员'列表",
    }
]

PROMPT_HINTS = {
    "athletes_list": (
        "请抽取模型作为最终答案明确列出的运动员名单。\n\n"
        "**抽取规则：**\n"
        "- **只**抽取模型明确作为答案列出的运动员；不要抽取仅在讨论 / 排除 /"
        "对比语境中提到但非答案的人物（例如模型说'某某有金牌不算'的那个人）。\n"
        "- 统一输出**中文姓名**：英文名请按常见对应转回中文"
        "（例如 'Huang Xuechen' → '黄雪辰'、'Pang Jiaying' → '庞佳颖'）。\n"
        "- 注意区分易混名：杨阳（无金，正确）≠ 杨扬（有金，错误）；"
        "李敬（正确）≠ 李静（错误，不同人）。模型怎么写就怎么记，由对齐层判定。\n"
        "- 模型若标注'不确定'/'存疑'，仍视为主张，将 uncertain 设为 true。\n"
        "- 模型若明确表示无法回答 / 找不到，返回空数组，not_mentioned 设为 true。"
    ),
}

VALUE_SCHEMA = """{
  "value": [
    {
      "name": "<运动员中文姓名>",
      "sport": "<项目（如游泳/短道速滑/体操/...）/ null>",
      "medal_count": "<奖牌总数（数字）/ null>",
      "uncertain": <true|false>
    }
  ],
  "not_mentioned": <true 仅当模型完全没列出任何运动员；否则 false>,
  "supporting_span": "<原文片段 30-200 字，证明列表来自模型回答>",
  "confidence": "<high|medium|low>"
}"""
