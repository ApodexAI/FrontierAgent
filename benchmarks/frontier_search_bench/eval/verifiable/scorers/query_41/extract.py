"""Query 41 — Empire State Building direction puzzle + visible water body.

Question (Chinese): standing on the Empire State Building observation deck
facing north, after a sequence of 4 turns (right 90°, right 45°, left 200°,
right 15°):
  Part A — what compass direction are you facing?
  Part B — looking out, what body of water do you see?

Reference answer:
  A: 310° / NW / Northwest / 西北 (within ±5° of 310°)
  B: Hudson River (Hudson / 哈德逊河 / 哈得孙河 / 哈得逊河 / 哈德森河)

T1 granularity — one composite value (degree + compass + water body).
"""

from __future__ import annotations

QUERY_ID = 41

QUERY_TEXT = (
    "你站在Empire State Building的观景台上，面朝正北方（朝向中央公园方向）。"
    "朋友在电话里给你指令：1. 向右转 90° 2. 再向右转 45° 3. 向左转 200° "
    "4. 再向右转 15°。问题A：你现在面朝哪个罗盘方向？"
    "问题B：你往远处看，能看到一片水域，那是什么水域？"
)

ENTITIES = [
    {
        "id": "direction_water_answer",
        "name": (
            "模型对两小问的最终答案：A 罗盘方向（度数和/或文字方向名）"
            "+ B 远处可见水域名"
        ),
    }
]

PROMPT_HINTS = {
    "direction_water_answer": (
        "请抽取模型对题目两小问的最终答案。\n\n"
        "**Part A — 罗盘方向：**\n"
        "- **degree**：模型给出的最终度数（整数，0-359）。如果模型只给文字方向"
        "  没给度数，degree 留 null。\n"
        "- **compass_name**：模型给出的方向文字名"
        "（如 'NW'、'Northwest'、'西北'、'北偏西50°'、'WNW'、'西北偏西' 等）。\n"
        "- 模型在推导过程中提到的中间值不抽，只抽最终结论。\n\n"
        "**Part B — 水域：**\n"
        "- **water_body**：模型最终给出的水域名"
        "（如 'Hudson River'、'哈德逊河'、'East River'、'东河' 等）。\n"
        "- 如果模型同时提到多个水域，以模型主张'你能看到的'那个为准；"
        "明确排除的不算。\n\n"
        "若模型完全没回答某一小问，对应字段留 null。"
    ),
}

VALUE_SCHEMA = """{
  "value": {
    "part_a": {
      "degree": <模型最终给出的度数（整数 0-359）或 null>,
      "compass_name": "<模型给出的方向文字名或 null>"
    },
    "part_b": {
      "water_body": "<模型最终给出的水域名或 null>"
    }
  },
  "not_mentioned": <true 仅当模型完全没回答两个小问；否则 false>,
  "supporting_span": "<原文片段 30-200 字，证明上面字段确实来自模型回答>",
  "confidence": "<high|medium|low>"
}"""
