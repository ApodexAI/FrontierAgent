"""
Query 02 — 中国国家级新区 GDP 增速差值 / 由正转负判断

Task: 对 2005-01-01 ~ 2026-01-01 间获批的 18 个国家级新区，模型应给出
  (1) 历年 GDP 增速 vs 全国增速差值；
  (2) 哪些新区差值曾由正转负；
  (3) 首次转负年份。

抽取层 spec：每个模型回答中按"新区为单位"抽出三种判断之一：
  - flipped         : 主张该新区差值"曾由正转负"（可附年份）
  - not_flipped     : 主张该新区"从未转负"（即始终保持正差值）
  - indeterminate   : 主张"无法严谨判断"或"数据不完整"

This module exports constants only. The runner is auto_scorer.py.
"""

from __future__ import annotations

QUERY_ID = 2

QUERY_TEXT = (
    "中国在2005年1月1日到2026年1月1日之间获批的所有国家级新区，"
    "各自历年GDP增速与同期全国GDP增速的差值是多少？"
    "其中，哪些新区的差值曾由正转负？首次转负分别发生在哪一年？"
)


ENTITIES = [
    {
        "id": "area_judgments",
        "name": (
            "模型对各国家级新区'GDP 增速差值是否曾由正转负'的判断清单。"
            "每个新区一条记录，含 area_name / claim_type / first_flip_year(可选) / reason(可选)"
        ),
    }
]


PROMPT_HINTS = {
    "area_judgments": (
        "请抽出模型回答里**针对每个国家级新区的明确判断**。每个新区一条 JSON 记录。\n\n"
        "**三种 claim_type：**\n\n"
        "1. `flipped` — 模型主张该新区差值'曾由正转负'\n"
        "   - 例：'湘江新区在 2021 年首次由正转负' → flipped + first_flip_year='2021'\n"
        "   - 例：'南沙新区差值历史上转过负'（无具体年份） → flipped + first_flip_year=null\n"
        "   - 模型若给出'差值序列'里某年从正变负，按 flipped 抽取；first_flip_year 填**模型主张的首次转负年**\n\n"
        "2. `not_flipped` — 模型主张该新区'从未发生由正转负'\n"
        "   - 例：'天府新区始终保持正差值' → not_flipped\n"
        "   - 例：'江北新区在观察期内未发生转负' → not_flipped\n"
        "   - 例：'X 新区差值持续为正' → not_flipped\n"
        "   - **注意**：模型只是没把某新区列入'转负名单'，**不算** not_flipped 主张；必须有明确文字否定才算\n\n"
        "3. `indeterminate` — 模型主张'无法严谨判断 / 数据不完整 / 口径多变'\n"
        "   - 例：'雄安新区由于无独立 GDP 数据，无法判断' → indeterminate\n"
        "   - 例：'西咸新区因口径多变，难以严谨给出转负年份' → indeterminate\n"
        "   - 例：'兰州新区缺乏逐年增速点值，无法判定首次转负' → indeterminate\n\n"
        "**抽取范围约束（重要）：**\n"
        "- 只抽取模型对**国家级新区**的判断；省级新区/经开区/自贸试验区/高新区**不抽**\n"
        "- 浦东新区（1992 获批）**虽不在 query 时间窗内**，但若模型把它列出仍要抽出"
        "（下游会判越界扣分）\n"
        "- 同一新区被模型多次描述（先 flipped 后又说 indeterminate 等）→ **每个独立判断抽一条**\n"
        "- 模型若仅列了 GDP 数据表但**没作'转负与否'判断**的新区，不抽（无主张）\n"
        "- 不要从外部知识补全：模型没说的事实不要填\n\n"
        "**字段说明：**\n"
        "- `area_name`: 新区中文名（保留模型原文，如'湘江新区'/'湖南湘江新区'/'长沙湘江新区'）\n"
        "- `claim_type`: flipped / not_flipped / indeterminate（三选一）\n"
        "- `first_flip_year`: 仅 claim_type=flipped 且模型给出年份时填字符串（如'2021'）；否则 null\n"
        "- `reason`: 模型给出的简短理由 / 关键论据（可选，<= 80 字）\n\n"
        "**抽取范围（语气）：**\n"
        "- 只抽取模型以**肯定或接近肯定**口吻的判断\n"
        "- 模型在'不可考'、'存疑'、'不确定但倾向于'等半肯定语境下：\n"
        "    - 若是关于'是否转负'的犹豫 → 按 indeterminate 抽\n"
        "    - 若是关于'年份'的犹豫但确认转负 → 按 flipped + first_flip_year=null（或最可能年）\n"
    ),
}


VALUE_SCHEMA = """{
  "value": [
    {
      "area_name": "<新区中文名（保留模型原文，如'湘江新区'/'湖南湘江新区'）>",
      "claim_type": "<flipped|not_flipped|indeterminate>",
      "first_flip_year": "<年份字符串如 '2021'；仅 claim_type=flipped 且模型给出年份时填，否则 null>",
      "reason": "<模型给出的简短理由 / 关键论据，<=80 字 / null>"
    }
  ],
  "not_mentioned": <true 仅当模型完全未对任何新区作判断时为 true；否则 false>,
  "supporting_span": "<原文片段 30-200 字，能证明你抽出的列表来自回答>",
  "confidence": "<high|medium|low>"
}"""
