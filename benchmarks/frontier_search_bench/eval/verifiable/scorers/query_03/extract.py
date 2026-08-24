"""Query 03 — SpaceX Starship integrated flight tests (IFT-1 to IFT-11).

The new query (locked 2026-04-29) asks only for:
  1. Total IFT count completed by 2026-01-01
  2. For EACH flight, list its key progress points and failure points.

Unlike the v2 design (5 entities: A/B/C/D/E for flight count + Mars
extrapolation + Musk delay multiplier), the v3 design uses 12 entities:
  - A_flight_count        — single integer
  - IFT_1_facts .. IFT_11_facts — composite {progress, failures} per flight
"""

from __future__ import annotations

QUERY_ID = 3

QUERY_TEXT = (
    "截至2026年1月1日，SpaceX 星舰一共完成了几次飞行测试？"
    "逐次列出每次的关键进展和失败点。"
)

ENTITIES = [
    {
        "id": "A_flight_count",
        "name": "星舰已完成飞行测试总次数（截至2026-01-01的整数）",
    },
    {
        "id": "IFT_1_facts",
        "name": "模型对 IFT-1（Integrated Flight Test 1）的关键进展和失败点描述",
    },
    {"id": "IFT_2_facts", "name": "模型对 IFT-2 的关键进展和失败点描述"},
    {"id": "IFT_3_facts", "name": "模型对 IFT-3 的关键进展和失败点描述"},
    {"id": "IFT_4_facts", "name": "模型对 IFT-4 的关键进展和失败点描述"},
    {"id": "IFT_5_facts", "name": "模型对 IFT-5 的关键进展和失败点描述"},
    {"id": "IFT_6_facts", "name": "模型对 IFT-6 的关键进展和失败点描述"},
    {"id": "IFT_7_facts", "name": "模型对 IFT-7 的关键进展和失败点描述"},
    {"id": "IFT_8_facts", "name": "模型对 IFT-8 的关键进展和失败点描述"},
    {"id": "IFT_9_facts", "name": "模型对 IFT-9 的关键进展和失败点描述"},
    {"id": "IFT_10_facts", "name": "模型对 IFT-10 的关键进展和失败点描述"},
    {"id": "IFT_11_facts", "name": "模型对 IFT-11 的关键进展和失败点描述"},
]

# ─────────────────────────────────────────────────────────────────────────
# Per-entity prompt hints
# ─────────────────────────────────────────────────────────────────────────

_IFT_FACTS_HINT = (
    "请抽出模型回答中**针对该次 IFT** 的关键进展（progress）和失败点/异常（failures）。\n\n"
    "**抽取规则：**\n"
    "- 每条事实是一个简短的字符串。例：'33 台 Raptor 全部点火' / 'LOX 滤网堵塞导致助推器爆炸'\n"
    "- 进展（progress）= 成功事项 / 首次完成的里程碑 / 数据指标达成（如速度、高度、点火数）\n"
    "- 失败点（failures）= 异常 / 故障 / 损伤 / 解体 / 任务中止 / 偏差等\n"
    "- 如果模型对该 IFT 完全没提到、或仅在引言/总结提到 IFT 编号但没具体细节，progress 和 failures 都置为 []\n"
    "- 不要替模型补充：模型没写的事实**不要从外部知识填充**\n"
    "- 模型如果把某条事实归错了 IFT 号，按模型实际写在哪段算（即模型说 'IFT-N: ...' 就归到 IFT_N_facts）\n"
    "- '溅落后爆炸（计划内）' 这种属于失败/异常类，归到 failures（即使是预期的）\n"
    "- 如果模型对某条事实给了具体数字（高度 km / 速度 km/h / 发动机数 / 时间戳）请保留\n\n"
    "**抽取范围约束：**\n"
    "- 只抽模型以肯定/接近肯定口吻的描述\n"
    "- 模型在'排除'、'不确定'、'据传'等强否定语境下的内容不抽"
)

PROMPT_HINTS = {
    "A_flight_count": (
        "模型声称的星舰已完成飞行测试总次数（IFT count）。**只输出一个整数**。\n"
        "如果模型说'~11次' / '约 11 次' / '11 次 IFT' 都按 11 输出。"
        "如果模型说'10-12 次范围'，输出代表性数字。"
        "如果模型完全没明确给出数字（仅枚举但不汇总），value 置 null 并 not_mentioned=true。"
    ),
    "IFT_1_facts": _IFT_FACTS_HINT,
    "IFT_2_facts": _IFT_FACTS_HINT,
    "IFT_3_facts": _IFT_FACTS_HINT,
    "IFT_4_facts": _IFT_FACTS_HINT,
    "IFT_5_facts": _IFT_FACTS_HINT,
    "IFT_6_facts": _IFT_FACTS_HINT,
    "IFT_7_facts": _IFT_FACTS_HINT,
    "IFT_8_facts": _IFT_FACTS_HINT,
    "IFT_9_facts": _IFT_FACTS_HINT,
    "IFT_10_facts": _IFT_FACTS_HINT,
    "IFT_11_facts": _IFT_FACTS_HINT,
}

# ─────────────────────────────────────────────────────────────────────────
# Schemas
# ─────────────────────────────────────────────────────────────────────────

# A_flight_count: simple integer entity
_SCHEMA_A = """{
  "value": <整数（如 11）或 null>,
  "not_mentioned": <true|false>,
  "supporting_span": "<原文片段 30-200 字>",
  "confidence": "<high|medium|low>"
}"""

# IFT_N_facts: composite with two lists
_SCHEMA_IFT = """{
  "value": {
    "progress": ["<事实字符串1>", "<事实字符串2>", ...],
    "failures": ["<事实字符串1>", "<事实字符串2>", ...]
  },
  "not_mentioned": <true 仅当模型完全没提该 IFT 时；否则 false>,
  "supporting_span": "<原文片段 30-200 字，证明你的抽取来自模型回答>",
  "confidence": "<high|medium|low>"
}"""

# extraction_pipeline.run_pipeline accepts a single schema string. We use
# IFT schema as default and attach the simpler A schema via per-entity
# override if the pipeline supports it. To stay compatible with the existing
# run_pipeline signature (one schema for all entities), we use IFT schema —
# the analyzer will down-cast to integer for A_flight_count by reading the
# `value` field tolerantly.
VALUE_SCHEMA = _SCHEMA_IFT

# Per-entity schema map — auto_scorer.py can pass this via `schemas=` if the
# extraction_pipeline supports per-entity schema overrides; otherwise the
# default VALUE_SCHEMA is used.
SCHEMAS_BY_ENTITY = {
    "A_flight_count": _SCHEMA_A,
    "IFT_1_facts": _SCHEMA_IFT,
    "IFT_2_facts": _SCHEMA_IFT,
    "IFT_3_facts": _SCHEMA_IFT,
    "IFT_4_facts": _SCHEMA_IFT,
    "IFT_5_facts": _SCHEMA_IFT,
    "IFT_6_facts": _SCHEMA_IFT,
    "IFT_7_facts": _SCHEMA_IFT,
    "IFT_8_facts": _SCHEMA_IFT,
    "IFT_9_facts": _SCHEMA_IFT,
    "IFT_10_facts": _SCHEMA_IFT,
    "IFT_11_facts": _SCHEMA_IFT,
}
