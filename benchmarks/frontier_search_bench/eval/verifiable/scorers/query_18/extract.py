"""
Query 18 — 建筑师谜题：找出符合多重线索的摩天大楼现名

T1 single-answer: model is asked for ONE building (the current name).
The entity holds at most one claim. Schema mirrors query_06/10 conventions
so downstream auto_scorer can consume `extraction.json.canonical.value`
uniformly.

Correct answer: OUE Downtown 1（原 DBS Building Tower 1，2010 改名）
- Architect: Lim Chong Keat (MIT)
- Inventor friend: Buckminster Fuller (geodesic dome)
- Island meetings: Campuan Meetings (Bali / Penang)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS.parent))

from pipeline.extraction_pipeline import run_pipeline  # noqa: E402

QUERY_ID = 18
QUERY_TEXT = (
    "有一位建筑师，他曾在美国一所著名理工大学学习建筑，后来和一位以某种球形建筑结构闻名的"
    "美国发明家成为终生好友，两人还在一个岛屿组织过一系列知识交流会。这位建筑师设计了一栋"
    "摩天大楼，地基打了巨型圆形沉箱。这栋楼现在的名字是什么？"
)


ENTITIES = [
    {
        "id": "skyscraper_current_name",
        "name": "模型给出的'这栋摩天大楼现在的名字'最终答案",
    },
]


PROMPT_HINTS = {
    "skyscraper_current_name": (
        "请抽出模型回答里**最终主张**的'这栋摩天大楼现在的名字'。"
        "本题答案只有一栋楼，**最多抽 1 条**。\n\n"
        "**抽取范围约束（重要）：**\n"
        "- 题目硬限定'**现在的名字**'（current name），所以：\n"
        "  - 模型同时给历史名+现名（如 '原 DBS Building，现 OUE Downtown 1'）→ 抽取**现名**\n"
        "  - 模型只给历史名（如 'DBS Building' / '星展大厦'）未提现名 → 仍然抽取它给的历史名"
        "（下游评分会判错；这是模型违反'现在的名字'限定的考点）\n"
        "- 只抽取模型以**肯定或接近肯定**口吻给出的最终答案。'据传'、'可能'、'有说法称'"
        "这类半肯定按肯定处理（保留）\n"
        "- **不要**抽取模型自己**否定**或**hedge**的答案：\n"
        "  - 模型说'无法确定'、'信息不足'、'无法给出确定答案'、'更负责任的说法是无法确定' "
        "→ not_mentioned=true（视为未答）\n"
        "  - 模型在推理过程中提到 X 但最终结论是'不能确认是 X' → 不抽\n"
        "  - 模型列出多个候选（A、B、C）但未择其一 → 不抽\n"
        "- 模型给出的可能是错误建筑（如 KOMTAR、香港中银大厦、Aon Center 等）"
        "  只要模型当作肯定主张给出，仍要抽取（下游对齐层会判对错）\n\n"
        "**示例：**\n"
        "  - 模型说'答案是 OUE Downtown 1（原 DBS Building）' → 抽 building_name_now='OUE Downtown 1'，"
        "building_name_historical='DBS Building'\n"
        "  - 模型说'是 DBS Building' 只字未提改名 → 抽 building_name_now='DBS Building'\n"
        "  - 模型说'综合所有信息无法给出确定答案' → not_mentioned=true，value=[]\n"
        "  - 模型说'香港中银大厦（贝聿铭设计）' → 抽 building_name_now='香港中银大厦'，"
        "architect_claimed='贝聿铭'\n"
    ),
}


VALUE_SCHEMA = """{
  "value": [
    {
      "building_name_now": "<模型主张的建筑现名（中英文皆可，如 'OUE Downtown 1' / '大华城市中心一号' / 'DBS Building' / '香港中银大厦'）>",
      "building_name_historical": "<模型主张的历史名 / null（如模型只给一个名字未区分新旧）>",
      "architect_claimed": "<模型主张的建筑师姓名 / null>",
      "inventor_friend_claimed": "<模型主张的发明家朋友姓名 / null>",
      "location": "<建筑所在城市/国家 / null>",
      "note": "<其他关键信息（如改名年份、岛屿名等）/ null>"
    }
  ],
  "not_mentioned": <true 仅当模型完全未给出建筑名（含 hedge/无法确定）；否则 false>,
  "supporting_span": "<原文片段 30-200 字，能证明你抽出的内容来自回答（包含建筑名的关键句）>",
  "confidence": "<high|medium|low>"
}"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--primary-model", default=None)
    ap.add_argument("--secondary-model", default=None)
    ap.add_argument("--parallel-models", nargs="+", default=None)
    ap.add_argument("--analyzer-model", default=None)
    ap.add_argument("--concurrency", type=int, default=None)
    args = ap.parse_args()

    run_pipeline(
        query_id=QUERY_ID,
        query_text=QUERY_TEXT,
        entities=ENTITIES,
        prompt_hints=PROMPT_HINTS,
        schema=VALUE_SCHEMA,
        models_input=args.models,
        output_dir=Path(args.output_dir),
        primary=args.primary_model,
        secondary=args.secondary_model,
        parallel=args.parallel_models,
        analyzer=args.analyzer_model,
        concurrency=args.concurrency,
    )


if __name__ == "__main__":
    main()
