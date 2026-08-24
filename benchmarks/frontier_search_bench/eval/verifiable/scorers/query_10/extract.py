"""
Query 10 — 2016-2025 Chinese-language films meeting 3 specific criteria.

T1 granularity: one composite entity holding the full list of films.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS.parent))

from pipeline.extraction_pipeline import run_pipeline  # noqa: E402

QUERY_ID = 10
QUERY_TEXT = (
    "在2016-2025的华语电影中，有哪些电影同时满足以下三个条件："
    "（1）豆瓣评分≥8.0；（2）国内票房<1亿人民币；"
    "（3）获得了金鸡奖的最佳故事片提名（不含获奖）或者金像奖的最佳电影提名（不含获奖）。"
    "请列出这些电影，并说明每部电影的导演和上映年份。"
)

ENTITIES = [
    {
        "id": "qualifying_films",
        "name": "模型列出的满足三条件（豆瓣≥8 + 票房<1亿 + 金鸡/金像最佳故事片/电影提名不含获奖）的华语电影清单",
    },
]

PROMPT_HINTS = {
    "qualifying_films": (
        "请抽出模型回答中**明确主张**为'满足三条件的华语电影'的条目。每部电影一个 JSON 对象。\n\n"
        "**抽取范围约束（重要）：**\n"
        "- 只抽取模型以**肯定或接近肯定**口吻作为候选/答案列出的电影。'可能也算'、'据报道符合'这种半肯定表述，按肯定处理（保留）\n"
        "- **不要**抽取模型在'排除'、'不符合'、'存疑'、'差一点/擦肩'、'不在名单'、'应排除'、'虽然提名但条件不符'等语境下讨论的电影；这些是模型自己否定/排除的案例，不构成它的主张\n"
        "- 示例：\n"
        '    - 模型说"《X》符合三条件" → 抽取\n'
        '    - 模型说"《Y》豆瓣只有 7.9，差一点" → **不抽取**（模型自己说排除了）\n'
        '    - 模型说"《Z》虽然提名但并非最佳故事片类别，应排除" → **不抽取**\n'
        '    - 模型说"《W》我觉得也可以算" → 抽取（半肯定按肯定处理）\n\n'
        "若模型提到的电影**自身主张**满足但实际豆瓣/票房数据与基准不符（如模型以为豆瓣 8.1 其实 7.9），这仍属于"
        "模型的肯定主张，**要抽出**（下游评分由对齐/打分层按真实基准判定）。"
        "若模型说获奖了（而非提名），如实记录 award_won=true。"
    ),
}

VALUE_SCHEMA = """{
  "value": [
    {
      "film_name": "<电影名，去除《》书名号>",
      "director": "<导演>",
      "release_year": "<年份字符串，如 2019 或 2018/2019>",
      "douban_rating": "<豆瓣评分字符串或 null>",
      "box_office": "<票房描述或 null>",
      "award_nomination": "<具体奖项+届次，如 '第32届金鸡奖最佳故事片提名' 或 null>",
      "award_won": <true|false|null>,
      "note": "<其他关键信息 / null>"
    }
  ],
  "not_mentioned": <true 仅当模型完全未列任何电影时为 true；否则 false>,
  "supporting_span": "<原文片段 30-200 字，能证明抽出的列表>",
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
