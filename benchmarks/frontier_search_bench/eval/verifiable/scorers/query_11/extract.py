"""
Query 11 — As of 2026-01-01, papers submitted to ICLR/NeurIPS/ICML,
rejected, never accepted again, citation > 10000.

T1 granularity: one composite entity holding the full list of paper claims.
Schema mirrors query_12 / query_06 / query_10 conventions so downstream
auto_scorer can consume `extraction.json.canonical.value` uniformly.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS.parent))

from pipeline.extraction_pipeline import run_pipeline  # noqa: E402

QUERY_ID = 11
QUERY_TEXT = (
    "截至2026年1月1日，帮我找到有哪些paper投过三大会（iclr，neurips，icml）"
    "但是被拒之后不再中稿，但是citation超过10000的论文"
)


ENTITIES = [
    {
        "id": "rejected_high_citation_papers",
        "name": (
            "模型列出的'投过 ICLR/NeurIPS/ICML 三大会被拒、之后不再中稿、"
            "且 citation 超过 10000'的论文清单"
        ),
    },
]


PROMPT_HINTS = {
    "rejected_high_citation_papers": (
        "请抽出模型回答里**明确主张**为'投过 ICLR/NeurIPS/ICML 三大会、被拒之后不再中稿、"
        "且 citation 超过 10000'的论文。每篇一个字段齐全的 JSON 对象。\n\n"
        "**抽取范围约束（重要）：**\n"
        "- 只抽取模型以**肯定或接近肯定**口吻主张的案例。'据传'、'有说法称'、'可能'这类半肯定按肯定处理（保留）\n"
        "- **不要**抽取模型在'排除'、'不符合'、'已被接收'、'实际发表于某会议'、"
        "  '作为反例'、'边界案例'、'存疑'、'仅是 workshop 不算正式接收' 等语境下讨论的条目；"
        "  这些是模型自己否定的案例\n"
        "- 模型若把论文分组为'正确答案 vs 边界案例 vs 错误答案/反例'，"
        "  **只抽取'正确答案/确认满足条件'那一组**\n"
        "- 示例：\n"
        '    - 模型说"Word2Vec 投 ICLR 2013 被拒后再未中稿，引用 14 万+，满足条件" → 抽取\n'
        '    - 模型说"BERT 也曾被拒，但后来中了 NAACL 2019，故排除" → **不抽取**（模型自己排除了）\n'
        '    - 模型说"YOLO 严格说没投过三大会，作为反例" → **不抽取**\n\n'
        "**抽取范围说明（重要）：**\n"
        "- 只抽取模型主张为'机器学习/AI 论文'的条目；非论文条目（如某些课程或代码库）忽略\n"
        "- 同一篇论文若模型在文中以多种简称出现（如 Word2Vec 与 'Mikolov 2013'），"
        "  按字面合并为一条；下游对齐层会去重"
    ),
}


VALUE_SCHEMA = """{
  "value": [
    {
      "name": "<论文标题或公认简称（中英文皆可，例：Word2Vec / Distilling the Knowledge in a Neural Network）>",
      "year": "<论文年份字符串 / null>",
      "first_author": "<第一作者（例：Mikolov / Hinton）/ null>",
      "rejected_venue": "<被哪个会议拒：ICLR / NeurIPS / ICML / 多个 / null>",
      "current_citation": "<声称的引用数（整数或字符串，例：'14万+' / 25000）/ null>",
      "later_published_at": "<之后发表的会议/期刊/workshop 名称；'arXiv only'：仅留 arXiv / null>",
      "note": "<其他关键证据，例如年份、被拒理由、影响力评价等 / null>"
    }
  ],
  "not_mentioned": <true 仅当模型完全未列任何论文时为 true；否则 false>,
  "supporting_span": "<原文片段 30-200 字，能证明你抽出的列表来自回答>",
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
