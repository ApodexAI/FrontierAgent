"""
Query 21 — Proxy-accepted Nobel Prize year
Thin entry calling the shared extraction pipeline.

Usage:
    python query_21/extract.py \
        --models claude=/path/to/claude.json \
                 apodex=/path/to/apodex.json \
        --output-dir query_21/auto_scores/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure we can import pipeline when run from either repo root or this folder.
THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS.parent))

from pipeline.extraction_pipeline import run_pipeline  # noqa: E402

QUERY_ID = 21
QUERY_TEXT = (
    "截至2026年1月1日，有一个人代表获奖者人领了一个很重要的世界级奖项，"
    "这个奖项的颁奖时间是哪一年？我记得这个人的名字中有do这两个字母。"
    "被领奖人的名字中有in这两个字母。"
)

# T1 granularity — the whole answer is one extraction with a composite value.
ENTITIES = [
    {
        "id": "award_event",
        "name": "被代领的国际奖项事件（年份 + 奖项 + 代领人 + 被领奖人 + 缺席原因）",
    },
]

PROMPT_HINTS = {
    "award_event": (
        "题目问一个由他人代领的国际大奖：颁奖年份是哪一年？"
        "代领人名字含连续字母 'do'；被领奖人名字含连续字母 'in'。"
        "请抽出模型声明的：年份、奖项名称、代领人、被领奖人、缺席原因。"
        "若模型给了多个候选，全部列出。"
    ),
}

VALUE_SCHEMA = """{
  "value": {
    "year": <整数或 null>,
    "award_name": "<字符串或 null>",
    "proxy_name": "<字符串或 null; 代领人>",
    "winner_name": "<字符串或 null; 被领奖人>",
    "absence_reason": "<字符串或 null; 缺席原因>",
    "alternative_candidates": [
      { "year": ..., "award_name": ..., "proxy_name": ..., "winner_name": ..., "absence_reason": ... }
    ]
  },
  "not_mentioned": <true|false>,
  "supporting_span": "<原文片段 30-200 字>",
  "confidence": "<high|medium|low>"
}"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--models",
        nargs="+",
        required=True,
        help="name=path list of model answer JSON files.",
    )
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
