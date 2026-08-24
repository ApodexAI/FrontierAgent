"""
Query 24 — Did the female reporter in《等等》participate in other documentaries?

T1 granularity: one composite entity with sub-fields.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS.parent))

from pipeline.extraction_pipeline import run_pipeline  # noqa: E402

QUERY_ID = 24
QUERY_TEXT = (
    "中国大陆县中高考教育生态纪录片《等等》其中出现的女记者还参与过其他的纪录片拍摄吗？"
)

ENTITIES = [
    {
        "id": "reporter_docs",
        "name": "《等等》女记者身份 + 是否参与其他纪录片/影像作品",
    },
]

PROMPT_HINTS = {
    "reporter_docs": (
        "问题两层含义：(1) 模型是否识别出《等等》中的女记者是谁（姓名）；"
        "(2) 她是否还参与过其他纪录片或影像作品？若是，列出作品名。"
        "若模型明确声明'没有其他纪录片'，请抽出 'no_other_works' 为 true。"
    ),
}

VALUE_SCHEMA = """{
  "value": {
    "reporter_name": "<女记者姓名；未识别则 null>",
    "no_other_works": <true|false|null;
        true = 模型明确声称除《等等》外无其他纪录片;
        false = 模型声称还参与过其他作品;
        null = 模型未就此表态>,
    "other_works": ["<其他作品名 1>", "<其他作品名 2>", ...]
  },
  "not_mentioned": <true|false>,
  "supporting_span": "<原文片段 30-200 字>",
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
