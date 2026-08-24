"""
Query 35 — Haidilao farthest store pair per country.
T2 granularity: 18 entities (one per country).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS.parent))

from pipeline.extraction_pipeline import run_pipeline  # noqa: E402

QUERY_ID = 35
QUERY_TEXT = "帮我调研，截至2026年1月1日，海底捞在每个国家相距最远的两家店之间有多远，每个国家是哪两家店？"

COUNTRIES = [
    ("D1", "中国大陆"),
    ("D2", "香港"),
    ("D3", "澳门"),
    ("D4", "台湾"),
    ("D5", "美国"),
    ("D6", "加拿大"),
    ("D7", "英国"),
    ("D8", "澳大利亚"),
    ("D9", "日本"),
    ("D10", "韩国"),
    ("D11", "新加坡"),
    ("D12", "马来西亚"),
    ("D13", "泰国"),
    ("D14", "越南"),
    ("D15", "印度尼西亚"),
    ("D16", "菲律宾"),
    ("D17", "阿联酋"),
    ("D18", "柬埔寨"),
]

ENTITIES = [{"id": cid, "name": f"{name} 最远门店对 + 距离"} for cid, name in COUNTRIES]

PROMPT_HINTS = {
    cid: (
        f"抽出模型对 **{name}** 的声明："
        "最远的两家门店名（store_a / store_b）+ 它们之间的距离（km）。"
        "若模型说该国只有 1 家店，置 only_one_store=true，store_b 和 distance_km 均 null。"
        "若该国未被模型提及，not_mentioned=true。"
    )
    for cid, name in COUNTRIES
}

VALUE_SCHEMA = """{
  "value": {
    "store_a": "<第一家门店名（城市+商场/门店）；未提则 null>",
    "store_b": "<第二家门店名；若只有1家店则 null>",
    "distance_km": <距离数字（km）；未提则 null>,
    "only_one_store": <true|false|null>
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
