"""
Query 36 — GeoGuessr NMPZ photo coordinate guess.

Single-fact coordinate question: identify lat/lon for a Greek roadside
scene featuring Mickey/Minnie graffiti on a retaining wall, narrow
two-lane road with white dashed centerline, wooden utility pole, dense
vegetation, rolling green hills. Reference answer: 38.6449°N, 24.0229°E.

T1 granularity — the whole answer is one extraction with a composite
value. Multi-candidate aware: any candidate within ±10° in BOTH lat and
lon counts as a hit.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS.parent))

from pipeline.extraction_pipeline import run_pipeline  # noqa: E402

QUERY_ID = 36
QUERY_TEXT = (
    "I was playing GeoGuessr (NMPZ mode) with my friends and was puzzled by a "
    "photo featuring a narrow two-lane asphalt road in good condition with no "
    "major damage, marked by a single white dashed center line, which gently "
    "curves and rises slightly in the distance alongside a wooden utility/"
    "electricity pole on the right side with overhead wires running across. "
    "On both sides, there is dense, lush green vegetation with mixed green tones—"
    "predominantly broadleaf trees and shrubs that also appear dry and scrubby—"
    "where the trees on the left are particularly dense, overgrown, and almost "
    "encroaching on the road. Partially obscured by this vegetation in the "
    "mid-distance on the left side is a small white/light-coloured building that "
    "appears to be a modest, single-storey residential structure. Running along "
    "the right side of the road is a low stone/concrete retaining wall covered "
    "in informal, urban-style colorful graffiti in tag-style lettering with blue "
    "and yellow tones, located right next to a red, black, and white cartoon "
    "depiction of Mickey Mouse and Minnie Mouse in classic Disney style. The "
    "background reveals rolling green hills, indicating a terrain that is hilly "
    "rather than mountainous; based on all these details, can you give me the "
    "latitude and longitude coordinates for the answer?"
)


ENTITIES = [
    {
        "id": "geoguessr_coordinate_guess",
        "name": "模型给出的 GeoGuessr 题目坐标猜测（主答案 + 备选候选）",
    },
]


PROMPT_HINTS = {
    "geoguessr_coordinate_guess": (
        "题目是一道 GeoGuessr 坐标识别题。请抽出模型回答中**所有**给出的经纬度坐标，"
        "包括主答案、备选候选、范围中心、'约/大约/approximately' 形式的坐标等。\n\n"
        "**抽取要点：**\n"
        "- `lat`：纬度（数值）。北纬为正，南纬转为负数。\n"
        "- `lon`：经度（数值）。东经为正，西经转为负数。\n"
        "- `location_name`：模型声称的地点名称（如'希腊埃维亚岛'）/ null\n"
        "- `country`：国家 / null\n"
        "- `confidence_label`：模型自评置信度（如 high/medium/low）/ null\n"
        "- `refused_to_answer`：模型是否明确拒绝给出坐标（true/false）\n\n"
        "**抽取约束：**\n"
        "- 只抽数值坐标；若模型只给国家/城市名而无具体经纬度数字，**不抽**\n"
        "- 度分秒（DMS）形式请换算为十进制度，例：38°38'41\"N → 38.6447\n"
        "- 多个候选全部抽出，主答案放 `value`，备选放 `alternative_candidates`\n"
        "- 模型若拒绝给出坐标，主 value 各字段为 null，refused_to_answer=true，"
        "  alternative_candidates 为空"
    ),
}


VALUE_SCHEMA = """{
  "value": {
    "lat": <数值（北正南负） / null>,
    "lon": <数值（东正西负） / null>,
    "location_name": "<地点名称 / null>",
    "country": "<国家 / null>",
    "confidence_label": "<high|medium|low / null>",
    "refused_to_answer": <true|false>,
    "alternative_candidates": [
      {"lat": <数值/null>, "lon": <数值/null>, "location_name": ..., "country": ..., "confidence_label": ...}
    ]
  },
  "not_mentioned": <true 仅当模型未做任何坐标尝试时为 true；否则 false>,
  "supporting_span": "<原文片段 30-200 字，能证明你抽出的坐标来自回答>",
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
