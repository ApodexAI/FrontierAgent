"""
Query 34 — Peninsula hotels with To Summer (观夏) stores within 1km.
T1 granularity: one entity holding the candidate list of qualifying
Peninsula hotels the model claims have a To Summer (观夏) store within 1km.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS.parent))

from pipeline.extraction_pipeline import run_pipeline  # noqa: E402

QUERY_ID = 34
QUERY_TEXT = "帮我找到哪些半岛酒店附近1km内有观夏的店铺？"

ENTITIES = [
    {
        "id": "qualifying_peninsula_hotels",
        "name": (
            "模型主张为本题答案的半岛酒店（即附近 1km 内有观夏门店的那些半岛酒店）。"
            "包括酒店所在城市、酒店名称/地址、模型主张的就近观夏门店、距离表述。"
        ),
    }
]

PROMPT_HINTS = {
    "qualifying_peninsula_hotels": (
        "请抽取模型在回答中**真正作为答案纳入**的半岛酒店——即模型主张其 1km 内有观夏门店的那些。\n\n"
        "**抽取规则：**\n"
        "- 一家半岛酒店 = 一个对象。\n"
        "- **只抽取模型作为正向答案纳入的半岛酒店**；模型明确说\"X 半岛附近没有观夏\"或"
        "\"X 半岛 1km 内无观夏\"的酒店不抽，但可放进 `excluded_hotels` 字段（用于诊断）。\n"
        "- 若模型把题目理解错（比如列了非半岛酒店、列了观夏门店本身），照抓不误——交给 scorer 判错。\n"
        "- **city**：酒店所在城市（如 \"上海\" / \"Shanghai\" / \"北京\" / \"Beijing\" / \"香港\" 等）。\n"
        "- **hotel_name**：半岛酒店的官方名/原文写法（如 \"上海半岛酒店\" / \"The Peninsula Shanghai\" / \n"
        "  \"王府半岛\" / \"北京王府半岛酒店\" 等）。\n"
        "- **hotel_address**：酒店地址（如 \"中山东一路32号\" / \"金鱼胡同8号\"），原文写法即可。\n"
        "- **nearby_to_summer_store**：模型主张的就近观夏门店名/位置（如 \"观夏外滩源店\"、\n"
        "  \"观夏书阁 WF Central 店\"），原文写法即可。\n"
        "- **claimed_distance**：模型给出的距离描述（如 \"约 150-250 米\"、\"380 米\"、\"约 0.4 公里\"）。\n"
        "- 若模型明确说\"全球所有半岛酒店都没有 1km 内观夏\"或类似结论，items 返回空数组、"
        "not_mentioned=true。\n"
        "- 若模型只讨论观夏 / 半岛各自的门店列表却没给出最终答案配对，不抽。"
    ),
}

VALUE_SCHEMA = """{
  "value": {
    "items": [
      {
        "city": "<城市字符串或 null>",
        "hotel_name": "<半岛酒店名称字符串或 null>",
        "hotel_address": "<酒店地址字符串或 null>",
        "nearby_to_summer_store": "<就近观夏门店字符串或 null>",
        "claimed_distance": "<距离描述字符串或 null>"
      }
    ],
    "excluded_hotels": [
      {
        "city": "<被模型主动排除的半岛酒店所在城市>",
        "hotel_name": "<...>",
        "reason": "<模型给出的排除理由，可简写>"
      }
    ]
  },
  "not_mentioned": <true 仅当模型主张\\\"没有任何半岛 1km 内有观夏\\\"或完全没列任何半岛；否则 false>,
  "supporting_span": "<原文片段 30-200 字，证明 items 来自模型回答>",
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
