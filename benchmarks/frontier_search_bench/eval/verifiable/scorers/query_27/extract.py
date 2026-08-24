"""
Query 27 — High-order Mobius strip cutting / geometry & topology.

Multi-dimensional question: a 2 cm × 30 cm strip is twisted 540°
(three half-twists) and joined into a Mobius-like band, then cut
continuously along a track 0.5 cm from the edge. Identify how many
independent strips result, their lengths/widths, the half-twist count
of the widest strip, and whether the widest strip is interlocked with
the others.

Reference answer (5 binary sub-dimensions, 1 point each, max 5):
  D1: strip_count == 2
  D2: there is a strip with width≈0.5 cm and length≈60 cm
  D3: there is a strip with width≈1   cm and length≈30 cm
  D4: widest strip has 3 half-twists (i.e. the original 540° twist)
  D5: widest strip interlocks with the other strip(s)

T1 granularity — the whole answer is one extraction with a composite value.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS.parent))

from pipeline.extraction_pipeline import run_pipeline  # noqa: E402

QUERY_ID = 27
QUERY_TEXT = (
    "想象一条宽度为2厘米、长度为30厘米的纸带。你将其一端扭转540度（即三个半圈），"
    "然后将两端粘合，形成一个高阶的莫比乌斯带。接着，你用剪刀沿着距离纸带边缘恰好0.5厘米的轨迹，"
    "连续剪开整个纸带。剪完所有的0.5厘米边缘轨迹后，你会得到几条独立的纸带？"
    "它们各自的长度和宽度是多少？更关键的是，如果此时你在得到的这些纸带中，"
    "找出最宽的那一条，它在拓扑学上包含了几个半扭转（Half-twists）？"
    "它是否与其他的带子形成了互锁（Interlocking）结？"
)


ENTITIES = [
    {
        "id": "mobius_cut_result",
        "name": "莫比乌斯带（540°/3 半扭转）沿距边 0.5 cm 剪后产生的所有纸带 + 最宽纸带的拓扑性质",
    },
]


PROMPT_HINTS = {
    "mobius_cut_result": (
        "题目要求计算一条 2cm×30cm、扭转 540°（3 个半扭转）粘合的纸带，"
        "沿距边缘 0.5cm 剪开后的产物。请抽出模型回答中**明确给出的最终结论**：\n\n"
        "**抽取要点：**\n"
        "- `strip_count`：得到几条独立纸带（整数）\n"
        "- `strips`：列表，每条独立纸带 `{width, length}`，单位 cm。\n"
        "  数值字段抽数字（例：`width: 0.5`），不要带单位字符串\n"
        "- `widest_half_twists`：最宽那条纸带在拓扑上包含的半扭转数（整数）\n"
        "- `interlocking`：最宽纸带是否与其它纸带形成互锁结（true/false/null）\n"
        "- `reasoning_summary`：模型推理过程的极简摘要（≤120 字）\n\n"
        "**抽取约束：**\n"
        "- 只看模型最终结论，忽略中间推导。模型若反复修正，取最后给出的版本\n"
        "- 模型若说某条纸带'与原长相同'/'2 倍原长'，请按 30cm/60cm 换算为数字\n"
        "- 'interlocking'/'互锁'/'交链'/'linked'/'topologically linked' 一律映射为 true；"
        "  '不互锁'/'分离'/'unlinked'/'separate' 映射为 false；模型未明示则为 null\n"
        "- 模型若给出多组互相矛盾的答案，按最终给出的那组抽取"
    ),
}


VALUE_SCHEMA = """{
  "value": {
    "strip_count": <整数 / null>,
    "strips": [
      {"width": <数值（cm）/ null>, "length": <数值（cm）/ null>}
    ],
    "widest_half_twists": <整数 / null>,
    "interlocking": <true|false|null>,
    "reasoning_summary": "<≤120字摘要 / null>"
  },
  "not_mentioned": <true 仅当模型完全未给出任何最终结论时为 true；否则 false>,
  "supporting_span": "<原文片段 30-200 字，能证明你抽出的内容来自回答>",
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
