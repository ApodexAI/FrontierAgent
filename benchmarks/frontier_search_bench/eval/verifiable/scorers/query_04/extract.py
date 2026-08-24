"""
Query 04 — Global INES level 4+ nuclear accidents, 1945-2026.
T1 granularity: one entity holding the candidate accident list.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS.parent))

from pipeline.extraction_pipeline import run_pipeline  # noqa: E402

QUERY_ID = 4
QUERY_TEXT = (
    "从1945年到2026年，全球所有核电站累计发生过多少次INES 4级及以上的核事故？"
    "请逐一列出每次事故的时间、地点、反应堆型号、事故等级、直接原因，"
    "以及事故后该国核电政策的具体变化（如立法、关停、新建冻结等）。"
    "对比这些事故前后5年内该国的核电发电量占比变化数据。"
)

ENTITIES = [
    {
        "id": "accident_candidates",
        "name": "候选核事故列表（每项含名称/地点/时间/堆型/等级/直接原因/政策变化/前后5年核电占比变化）",
    },
]

PROMPT_HINTS = {
    "accident_candidates": (
        "提取模型在回答中作为答案纳入的所有核事故项。\n\n"
        "**抽取规则：**\n"
        "- 一个事故项对应一个对象。若模型把多个事件合并为一项叙述，保留为一项，不要自行拆分。\n"
        "- 字段尽量贴近模型原文说法；模型未明确提到的字段填 \"未明确说明\"。\n"
        "- 不要使用外部知识补全或纠正模型未说的内容；不做单位换算。\n"
        "- 模型明确**排除**或仅作背景/反例提到的事故（例如 \"切尔诺贝利不算 INES 4 级因为是 7 级，但仍列入\" 这种边界情况，"
        "只要被作为答案列出就抽取；明确说 \"这不算我答案\" 的不抽）。\n"
        "- 若模型只讨论统计口径或方法却未列任何具体事故，"
        "value.items 返回空数组并把 not_mentioned 设为 true。"
    ),
}

VALUE_SCHEMA = """{
  "value": {
    "items": [
      {
        "事故名称": "<事故名 / 机组名>",
        "事故等级": "<INES 等级，如 \\"INES 4\\" 或 \\"5 级\\">",
        "事故地点": "<地点 / 国家>",
        "事故时间": "<日期或年份字符串>",
        "反应堆型号": "<堆型 / 机组型号>",
        "直接原因": "<直接原因简述>",
        "事故后该国核电政策具体变化": "<政策变化简述>",
        "事故前后5年内核电发电量占比变化数据": "<核电占比变化数据或定性描述>"
      }
    ]
  },
  "not_mentioned": <true 仅当模型完全没列任何事故；否则 false>,
  "supporting_span": "<原文片段 30-200 字，证明上面 items 确实来自模型回答>",
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
