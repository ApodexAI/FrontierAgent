"""
Query 06 — Nobel laureates whose key paper was rejected by journals.

T1 granularity: one composite entity holding the full list of laureate claims.
Schema is intentionally flat: each laureate is a top-level dict with
laureate name, year, paper title, rejecting journal(s), and supporting note.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS.parent))

from pipeline.extraction_pipeline import run_pipeline  # noqa: E402

QUERY_ID = 6
QUERY_TEXT = "截至2026年1月1日，哪些诺贝尔奖得主的获奖论文最初被期刊拒稿过？"

ENTITIES = [
    {
        "id": "rejected_list",
        "name": "模型列出的'获奖论文被期刊拒稿过'的诺贝尔奖得主清单",
    },
]

PROMPT_HINTS = {
    "rejected_list": (
        "请抽出模型回答里**明确主张**为'其获奖论文/关键论文最初被期刊拒稿过'的诺贝尔奖得主。"
        "每人一个字段齐全的 JSON 对象。若模型只给人名未提期刊，其余字段写 null。\n\n"
        "**抽取范围约束（重要）：**\n"
        "- 只抽取模型以**肯定或接近肯定**口吻主张的案例。'据传'、'据说'、'可能也算'、'有说法称'这种半肯定表述，按肯定处理（保留）\n"
        "- **不要**抽取模型在'排除'、'不符合'、'存疑'、'差一点'、'不是期刊拒稿'、'无法验证'、'查无实证'等语境下讨论的条目；这些是模型自己否定的案例，不构成它的主张\n"
        "- 示例：\n"
        '    - 模型说"Krebs 被 Nature 拒" → 抽取\n'
        '    - 模型说"X 据传被拒但无直接证据" → 抽取（带证据限定的半肯定，仍视为主张）\n'
        '    - 模型说"Y 案例不成立，因为那不是期刊拒稿而是国家审查" → **不抽取**（模型自己排除了）\n'
        '    - 模型说"Z 在名单上但条件不符，应排除" → **不抽取**\n\n'
        "**共同获奖者并列场景（重要）：**\n"
        "当多人作为**同一诺奖的共同获奖者**被模型并列提到，且模型**仅对其中一人有明确独立的拒稿主张**"
        "（期刊名、拒稿年份、被拒论文主题等具体事件），其他共同获奖者**仅作为并列人物出现**"
        "（没有自己独立的拒稿事件描述）时，**只抽取那位有独立拒稿主张的得主一人**，其他并列者**不抽为独立 claim**。\n"
        "- 示例：\n"
        '    - 模型说"Karikó 和 Weissman 共同获 2023 医学奖，他们的 mRNA 论文被 Nature 拒稿" → 只抽 Karikó（mRNA 疫苗共同作者但独立拒稿事件归属 Karikó），Weissman 不抽\n'
        '    - 模型说"Ratcliffe、Kaelin、Semenza 共同获 2019 医学奖，Ratcliffe 的 HIF 论文被 Nature 拒稿" → 只抽 Ratcliffe（拒稿事件归属他）\n'
        '    - 反例：模型说"Kaelin **本人的** VHL 论文也被 Science 拒过" → 这是 Kaelin 独立的拒稿主张，仍要抽 Kaelin\n\n'
        "模型顺带提到的**非诺奖人物**（如 Lynn Margulis 内共生、Theodore Maiman 激光器）"
        "若被模型作为肯定案例列出，仍要抽出（下游对齐层会判池内/池外；这是任务覆盖的'模型是否错列非诺奖得主'评测点）。"
    ),
}

VALUE_SCHEMA = """{
  "value": [
    {
      "name": "<得主姓名（中英文皆可）>",
      "nobel_year": "<获奖年份字符串或 null>",
      "nobel_field": "<生理学或医学 / 物理学 / 化学 / 经济学 / 文学 / 和平 / 非诺奖 / null>",
      "rejected_paper_topic": "<被拒论文的主题或核心描述 / null>",
      "rejecting_journal": "<拒稿期刊名 / null>",
      "rejection_reason": "<原因或 null>",
      "final_publication": "<最终发表期刊 / null>",
      "note": "<其他关键信息（如拒稿时间、审稿细节）/ null>"
    }
  ],
  "not_mentioned": <true 仅当模型完全未列任何得主时为 true；否则 false>,
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
