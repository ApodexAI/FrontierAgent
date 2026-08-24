"""
Query 17 — Paper-to-stock transmission events within 48h, 2025-2026.
T1 granularity: one entity holding the candidate list of paper→stock events.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS.parent))

from pipeline.extraction_pipeline import run_pipeline  # noqa: E402

QUERY_ID = 17
QUERY_TEXT = (
    "回顾2025年1月1日到2026年1月1日的AI论文发布和对应公司股价，"
    "哪些论文发布后48小时内引发了相关公司股价超过5%的异常波动？"
    "这种\"论文→股价\"传导效应在哪些子领域最显著？"
)

ENTITIES = [
    {
        "id": "paper_stock_events",
        "name": (
            "候选 \"论文→股价\" 事件列表："
            "每项含论文标题/标识、所涉公司、股价变动幅度、48h 窗口、所属子领域"
        ),
    }
]

PROMPT_HINTS = {
    "paper_stock_events": (
        "抽取模型在回答中**作为最终答案纳入**的所有 \"论文→股价\" 事件。\n\n"
        "**抽取规则：**\n"
        "- 一个事件 = 一个对象。若模型把多个事件合并叙述，按文中实际个数拆开。\n"
        "- 模型明确**排除**或仅作反例提到的事件不抽（例如 \"DeepSeek R1 不算因为差 5 天\"）。\n"
        "- 模型在背景介绍 / 概念解释 / 子领域趋势中泛泛提到的论文，若没有作为具体答案事件，不抽。\n"
        "- 字段尽量贴近原文写法；模型没明确说的字段填 null，不要补全。\n"
        "- **paper_title**：论文标题，原文写法（含中英文混写情况）。\n"
        "- **paper_id**：arXiv id / DOI / 其他正式标识（如 \"arXiv:2502.11089\"）；没有则 null。\n"
        "- **publish_date**：论文公开日期（原文写法即可）。\n"
        "- **company**：受影响的具体上市公司（不是板块/指数）；含多家公司时填主受益公司，"
        "其余进 `other_companies`。\n"
        "- **stock_change_pct**：模型给出的股价变动百分比，原文写法（如 \"+17.96%\"、\"+9.30%\"）。\n"
        "- **direction**：`up` / `down` / `unknown`。\n"
        "- **subfield**：模型主张的子领域归类（如 \"稀疏注意力\"、\"大模型/多模态\" 等）。\n"
        "- **claimed_within_48h**：模型是否主张该波动在论文公开后 48h 内发生。\n"
        "- 若模型完全没列任何事件（只讨论方法或泛泛而谈），items 返回空数组、not_mentioned=true。"
    ),
}

VALUE_SCHEMA = """{
  "value": {
    "items": [
      {
        "paper_title": "<论文标题字符串或 null>",
        "paper_id": "<arXiv id / DOI 字符串或 null>",
        "publish_date": "<公开日期字符串或 null>",
        "company": "<主受益上市公司字符串或 null>",
        "other_companies": ["<其他被提到的公司>"],
        "stock_change_pct": "<股价变动百分比字符串或 null>",
        "direction": "<up | down | unknown>",
        "subfield": "<子领域字符串或 null>",
        "claimed_within_48h": <true|false|null>
      }
    ],
    "subfield_summary": "<模型对 \\\"哪个子领域最显著\\\" 给出的总结，可为 null>"
  },
  "not_mentioned": <true 仅当模型未列出任何具体事件；否则 false>,
  "supporting_span": "<原文片段 30-200 字，能证明上面 items 来自模型回答>",
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
