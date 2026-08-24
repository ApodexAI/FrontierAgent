"""
Query 32 — DOCBENCH (Zou et al., 2025) GT answer average length.
T1 granularity: one entity holding the model's main claim about average length.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS.parent))

from pipeline.extraction_pipeline import run_pipeline  # noqa: E402

QUERY_ID = 32
QUERY_TEXT = (
    "看看 DOCBENCH (Zou et al., 2025) 这篇论文的数据，"
    "帮我分析一下他们 bench 的 gt 答案的平均长度是多少？"
)

ENTITIES = [
    {
        "id": "avg_length_claim",
        "name": "模型对 DOCBENCH GT 答案平均长度给出的最终主结论（含数值/区间、口径、性质）",
    },
]

PROMPT_HINTS = {
    "avg_length_claim": (
        "提取模型对 \"DOCBENCH 的 GT 答案平均长度是多少\" 这一问题给出的**最终主结论**。\n\n"
        "**抽取规则：**\n"
        "- 优先以模型在 short_answer / 总结部分给出的最终值为准；若主结论与展开分析冲突，以最终结论为准。\n"
        "- 不要把背景介绍、论文摘要、示例答案、推理过程、附带建议、数据获取路径当作主答案。\n"
        "- **final_claim_type**：\n"
        "  · 明确给出单值（如 \"62.18 个字符\"、\"4.3 tokens\"）→ `numeric`\n"
        "  · 明确给出区间/范围（如 \"60-65 字符\"、\"3-5 tokens\"）→ `range`\n"
        "  · 核心结论是 \"论文未报告 / 公开资料没有 / 当前无法可靠确定 / 需要先下载原始数据\" → `unavailable`\n"
        "  · 同时给出多个彼此冲突且没有明显主次的数值/范围 → `ambiguous`\n"
        "  · 其它 → `other`\n"
        "- **normalized_metric**：能明确判断时填 `character` / `token` / `word`；否则填 `unknown`。\n"
        "- **unit_text**：原文出现的单位字符串（如 \"字符\"、\"tokens\"、\"个字\"）。\n"
        "- **single_value**：单值答案时填数字；其余填 null。\n"
        "- **range_min / range_max**：区间答案时填两端数字；其余填 null。\n"
        "- **estimate_nature**：`official_or_computed` / `rough_estimate` / `speculative` / `not_applicable` / `unknown`。\n"
        "- **says_paper_does_not_report**：模型是否明确声称论文未报告该统计。\n"
        "- **says_cannot_determine_without_dataset**：模型是否明确声称需要原始数据才能算。\n"
        "- 即使文中顺带给了粗略猜测，只要回答没把该猜测当作最终结论，就不要把它当主答案。\n"
        "- 拼写、单位、限定词（\"约\"\"大概\"\"tokens\"\"字符\" 等）保持模型原文写法。"
    ),
}

VALUE_SCHEMA = """{
  "value": {
    "final_claim_type": "<numeric | range | unavailable | ambiguous | other>",
    "raw_final_answer": "<尽量简短地忠实复述模型最终答案>",
    "normalized_metric": "<character | token | word | unknown>",
    "unit_text": "<原文单位字符串，可为空>",
    "single_value": <数字或 null>,
    "range_min": <数字或 null>,
    "range_max": <数字或 null>,
    "estimate_nature": "<official_or_computed | rough_estimate | speculative | not_applicable | unknown>",
    "says_paper_does_not_report": <true|false>,
    "says_cannot_determine_without_dataset": <true|false>
  },
  "not_mentioned": <true 仅当模型完全没就该题给出任何主结论；否则 false>,
  "supporting_span": "<原文片段 30-200 字，最能支撑 raw_final_answer 的一段>",
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
