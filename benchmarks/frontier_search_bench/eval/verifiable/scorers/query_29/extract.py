"""
Query 29 — FlashAttention CUDA kernel optimization origin.
T1 granularity: one composite entity holding
(论文标题 / 技巧名称 / 首次应用版本 / 首次应用 kernel 文件).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS.parent))

from pipeline.extraction_pipeline import run_pipeline  # noqa: E402

QUERY_ID = 29
QUERY_TEXT = (
    "FlashAttention系列从v1到v3的演进中，有一个特定的CUDA kernel优化技巧最初"
    "来源于一篇非attention领域的论文。请找出这篇论文的标题、该技巧的具体名称，"
    "以及它在FlashAttention哪个版本的哪个kernel文件中首次被应用？"
)

ENTITIES = [
    {
        "id": "fa_origin_answer",
        "name": (
            "模型主张为本题答案的主答案四要素："
            "论文标题 / 技巧名称 / 首次应用版本 / 首次应用 kernel 文件"
        ),
    }
]

PROMPT_HINTS = {
    "fa_origin_answer": (
        "请抽取模型作为本题**最终答案**给出的主答案四要素。\n\n"
        "**抽取规则：**\n"
        "- 模型可能讨论多个候选，但通常会明确选定一个作为答案——把那个填到主字段；"
        "其他候选放进 alternative_candidates 列表。\n"
        "- 若模型明确说\"无法确定\"或\"找不到\"，主字段全部留 null，并把 not_mentioned 设为 true。\n"
        "- **paper_title**：模型主张的论文标题（如 \"CudaDMA: Optimizing GPU Memory Bandwidth "
        "via Warp Specialization\"）；保留原文写法和大小写，不要扩写或纠错。\n"
        "- **technique_name**：模型主张的技巧名称（如 \"warp specialization\"）；保留原文术语。\n"
        "- **first_applied_version**：模型主张该技巧首次出现的 FlashAttention 版本\n"
        "  （如 \"FlashAttention-3\" / \"FlashAttention 3\" / \"FlashAttention v3\" / \"FA3\"）；"
        "原文写法即可。\n"
        "- **first_applied_kernel_file**：模型主张该技巧首次出现的 kernel 文件路径\n"
        "  （如 \"hopper/mainloop_fwd_sm90_tma_gmma_ws.hpp\"，或不带前缀的 "
        "\"mainloop_fwd_sm90_tma_gmma_ws.hpp\"）；保留原文路径分隔符与大小写。\n"
        "- 模型在排除 / 否定语境中提到的项目不抽（例如 \"不是 xxx 因为...\"）。\n"
        "- 若模型同时提到了多组候选，把模型最终选定的那一组填到主字段，"
        "其余每组完整放进 alternative_candidates。"
    ),
}

VALUE_SCHEMA = """{
  "value": {
    "paper_title": "<论文标题字符串或 null>",
    "technique_name": "<技巧名称字符串或 null>",
    "first_applied_version": "<FlashAttention 版本字符串或 null>",
    "first_applied_kernel_file": "<kernel 文件路径字符串或 null>",
    "alternative_candidates": [
      {
        "paper_title": "...",
        "technique_name": "...",
        "first_applied_version": "...",
        "first_applied_kernel_file": "..."
      }
    ]
  },
  "not_mentioned": <true 仅当模型完全没给出任何候选；否则 false>,
  "supporting_span": "<原文片段 30-200 字，能证明上面字段确实来自模型回答>",
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
