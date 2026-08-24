"""
Query 23 — slime megatron issue impacting MoE training (in 2025).

Single-fact question: identify a specific GitHub issue from the `slime` repo
that affected MoE training with Megatron-Core (mcore). Reference answer:
issue #958 ("[bug] while train moe model with mcore, it seems only return
moe optimizer in setup_model_and_optimizer").

T1 granularity — the whole answer is one extraction with a composite value.
Schema mirrors query_21 conventions (single-fact + alternative_candidates).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS.parent))

from pipeline.extraction_pipeline import run_pipeline  # noqa: E402

QUERY_ID = 23
QUERY_TEXT = "slime 在2025年出现的一个重大megatron问题，对moe训练有影响，是哪个"


ENTITIES = [
    {
        "id": "slime_megatron_issue",
        "name": "slime 仓库 2025 年出现、影响 MoE 训练的 Megatron 相关 issue/PR",
    },
]


PROMPT_HINTS = {
    "slime_megatron_issue": (
        "题目问 slime 项目在 2025 年出现的一个对 MoE 训练有影响的 Megatron 相关问题。"
        "请抽出模型回答里**明确指认**的那个 issue/PR，以及辅助证据字段。\n\n"
        "**抽取要点：**\n"
        "- `repo`：仓库名（例：`THUDM/slime` / `slime` / 模型给出的全名）\n"
        "- `issue_number`：issue 或 PR 的编号（整数；例：958）\n"
        "- `title`：issue/PR 标题（原文片段即可）\n"
        "- `link`：issue/PR 的完整 URL（若回答中给出）\n"
        "- `affects_moe`：模型是否声称该问题影响 MoE 训练（true/false）\n"
        "- `description`：问题简要描述\n\n"
        "**抽取范围约束：**\n"
        "- 模型若给出多个候选 issue，主答案放 `value`，其余放 `alternative_candidates`\n"
        "- 仅抽取模型**明确主张为答案**的 issue；以反例/排除语境讨论的 issue **不抽**"
    ),
}


VALUE_SCHEMA = """{
  "value": {
    "repo": "<仓库名 / null>",
    "issue_number": <整数 / null>,
    "title": "<issue/PR 标题 / null>",
    "link": "<完整 URL / null>",
    "affects_moe": <true|false|null>,
    "description": "<问题简要描述 / null>",
    "alternative_candidates": [
      { "repo": ..., "issue_number": ..., "title": ..., "link": ..., "affects_moe": ..., "description": ... }
    ]
  },
  "not_mentioned": <true 仅当模型完全未指认任何 issue 时为 true；否则 false>,
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
