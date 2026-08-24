"""
Query 19 — Maekawa Kunio's 1935 unbuilt competition at Le Corbusier's atelier.
T1 granularity: one composite entity holding (city, building, position).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS.parent))

from pipeline.extraction_pipeline import run_pipeline  # noqa: E402

QUERY_ID = 19
QUERY_TEXT = (
    "1935年，日本建筑师前川國男为勒·柯布西耶的巴黎事务所工作期间，"
    "曾参与了一个未建成的竞赛方案。请问：这个方案是为哪座城市的哪栋建筑设计的？"
    "前川在其中具体负责了哪一部分？"
)

ENTITIES = [
    {
        "id": "maekawa_project_answer",
        "name": (
            "模型主张为本题答案的方案，含三项关键事实："
            "目标城市 / 建筑或项目名称 / 前川在其中的具体职位或角色"
        ),
    }
]

PROMPT_HINTS = {
    "maekawa_project_answer": (
        "请抽取模型作为本题**最终答案**给出的方案的三项关键事实：city / building / position。\n\n"
        "**抽取规则：**\n"
        "- 模型可能讨论多个候选项目，但通常会明确选定一个作为答案——把那个填到主字段；"
        "其他候选放进 alternative_candidates 列表。\n"
        "- 若模型明确说\"无法确定\"或\"找不到\"，主字段全部留 null，并把 not_mentioned 设为 true。\n"
        "- **city**：方案所在的城市，原文怎么写就怎么填\n"
        "  （如 \"Geneva\" / \"日内瓦\" / \"Genève\" / \"Genova\" 等）。\n"
        "- **building**：建筑 / 项目名称，原文写法\n"
        "  （如 \"Mundaneum\" / \"Cité Mondiale\" / \"世界城市\" / \"World City\" / \"国际联盟总部\" / \n"
        "  \"Palais des Nations\" / \"Centre Mondial\" 等）；保留原文拼写，不要纠错或扩展。\n"
        "- **position**：前川在该方案中的具体职位 / 角色 / 负责部分\n"
        "  （如 \"unpaid draftsman\" / \"无薪绘图员\" / \"绘图员\" / \"制图员\" / \"实习生\" / \n"
        "  \"学徒\" / \"volunteer\" / \"负责立面图\" / \"负责平面布局\" 等）。\n"
        "  尽量保留原文细节（薪资状态、绘图 vs 设计、具体负责的部分等），不要替模型脑补。\n"
        "- 模型在排除 / 否定语境中提到的项目不抽（例如 \"不是国际联盟总部因为...\"）。\n"
        "- 若模型同时提到了多组候选 (city, building, position)，把模型最终选定的那一组填到主字段，"
        "其余每组完整放进 alternative_candidates。"
    ),
}

VALUE_SCHEMA = """{
  "value": {
    "city": "<城市字符串或 null>",
    "building": "<建筑/项目名称字符串或 null>",
    "position": "<前川在方案中的具体职位/角色字符串或 null>",
    "alternative_candidates": [
      {
        "city": "...",
        "building": "...",
        "position": "..."
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
