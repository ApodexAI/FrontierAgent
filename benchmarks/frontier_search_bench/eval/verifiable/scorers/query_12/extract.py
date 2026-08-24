"""
Query 12 — Nobel laureates in Literature who were officially expelled or
exiled by their own government.

T1 granularity: one composite entity holding the full list of laureate
claims. Schema mirrors query_06 / query_10 conventions so downstream
auto_scorer can consume `extraction.json.canonical.value` uniformly.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS.parent))

from pipeline.extraction_pipeline import run_pipeline  # noqa: E402

QUERY_ID = 12
QUERY_TEXT = "获得过诺贝尔文学奖的作家中，有哪些人曾被本国政府正式驱逐出境或流放？"


ENTITIES = [
    {
        "id": "expelled_laureates",
        "name": "模型列出的'曾被本国政府正式驱逐出境或流放'的诺贝尔文学奖作家清单",
    },
]


PROMPT_HINTS = {
    "expelled_laureates": (
        "请抽出模型回答里**明确主张**为'曾被本国政府正式驱逐出境或流放'的诺贝尔文学奖作家。"
        "每人一个字段齐全的 JSON 对象。\n\n"
        "**抽取范围约束（重要）：**\n"
        "- 只抽取模型以**肯定或接近肯定**口吻主张的案例。'据传'、'有说法称'、'可能'这类半肯定按肯定处理（保留）\n"
        "- **不要**抽取模型在'排除'、'不符合'、'不算正式驱逐'、'仅是自我流亡而非被驱逐'、"
        "  '差一点'、'存疑'、'仅受迫害但未被驱逐'等语境下讨论的条目；这些是模型自己否定的案例\n"
        "- 模型若把人物分组为'正确答案 vs 边界案例 vs 错误答案'，**只抽取'正确答案/确认被驱逐'那一组**\n"
        "- 示例：\n"
        '    - 模型说"索尔仁尼琴 1974 年被苏联剥夺国籍并驱逐" → 抽取\n'
        '    - 模型说"高行健 1987 年自行流亡法国后被宣布不受欢迎" → **如果模型把他列为正确答案则抽取**；'
        '      如果模型说"严格意义上不算被驱逐，应排除" → **不抽取**\n'
        '    - 模型说"帕斯捷尔纳克受威胁但未被实际驱逐，故不计入" → **不抽取**（模型自己排除了）\n\n'
        "**抽取范围说明（重要）：**\n"
        "- 只抽取模型主张为'诺贝尔文学奖得主'的人物。如果模型顺带提到的非诺贝尔文学奖作家"
        "  （如哲学家、艺术家、政治家等）被作为肯定案例列出，仍要抽出"
        "  （下游对齐层会判池内/池外；这是评测'模型是否错列非文学奖人物'的考点）\n"
        "- 共同获奖者并列场景（如 1974 文学奖给 Eyvind Johnson + Harry Martinson 共享）"
        "  通常不影响本题（驱逐事件是个人针对性的）；按字面抽取即可\n"
    ),
}


VALUE_SCHEMA = """{
  "value": [
    {
      "name": "<作家姓名（中英文皆可，例如：Aleksandr Solzhenitsyn / 索尔仁尼琴）>",
      "nobel_year": "<诺贝尔文学奖获奖年份字符串 / null>",
      "country": "<作家所属国家（驱逐发生时的国籍国），例如：苏联 / 纳粹德国 / 法国维希政府 / 危地马拉 / null>",
      "government_action_type": "<政府行为类型：剥夺国籍 / 驱逐令 / 没收财产 / 缺席判决 / 通缉 / 其他 / null>",
      "action_year": "<政府行为发生的年份字符串 / null>",
      "note": "<其他关键信息，例如颁令机构、驱逐目的国、是否后来恢复国籍等 / null>"
    }
  ],
  "not_mentioned": <true 仅当模型完全未列任何作家时为 true；否则 false>,
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
