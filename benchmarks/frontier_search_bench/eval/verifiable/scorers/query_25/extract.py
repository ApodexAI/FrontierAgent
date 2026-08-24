"""
Query 25 — Starbucks latte supply-chain profit distribution.
T2 granularity: 9 entities (8 stages + Ethiopian farmer per-cup).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS.parent))

from pipeline.extraction_pipeline import run_pipeline  # noqa: E402

QUERY_ID = 25
QUERY_TEXT = (
    "一杯星巴克大杯拿铁，从咖啡种子到端到消费者手中，"
    "完整供应链上每个环节（种植、采摘、加工、运输、烘焙、包装、门店运营、税费）"
    "各自分走了多少比例的利润？埃塞俄比亚咖啡农每杯实际获得多少钱？"
)

ENTITIES = [
    {"id": "S1_planting", "name": "种植环节比例"},
    {"id": "S2_picking", "name": "采摘环节比例"},
    {"id": "S3_processing", "name": "加工环节比例"},
    {"id": "S4_transport", "name": "运输环节比例"},
    {"id": "S5_roasting", "name": "烘焙环节比例"},
    {"id": "S6_packaging", "name": "包装环节比例"},
    {"id": "S7_store_ops", "name": "门店运营环节比例"},
    {"id": "S8_taxes", "name": "税费环节比例"},
    {"id": "E_ethiopia_cup", "name": "埃塞俄比亚咖啡农每杯实际获得金额"},
]

PROMPT_HINTS = {
    "S1_planting": (
        "种植环节 = farm-gate 农户净得（咖啡树栽培/肥料/灌溉）。"
        "若模型把种植+采摘+加工合并为'农户/产地'总项，"
        "请在 value 抽出该合并总项比例，并在 note 字段标注 combined。"
    ),
    "S2_picking": "采摘 = 咖啡樱桃采摘工资 / 机械成本。若和种植合并则同 S1 处理。",
    "S3_processing": "加工 = 水洗/日晒/去壳/分级 + 合作社/加工厂利润。若被合并则同 S1 处理。",
    "S4_transport": "运输 = 产地→港→海运→进口清关→到烘焙厂仓储。",
    "S5_roasting": "烘焙 = 烘焙成本（能源/劳务/损耗）+ 烘焙商净利。",
    "S6_packaging": "包装 = 杯/盖/套/搅拌棒/吸管/辅料。",
    "S7_store_ops": "门店运营 = 店员工资+福利 + 门店租金/水电 + 折旧/维护（可含 G&A）。",
    "S8_taxes": "税费 = 企业所得税 / 销售税 / 关税 / 财产税（任一或综合）。",
    "E_ethiopia_cup": "埃塞俄比亚咖啡农**每杯**实际获得的金额（美元或人民币均可）。",
}

VALUE_SCHEMA = """{
  "value": {
    "percent": "<比例，如 \\"25%\\" 或 25 数字；S1/S2/S3 被合并则此处是合并比例>",
    "amount_per_cup": "<金额（仅 E_ethiopia_cup 需要），如 \\"$0.05\\" 或 \\"¥0.35\\"；其他环节可为 null>",
    "note": "<合并情况等简短注记，如 \\"combined S1+S2+S3\\";没有则 null>"
  },
  "not_mentioned": <true|false>,
  "supporting_span": "<原文片段 30-200 字>",
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
