"""
Query 16 — Nobel Prize waiting time statistics (2000-2025).
T2 granularity: 9 stat entities + 2 extremes lists (total 11).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS.parent))

from pipeline.extraction_pipeline import run_pipeline  # noqa: E402

QUERY_ID = 16
QUERY_TEXT = (
    "请查找 2000–2025 年间所有诺贝尔物理学奖、化学奖和生理学或医学奖的获奖者，"
    "对于每位获奖者，找到其获奖所依据的核心论文（或核心发现的最早公开发表时间），"
    "计算从发表到获奖的间隔年数。按三个学科分别统计中位数、均值和趋势（是否在缩短），"
    "并列出等待时间最长和最短的各 3 个案例。"
)

ENTITIES = [
    {"id": "phys_median", "name": "物理学奖：等待时间中位数（年）"},
    {"id": "phys_mean", "name": "物理学奖：等待时间均值（年）"},
    {"id": "phys_trend", "name": "物理学奖：等待时间趋势方向"},
    {"id": "chem_median", "name": "化学奖：等待时间中位数（年）"},
    {"id": "chem_mean", "name": "化学奖：等待时间均值（年）"},
    {"id": "chem_trend", "name": "化学奖：等待时间趋势方向"},
    {"id": "med_median", "name": "生理医学奖：等待时间中位数（年）"},
    {"id": "med_mean", "name": "生理医学奖：等待时间均值（年）"},
    {"id": "med_trend", "name": "生理医学奖：等待时间趋势方向"},
    {"id": "extreme_longest", "name": "等待时间最长的 3 个案例（人名列表）"},
    {"id": "extreme_shortest", "name": "等待时间最短的 3 个案例（人名列表）"},
]

PROMPT_HINTS = {
    "phys_median": "物理学奖等待时间（论文发表到获奖）的**中位数**，数字单位年。",
    "phys_mean": "物理学奖等待时间的**均值**。",
    "phys_trend": (
        "物理学奖等待时间的**趋势方向**。只抽出一个词："
        "`lengthening/increasing/变长/延长` 或 "
        "`shortening/decreasing/缩短` 或 `stable/不变`。"
    ),
    "chem_median": "化学奖等待时间中位数。",
    "chem_mean": "化学奖等待时间均值。",
    "chem_trend": "化学奖等待时间趋势方向（同上）。",
    "med_median": "生理/医学奖等待时间中位数。",
    "med_mean": "生理/医学奖等待时间均值。",
    "med_trend": "生理/医学奖等待时间趋势方向（同上）。",
    "extreme_longest": (
        "模型列出的等待时间**最长**的 3 个获奖者姓名（只抽名单，"
        "每人一个姓或全名即可）。"
    ),
    "extreme_shortest": ("模型列出的等待时间**最短**的 3 个获奖者姓名。"),
}

VALUE_SCHEMA = """{
  "value": <对于 median/mean 是数字；对于 trend 是字符串关键词；对于 extreme_* 是字符串数组>,
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
