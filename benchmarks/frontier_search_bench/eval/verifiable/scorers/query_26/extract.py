"""
Query 26 — A-share stock identification from price/volume clues.

Single-fact question: identify a specific Chinese A-share stock matching
the price/volume pattern described in the query (low ≈ 5.50 in Dec 2025,
peak ≈ 9.45 in early Jan 2026, oscillating around 7+ yuan; one active day
volume ≈ 184.78 万手, turnover ≈ 10%). Reference answer: 天下秀 (ticker 600556).

T1 granularity — the whole answer is one extraction with a composite value.
Schema mirrors query_21 / query_23 conventions (single-fact + alternatives).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS.parent))

from pipeline.extraction_pipeline import run_pipeline  # noqa: E402

QUERY_ID = 26
QUERY_TEXT = (
    "请在中国A股市场中识别一只股票：它在2025年12月附近曾跌至约5.50的阶段低点，"
    "随后在2026年1月初快速上涨，并在短时间内触及约9.45的阶段高点；"
    "其后股价有所回落，但并未完全回吐涨幅，而是在7元多的区间继续震荡。"
    "公开交易数据还显示，该股某一活跃交易日的成交量约为184.78万手，换手率约为10%。"
    "请判断这只股票最可能是哪只，并给出证据链。"
)


ENTITIES = [
    {
        "id": "stock_identification",
        "name": "模型识别出的中国 A 股股票（股名 + 代码 + 证据链）",
    },
]


PROMPT_HINTS = {
    "stock_identification": (
        "题目要求识别一只中国 A 股股票，请抽出模型回答里**明确指认为答案**的那只股票，"
        "以及辅助证据字段。\n\n"
        "**抽取要点：**\n"
        "- `stock_name`：中文股名（例：`天下秀`）\n"
        "- `ticker`：股票代码（例：`600556`）\n"
        "- `exchange`：交易所（上交所/深交所）/ null\n"
        "- `low_price`：模型声称的阶段低点价格（约 5.50）/ null\n"
        "- `peak_price`：模型声称的阶段高点价格（约 9.45）/ null\n"
        "- `volume_wanshou`：模型声称的活跃交易日成交量（万手；约 184.78）/ null\n"
        "- `turnover_rate`：模型声称的换手率（百分比；约 10）/ null\n"
        "- `evidence_summary`：模型给出的证据链摘要\n\n"
        "**抽取范围约束：**\n"
        "- 模型若给出多个候选股票，主答案放 `value`，其余放 `alternative_candidates`\n"
        "- 仅抽取模型**明确主张为答案**的股票；以反例/排除语境讨论的不抽\n"
        "- 模型若用 `xxx（xxxxxx）` 形式同时给出股名与代码，按字面拆分到两个字段"
    ),
}


VALUE_SCHEMA = """{
  "value": {
    "stock_name": "<中文股名 / null>",
    "ticker": "<股票代码字符串 / null>",
    "exchange": "<上交所/深交所 / null>",
    "low_price": <数值 / null>,
    "peak_price": <数值 / null>,
    "volume_wanshou": <数值（万手）/ null>,
    "turnover_rate": <数值（百分比）/ null>,
    "evidence_summary": "<证据链摘要 / null>",
    "alternative_candidates": [
      { "stock_name": ..., "ticker": ..., "exchange": ..., "low_price": ..., "peak_price": ..., "volume_wanshou": ..., "turnover_rate": ..., "evidence_summary": ... }
    ]
  },
  "not_mentioned": <true 仅当模型完全未识别任何股票时为 true；否则 false>,
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
