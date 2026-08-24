"""
Query 13 — Tech-stock earnings ~5% drop → upstream/downstream supply-chain
single-day max drop.

Cold-start mode: baseline starts empty. Stage C web-verification agent
checks each model claim against 6 conditions and produces resolutions
(or builds GT inline) — cold-start mode: baseline starts empty, Stage C agent fills it.


"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS.parent))

from pipeline.extraction_pipeline import run_pipeline  # noqa: E402

QUERY_ID = 13
QUERY_TEXT = (
    "2016年1月1日到2026年1月1日之间，查找每次财报后，股价跌幅在5%左右的科技股，"
    "其行业上下游产业中，单日跌幅最大的股票是哪个？"
)

ENTITIES = [
    {
        "id": "answer_events",
        "name": (
            "模型主张为本题答案的'科技股财报后跌约5%事件 + 上下游产业链中"
            "单日跌幅最大的股票'完整事件清单"
        ),
    },
]

PROMPT_HINTS = {
    "answer_events": (
        "请抽出模型回答中**明确主张**为本题答案的事件。每个事件 = 一个 JSON 对象。\n\n"
        "**一个 claim = 一个完整事件**，包含：\n"
        "- 科技股母公司（parent_company / parent_ticker）\n"
        "- 财报日期（earnings_date）\n"
        "- 母公司财报后跌幅（parent_drop_pct，带负号）\n"
        "- 上下游答案股票（answer_ticker / answer_company_name）\n"
        "- 该答案股票当日跌幅（answer_drop_pct）\n"
        "- 答案股票跌幅日期（answer_drop_date）\n"
        "- 上下游关系类型与依据（relationship_type / relationship_evidence）\n\n"
        "**抽取范围约束（重要）：**\n"
        "- 模型如列出**多个**候选事件（如'英伟达财报后…'、'苹果财报后…'），"
        "分别抽成多条独立 claim\n"
        '- 只抽取模型以**肯定或接近肯定**口吻列为答案的事件。"可能"、"或许"'
        "等半肯定按肯定处理（保留）\n"
        '- **不要**抽取模型在"排除"、"不符合条件"、"存疑"、"不算"、"仅作对照"、'
        '"不是答案"等语境下讨论的事件；这些是模型自己否定/排除的案例\n'
        '- 模型给的是**产业类别**（如"VR 设备制造商整体 12.5%"）而非具体股票代码时，'
        "**仍要抽出**，answer_ticker 字段填该类别描述、answer_company_name 留 null。"
        "下游条件 C5/C6 会按事实核查判定（无法核查具体股票则归 unresolved）\n"
        "- 模型即使**把跌幅数据写错**（如说苹果跌 5% 实际跌 4%），或把母公司"
        "归错产业，仍属其主张要抽出（下游 Stage C agent 按真实数据判定对错）\n"
        '- 模型若仅给"模式总结"（如"供应商比客户跌得更狠"）但没给具体事件案例，'
        "**不抽**——本题要的是具体事件\n\n"
        "**示例：**\n"
        '- 模型说"NVDA 2026-02-26 财报后跌 5.46%，盛美上海当日跌 8.74%" → 抽取，'
        'answer_ticker="688082", answer_drop_pct="-8.74"\n'
        '- 模型说"Lumentum 在 2018-11 苹果削减出货量后跌 32%" → 抽取（即使母公司'
        "跌幅可能不在 5% 区间，下游 C3 会判 fail）\n"
        '- 模型说"Roku 在 Netflix 财报后跌 9.1%（但 Netflix 自己跌 35%，明显不是 5% 案例）" → '
        "**仍抽出**（事实核查由下游做）\n"
        '- 模型说"VR 设备制造商整体 12.5%" → 抽取，answer_ticker="VR 设备制造商整体"'
    ),
}

VALUE_SCHEMA = """{
  "value": [
    {
      "parent_company": "<科技股母公司名称，如 NVIDIA / 苹果 / 英伟达>",
      "parent_ticker": "<母公司股票代码，如 NVDA、AAPL；模型未明确给则 null>",
      "earnings_date": "<财报发布日期 YYYY-MM-DD；模型仅给月份则填 YYYY-MM；不明则 null>",
      "parent_drop_pct": "<母公司财报后跌幅，数值字符串带负号如 '-5.46'，或文字描述>",
      "parent_drop_date": "<母公司跌幅发生的日期 YYYY-MM-DD（一般 = earnings_date 当日或次日）；不明则 null>",
      "answer_ticker": "<被声称为'上下游单日跌幅最大'的股票代码或名称；产业类别也填这里>",
      "answer_company_name": "<答案股票的公司全称；模型未明确则 null>",
      "answer_drop_pct": "<答案股票当日跌幅，数值字符串带负号如 '-8.74'，或文字描述>",
      "answer_drop_date": "<答案股票跌幅发生的日期 YYYY-MM-DD；不明则 null>",
      "relationship_type": "<'上游'|'下游'|'竞争对手'|'同业'|'其他'；或模型用的原文描述>",
      "relationship_evidence": "<模型给出的关系依据，如 '台积电是 NVIDIA 主要代工厂'；不明则 null>",
      "note": "<其他关键信息或限定，如 '盘前跌幅' / '次日跌幅' / '盘后' / null>"
    }
  ],
  "not_mentioned": <true 仅当模型完全未给出任何具体事件案例时为 true；否则 false>,
  "supporting_span": "<原文片段 30-200 字，能证明抽出的事件来自回答>",
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
