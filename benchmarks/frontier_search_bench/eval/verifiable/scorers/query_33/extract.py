"""
Query 33 — Denver, CO 沿正南方向移动至南极点路径上依次经过的 US Counties。

T1 granularity：单实体（包含一个有序的 US Counties 列表）。
schema 沿用 query_06 / query_12 风格，下游 auto_scorer 通过
`extraction.json.canonical.value` 拿到这个列表。

GT 来源：FCC Census Block API + OSM Nominatim 双源验证（沿 -104.99°W 经线
21 个采样点，纬度范围 38.85°N → 31.45°N）。共 17 个县，CO 9 + NM 7 + TX 1。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS.parent))

from pipeline.extraction_pipeline import run_pipeline  # noqa: E402

QUERY_ID = 33  # 与仓库 queries/verifiable.json 中 id=33 对齐

QUERY_TEXT = (
    "如果我从美国科罗拉多州的丹佛市（Denver, CO）一直向正南方移动，"
    "直到到达南极点。请列出我将依次经过的所有县（Counties）的名字。"
)


ENTITIES = [
    {
        "id": "us_counties_path",
        "name": (
            "模型主张的、从丹佛沿正南至南极路径上依次经过的"
            "**美国 Counties**（按从北到南顺序）"
        ),
    },
]


PROMPT_HINTS = {
    "us_counties_path": (
        "请抽取模型回答里**明确列为'依次经过'**的美国 Counties，按"
        "**模型给出的顺序**保留为有序列表。每个县一条结构化记录。\n\n"
        "**抽取范围（重要）：**\n"
        "- 仅抽取**美国 (US)** 的 Counties。墨西哥 estados / municipios / "
        "  其它国家行政区**一律不抽**（题目问的是 'Counties'，而 'County' "
        "  在题面语境下专指美国/英语国家的县级行政区；墨西哥用的是 "
        "  municipio，不同概念）。\n"
        "- 模型若按 '美国境内 / 跨境之后' 分段，**只抽美国境内那段**。\n"
        "- 若模型同时给了 'I-25 高速沿线' vs '正南经线沿线' 两个版本，"
        "  以**正南经线沿线**那个版本为准（题面要求是'正南方'）。\n"
        "- 不抽 state（州）级别的实体；那不是 county。\n\n"
        "**name + state 的强制要求：**\n"
        "- 美国有大量同名县（如 El Paso 在 CO 和 TX 各有一个；San Miguel "
        "  在 CA 和 NM 各有一个；Lincoln 在多个州都有；Otero 在 CO 和 NM "
        "  都有）。**模型必须明确写出 state 才能区分**。\n"
        "- 模型若上下文清晰指明所在州（例如标题是 'New Mexico:' 然后下面"
        "  列了 Lincoln County），抽取时把该 state 填进 state 字段。\n"
        "- 模型若没明示 state（仅写 'Lincoln County'），state 字段填 "
        "  'unspecified'。\n\n"
        "**顺序约束（关键）：**\n"
        "- 输出 list 的顺序必须严格反映模型在原文中给出的顺序"
        "（北→南）。不要按字母序、按州分组等其它排列重新排序。\n"
        "- 若模型按州分组列出（先 CO 段，再 NM 段，再 TX 段），按"
        "  '段内顺序 + 段间顺序' 拼接成一条 list 即可。\n\n"
        "**示例：**\n"
        "- 模型说 '科罗拉多州：Denver County → Arapahoe County → Douglas "
        "  County → ... → Las Animas County；新墨西哥州：Colfax → ...'\n"
        "  → 抽取为 list = [\n"
        "      {name: 'Denver County', state: 'CO'},\n"
        "      {name: 'Arapahoe County', state: 'CO'},\n"
        "      ...,\n"
        "      {name: 'Colfax County', state: 'NM'},\n"
        "      ...]\n"
        "- 模型说 '依次经过 Denver, Pueblo, Colfax, Hudspeth' （未分州）\n"
        "  → 抽取为 list = [\n"
        "      {name: 'Denver County', state: 'unspecified'},\n"
        "      {name: 'Pueblo County', state: 'unspecified'},\n"
        "      ...]\n"
        "- 模型说 '跨过美墨边界后进入 Chihuahua state'\n"
        "  → 不抽（墨西哥州，非 US County）\n"
    ),
}


VALUE_SCHEMA = """{
  "value": [
    {
      "name": "<县名（英文优先，例如：Denver County / Las Animas County / Hudspeth County）；中文或带 '县' 字样也接受>",
      "state": "<两字州代码：CO / NM / TX；若模型未指明则填 'unspecified'>",
      "note": "<其它有助于消歧的信息（如纬度、紧邻县名）；可空>"
    }
  ],
  "not_mentioned": <true 仅当模型完全未列出任何 US County 时为 true；否则 false>,
  "supporting_span": "<原文片段 30-200 字，能证明你抽出的列表与顺序确实来自回答>",
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
