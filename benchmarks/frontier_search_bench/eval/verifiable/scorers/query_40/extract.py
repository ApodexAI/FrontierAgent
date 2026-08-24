"""
Query 40 — Route 66 elevation profile (Chicago → Los Angeles).

T2 granularity: 9 entities (one per fact slot):
  - HIGH:          highest elevation point on the route (name + elevation)
  - LOW:           lowest  elevation point on the route (name + elevation)
  - BORDER_IL_MO:  IL/MO border crossing (Chain of Rocks Bridge area)
  - BORDER_MO_KS:  MO/KS border (Joplin/Galena)
  - BORDER_KS_OK:  KS/OK border (Baxter Springs/Quapaw)
  - BORDER_OK_TX:  OK/TX border (Texola/Shamrock)
  - BORDER_TX_NM:  TX/NM border (Glenrio ghost town)
  - BORDER_NM_AZ:  NM/AZ border (Lupton)
  - BORDER_AZ_CA:  AZ/CA border (Topock/Needles, Colorado River)

Each entity asks the extractor to pull a (location_name, elevation_ft,
elevation_m) triple from the model's response. The auto_scorer then
matches the location against a per-slot GT keyword list and grades the
elevation by tolerance bands (0/1/2/3 per slot, 27 total).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS.parent))

from pipeline.extraction_pipeline import run_pipeline  # noqa: E402

QUERY_ID = 40
QUERY_TEXT = (
    "沿着美国66号公路（Route 66）从芝加哥驾车到洛杉矶，沿途海拔的最高点"
    "是多少、在哪里？最低点是多少、在哪里？以及每个州界处的海拔分别是多少？"
)


# (entity_id, slot_label, slot_description) — slot_label is shown to the LLM
SLOTS: list[tuple[str, str, str]] = [
    ("HIGH", "全程最高点", "Route 66 全程沿途海拔的最高点（地名 + 海拔）"),
    ("LOW", "全程最低点", "Route 66 全程沿途海拔的最低点（地名 + 海拔）"),
    (
        "BORDER_IL_MO",
        "IL↔MO 州界（伊利诺伊/密苏里）",
        "Route 66 跨越伊利诺伊州与密苏里州的州界处（一般在 Chain of Rocks Bridge 跨密西西比河，圣路易斯附近）的海拔",
    ),
    (
        "BORDER_MO_KS",
        "MO↔KS 州界（密苏里/堪萨斯）",
        "Route 66 跨越密苏里州与堪萨斯州的州界处（一般在 Joplin / Galena 一带）的海拔",
    ),
    (
        "BORDER_KS_OK",
        "KS↔OK 州界（堪萨斯/俄克拉荷马）",
        "Route 66 跨越堪萨斯州与俄克拉荷马州的州界处（一般在 Baxter Springs / Quapaw 一带）的海拔",
    ),
    (
        "BORDER_OK_TX",
        "OK↔TX 州界（俄克拉荷马/得克萨斯）",
        "Route 66 跨越俄克拉荷马州与得克萨斯州的州界处（一般在 Texola / Shamrock 一带，得州狭长地带）的海拔",
    ),
    (
        "BORDER_TX_NM",
        "TX↔NM 州界（得克萨斯/新墨西哥）",
        "Route 66 跨越得克萨斯州与新墨西哥州的州界处（一般在 Glenrio 跨州鬼镇）的海拔",
    ),
    (
        "BORDER_NM_AZ",
        "NM↔AZ 州界（新墨西哥/亚利桑那）",
        "Route 66 跨越新墨西哥州与亚利桑那州的州界处（一般在 Lupton, AZ 紧邻边界）的海拔",
    ),
    (
        "BORDER_AZ_CA",
        "AZ↔CA 州界（亚利桑那/加利福尼亚）",
        "Route 66 跨越亚利桑那州与加利福尼亚州的州界处（一般在 Topock / Needles 跨科罗拉多河）的海拔",
    ),
]


ENTITIES = [{"id": sid, "name": f"{label} — {desc}"} for sid, label, desc in SLOTS]


def _hint(sid: str, label: str, desc: str) -> str:
    base = (
        f"请抽出模型对 **{label}** 的声明：\n"
        f"  - location_name: 模型给出的具体地名（地标/城市/山口名）\n"
        f"  - elevation_ft: 海拔英尺数值（数字，无单位）\n"
        f"  - elevation_m:  海拔米数值（数字，无单位）\n\n"
        f"槽位定义：{desc}\n\n"
        "**抽取规则：**\n"
        "- 模型若给出**多个候选**（例如最高点说既有 1937 前的 Glorieta Pass，"
        "  又有 1937 后的 Brannigan Park），抽取**模型主推/最终结论**的那一个；"
        "  如果模型并列推荐两者，抽取**第一个出现**的那个\n"
        "- 模型若同时给 ft 和 m 两个数字，**两个都填**；只给一个则另一个填 null"
        "（**不要**自己换算填补，避免引入误差）\n"
        "- 海拔以数字形式抽出（如 7400, 7,400, 7400 ft 都抽 7400），"
        "  忽略单位文字\n"
        "- 模型若说该槽位无法确定 / 数据缺失，置 not_mentioned=true，"
        "  其余字段全 null\n"
        "- 模型若把该槽位的地点说错（如 OK/TX 边界说在 Amarillo），"
        "  仍按字面抽取（下游 scorer 会判分）；**不要替模型纠错**\n"
        "- 模型在'排除/不是/反例'语境下提到的地点（如'有人误以为 Sitgreaves "
        "Pass 是最高，其实不是'）→ 不抽取该地点；抽取模型主张的正确答案\n"
    )
    if sid == "HIGH":
        base += (
            "\n**最高点专项提示**：\n"
            "- 题目问 Route 66 沿途**最高**海拔，公认有三种合法答案：\n"
            "  ① Brannigan Park / 49 Hill / Fortynine Hill, AZ（≈7,400 ft，1937 后主线最高）\n"
            "  ② Glorieta Pass, NM（≈7,500 ft，1926-1937 原线路最高）\n"
            "  ③ Continental Divide / Campbell Pass, NM（≈7,263 ft，常被旅游材料误称'最高'）\n"
            "- 抽取模型主张的那一个（不要替模型选）\n"
        )
    if sid == "LOW":
        base += (
            "\n**最低点专项提示**：\n"
            "- 题目问 Route 66 沿途**最低**海拔，公认答案是 Santa Monica（西端终点，太平洋岸），约 0 ft / 海平面\n"
            "- 模型若答 Santa Monica Pier / Pacific Ocean / Ocean Avenue 等都对应该槽\n"
            "- 模型若答 Chicago 起点（约 595 ft）或 Topock（约 456-499 ft）"
            "  作为'最低点'，仍按字面抽取（下游会判错）\n"
        )
    return base


PROMPT_HINTS = {sid: _hint(sid, label, desc) for sid, label, desc in SLOTS}


VALUE_SCHEMA = """{
  "value": {
    "location_name": "<模型给出的地名（中文/英文皆可）；未提则 null>",
    "elevation_ft": <数字，海拔英尺；未提则 null>,
    "elevation_m": <数字，海拔米；未提则 null>
  },
  "not_mentioned": <true 仅当模型完全未答此槽位时为 true；否则 false>,
  "supporting_span": "<原文片段 30-200 字，能证明你抽出的字段来自回答>",
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
