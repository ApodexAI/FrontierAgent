"""
Query 12 — Nobel-laureate (Literature) officially expelled-or-exiled
auto-scorer.

Scoring rule:
  ✅ 正确声明（被本国政府正式驱逐出境/剥夺国籍）→ +1.0
  ⚠️ 边界案例（如群体性国籍法令而非个人针对）  →  0.0
  ❌ 错误声明（自我流亡 / 政治避难 / 逃亡 / 受威胁未被驱逐）→ -1.0
  __HALLUCINATION__（Stage C 验证为虚构）          → -1.0
  null + Stage C unresolved（找不到证据 / 闭合集合外）→ -1.0
  alignment_confidence = "needs_review"            →  0.0（不计分，特殊路径）
  MAX_SCORE = 所有 ✅ 基准分之和（仅累加正分）
  total_score 可为负

Pipeline:
  1. Extraction — v2 pipeline (primary Claude-Sonnet-4 + secondary GPT-5
     + analyzer Claude-Opus-4.6).
  2. Alignment — 3-model vote + field cross-check + kw sanity + judge LLM
     (see pipeline/alignment.py).
  3. Null verification — manual + Claude-Code-assisted; produce
     null_resolutions.json (Stage C, see pipeline/alignment.py).
  4. Scoring — deterministic lookup from canonical_id to benchmark score.
"""

from __future__ import annotations

import argparse
import json
import re as _re
import sys
from pathlib import Path

THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS.parent))

from extract import (  # noqa: E402
    ENTITIES,
    PROMPT_HINTS,
    QUERY_ID,
    QUERY_TEXT,
    VALUE_SCHEMA,
)
from pipeline.alignment import (  # noqa: E402
    align_claims,
    apply_null_resolutions,
    export_null_claims_for_review,
    persist_new_baseline_entries,
)
from pipeline.extraction_pipeline import get_client, run_pipeline  # noqa: E402

# ═══════════════════════════════════════════════════════════════════════════
# Benchmark — 15 curated dimensions (D1-D15).
#   D1, D2, D4   : ✅ 政府正式驱逐令 + 强制押送出境（最严格满足题意）
#   D3, D5, D6,  : ⚠️ 政府正式法令剥夺国籍但本人非被押送（D3/D5）；
#   D14            或群体性国籍法令非个人针对（D6 Sachs / D14 Bunin）
#   D7-D13, D15  : ❌ 自我流亡 / 政治避难 / 逃亡 / 政府正式行为但非驱逐
#                  令（如通缉/审判/护照没收/不受欢迎人物/准许出境）
# Cross-verified 2026-04-29 against NobelPrize.org / Wikipedia / Britannica
# / USHMM Holocaust Encyclopedia / Saint-John Perse Foundation.
# Strict-literal re-judgement 2026-04-29: D3 Mann / D5 Perse 由 ✅ 降为 ⚠️
# （政府正式法令剥夺国籍 ≠ 题目要求的"正式驱逐出境或流放"）。
# ═══════════════════════════════════════════════════════════════════════════

SNAPSHOT_DATE = "2026-04-29"
SCORING_MODE = "A2+C-strict"  # null + unresolved → -1（与 Q07 不同）

DIMS = [
    # ─── ✅ 正确答案（3 条，+1）：政府正式驱逐令 + 强制押送出境 ───
    (
        "D1",
        "Aleksandr Solzhenitsyn (1970文学) / 苏联 / "
        "1974-02-12 苏联最高苏维埃主席团颁令剥夺国籍，"
        "次日由KGB强制驱逐至西德法兰克福",
        "✅",
        1.0,
        [
            "solzhenitsyn",
            "索尔仁尼琴",
            "古拉格群岛",
            "gulag archipelago",
            "苏联",
        ],
    ),
    (
        "D2",
        "Joseph Brodsky (1987文学) / 苏联 / "
        "1972-06-04 被苏联当局押上飞机驱逐至维也纳，"
        "并使其成为无国籍人(stateless 1972-1977)",
        "✅",
        1.0,
        [
            "brodsky",
            "约瑟夫·布罗茨基",
            "布罗茨基",
            "iosif brodsky",
            "ленинград",
        ],
    ),
    (
        "D3",
        "Thomas Mann (1929文学) / 纳粹德国 / "
        "1933-02 在瑞士度假时希特勒上台，子女警告其不要回德国，"
        "自此自行留在国外；1936 年纳粹政府正式剥夺其本人及妻儿的"
        "德国国籍，没收慕尼黑家产，波恩大学撤销荣誉博士。"
        "性质：政府正式法令剥夺国籍（denaturalization decree），"
        "但法令内容非'驱逐令'，且本人非被押送出境（自行已在国外）。"
        "学界惯用 'exile by Nazis' 但属广义；按题目'正式驱逐出境或流放'"
        "的字面严格判定属 ⚠️ 边界",
        "⚠️",
        0.0,
        [
            "thomas mann",
            "托马斯·曼",
            "魔山",
            "buddenbrooks",
            "布登勃洛克",
            "magic mountain",
        ],
    ),
    (
        "D4",
        "Miguel Ángel Asturias (1967文学) / 危地马拉 / "
        "1954 年阿本斯政权倒台后，Castillo Armas 政权"
        "剥夺其危地马拉国籍并驱逐；流亡布宜诺斯艾利斯 8 年；"
        "1966 年 Méndez Montenegro 政府恢复其国籍",
        "✅",
        1.0,
        [
            "asturias",
            "阿斯图里亚斯",
            "miguel angel asturias",
            "危地马拉",
            "总统先生",
            "el señor presidente",
        ],
    ),
    (
        "D5",
        "Saint-John Perse / Alexis Léger (1960文学) / 法国维希政府 / "
        "1940-07 法国沦陷后自行赴美定居华盛顿；"
        "1940-10 维希政府颁令剥夺其法国国籍、撤销荣誉军团勋位、没收财产。"
        "性质：政府正式法令剥夺国籍（denaturalization decree），"
        "但法令内容非'驱逐令'，且本人非被押送出境（自行已在国外）。"
        "学界惯用 'exiled by Vichy' 但属广义；按题目'正式驱逐出境或流放'"
        "的字面严格判定属 ⚠️ 边界（与 D3 Mann 同档）",
        "⚠️",
        0.0,
        [
            "saint-john perse",
            "圣-琼·佩斯",
            "圣琼佩斯",
            "alexis léger",
            "alexis leger",
            "莱热",
            "perse",
        ],
    ),
    # ─── ⚠️ 边界（与 D3/D5 同档：政府正式法令但非驱逐令本身；0 分）───
    (
        "D6",
        "Nelly Sachs (1966文学) / 纳粹德国 / "
        "1940 年逃往瑞典；纳粹《纽伦堡法案》体系性剥夺犹太人国籍"
        "属于群体性种族法令而非针对她个人的专门驱逐令；"
        "她本人并非因言论被点名驱逐",
        "⚠️",
        0.0,
        [
            "nelly sachs",
            "奈莉·萨克斯",
            "萨克斯",
            "selma lagerlöf",
            "犹太裔",
        ],
    ),
    # ─── ❌ 错误答案：自我流亡 / 政治避难 / 逃亡（7 条，-1） ───
    (
        "D7",
        "Gao Xingjian (2000文学) / 中国 / "
        "1987 年自行赴法定居，作品被禁、被宣布为不受欢迎人物，"
        "但中国政府从未颁布对其个人的剥夺国籍令或驱逐令；"
        "属于'自我流亡 + 后被宣布 persona non grata'，非正式驱逐",
        "❌",
        -1.0,
        [
            "gao xingjian",
            "高行健",
            "灵山",
            "soul mountain",
            "逃亡",
        ],
    ),
    (
        "D8",
        "Czesław Miłosz (1980文学) / 波兰 / "
        "1951 年自巴黎使馆任上'出走'公开宣告与波共政权决裂，"
        "在法国申请政治避难；属于主动叛逃 + 政治避难，非被官方驱逐",
        "❌",
        -1.0,
        [
            "milosz",
            "miłosz",
            "切斯瓦夫·米沃什",
            "米沃什",
            "captive mind",
            "被禁锢的头脑",
        ],
    ),
    (
        "D9",
        "Herta Müller (2009文学) / 罗马尼亚（齐奥塞斯库政权）/ "
        "因拒绝与 Securitate 秘密警察合作长期受迫害，"
        "1987 年获罗马尼亚政府准许移居西德；"
        "属于'受迫害后获准出境'，非被正式驱逐",
        "❌",
        -1.0,
        [
            "herta müller",
            "herta muller",
            "赫塔·米勒",
            "赫塔·穆勒",
            "securitate",
            "罗马尼亚",
        ],
    ),
    (
        "D10",
        "Pablo Neruda (1971文学) / 智利 / "
        "1948 年 González Videla 政府禁共后对其发出逮捕令，"
        "他藏匿后骑马翻越安第斯山逃往阿根廷；"
        "属于'遭通缉后越境逃亡'，非被官方驱逐",
        "❌",
        -1.0,
        [
            "neruda",
            "巴勃罗·聂鲁达",
            "聂鲁达",
            "二十首情诗",
            "canto general",
            "智利",
        ],
    ),
    (
        "D11",
        "Boris Pasternak (1958文学) / 苏联 / "
        "苏联作协曾联名请愿要求剥夺其国籍并驱逐，"
        "苏联政府以'若出国领奖即不得回国'相威胁；"
        "他以拒领诺奖换取留境；最终未被实际驱逐或剥夺国籍",
        "❌",
        -1.0,
        [
            "pasternak",
            "鲍里斯·帕斯捷尔纳克",
            "帕斯捷尔纳克",
            "doctor zhivago",
            "日瓦戈医生",
        ],
    ),
    (
        "D12",
        "Wole Soyinka (1986文学) / 尼日利亚（阿巴查军政府）/ "
        "1994 年阿巴查没收其护照，他骑摩托经贝宁逃出境；"
        "1997 年被以叛国罪缺席判处死刑；"
        "属于'逃亡避难 + 缺席判决'，非政府颁布个人驱逐令",
        "❌",
        -1.0,
        [
            "wole soyinka",
            "索因卡",
            "soyinka",
            "abacha",
            "阿巴查",
            "尼日利亚",
        ],
    ),
    (
        "D13",
        "Svetlana Alexievich (2015文学) / 白俄罗斯 / "
        "因卢卡申科政府迫害，2000 年自我流亡至巴黎/哥德堡/柏林；"
        "2011 年自愿回国；2020 年大选抗议后再次主动出境赴德；"
        "白俄罗斯政府从未颁布剥夺国籍或驱逐令",
        "❌",
        -1.0,
        [
            "alexievich",
            "斯维特拉娜·阿列克谢耶维奇",
            "阿列克谢耶维奇",
            "亚历塞维奇",
            "lukashenko",
            "卢卡申科",
            "白俄罗斯",
        ],
    ),
    # web-search-verified 2026-04-29 — added from null_resolutions.json
    (
        "D14",
        "Ivan Bunin (1933文学) / 苏俄→苏联 / 1920 年随白军溃败逃离克里米亚至法国（主动流亡）；苏联 1922 年成立后颁布去国籍化法令系统性剥夺已离境白俄流亡者的国籍（群体性法令而非针对蒲宁个人）；其后持南森护照作为无国籍人。性质类比 D6 Sachs（群体性国籍法令而非个人针对）",
        "⚠️",
        0.0,
        [
            "bunin",
            "蒲宁",
            "ivan bunin",
            "伊万·蒲宁",
            "南森护照",
            "nansen passport",
            "白俄",
            "白军",
        ],
    ),
    # web-search-verified 2026-04-29 — added from null_resolutions.json
    (
        "D15",
        "Doris Lessing (2007文学) / 英国（本国）/ 1956 年被南罗得西亚（殖民地政府，禁令至 1982）和南非（外国，禁令至 1980 年代末）宣布为 prohibited immigrant；并非'被本国（英国）政府正式驱逐'。'禁止入境(prohibited alien)' 性质为'拒绝入境'而非'驱逐出境'；驱逐方亦非其本国政府。",
        "❌",
        -1.0,
        [
            "doris lessing",
            "莱辛",
            "多丽丝·莱辛",
            "rhodesia",
            "南罗得西亚",
            "津巴布韦",
            "prohibited immigrant",
            "禁止入境",
        ],
    ),
]

DIM_MAP = {
    d[0]: {"id": d[0], "name": d[1], "judgment": d[2], "score": d[3], "kw": d[4]}
    for d in DIMS
}
MAX_SCORE = sum(d[3] for d in DIMS if d[3] > 0)  # 仅累加 ✅ 正分基准 = 5.0


# ═══════════════════════════════════════════════════════════════════════════
# DIMS → alignment baseline spec
# ═══════════════════════════════════════════════════════════════════════════


def _derive_match_fields(description: str) -> dict:
    """Parse nobel_year and country out of the DIMS description string.

    Format expected: "<Name> (<year>文学) / <country> / <event>". Returns an
    empty dict if regex doesn't match (alignment cross-check degrades to
    LLM-only)."""
    fields: dict = {}
    m_year = _re.search(r"\((\d{4})文学\)", description)
    if m_year:
        fields["nobel_year"] = m_year.group(1)
    m_country = _re.search(r"文学\)\s*/\s*([^/]+?)\s*/", description)
    if m_country:
        fields["country"] = m_country.group(1).strip()
    return fields


def build_baselines() -> list[dict]:
    """Convert DIMS into the baseline spec consumed by pipeline.alignment."""
    out = []
    for d_id, desc, judgment, score, kw in DIMS:
        out.append(
            {
                "id": d_id,
                "description": f"{desc} [基准: {judgment}]",
                "match_fields": _derive_match_fields(desc),
                "kw": kw,
                "judgment": judgment,
                "score": score,
            }
        )
    return out


# ═══════════════════════════════════════════════════════════════════════════
# Load raw claims from v2 extraction output
# ═══════════════════════════════════════════════════════════════════════════


def load_raw_claims(extraction_dir: Path) -> dict[str, list[dict]]:
    """Read each model's `{dir}/{model}/extraction.json` and return the raw
    `entities[0].canonical.value` list."""
    out: dict[str, list[dict]] = {}
    for d in Path(extraction_dir).iterdir():
        if not d.is_dir():
            continue
        ext_file = d / "extraction.json"
        if not ext_file.exists():
            continue
        payload = json.loads(ext_file.read_text(encoding="utf-8"))
        entities = payload.get("entities", [])
        if not entities:
            out[d.name] = []
            continue
        canonical_value = (entities[0].get("canonical") or {}).get("value") or []
        if not isinstance(canonical_value, list):
            canonical_value = []
        out[d.name] = canonical_value
    return out


# ═══════════════════════════════════════════════════════════════════════════
# Scoring from aligned claims
# ═══════════════════════════════════════════════════════════════════════════


def score_aligned(aligned_by_model: dict[str, list[dict]]) -> dict[str, dict]:
    """Aggregate scores from alignment output. Each aligned claim carries a
    `canonical_id` (or None) plus `alignment_confidence`. Scoring rules
    (Q12-strict, differs from Q07):

      - canonical_id ∈ DIM_MAP (first occurrence)            → DIM_MAP[cid].score
      - canonical_id == "__HALLUCINATION__"                  → -1.0
      - canonical_id is None and confidence != needs_review  → -1.0  (☆ Q12 only)
      - alignment_confidence == "needs_review"               →  0.0 (special path)
      - duplicates dropped (only first occurrence per cid counts)

    Totals can be negative.
    """
    all_scores: dict[str, dict] = {}
    for model_name, claims in aligned_by_model.items():
        seen_dims: set[str] = set()
        scored: list[dict] = []
        unverified: list[dict] = []
        total = 0.0
        for c in claims:
            raw = c.get("raw", {}) or {}
            cid = c.get("canonical_id")
            conf = c.get("alignment_confidence", "medium")
            reason = c.get("alignment_reasoning", "")
            judge_invoked = c.get("judge_invoked", False)
            person = raw.get("name", "") or raw.get("laureate_name", "")

            if conf == "needs_review":
                unverified.append(
                    {
                        "name": person,
                        "country": raw.get("country", ""),
                        "canonical_id_tentative": cid,
                        "reason": f"needs_review: {reason}",
                        "judge_invoked": judge_invoked,
                        "score": 0.0,
                    }
                )
                continue

            if cid == "__HALLUCINATION__":
                scored.append(
                    {
                        "id": "__HALLUCINATION__",
                        "name": "(verified hallucination)",
                        "person": person,
                        "score": -1.0,
                        "judgment": "❌",
                        "confidence": conf,
                        "judge_invoked": judge_invoked,
                        "reason": reason,
                    }
                )
                total += -1.0
                continue

            if cid and cid in DIM_MAP:
                if cid in seen_dims:
                    continue  # dedup
                seen_dims.add(cid)
                d = DIM_MAP[cid]
                scored.append(
                    {
                        "id": cid,
                        "name": d["name"],
                        "person": person,
                        "score": d["score"],
                        "judgment": d["judgment"],
                        "confidence": conf,
                        "judge_invoked": judge_invoked,
                        "reason": reason or f"对齐到 {cid}（基准 {d['judgment']}）",
                    }
                )
                total += d["score"]
            else:
                # null + Stage C unresolved → -1（Q12 strict 规则）
                scored.append(
                    {
                        "id": "__UNRESOLVED_NULL__",
                        "name": "(null + unresolved, treated as wrong)",
                        "person": person,
                        "score": -1.0,
                        "judgment": "❌",
                        "confidence": conf,
                        "judge_invoked": judge_invoked,
                        "reason": reason or "未对齐到任何基准条目，且 Stage C 未能验证",
                    }
                )
                total += -1.0

        n_answered = len(scored)
        all_scores[model_name] = {
            "total_score": total,
            "max_score": MAX_SCORE,
            "score_rate": round(total / n_answered, 4) if n_answered else 0.0,
            "total_rate": round(total / MAX_SCORE, 4) if MAX_SCORE else 0.0,
            "dimensions_answered": n_answered,
            "per_dimension": sorted(scored, key=lambda x: x["id"]),
            "unverified_claims": unverified,
        }
    return all_scores


# ═══════════════════════════════════════════════════════════════════════════
# Outputs
# ═══════════════════════════════════════════════════════════════════════════


def build_scores_json(all_scores: dict[str, dict]) -> dict:
    ranking = sorted(all_scores.items(), key=lambda x: -x[1]["total_score"])
    unverified_all = []
    for m, s in all_scores.items():
        for uv in s.get("unverified_claims", []):
            unverified_all.append({"model": m, **uv, "action_needed": "需人工核实"})
    return {
        "query_id": QUERY_ID,
        "scoring_mode": SCORING_MODE,
        "snapshot_date": SNAPSHOT_DATE,
        "max_score": MAX_SCORE,
        "extraction_pipeline": "v2 (primary=claude-sonnet-4, secondary=gpt-5, analyzer=claude-opus-4.6)",
        "results": all_scores,
        "ranking": [
            {
                "rank": i + 1,
                "model": m,
                "score": s["total_score"],
                "rate": f"{s['total_rate'] * 100:.1f}%",
            }
            for i, (m, s) in enumerate(ranking)
        ],
        "unverified_items": unverified_all,
    }


def build_ranking_md(all_scores: dict[str, dict]) -> str:
    ranked = sorted(
        all_scores.items(),
        key=lambda x: (-x[1]["total_score"], -x[1].get("score_rate", 0)),
    )
    lines = [
        "# Query 12 排名报告（被本国政府正式驱逐/流放的诺奖文学得主）",
        "",
        f"> Snapshot: {SNAPSHOT_DATE}  Mode: {SCORING_MODE}  Max: {MAX_SCORE}",
        "> 抽取流水线：Primary=claude-sonnet-4, Secondary=gpt-5, Analyzer=claude-opus-4.6",
        "> 评分：✅+1 / ⚠️0 / ❌-1 / null+unresolved -1 / __HALLUCINATION__ -1",
        "",
        "| Rank | Model | Score | Rate | Answered | Unverified |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for i, (m, s) in enumerate(ranked, 1):
        lines.append(
            f"| {i} | {m} | {s['total_score']}/{s['max_score']} "
            f"| {s['total_rate'] * 100:.1f}% "
            f"| {s['dimensions_answered']} "
            f"| {len(s['unverified_claims'])} |"
        )
    return "\n".join(lines) + "\n"


# ═══════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════


def _load_alignment(out_dir: Path) -> dict[str, list[dict]]:
    """Reload alignment.json from disk when --skip-align is set."""
    aligned: dict[str, list[dict]] = {}
    for d in Path(out_dir).iterdir():
        if not d.is_dir():
            continue
        af = d / "alignment.json"
        if not af.exists():
            continue
        payload = json.loads(af.read_text(encoding="utf-8"))
        aligned[d.name] = payload.get("aligned_claims", [])
    return aligned


def main() -> None:
    ap = argparse.ArgumentParser(description="Query 12 auto-scorer (v2+align)")
    ap.add_argument(
        "--models", nargs="+", help="name=path list of model answer JSON files."
    )
    ap.add_argument("--output-dir", default=str(THIS / "auto_scores"))
    ap.add_argument("--primary-model", default=None)
    ap.add_argument("--secondary-model", default=None)
    ap.add_argument("--parallel-models", nargs="+", default=None)
    ap.add_argument("--analyzer-model", default=None)
    ap.add_argument("--aligner-models", nargs="+", default=None)
    ap.add_argument("--judge-model", default=None)
    ap.add_argument("--concurrency", type=int, default=None)
    ap.add_argument(
        "--skip-extract",
        action="store_true",
        help="Skip extraction; reuse existing extraction.json.",
    )
    ap.add_argument(
        "--skip-align",
        action="store_true",
        help="Skip alignment; reuse existing alignment.json.",
    )
    args = ap.parse_args()

    if not args.models:
        sys.exit("[ERROR] --models is required.")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Stage 1: extraction
    if not args.skip_extract:
        run_pipeline(
            query_id=QUERY_ID,
            query_text=QUERY_TEXT,
            entities=ENTITIES,
            prompt_hints=PROMPT_HINTS,
            schema=VALUE_SCHEMA,
            models_input=args.models,
            output_dir=out_dir,
            primary=args.primary_model,
            secondary=args.secondary_model,
            parallel=args.parallel_models,
            analyzer=args.analyzer_model,
            concurrency=args.concurrency,
        )

    # Stage 2: alignment
    baselines = build_baselines()
    if args.skip_align:
        aligned = _load_alignment(out_dir)
    else:
        client = get_client()
        overrides: dict = {}
        if args.aligner_models:
            overrides["aligner_models"] = args.aligner_models
        if args.judge_model:
            overrides["judge_model"] = args.judge_model
        if args.concurrency:
            overrides["concurrency"] = args.concurrency
        raw_claims = load_raw_claims(out_dir)
        aligned = align_claims(
            client,
            claims_by_model=raw_claims,
            baselines=baselines,
            query_text=QUERY_TEXT,
            output_dir=out_dir,
            overrides=overrides,
        )

    # Stage 3a: export null claims for manual + Claude-Code-assisted verification
    null_items = export_null_claims_for_review(
        aligned,
        out_dir / "null_review.json",
        query_id=QUERY_ID,
        query_text=QUERY_TEXT,
        models_input=args.models,
    )
    print(
        f"\n[*] Exported {len(null_items)} null/needs_review claims → null_review.json"
    )

    # Stage 3b: apply null_resolutions.json if present
    resolutions_path = out_dir / "null_resolutions.json"
    if resolutions_path.exists():
        print("[*] Found null_resolutions.json, applying …")
        new_baseline_entries = apply_null_resolutions(
            aligned, resolutions_path, dims_ref=DIMS
        )
        if new_baseline_entries:
            print(
                f"[*] {len(new_baseline_entries)} new baseline entries from resolutions"
            )
            for e in new_baseline_entries:
                DIM_MAP[e["id"]] = {
                    "id": e["id"],
                    "name": e.get("description", e["id"]),
                    "judgment": e.get("judgment", "⚠️"),
                    "score": e.get("score", 0.0),
                    "kw": e.get("kw", []),
                }
            persist_new_baseline_entries(new_baseline_entries, __file__)
    else:
        print(
            f"[*] No null_resolutions.json yet."
            f"\n    → 用 Claude Code 读 {resolutions_path.parent}/null_review.json"
            f"\n      逐条 web 验证后产出 {resolutions_path.name}"
            f"\n      然后重跑：python3 {Path(__file__).name} "
            f"--skip-extract --skip-align --models …"
        )

    # Stage 4: scoring
    scores = score_aligned(aligned)

    (out_dir / "scores.json").write_text(
        json.dumps(build_scores_json(scores), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (out_dir / "ranking_report.md").write_text(
        build_ranking_md(scores),
        encoding="utf-8",
    )

    print("\n" + "─" * 62)
    print("Query 12 scoring done.")
    for i, (m, s) in enumerate(
        sorted(scores.items(), key=lambda x: -x[1]["total_score"]), 1
    ):
        print(
            f"  {i}. {m:28s} {s['total_score']:>5.1f}/{s['max_score']:.1f}"
            f"  answered={s['dimensions_answered']}"
            f"  unverified={len(s['unverified_claims'])}"
        )


if __name__ == "__main__":
    main()
