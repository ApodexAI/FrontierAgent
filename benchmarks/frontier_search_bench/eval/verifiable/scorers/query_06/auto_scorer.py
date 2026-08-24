"""
Query 06 — Nobel-laureate paper-rejection auto-scorer.

Scoring rule:
  ✅ 正确声明          → +1.0
  ⚠️ 部分正确/证据不充分 → 0.0（不加不扣，但计入作答数）
  ❌ 错误/无关声明     → -1.0（扣分）
  未对齐到 D1-D47      → 进 unverified_claims，不计分
  MAX_SCORE            = 所有 ✅ 基准分之和（仅累加正分）
  total_score          可为负。

Pipeline:
  1. Extraction — v2 pipeline (primary Claude-Sonnet-4 + secondary GPT-5 +
     optional phase-4 analyzer), produces per-model canonical claim lists.
  2. Alignment — 3-model vote + field cross-check + kw sanity + optional
     judge LLM, produces canonical_id per claim (see pipeline/alignment.py).
  3. Scoring — deterministic lookup from canonical_id to benchmark score.
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
# Benchmark — 43 curated dimensions (D1-D43), judgment ✅/⚠️/❌ tagged by the
# Verified against external sources.
# ═══════════════════════════════════════════════════════════════════════════

SNAPSHOT_DATE = "2026-05-11"
SCORING_MODE = "A2+C"

DIMS = [
    (
        "D1",
        "Hans Krebs / Nature拒 / Enzymologia (1953医学)",
        "✅",
        1.0,
        ["krebs", "克雷布斯", "柠檬酸循环", "citric acid", "enzymologia"],
    ),
    (
        "D2",
        "Rosalyn Yalow / Science+JCI拒 (1977医学)",
        "⚠️",
        0.0,
        ["yalow", "耶洛", "雅洛", "radioimmunoassay", "放射免疫", "ria"],
    ),
    (
        "D3",
        "Peter Ratcliffe / Nature拒 / PNAS (2019医学)",
        "✅",
        1.0,
        ["ratcliffe", "拉特克利夫", "oxygen sensing", "低氧感应", "hif"],
    ),
    (
        "D4",
        "Paul Lauterbur / Nature拒后申诉 (2003医学)",
        "✅",
        1.0,
        ["lauterbur", "劳特布尔", "mri", "磁共振成像"],
    ),
    (
        "D5",
        "Katalin Karikó / Nature+Science拒 / Immunity (2023医学)",
        "✅",
        1.0,
        ["kariko", "karikó", "卡里科", "考里科", "mrna核苷", "weissman", "韦斯曼"],
    ),
    (
        "D6",
        "Barry Marshall / 学会拒 (2005医学)",
        "⚠️",
        0.0,
        ["marshall", "马歇尔", "pylori", "幽门螺杆菌"],
    ),
    (
        "D7",
        "Leland Hartwell / Nature desk-reject (2001医学)",
        "⚠️",
        0.0,
        ["hartwell", "哈特韦尔", "cell cycle gene", "细胞周期控制"],
    ),
    (
        "D8",
        "Baruch Blumberg / Annals拒 (1976医学)",
        "⚠️",
        0.0,
        ["blumberg", "布隆伯格", "hepatitis b", "乙肝"],
    ),
    (
        "D9",
        "Tim Hunt拒稿经历 (2001医学)",
        "⚠️",
        0.0,
        ["tim hunt", "蒂姆·亨特", "cyclin发现"],
    ),
    (
        "D10",
        "Richard Ernst / JCP拒两次 / RSI (1991化学)",
        "✅",
        1.0,
        ["ernst", "恩斯特", "ft-nmr", "fourier transform nmr", "傅里叶变换核磁"],
    ),
    (
        "D11",
        "Dan Shechtman / JAP拒 / PRL (2011化学)",
        "✅",
        1.0,
        ["shechtman", "谢赫特曼", "quasicrystal", "准晶"],
    ),
    (
        "D12",
        "Kary Mullis / Nature+Science拒 / Methods Enzymol (1993化学)",
        "✅",
        1.0,
        ["mullis", "穆利斯", "pcr", "聚合酶链式反应"],
    ),
    (
        "D13",
        "John Polanyi / PRL拒 (1986化学)",
        "⚠️",
        0.0,
        ["polanyi", "波拉尼", "chemical laser", "化学激光", "化学反应动力学"],
    ),
    (
        "D14",
        "Aaron Ciechanover / JBC拒 / BBRC (2004化学)",
        "⚠️",
        0.0,
        ["ciechanover", "切哈诺沃", "ubiquitin", "泛素"],
    ),
    (
        "D15",
        "Paul Boyer / JBC拒 / PNAS (1997化学)",
        "✅",
        1.0,
        ["boyer", "博耶", "atp synthase", "atp合酶", "旋转催化"],
    ),
    (
        "D16",
        "Michael Smith / Cell拒 / JBC (1993化学)",
        "✅",
        1.0,
        ["michael smith", "迈克尔·史密斯", "site-directed mutagenesis", "定点诱变"],
    ),
    (
        "D17",
        "Sidney Altman / Nature拒 / PNAS (1989化学)",
        "✅",
        1.0,
        ["altman", "奥尔特曼", "ribozyme", "rna catalytic", "rnase p"],
    ),
    (
        "D18",
        "Deisenhofer/Huber/Michel / Nature拒 (1988化学)",
        "❌",
        -1.0,
        [
            "deisenhofer",
            "huber",
            "michel",
            "photosynthesis reaction center",
            "光合作用反应中心",
        ],
    ),
    (
        "D19",
        "Enrico Fermi / Nature拒β衰变 (1938物理)",
        "⚠️",
        0.0,
        ["fermi", "费米", "beta decay", "β衰变", "弱相互作用"],
    ),
    (
        "D20",
        "Peter Higgs / Physics Letters拒 / PRL (2013物理)",
        "✅",
        1.0,
        ["higgs", "希格斯", "symmetry breaking", "对称性破缺", "玻色子"],
    ),
    (
        "D21",
        "Herbert Kroemer / APL拒 / Proc IEEE (2000物理)",
        "✅",
        1.0,
        ["kroemer", "克罗默", "heterostructure", "异质结构"],
    ),
    (
        "D22",
        "Lee/Osheroff/Richardson / PRL申诉后接收 (1996物理)",
        "✅",
        1.0,
        ["osheroff", "richardson", "superfluid helium-3", "超流氦-3", "超流体"],
    ),
    (
        "D23",
        "Robert Laughlin / PRL拒FQHE (1998物理)",
        "⚠️",
        0.0,
        ["laughlin", "劳克林", "fractional quantum hall", "分数量子霍尔"],
    ),
    # CERN 官网 + Santa Fe Institute (Helen
    # Tuck 亲述) 证实 PRL 拒稿；二手档案但可信，折中判定 ⚠️。
    (
        "D24",
        "Murray Gell-Mann / Physical Review拒 (1969物理)",
        "⚠️",
        0.0,
        ["gell-mann", "盖尔曼", "quark", "夸克", "curious particles", "奇特粒子"],
    ),
    (
        "D25",
        "Binnig & Rohrer / 拒STM (1986物理)",
        "⚠️",
        0.0,
        ["binnig", "rohrer", "宾尼希", "scanning tunneling", "扫描隧道显微"],
    ),
    (
        "D26",
        "Hideki Yukawa / Nature拒 (1949物理)",
        "⚠️",
        0.0,
        ["yukawa", "汤川", "meson theory", "介子理论"],
    ),
    (
        "D27",
        "Theodore Maiman / PRL拒激光器(非诺奖)",
        "❌",
        -1.0,
        ["maiman", "梅曼", "first laser", "首台激光"],
    ),
    (
        "D28",
        "William Sharpe / J Finance拒 (1990经济学)",
        "✅",
        1.0,
        ["sharpe", "夏普", "capm", "资本资产定价"],
    ),
    (
        "D29",
        "George Akerlof / AER+RES+JPE拒 / QJE (2001经济学)",
        "✅",
        1.0,
        ["akerlof", "阿克尔洛夫", "lemon", "柠檬市场", "信息不对称"],
    ),
    (
        "D30",
        "Black-Scholes / JPE+RES拒 (1997经济学)",
        "✅",
        1.0,
        ["black-scholes", "scholes", "期权定价", "options pricing"],
    ),
    (
        "D31",
        "Robert Lucas / AER拒 (1995经济学)",
        "✅",
        1.0,
        ["lucas", "卢卡斯", "rational expectations", "理性预期"],
    ),
    (
        "D32",
        "Richard Thaler / 多家期刊拒 (2017经济学)",
        "✅",
        1.0,
        ["thaler", "塞勒", "consumer choice", "行为经济"],
    ),
    (
        "D33",
        "William Golding / ~21出版商拒 (1983文学)",
        "⚠️",
        0.0,
        ["golding", "戈尔丁", "lord of the flies", "蝇王"],
    ),
    # Novy Mir 是苏联文学期刊，1956-09 编辑部
    # 签署正式拒稿信（Wikipedia + GLLI-US 独立来源）。属于期刊拒稿 → ✅。
    (
        "D34",
        "Boris Pasternak / Novy Mir 1956期刊拒 (1958文学)",
        "✅",
        1.0,
        ["pasternak", "帕斯捷尔纳克", "zhivago", "日瓦戈", "novy mir"],
    ),
    (
        "D35",
        "Arthur Kornberg / JBC审稿人推荐拒绝→作者撤回 (1959医学)",
        "⚠️",
        0.0,
        ["kornberg", "科恩伯格", "dna polymerase", "dna聚合酶"],
    ),
    (
        "D36",
        "Thomas Cech / Cell审稿人反对但编辑发表(非拒稿) (1989化学)",
        "❌",
        -1.0,
        ["thomas cech", "切赫", "self-splicing", "自剪接", "ribozyme催化rna"],
    ),
    (
        "D37",
        "Carolyn Bertozzi / Nature拒(无法验证) (2022化学)",
        "❌",
        -1.0,
        [
            "bertozzi",
            "贝尔托齐",
            "bioorthogonal",
            "生物正交",
            "click chemistry点击化学",
        ],
    ),
    (
        "D38",
        "Pavel Cherenkov / Nature拒 / Physical Review (1958物理)",
        "✅",
        1.0,
        ["cherenkov", "切伦科夫", "cherenkov radiation", "切伦科夫辐射"],
    ),
    (
        "D39",
        "Leo Esaki / Physical Review拒超晶格论文(非诺奖隧穿二极管) (1973物理)",
        "❌",
        -1.0,
        [
            "esaki",
            "江崎",
            "tunnel diode",
            "隧穿二极管",
            "tunneling semiconductor",
            "superlattice",
            "超晶格",
        ],
    ),
    # Einstein-Rosen 1936 引力波论文确被
    # Physical Review 拒（Physics Today archive）。但与 1921 光电效应诺奖
    # 主题不相关 → 套用"拒稿属实但与获奖非直接对应"规则（同 D13/D14），
    # 判定 ⚠️ 保持规则一致性。
    (
        "D40",
        "Albert Einstein / Physical Review拒引力波 (1921物理, 非获奖论文)",
        "⚠️",
        0.0,
        [
            "einstein gravitational",
            "爱因斯坦引力波",
            "gravitational wave paper rejected",
            "einstein rosen",
        ],
    ),
    (
        "D41",
        "Paul Samuelson / 期刊拒稿(无法验证) (1970经济学)",
        "❌",
        -1.0,
        ["samuelson拒稿", "萨缪尔森拒稿"],
    ),
    (
        "D42",
        "Lynn Margulis / ~15家期刊拒内共生论文 (非诺奖)",
        "❌",
        -1.0,
        ["margulis", "马古利斯", "endosymbiosis", "内共生"],
    ),
    (
        "D43",
        "Chen-Ning Yang / PRL拒Yang-Baxter方程论文 (1957物理)",
        "❌",
        -1.0,
        ["chen-ning yang", "杨振宁", "yang-baxter", "杨-巴克斯特"],
    ),
    # web-search-verified 2026-04-24 — added from null_resolutions.json
    (
        "D44",
        "William Kaelin Jr. / 三人组并列主张 Nature 拒稿(Kaelin 本人获奖论文无期刊拒稿记录) (2019医学)",
        "⚠️",
        0.0,
        ["kaelin", "凯林", "vhl", "hif", "oxygen sensing"],
    ),
    # web-search-verified 2026-04-24 — added from null_resolutions.json
    (
        "D45",
        "Gregg L. Semenza / 三人组并列主张 Nature 拒稿(Semenza 本人获奖论文无期刊拒稿记录) (2019医学)",
        "⚠️",
        0.0,
        ["semenza", "塞门扎", "hif-1", "oxygen sensing"],
    ),
    # web-search-verified 2026-05-11 — added from null_resolutions.json
    (
        "D46",
        "Klaus von Klitzing / Physical Review Letters 初稿拒 / PRL 修改后接受 (1985物理) — 原题 'Realization of a resistance standard based on natural constants' 被审稿人反对，修改为 'New method for high-accuracy determination of the fine-structure constant based on quantized Hall resistance' 后接受",
        "✅",
        1.0,
        [],
    ),
    # web-search-verified 2026-05-11 — added from null_resolutions.json
    (
        "D47",
        "William Nordhaus / 首份气候变化经济学主要论文被经济期刊拒 / Science 1992 发表 (2018经济) — DICE 模型前身论文，二手记载，缺乏具体期刊名称和拒稿信函原始文件",
        "⚠️",
        0.0,
        [],
    ),
]

DIM_MAP = {
    d[0]: {"id": d[0], "name": d[1], "judgment": d[2], "score": d[3], "kw": d[4]}
    for d in DIMS
}
MAX_SCORE = sum(d[3] for d in DIMS if d[3] > 0)  # 仅累加 ✅ 正分基准


# ═══════════════════════════════════════════════════════════════════════════
# DIMS → alignment baseline spec
# ═══════════════════════════════════════════════════════════════════════════


def _derive_match_fields(description: str) -> dict:
    """Parse year/field out of the DIMS description string for cross-check."""
    fields: dict = {}
    m = _re.search(r"(\d{4})", description)
    if m:
        fields["nobel_year"] = m.group(1)
    for f in ["生理学或医学", "医学", "物理", "化学", "经济学", "文学", "和平"]:
        if f in description:
            fields["nobel_field"] = f
            break
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
                # keep for downstream scoring:
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
    `entities[0].canonical.value` list (no keyword matching here; alignment
    is a separate stage)."""
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
    `canonical_id` (or None) plus `alignment_confidence`. Scoring:
      - canonical_id in DIM_MAP and first-occurrence → apply benchmark score
      - canonical_id = None → unverified (unclassified by baseline)
      - alignment_confidence = "needs_review" → unverified (separate reason)
      - duplicates dropped
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

            if conf == "needs_review":
                unverified.append(
                    {
                        "name": raw.get("name", "") or raw.get("laureate_name", ""),
                        "summary": raw.get("rejected_paper_topic", ""),
                        "canonical_id_tentative": cid,
                        "reason": f"needs_review: {reason}",
                        "judge_invoked": judge_invoked,
                    }
                )
                continue

            if cid == "__HALLUCINATION__":
                # null_resolution marked this as verified hallucination → -1
                scored.append(
                    {
                        "id": "__HALLUCINATION__",
                        "name": "(verified hallucination)",
                        "person": raw.get("name", "") or raw.get("laureate_name", ""),
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
                        "person": raw.get("name", "") or raw.get("laureate_name", ""),
                        "score": d["score"],
                        "judgment": d["judgment"],
                        "confidence": conf,
                        "judge_invoked": judge_invoked,
                        "reason": reason or f"对齐到 {cid}（基准 {d['judgment']}）",
                    }
                )
                total += d["score"]
            else:
                # canonical_id is null (not in baseline) — does not count
                unverified.append(
                    {
                        "name": raw.get("name", "") or raw.get("laureate_name", ""),
                        "summary": raw.get("rejected_paper_topic", ""),
                        "canonical_id_tentative": cid,
                        "reason": reason or "未匹配到 D1-D43",
                        "judge_invoked": judge_invoked,
                    }
                )

        n_answered = len(scored)
        all_scores[model_name] = {
            "total_score": total,
            "max_score": MAX_SCORE,
            "score_rate": round(total / n_answered, 4) if n_answered else 0.0,
            "total_rate": round(total / MAX_SCORE, 4),
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
        "extraction_pipeline": "v2.1 (primary=claude-sonnet-4, secondary=gpt-5, analyzer=claude-opus-4.6)",
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
        "# Query 06 排名报告（v2.1 抽取）",
        "",
        f"> Snapshot: {SNAPSHOT_DATE}  Mode: {SCORING_MODE}  Max: {MAX_SCORE}",
        "> 抽取流水线：Primary=claude-sonnet-4, Secondary=gpt-5, Analyzer=claude-opus-4.6",
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


def main() -> None:
    ap = argparse.ArgumentParser(description="Query 06 auto-scorer (v2.1+align)")
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

    # Stage 2: alignment — canonical_id per claim via vote + checks + judge
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

    # Stage 3a: export null claims for external verification (with context_span
    # from raw model responses so the verifier can recover from extraction drops)
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

    # Stage 3b: apply null_resolutions.json if present (verification feedback)
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
            # merge into runtime DIM_MAP so score_aligned recognizes the new ids
            for e in new_baseline_entries:
                DIM_MAP[e["id"]] = {
                    "id": e["id"],
                    "name": e.get("description", e["id"]),
                    "judgment": e.get("judgment", "⚠️"),
                    "score": e.get("score", 0.0),
                    "kw": e.get("kw", []),
                }
            # persist back to auto_scorer.py (DIMS extension) for future runs
            persist_new_baseline_entries(new_baseline_entries, __file__)
    else:
        print("[*] No null_resolutions.json (run web-search agent to produce one).")

    # Stage 4: scoring (with resolutions applied, if any)
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
    print("Query 06 scoring done.")
    for i, (m, s) in enumerate(
        sorted(scores.items(), key=lambda x: -x[1]["total_score"]), 1
    ):
        print(
            f"  {i}. {m:28s} {s['total_score']:>5.1f}/{s['max_score']:.1f}"
            f"  answered={s['dimensions_answered']}"
            f"  unverified={len(s['unverified_claims'])}"
        )


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


if __name__ == "__main__":
    main()
