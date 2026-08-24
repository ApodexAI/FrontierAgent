"""Query 19 — Maekawa Kunio / Mundaneum auto-scorer.

Reference answer (rubric: 一点一分):
  - 城市     : 日内瓦 / Geneva
  - 建筑/项目: Mundaneum
  - 职位     : 无薪绘图员 / unpaid draftsman

Scoring rule (3 independent dimensions, each 0/1, max 3, min 0):
  A_city     : claim contains Geneva alias                   → +1
  B_building : claim contains Mundaneum (or 'Cité Mondiale' /
               'World City' / 'Centre Mondial' — historically
               the same project)                              → +1
  C_position : claim contains 'unpaid' / 无薪 / 无偿 / volunteer
               AND a drafting / drawing term (绘图 / 制图 /
               draftsman / draftsperson / drafter)            → +1
               若只有 drafting 但没说 unpaid → defer to LLM judge
               若只有 unpaid 但没说 drafting → defer to LLM judge

Pipeline (Stage 1 extract via pipeline → Stage 2 per-dim rule match →
Stage 3 LLM judge fallback for ambiguous dims → Stage 4 score).
"""

from __future__ import annotations

import argparse
import contextlib
import json
import re
import sys
import unicodedata
from dataclasses import dataclass, field
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
from pipeline.extraction_pipeline import (  # noqa: E402
    _extract_json,
    call_llm,
    get_client,
    run_pipeline,
)

# ═══════════════════════════════════════════════════════════════════════════
# Benchmark / alias lists
# ═══════════════════════════════════════════════════════════════════════════

BENCHMARK: dict = {
    "version": "1.0",
    "locked_at": "2026-05-08",
    # City — Geneva, Switzerland.
    "city_keys": [
        "geneva",
        "genève",
        "geneve",
        "genf",
        "ginebra",
        "ginevra",
        "日内瓦",
        "日內瓦",
    ],
    # Building / project — Mundaneum (later iterations also called
    # "Cité Mondiale" / "World City" / "Centre Mondial"; these are
    # historically the same project that Le Corbusier carried forward).
    "building_correct_keys": [
        "mundaneum",
        "cité mondiale",
        "cite mondiale",
        "cité mondial",
        "cite mondial",
        "centre mondial",
        "world city",
        "world centre",
        "world center",
        "世界城市",
        "世界中心",
        "蒙达内乌姆",
        "蒙达涅姆",
        "蒙达诺姆",
    ],
    # Building — clearly wrong but historically conflated.
    "building_wrong_keys": [
        "palais des nations",
        "palais de la société des nations",
        "league of nations",
        "国际联盟总部",
        "国联总部",
        "万国宫",
        "weissenhof",
        "villa savoye",
    ],
    # Position — explicit "unpaid" markers.
    "position_unpaid_keys": [
        "unpaid",
        "without pay",
        "no pay",
        "no salary",
        "without salary",
        "volunteer",
        "voluntary",
        "无薪",
        "无偿",
        "义务",
        "免费",
        "未支薪",
    ],
    # Position — explicit drafting / drawing markers.
    "position_drafting_keys": [
        "draftsman",
        "draftsperson",
        "draftman",
        "drafter",
        "drafting",
        "tracer",
        "draughtsman",
        "draughtsperson",
        "绘图员",
        "制图员",
        "绘图",
        "制图",
        "描图员",
        "描图",
    ],
    # Other commonly seen role descriptions — defer to LLM (could still
    # be acceptable if context implies unpaid drafting).
    "position_neutral_keys": [
        "intern",
        "apprentice",
        "trainee",
        "学徒",
        "实习",
        "实习生",
        "见习",
        "助手",
        "assistant",
    ],
}

LLM_JUDGE_MODEL = "anthropic/claude-sonnet-4"
LLM_JUDGE_SYSTEM = "你是严格的答题评分员。只输出 JSON，不要多余文字。"


# ═══════════════════════════════════════════════════════════════════════════
# Normalization helpers
# ═══════════════════════════════════════════════════════════════════════════


def _norm(value) -> str:
    """NFKC + lower + collapse whitespace. Keep CJK / Latin letters,
    strip surrounding punct. Hyphens/spaces kept so multi-word aliases
    still match by substring."""
    if value is None:
        return ""
    v = unicodedata.normalize("NFKC", str(value)).lower().strip()
    v = v.replace("–", "-").replace("—", "-")
    v = re.sub(r"\s+", " ", v)
    v = v.strip("\"'`“”‘’,，。；：:（）()、/")
    return v


def _contains_any(text: str, keys: list[str]) -> str | None:
    for k in keys:
        if k and k in text:
            return k
    return None


# ═══════════════════════════════════════════════════════════════════════════
# Per-dimension rule matching
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class DimVerdict:
    score: int  # 0 or 1, or -1 to mark "defer to LLM"
    reason: str
    llm_used: bool = False
    llm_log: dict | None = None


def _rule_city(city_raw: str | None) -> DimVerdict:
    n = _norm(city_raw)
    if not n:
        return DimVerdict(0, "city empty")
    hit = _contains_any(n, BENCHMARK["city_keys"])
    if hit:
        return DimVerdict(1, f"city='{n}' contains '{hit}'")
    return DimVerdict(0, f"city='{n}' not Geneva")


def _rule_building(building_raw: str | None) -> DimVerdict:
    n = _norm(building_raw)
    if not n:
        return DimVerdict(0, "building empty")
    wrong = _contains_any(n, BENCHMARK["building_wrong_keys"])
    correct = _contains_any(n, BENCHMARK["building_correct_keys"])
    if correct:
        return DimVerdict(1, f"building='{n}' contains '{correct}'")
    if wrong:
        return DimVerdict(
            0,
            f"building='{n}' is '{wrong}' (commonly confused but wrong project)",
        )
    # Unknown — let the LLM decide (covers exotic aliases / translations).
    return DimVerdict(-1, f"building='{n}' not in alias list, defer to LLM")


def _rule_position(position_raw: str | None) -> DimVerdict:
    """Position semantics:
    - explicit unpaid + drafting  → 1
    - drafting only (no pay info) → defer
    - unpaid only (no drafting)   → defer
    - neutral role (intern etc.)  → defer
    - none of the above           → 0
    """
    n = _norm(position_raw)
    if not n:
        return DimVerdict(0, "position empty")
    unpaid = _contains_any(n, BENCHMARK["position_unpaid_keys"])
    drafting = _contains_any(n, BENCHMARK["position_drafting_keys"])
    neutral = _contains_any(n, BENCHMARK["position_neutral_keys"])

    if unpaid and drafting:
        return DimVerdict(
            1,
            f"position='{n}' has unpaid marker '{unpaid}' + drafting marker '{drafting}'",
        )
    if drafting:
        return DimVerdict(
            -1,
            f"position='{n}' has drafting '{drafting}' but no explicit unpaid marker — defer to LLM",
        )
    if unpaid:
        return DimVerdict(
            -1,
            f"position='{n}' has unpaid '{unpaid}' but no drafting marker — defer to LLM",
        )
    if neutral:
        return DimVerdict(
            -1,
            f"position='{n}' has neutral role '{neutral}' — defer to LLM",
        )
    return DimVerdict(0, f"position='{n}' has no relevant markers")


# ═══════════════════════════════════════════════════════════════════════════
# LLM judge fallback
# ═══════════════════════════════════════════════════════════════════════════


_JUDGE_PROMPT_TEMPLATE = """题干：{query}

参考答案：
- 城市：日内瓦 (Geneva)
- 建筑 / 项目：Mundaneum（也接受历史上同一项目的别名：Cité Mondiale / World City / Centre Mondial / 世界城市 等）
- 职位：unpaid draftsman / 无薪绘图员（核心是\"无薪\" + \"绘图/制图\"两层语义；
  不接受只是\"实习生\"\"助手\" 等不含绘图职能的说法，也不接受领薪的正式职位）。

请只根据下面给出的模型抽取项判定三个独立维度（每项 0 或 1）：
- A_city      : 模型主张的城市是否实质上等同于日内瓦 / Geneva（含其他语言写法或拼写差异）
- B_building  : 模型主张的建筑 / 项目是否实质上指 Mundaneum 那个项目（含同项目别名 / 翻译）
- C_position  : 模型主张的职位 / 角色是否实质上等同于 \"unpaid draftsman / 无薪绘图员\"
  （绘图职能 + 无薪/义务/志愿性质；含义涵盖即判 1，措辞不必完全一致；
   只说 \"实习生 / 助手\" 而没有绘图职能 → 0；
   只说 \"绘图员\" 但明显是领薪正式员工 → 0）

模型抽取：
- city     : {city!r}
- building : {building!r}
- position : {position!r}

请按以下 JSON 输出：
{{
  "A_city":     0|1,
  "B_building": 0|1,
  "C_position": 0|1,
  "rationale":  "<≤80 字简短说明三项的判定依据>"
}}
"""


def _llm_judge(client, city: str, building: str, position: str) -> dict:
    prompt = _JUDGE_PROMPT_TEMPLATE.format(
        query=QUERY_TEXT,
        city=city or "",
        building=building or "",
        position=position or "",
    )
    raw = call_llm(
        client,
        LLM_JUDGE_MODEL,
        LLM_JUDGE_SYSTEM,
        prompt,
        max_tokens=512,
        temperature=0,
    )
    parsed = _extract_json(raw) or {}
    return {
        "A_city": int(parsed.get("A_city", 0) or 0),
        "B_building": int(parsed.get("B_building", 0) or 0),
        "C_position": int(parsed.get("C_position", 0) or 0),
        "rationale": parsed.get("rationale", ""),
        "raw_output": raw,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Per-model scoring
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class ModelScore:
    model: str
    A: DimVerdict
    B: DimVerdict
    C: DimVerdict
    total: int
    raw_claim: dict
    alternative_candidates: list[dict] = field(default_factory=list)
    llm_judge_log: dict | None = None
    resolution: str = "unknown"
    notes: str = ""


def _score_one_model(
    model_name: str, payload: dict, client
) -> ModelScore:
    entities = payload.get("entities", [])
    if not entities:
        empty = DimVerdict(0, "no extraction entities")
        return ModelScore(
            model=model_name,
            A=empty,
            B=empty,
            C=empty,
            total=0,
            raw_claim={},
            resolution="empty",
            notes="no entities in extraction.json",
        )

    ent = entities[0]
    canonical = (ent.get("canonical") or {}).get("value") or {}
    if not isinstance(canonical, dict):
        canonical = {}

    city = canonical.get("city")
    building = canonical.get("building")
    position = canonical.get("position")
    alternatives = canonical.get("alternative_candidates") or []
    if not isinstance(alternatives, list):
        alternatives = []

    a = _rule_city(city)
    b = _rule_building(building)
    c = _rule_position(position)

    llm_log: dict | None = None
    if any(v.score == -1 for v in (a, b, c)):
        if client is None:
            client = get_client()
        judge = _llm_judge(client, city or "", building or "", position or "")
        llm_log = {
            "asked_for": {
                "A_city": a.score == -1,
                "B_building": b.score == -1,
                "C_position": c.score == -1,
            },
            "judge_result": judge,
        }
        # Only adopt LLM verdict where the rule deferred.
        if a.score == -1:
            a = DimVerdict(
                judge["A_city"],
                f"LLM judge: {judge['rationale']}",
                llm_used=True,
            )
        if b.score == -1:
            b = DimVerdict(
                judge["B_building"],
                f"LLM judge: {judge['rationale']}",
                llm_used=True,
            )
        if c.score == -1:
            c = DimVerdict(
                judge["C_position"],
                f"LLM judge: {judge['rationale']}",
                llm_used=True,
            )

    total = a.score + b.score + c.score
    return ModelScore(
        model=model_name,
        A=a,
        B=b,
        C=c,
        total=total,
        raw_claim={"city": city, "building": building, "position": position},
        alternative_candidates=alternatives,
        llm_judge_log=llm_log,
        resolution=ent.get("resolution", "unknown"),
    )


# ═══════════════════════════════════════════════════════════════════════════
# I/O
# ═══════════════════════════════════════════════════════════════════════════


def _dim_to_dict(d: DimVerdict) -> dict:
    return {"score": d.score, "reason": d.reason, "llm_used": d.llm_used}


def _write_model_score(out_dir: Path, r: ModelScore) -> None:
    path = out_dir / r.model / "score.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "query_id": QUERY_ID,
                "model": r.model,
                "score": r.total,
                "max_score": 3,
                "min_score": 0,
                "total_rate": r.total / 3.0,
                "dimensions": {
                    "A_city": _dim_to_dict(r.A),
                    "B_building": _dim_to_dict(r.B),
                    "C_position": _dim_to_dict(r.C),
                },
                "raw_claim": r.raw_claim,
                "alternative_candidates": r.alternative_candidates,
                "llm_judge_log": r.llm_judge_log,
                "scoring_rule": (
                    "A_city ±1 (Geneva alias), "
                    "B_building ±1 (Mundaneum/Cité Mondiale alias), "
                    "C_position ±1 (unpaid + drafting; LLM judges ambiguous cases)"
                ),
                "benchmark_version": BENCHMARK["version"],
                "extraction_resolution": r.resolution,
                "notes": r.notes,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def _write_scores_json(out_dir: Path, results: list[ModelScore]) -> None:
    ranked = sorted(results, key=lambda r: (-r.total, r.model))
    (out_dir / "scores.json").write_text(
        json.dumps(
            {
                "query_id": QUERY_ID,
                "max_score": 3,
                "min_score": 0,
                "scoring_rule": (
                    "A_city ±1 (Geneva alias), "
                    "B_building ±1 (Mundaneum / Cité Mondiale alias), "
                    "C_position ±1 (unpaid + drafting; LLM judges ambiguous cases)."
                ),
                "benchmark_version": BENCHMARK["version"],
                "results": {
                    r.model: {
                        "total_score": r.total,
                        "total_rate": r.total / 3.0,
                        "A_city": r.A.score,
                        "B_building": r.B.score,
                        "C_position": r.C.score,
                        "llm_judge_used": any(
                            (r.A.llm_used, r.B.llm_used, r.C.llm_used)
                        ),
                        "raw_claim": r.raw_claim,
                        "extraction_resolution": r.resolution,
                    }
                    for r in results
                },
                "ranking": [
                    {
                        "rank": i + 1,
                        "model": r.model,
                        "score": r.total,
                        "A": r.A.score,
                        "B": r.B.score,
                        "C": r.C.score,
                    }
                    for i, r in enumerate(ranked)
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def _write_ranking_report(out_dir: Path, results: list[ModelScore]) -> None:
    ranked = sorted(results, key=lambda r: (-r.total, r.model))
    lines: list[str] = [
        f"# Query {QUERY_ID} Ranking Report",
        "",
        f"> Benchmark v{BENCHMARK['version']} (locked {BENCHMARK['locked_at']})  ",
        "> Scoring: A_city ±1 · B_building ±1 · C_position ±1 — independent.  ",
        "> Reference answer: **Geneva / Mundaneum / unpaid draftsman**.",
        "",
        "| Rank | Model | Total | A city | B building | C position | LLM used |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for i, r in enumerate(ranked, start=1):
        any_llm = any((r.A.llm_used, r.B.llm_used, r.C.llm_used))
        lines.append(
            f"| {i} | {r.model} | {r.total}/3 | {r.A.score} | {r.B.score} | {r.C.score} "
            f"| {'✅' if any_llm else '—'} |"
        )
    lines.append("")
    lines.append("## 各模型主张")
    lines.append("")
    lines.append("| Model | city | building | position |")
    lines.append("|---|---|---|---|")
    for r in ranked:
        rc = r.raw_claim or {}
        lines.append(
            f"| {r.model} "
            f"| {str(rc.get('city') or '—')[:40]} "
            f"| {str(rc.get('building') or '—')[:40]} "
            f"| {str(rc.get('position') or '—')[:60]} |"
        )
    lines.append("")
    lines.append("## 维度判定理由")
    lines.append("")
    for r in ranked:
        lines.append(f"### {r.model}  (total={r.total}/3)")
        lines.append(f"- A_city ({r.A.score}): {r.A.reason}")
        lines.append(f"- B_building ({r.B.score}): {r.B.reason}")
        lines.append(f"- C_position ({r.C.score}): {r.C.reason}")
        lines.append("")
    (out_dir / "ranking_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


# ═══════════════════════════════════════════════════════════════════════════
# Entry
# ═══════════════════════════════════════════════════════════════════════════


def main() -> None:
    global LLM_JUDGE_MODEL
    ap = argparse.ArgumentParser(description=f"Query {QUERY_ID} auto-scorer")
    ap.add_argument("--models", nargs="+", required=True, help="name=path list")
    ap.add_argument("--output-dir", default=str(THIS / "auto_scores"))
    ap.add_argument("--primary-model", default=None)
    ap.add_argument("--secondary-model", default=None)
    ap.add_argument("--parallel-models", nargs="+", default=None)
    ap.add_argument("--analyzer-model", default=None)
    ap.add_argument("--concurrency", type=int, default=None)
    ap.add_argument("--skip-extract", action="store_true")
    ap.add_argument("--judge-model", default=LLM_JUDGE_MODEL)
    args = ap.parse_args()
    LLM_JUDGE_MODEL = args.judge_model

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.skip_extract:
        all_results: dict[str, dict] = {}
        for spec in args.models:
            name = spec.split("=", 1)[0].strip()
            ep = out_dir / name / "extraction.json"
            if not ep.exists():
                sys.exit(f"[ERROR] missing extraction: {ep}")
            all_results[name] = json.loads(ep.read_text(encoding="utf-8"))
    else:
        all_results = run_pipeline(
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

    # Lazy LLM client — only created if any model triggers a deferred dim.
    client = None
    with contextlib.suppress(SystemExit):
        client = get_client()

    results: list[ModelScore] = []
    for model_name, payload in all_results.items():
        r = _score_one_model(model_name, payload, client=client)
        _write_model_score(out_dir, r)
        results.append(r)

    _write_ranking_report(out_dir, results)
    _write_scores_json(out_dir, results)

    print("\n" + "─" * 64)
    print(f"Query {QUERY_ID} scoring done.")
    for r in sorted(results, key=lambda x: (-x.total, x.model)):
        mark = "✅" if r.total == 3 else ("🟡" if r.total > 0 else "❌")
        print(
            f"  {mark} {r.model:28s} {r.total}/3  "
            f"A={r.A.score} B={r.B.score} C={r.C.score}"
        )


if __name__ == "__main__":
    main()
