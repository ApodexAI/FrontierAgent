"""Query 04 — Global INES level 4+ nuclear accidents auto-scorer.

Reference answer:
  8 standard accidents in `id4_standard_database.json`
  (6 strict + 2 lenient; aliases inline + MANUAL_ALIASES below).

Scoring rule (per matched accident, max 8 dims):
  D1 name              : matched against DB name/alias → +1
  D2 ines_level        : parsed INES level == DB level (or >= for `>=` rule)
  D3 location          : LLM judge vs DB location
  D4 year              : extracted year == DB date year
  D5 reactor_type      : LLM judge vs DB reactor_type
  D6 direct_cause      : LLM judge vs DB direct_cause
  D7 policy_change     : LLM judge vs DB policy_change
  D8 power_share_change: LLM judge vs DB power_share_change (with caution rule)
Extra (unmatched) accident: -5 each.

Pipeline (Stage 1 extract via pipeline → Stage 2 name match →
Stage 3 LLM judge per matched pair → Stage 4 score).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
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
# Benchmark / GT
# ═══════════════════════════════════════════════════════════════════════════

DEFAULT_DB_PATH = THIS / "id4_standard_database.json"
EXTRA_EVENT_PENALTY = -5
PER_ITEM_MAX_DIMS = 8

# Manual cross-language aliases (in addition to those in the DB).
MANUAL_ALIASES: dict[str, list[str]] = {
    "Chernobyl Unit 4": ["切尔诺贝利"],
    "Three Mile Island Unit 2": [
        "三哩岛2号机",
        "three mile island 2",
        "three mile island unit 2",
    ],
    "Fukushima Daiichi": ["fukushima daiichi nuclear power station", "福岛第一"],
    "Saint-Laurent A1": ["saint laurent a1", "saint-laurent a1"],
    "Saint-Laurent A2": ["saint laurent a2", "saint-laurent a2"],
    "Bohunice A-1": ["bohunice a1", "bohunice a-1"],
    "Fermi-1": ["fermi 1", "enrico fermi unit 1"],
    "Lucens": ["lucens reactor", "lucens nuclear plant", "lucens nuclear power plant"],
}

LLM_JUDGE_MODEL = "anthropic/claude-sonnet-4"
LLM_JUDGE_SYSTEM = "你是严格的核事故评分器。只输出 JSON，不要多余文字。"


# ═══════════════════════════════════════════════════════════════════════════
# Normalization / parsing
# ═══════════════════════════════════════════════════════════════════════════


def _norm(s) -> str:
    if s is None:
        return ""
    v = str(s).lower().strip()
    v = v.replace("–", "-").replace("—", "-")
    v = re.sub(r"[\s_]+", "", v)
    v = re.sub(r"[\"'`“”‘’,，。；：:（）()、/]+", "", v)
    return v


def extract_year(value: str | None) -> str | None:
    if not value:
        return None
    m = re.search(r"(19|20)\d{2}", value)
    return m.group(0) if m else None


def extract_ines_level(value: str | None) -> int | None:
    if not value:
        return None
    m = re.search(r"([0-7])", value)
    return int(m.group(1)) if m else None


# ═══════════════════════════════════════════════════════════════════════════
# Name index + matching
# ═══════════════════════════════════════════════════════════════════════════


def build_name_index(db: dict) -> tuple[dict[str, dict], dict[str, str]]:
    items_by_name: dict[str, dict] = {}
    alias_to_name: dict[str, str] = {}
    for item in db["items"]:
        name = item["name"]
        items_by_name[name] = item
        aliases = [name, *item.get("aliases", []), *MANUAL_ALIASES.get(name, [])]
        for a in aliases:
            alias_to_name[_norm(a)] = name
    return items_by_name, alias_to_name


def match_item_name(name: str | None, alias_to_name: dict[str, str]) -> str | None:
    if not name:
        return None
    n = _norm(name)
    if not n:
        return None
    if n in alias_to_name:
        return alias_to_name[n]
    # Substring match (guard against very short queries).
    if len(n) >= 4:
        for alias, canonical in alias_to_name.items():
            if alias and (alias in n or n in alias):
                return canonical
    return None


# ═══════════════════════════════════════════════════════════════════════════
# Rule-based dimension scoring
# ═══════════════════════════════════════════════════════════════════════════


def score_level(extracted_level: str | None, standard_level) -> int:
    n = extract_ines_level(extracted_level)
    if n is None:
        return 0
    if isinstance(standard_level, int):
        return int(n == standard_level)
    text = str(standard_level)
    if ">=" in text:
        thresh = extract_ines_level(text)
        return int(thresh is not None and n >= thresh)
    s = extract_ines_level(text)
    return int(s is not None and n == s)


def score_year(extracted_time: str | None, standard_date: str | None) -> int:
    ey, sy = extract_year(extracted_time), extract_year(standard_date)
    return int(bool(ey and sy and ey == sy))


# ═══════════════════════════════════════════════════════════════════════════
# LLM judge (5 semantic dimensions)
# ═══════════════════════════════════════════════════════════════════════════


_JUDGE_PROMPT_TEMPLATE = """题干：{query}

请只根据下面给出的标准数据库条目与抽取项做比较，不要使用外部知识。

## 标准数据库条目
- 名称: {std_name}
- 地点: {std_location}
- 反应堆型号: {std_reactor_type}
- 直接原因: {std_direct_cause}
- 事故后该国核电政策具体变化: {std_policy_change}
- 事故前后5年内核电发电量占比变化数据: {std_power_share_change}

## 抽取项
- 名称: {ex_name}
- 地点: {ex_location}
- 反应堆型号: {ex_reactor_type}
- 直接原因: {ex_direct_cause}
- 事故后该国核电政策具体变化: {ex_policy_change}
- 事故前后5年内核电发电量占比变化数据: {ex_power_share_change}

请对以下 5 个语义维度逐一判定（每项 0 或 1）：
- location：地点是否基本正确
- reactor_type：反应堆型号 / 堆型是否基本正确
- direct_cause：直接原因是否抓住数据库中的关键机制
- policy_change：事故后该国核电政策变化是否与数据库一致
- power_share_change：事故前后5年核电占比变化是否与数据库一致

特别规则：
- 若标准条目的 power_share_change 明确说 "只能做谨慎判断 / 未完全核定 / 不要接受精确数值"，
  那么只有 "谨慎且方向一致" 的回答才算正确；没有来源支持的精确百分比不算正确。
- 抽取项的说法和标准条目在评估上可接受即判 1，不要求措辞完全一致。
- 抽取项过于笼统、未达到可核验程度，应判 0。

输出 JSON：
{{
  "location": 0|1,
  "reactor_type": 0|1,
  "direct_cause": 0|1,
  "policy_change": 0|1,
  "power_share_change": 0|1,
  "notes": "<≤60 字简短说明>"
}}
"""


def _judge_pair_via_llm(client, query: str, std: dict, extracted: dict) -> dict:
    prompt = _JUDGE_PROMPT_TEMPLATE.format(
        query=query,
        std_name=std.get("name", ""),
        std_location=std.get("location", ""),
        std_reactor_type=std.get("reactor_type", ""),
        std_direct_cause=std.get("direct_cause", ""),
        std_policy_change=std.get("policy_change", ""),
        std_power_share_change=std.get("power_share_change", ""),
        ex_name=extracted.get("事故名称", ""),
        ex_location=extracted.get("事故地点", ""),
        ex_reactor_type=extracted.get("反应堆型号", ""),
        ex_direct_cause=extracted.get("直接原因", ""),
        ex_policy_change=extracted.get("事故后该国核电政策具体变化", ""),
        ex_power_share_change=extracted.get(
            "事故前后5年内核电发电量占比变化数据", ""
        ),
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
        "location": int(parsed.get("location", 0) or 0),
        "reactor_type": int(parsed.get("reactor_type", 0) or 0),
        "direct_cause": int(parsed.get("direct_cause", 0) or 0),
        "policy_change": int(parsed.get("policy_change", 0) or 0),
        "power_share_change": int(parsed.get("power_share_change", 0) or 0),
        "notes": parsed.get("notes", ""),
        "raw_output": raw,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Per-model scoring
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class MatchedPair:
    matched_standard_name: str
    standard_item: dict
    extracted_item: dict
    dimensions: dict
    score: int
    accept_level: str
    notes: str = ""


@dataclass
class ModelScore:
    model: str
    matched_items: list[MatchedPair] = field(default_factory=list)
    duplicates_ignored: list[MatchedPair] = field(default_factory=list)
    extra_items: list[dict] = field(default_factory=list)
    positive_score: int = 0
    penalty_score: int = 0
    total_score: int = 0
    resolution: str = "unknown"


def _score_model(
    model_name: str,
    payload: dict,
    db: dict,
    items_by_name: dict[str, dict],
    alias_to_name: dict[str, str],
    client,
) -> ModelScore:
    entities = payload.get("entities", [])
    if not entities:
        return ModelScore(model=model_name, resolution="empty")
    ent = entities[0]
    canonical = (ent.get("canonical") or {}).get("value") or {}
    items = canonical.get("items") if isinstance(canonical, dict) else None
    if not isinstance(items, list):
        items = []

    scored_pairs: list[MatchedPair] = []
    extra_items: list[dict] = []

    for idx, ex in enumerate(items):
        if not isinstance(ex, dict):
            continue
        canonical_name = match_item_name(ex.get("事故名称"), alias_to_name)
        if not canonical_name:
            extra_items.append({"item_index": idx, "extracted_item": ex})
            continue
        std = items_by_name[canonical_name]
        llm = _judge_pair_via_llm(client, QUERY_TEXT, std, ex)
        dims = {
            "name": 1,
            "level": score_level(ex.get("事故等级"), std.get("ines_level")),
            "location": llm["location"],
            "year": score_year(ex.get("事故时间"), std.get("date")),
            "reactor_type": llm["reactor_type"],
            "direct_cause": llm["direct_cause"],
            "policy_change": llm["policy_change"],
            "power_share_change": llm["power_share_change"],
        }
        scored_pairs.append(
            MatchedPair(
                matched_standard_name=canonical_name,
                standard_item=std,
                extracted_item=ex,
                dimensions=dims,
                score=sum(dims.values()),
                accept_level=std.get("accept_level", "unknown"),
                notes=llm.get("notes", ""),
            )
        )

    # Keep best-scoring extracted pair per standard accident; the rest go to duplicates.
    best_by_std: dict[str, MatchedPair] = {}
    duplicates: list[MatchedPair] = []
    for p in scored_pairs:
        cur = best_by_std.get(p.matched_standard_name)
        if cur is None or p.score > cur.score:
            if cur is not None:
                duplicates.append(cur)
            best_by_std[p.matched_standard_name] = p
        else:
            duplicates.append(p)

    matched = sorted(
        best_by_std.values(), key=lambda p: (-p.score, p.matched_standard_name)
    )
    positive = sum(p.score for p in matched)
    penalty = EXTRA_EVENT_PENALTY * len(extra_items)

    return ModelScore(
        model=model_name,
        matched_items=matched,
        duplicates_ignored=duplicates,
        extra_items=extra_items,
        positive_score=positive,
        penalty_score=penalty,
        total_score=positive + penalty,
        resolution=ent.get("resolution", "unknown"),
    )


# ═══════════════════════════════════════════════════════════════════════════
# I/O
# ═══════════════════════════════════════════════════════════════════════════


NORMALIZATION_REFERENCE_MAX = PER_ITEM_MAX_DIMS  # 8 dims per matched item


def _pair_to_dict(p: MatchedPair) -> dict:
    return {
        "matched_standard_name": p.matched_standard_name,
        "accept_level": p.accept_level,
        "score": p.score,
        "dimensions": p.dimensions,
        "notes": p.notes,
        "standard_item": p.standard_item,
        "extracted_item": p.extracted_item,
    }


def _write_model_score(out_dir: Path, r: ModelScore, db_summary: dict) -> None:
    path = out_dir / r.model / "score.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    # 跨题归一化时把上限拍到单个 8 分项；不是本题真实上限。
    max_for_norm = NORMALIZATION_REFERENCE_MAX * max(1, db_summary.get("strict_count", 6))
    total_rate = max(0.0, min(r.total_score / max_for_norm, 1.0))
    path.write_text(
        json.dumps(
            {
                "query_id": QUERY_ID,
                "model": r.model,
                "score": r.total_score,
                "max_score": max_for_norm,
                "total_rate": total_rate,
                "positive_score": r.positive_score,
                "penalty_score": r.penalty_score,
                "scoring_rule": {
                    "per_matched_item_max_dims": PER_ITEM_MAX_DIMS,
                    "extra_event_penalty": EXTRA_EVENT_PENALTY,
                    "dim_breakdown": (
                        "name + level + year by rule; "
                        "location + reactor_type + direct_cause + policy_change + "
                        "power_share_change by LLM judge."
                    ),
                },
                "database_summary": db_summary,
                "matched_items": [_pair_to_dict(p) for p in r.matched_items],
                "duplicates_ignored": [_pair_to_dict(p) for p in r.duplicates_ignored],
                "extra_items": r.extra_items,
                "extraction_resolution": r.resolution,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def _write_scores_json(
    out_dir: Path, results: list[ModelScore], db_summary: dict
) -> None:
    ranked = sorted(results, key=lambda r: (-r.total_score, r.model))
    max_for_norm = NORMALIZATION_REFERENCE_MAX * max(
        1, db_summary.get("strict_count", 6)
    )
    (out_dir / "scores.json").write_text(
        json.dumps(
            {
                "query_id": QUERY_ID,
                "max_score": max_for_norm,
                "min_score": None,
                "scoring_rule": (
                    "每 matched 标准事件 ≤ 8 分（name/level/year 规则评，"
                    "location/reactor_type/direct_cause/policy_change/power_share_change "
                    f"LLM 评）；每 extra 事件 {EXTRA_EVENT_PENALTY}。"
                ),
                "database_summary": db_summary,
                "results": {
                    r.model: {
                        "total_score": r.total_score,
                        "positive_score": r.positive_score,
                        "penalty_score": r.penalty_score,
                        "matched_count": len(r.matched_items),
                        "extra_count": len(r.extra_items),
                        "total_rate": max(
                            0.0, min(r.total_score / max_for_norm, 1.0)
                        ),
                        "extraction_resolution": r.resolution,
                    }
                    for r in results
                },
                "ranking": [
                    {
                        "rank": i + 1,
                        "model": r.model,
                        "score": r.total_score,
                        "matched": len(r.matched_items),
                        "extras": len(r.extra_items),
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
    ranked = sorted(results, key=lambda r: (-r.total_score, r.model))
    lines: list[str] = [
        f"# Query {QUERY_ID} Ranking Report",
        "",
        "> Scoring: each matched standard accident → up to 8 dims (name/level/location/"
        "year/reactor_type/direct_cause/policy_change/power_share_change);"
        f" each extra non-DB accident → {EXTRA_EVENT_PENALTY}.",
        "",
        "| Rank | Model | Total | Positive | Penalty | Matched | Extra |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for i, r in enumerate(ranked, start=1):
        lines.append(
            f"| {i} | {r.model} | {r.total_score:+d} | {r.positive_score} | "
            f"{r.penalty_score:+d} | {len(r.matched_items)} | {len(r.extra_items)} |"
        )
    lines.append("")
    lines.append("## 每模型 matched 明细")
    lines.append("")
    for r in ranked:
        lines.append(f"### {r.model} (total={r.total_score:+d})")
        if not r.matched_items:
            lines.append("- （无 matched）")
        for p in r.matched_items:
            dims = p.dimensions
            lines.append(
                f"- **{p.matched_standard_name}** [{p.accept_level}] score={p.score}/8 "
                f"(name={dims['name']} level={dims['level']} loc={dims['location']} "
                f"year={dims['year']} reactor={dims['reactor_type']} "
                f"cause={dims['direct_cause']} policy={dims['policy_change']} "
                f"share={dims['power_share_change']})"
            )
        if r.extra_items:
            extras = [
                str(e["extracted_item"].get("事故名称", "?"))[:40] for e in r.extra_items
            ]
            lines.append(f"- extras ({len(extras)}): {', '.join(extras)}")
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
    ap.add_argument("--database", default=str(DEFAULT_DB_PATH))
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

    db = json.loads(Path(args.database).read_text(encoding="utf-8"))
    items_by_name, alias_to_name = build_name_index(db)
    db_summary = {
        "strict_count": db.get("strict_count"),
        "lenient_total_count": db.get("lenient_total_count"),
        "active_db_size": len(items_by_name),
    }

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

    client = get_client()
    results: list[ModelScore] = []
    for model_name, payload in all_results.items():
        r = _score_model(model_name, payload, db, items_by_name, alias_to_name, client)
        _write_model_score(out_dir, r, db_summary)
        results.append(r)

    _write_ranking_report(out_dir, results)
    _write_scores_json(out_dir, results, db_summary)

    print("\n" + "─" * 64)
    print(f"Query {QUERY_ID} scoring done.")
    for r in sorted(results, key=lambda x: (-x.total_score, x.model)):
        print(
            f"  {r.model:28s} total={r.total_score:+d}  "
            f"matched={len(r.matched_items)}  extras={len(r.extra_items)}  "
            f"(+{r.positive_score} {r.penalty_score:+d})"
        )


if __name__ == "__main__":
    main()
