"""Query 07 — IMO Fields Medal country auto-scorer.

Reference answer (locked 2026-05-05):
  Country: Soviet Union (USSR) — accepts: 苏联 / Soviet Union / USSR /
           前苏联 / 苏联/俄罗斯 / Russia/USSR / 蘇聯
  Count:   4
  Names:   Grigory Margulis, Vladimir Drinfeld, Grigori Perelman,
           Stanislav Smirnov

Scoring rule (3 independent dimensions, max 9):
  A country : +3 if matches USSR aliases, else 0
  B count   : +2 if exactly 4, else 0
  C names   : +1 per correct name (max +4)
  Total max : 9
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
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
from pipeline.extraction_pipeline import run_pipeline  # noqa: E402

# ═══════════════════════════════════════════════════════════════════════════
# Benchmark
# ═══════════════════════════════════════════════════════════════════════════

BENCHMARK: dict = {
    "version": "1.0",
    "locked_at": "2026-05-05",
    "country_aliases": [
        "苏联",
        "Soviet Union",
        "USSR",
        "前苏联",
        "苏联/俄罗斯",
        "Russia/USSR",
        "蘇聯",
    ],
    "count": 4,
    "names": [
        {
            "canonical": "Grigory Margulis",
            "aliases": ["Margulis", "马尔古利斯", "格里戈里·马尔古利斯"],
        },
        {
            "canonical": "Vladimir Drinfeld",
            "aliases": ["Drinfeld", "德林费尔德", "弗拉基米尔·德林费尔德"],
        },
        {
            "canonical": "Grigori Perelman",
            "aliases": ["Perelman", "佩雷尔曼", "格里戈里·佩雷尔曼"],
        },
        {
            "canonical": "Stanislav Smirnov",
            "aliases": ["Smirnov", "斯米尔诺夫", "斯坦尼斯拉夫·斯米尔诺夫"],
        },
    ],
    "scoring": {
        "correct_country": 3,
        "correct_count": 2,
        "per_correct_name": 1,
        "max_name_score": 4,
        "max_score": 9,
    },
    "runner_up": {"country": "France", "count": 3},
}


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════


def _norm(s) -> str:
    if s is None:
        return ""
    return str(s).strip().lower()


def _contains_any(text: str, keys: list[str]) -> str | None:
    """Return the first matching key (substring, case-insensitive), else None."""
    t = _norm(text)
    if not t:
        return None
    for k in keys:
        if _norm(k) in t:
            return k
    return None


def _name_match(extracted: str, canonical: str, aliases: list[str]) -> bool:
    e = _norm(extracted)
    if not e:
        return False
    if _norm(canonical) == e or _norm(canonical) in e or e in _norm(canonical):
        return True
    return any(_norm(a) == e or _norm(a) in e or e in _norm(a) for a in aliases)


# ═══════════════════════════════════════════════════════════════════════════
# Per-dimension scoring
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class DimScore:
    score: int
    reason: str


def score_A_country(claim: dict) -> DimScore:
    country = claim.get("country") or ""
    hit = _contains_any(country, BENCHMARK["country_aliases"])
    if hit is not None:
        return DimScore(BENCHMARK["scoring"]["correct_country"],
                        f"country='{country}' matches '{hit}'")
    return DimScore(0, f"country='{country}' not USSR")


def score_B_count(claim: dict) -> DimScore:
    raw = claim.get("count")
    try:
        n = int(raw) if raw is not None else None
    except (ValueError, TypeError):
        n = None
    if n == BENCHMARK["count"]:
        return DimScore(BENCHMARK["scoring"]["correct_count"],
                        f"count={n} matches {BENCHMARK['count']}")
    return DimScore(0, f"count={raw} != {BENCHMARK['count']}")


def score_C_names(claim: dict) -> tuple[DimScore, list[str], list[str]]:
    """Return (DimScore, matched_canonical_names, missed_canonical_names)."""
    names = claim.get("names") or []
    if not isinstance(names, list):
        names = [names]
    matched: set[str] = set()
    extracted_unmatched: list[str] = []

    for ext in names:
        ext_str = str(ext).strip()
        if not ext_str:
            continue
        hit = False
        for gt in BENCHMARK["names"]:
            if gt["canonical"] in matched:
                continue
            if _name_match(ext_str, gt["canonical"], gt["aliases"]):
                matched.add(gt["canonical"])
                hit = True
                break
        if not hit:
            extracted_unmatched.append(ext_str)

    sc = BENCHMARK["scoring"]
    pts = min(len(matched) * sc["per_correct_name"], sc["max_name_score"])
    missed = [g["canonical"] for g in BENCHMARK["names"] if g["canonical"] not in matched]
    return (
        DimScore(
            pts,
            f"matched {len(matched)}/{len(BENCHMARK['names'])} names → +{pts}",
        ),
        sorted(matched),
        missed,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Per-model aggregator
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class ModelScore:
    model: str
    A: DimScore
    B: DimScore
    C: DimScore
    total: int
    matched_names: list[str]
    missed_names: list[str]
    raw_claim: dict
    notes: str = ""


def _score_one(model_name: str, claim: dict | None) -> ModelScore:
    if not claim:
        empty: dict = {}
        a = score_A_country(empty)
        b = score_B_count(empty)
        c, matched, missed = score_C_names(empty)
        return ModelScore(
            model=model_name,
            A=a, B=b, C=c,
            total=a.score + b.score + c.score,
            matched_names=matched,
            missed_names=missed,
            raw_claim={},
            notes="no extraction (empty / not_mentioned)",
        )
    a = score_A_country(claim)
    b = score_B_count(claim)
    c, matched, missed = score_C_names(claim)
    return ModelScore(
        model=model_name,
        A=a, B=b, C=c,
        total=a.score + b.score + c.score,
        matched_names=matched,
        missed_names=missed,
        raw_claim=claim,
    )


def load_raw_claims(out_dir: Path) -> dict[str, dict | None]:
    """Read each model's extraction.json and return a dict of raw claim values."""
    out: dict[str, dict | None] = {}
    for d in Path(out_dir).iterdir():
        if not d.is_dir():
            continue
        ext = d / "extraction.json"
        if not ext.exists():
            continue
        payload = json.loads(ext.read_text(encoding="utf-8"))
        entities = payload.get("entities", [])
        if not entities:
            out[d.name] = None
            continue
        canonical_value = (entities[0].get("canonical") or {}).get("value")
        if not isinstance(canonical_value, dict):
            out[d.name] = None
            continue
        # treat all-null primary fields as no answer
        if (
            not canonical_value.get("country")
            and canonical_value.get("count") in (None, "")
            and not canonical_value.get("names")
        ):
            out[d.name] = None
            continue
        out[d.name] = canonical_value
    return out


# ═══════════════════════════════════════════════════════════════════════════
# Output writers
# ═══════════════════════════════════════════════════════════════════════════


def _dim_to_dict(ds: DimScore) -> dict:
    return {"score": ds.score, "reason": ds.reason}


def write_outputs(out_dir: Path, results: list[ModelScore]) -> None:
    for r in results:
        (out_dir / r.model).mkdir(parents=True, exist_ok=True)
        (out_dir / r.model / "score.json").write_text(
            json.dumps(
                {
                    "query_id": QUERY_ID,
                    "model": r.model,
                    "total_score": r.total,
                    "max_score": BENCHMARK["scoring"]["max_score"],
                    "dimensions": {
                        "A_country": _dim_to_dict(r.A),
                        "B_count": _dim_to_dict(r.B),
                        "C_names": _dim_to_dict(r.C),
                    },
                    "matched_names": r.matched_names,
                    "missed_names": r.missed_names,
                    "raw_claim": r.raw_claim,
                    "notes": r.notes,
                    "benchmark_version": BENCHMARK["version"],
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    ranked = sorted(results, key=lambda r: (-r.total, r.model))
    (out_dir / "scores.json").write_text(
        json.dumps(
            {
                "query_id": QUERY_ID,
                "max_score": BENCHMARK["scoring"]["max_score"],
                "benchmark_version": BENCHMARK["version"],
                "scoring_rule": "A country +3, B count +2, C names +1 each (max +4)",
                "results": {
                    r.model: {
                        "total_score": r.total,
                        "A_country": r.A.score,
                        "B_count": r.B.score,
                        "C_names": r.C.score,
                        "matched_names": r.matched_names,
                        "missed_names": r.missed_names,
                        "claim": r.raw_claim,
                    }
                    for r in results
                },
                "ranking": [
                    {"rank": i + 1, "model": r.model, "score": r.total,
                     "A": r.A.score, "B": r.B.score, "C": r.C.score}
                    for i, r in enumerate(ranked)
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    lines = [
        f"# Query {QUERY_ID} 排名报告",
        "",
        f"> 基准 v{BENCHMARK['version']}（locked {BENCHMARK['locked_at']}）  ",
        "> 评分：A 国家 +3 · B 人数 +2 · C 人名 +1×N（≤4） · 满分 9  ",
        "> 标准答案：**苏联（USSR）— 4 人 — Margulis / Drinfeld / Perelman / Smirnov**",
        "",
        "## 排名",
        "",
        "| Rank | Model | Total | A 国家 | B 人数 | C 人名 |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for i, r in enumerate(ranked, 1):
        lines.append(
            f"| {i} | {r.model} | {r.total}/{BENCHMARK['scoring']['max_score']} "
            f"| {r.A.score} | {r.B.score} | {r.C.score} |"
        )
    lines.append("")
    lines.append("## 各模型主张")
    lines.append("")
    lines.append("| Model | Country | Count | Names |")
    lines.append("|---|---|---:|---|")
    for r in ranked:
        rc = r.raw_claim or {}
        names = rc.get("names") or []
        names_str = ", ".join(str(n) for n in names) if names else "—"
        lines.append(
            f"| {r.model} | {rc.get('country') or '—'} "
            f"| {rc.get('count') if rc.get('count') is not None else '—'} "
            f"| {names_str} |"
        )
    lines.append("")
    lines.append("## 维度判定理由")
    lines.append("")
    for r in ranked:
        lines.append(f"### {r.model}  (total={r.total}/{BENCHMARK['scoring']['max_score']})")
        lines.append(f"- A (country): **{r.A.score}** — {r.A.reason}")
        lines.append(f"- B (count): **{r.B.score}** — {r.B.reason}")
        lines.append(f"- C (names): **{r.C.score}** — {r.C.reason}")
        if r.missed_names:
            lines.append(f"  · missed: {', '.join(r.missed_names)}")
        lines.append("")
    (out_dir / "ranking_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


# ═══════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════


def main() -> None:
    ap = argparse.ArgumentParser(description=f"Query {QUERY_ID} auto-scorer")
    ap.add_argument("--models", nargs="+", required=True, help="name=path list")
    ap.add_argument("--output-dir", default=str(THIS / "auto_scores"))
    ap.add_argument("--primary-model", default=None)
    ap.add_argument("--secondary-model", default=None)
    ap.add_argument("--parallel-models", nargs="+", default=None)
    ap.add_argument("--analyzer-model", default=None)
    ap.add_argument("--concurrency", type=int, default=None)
    ap.add_argument("--skip-extract", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

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

    raw = load_raw_claims(out_dir)
    results = [_score_one(name, claim) for name, claim in raw.items()]
    write_outputs(out_dir, results)

    print("\n" + "─" * 64)
    print(f"Query {QUERY_ID} scoring done.")
    for r in sorted(results, key=lambda x: (-x.total, x.model)):
        mark = "✅" if r.total >= 7 else ("🟡" if r.total > 0 else "❌")
        print(
            f"  {mark} {r.model:30s} {r.total}/{BENCHMARK['scoring']['max_score']}  "
            f"A{r.A.score} B{r.B.score} C{r.C.score}"
        )


if __name__ == "__main__":
    main()
