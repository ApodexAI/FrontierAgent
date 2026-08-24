"""Query 41 — Empire State direction + water auto-scorer.

Reference answer (locked 2026-05-05):
  Part A — 310° (within ±5°) ⇔ NW / Northwest / 西北 / 北偏西50° / N50°W / NWbN
           Partial credit (2 pts): WNW / 西北偏西 / NNW / 西北偏北
  Part B — Hudson River (alias: Hudson / 哈德逊河 / 哈得孙河 / 哈得逊河 / 哈德森河)

Derivation: 0° → +90° → +45° → 135° → -200° → -65° ≡ 295° → +15° → 310° (NW).

Scoring rule (max 10):
  Part A: 5 pts full / 2 pts partial / 0
  Part B: 5 pts full / 0 (East River / Harlem River / others all 0)
"""

from __future__ import annotations

import argparse
import json
import re
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
    "part_a": {
        "correct_degree": 310,
        "degree_tolerance": 5,
        "correct_names": ["NW", "Northwest", "西北", "北偏西50", "N50W", "NWbN"],
        "partial_names": ["WNW", "西北偏西", "NNW", "西北偏北"],
        "points_full": 5,
        "points_partial": 2,
    },
    "part_b": {
        "correct_answer": "Hudson River",
        "aliases": ["Hudson", "哈德逊河", "哈得孙河", "哈得逊河", "哈德森河"],
        "wrong_answers": [
            {"name": "East River", "aliases": ["东河"]},
            {"name": "Harlem River", "aliases": ["哈莱姆河", "哈林河"]},
        ],
        "points_full": 5,
    },
    "max_score": 10,
}


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════


def _norm(s) -> str:
    if s is None:
        return ""
    return re.sub(r"\s+", "", str(s).strip().lower())


def _name_in(text: str, name: str) -> bool:
    t = _norm(text)
    n = _norm(name)
    return bool(n) and n in t


# ═══════════════════════════════════════════════════════════════════════════
# Per-dimension scoring
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class DimScore:
    score: int
    reason: str


def score_A_direction(part_a: dict | None) -> DimScore:
    sc = BENCHMARK["part_a"]
    if not part_a:
        return DimScore(0, "Part A not answered")

    # Try numeric degree first
    degree = part_a.get("degree")
    if degree is not None:
        try:
            d = int(degree) % 360
            target = sc["correct_degree"]
            delta = abs(d - target)
            delta = min(delta, 360 - delta)
            if delta <= sc["degree_tolerance"]:
                return DimScore(
                    sc["points_full"],
                    f"degree={d}° within ±{sc['degree_tolerance']}° of {target}°",
                )
        except (ValueError, TypeError):
            pass

    # Fall back to compass name
    compass = part_a.get("compass_name") or ""
    if compass:
        for nm in sc["correct_names"]:
            if _name_in(compass, nm):
                return DimScore(
                    sc["points_full"],
                    f"compass='{compass}' matches '{nm}'",
                )
        for nm in sc["partial_names"]:
            if _name_in(compass, nm):
                return DimScore(
                    sc["points_partial"],
                    f"compass='{compass}' partial match '{nm}'",
                )

    return DimScore(
        0, f"Part A: degree={degree}, compass='{compass}' (expected ~310°/NW)"
    )


def score_B_water(part_b: dict | None) -> DimScore:
    sc = BENCHMARK["part_b"]
    if not part_b:
        return DimScore(0, "Part B not answered")
    water = part_b.get("water_body") or ""
    if not water:
        return DimScore(0, "Part B: water_body empty")

    # Correct first
    if _name_in(water, sc["correct_answer"]) or any(
        _name_in(water, a) for a in sc["aliases"]
    ):
        return DimScore(
            sc["points_full"], f"water='{water}' matches Hudson River"
        )

    # Wrong (matched but explicitly wrong) — still 0
    for wa in sc["wrong_answers"]:
        if _name_in(water, wa["name"]) or any(
            _name_in(water, a) for a in wa.get("aliases", [])
        ):
            return DimScore(
                0, f"water='{water}' is {wa['name']} (wrong, expected Hudson)"
            )

    return DimScore(0, f"water='{water}' not Hudson River")


# ═══════════════════════════════════════════════════════════════════════════
# Per-model aggregator
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class ModelScore:
    model: str
    A: DimScore
    B: DimScore
    total: int
    raw_claim: dict
    notes: str = ""


def _score_one(model_name: str, claim: dict | None) -> ModelScore:
    if not claim:
        return ModelScore(
            model=model_name,
            A=DimScore(0, "no extraction"),
            B=DimScore(0, "no extraction"),
            total=0,
            raw_claim={},
            notes="no extraction (empty / not_mentioned)",
        )
    a = score_A_direction(claim.get("part_a"))
    b = score_B_water(claim.get("part_b"))
    return ModelScore(
        model=model_name, A=a, B=b, total=a.score + b.score, raw_claim=claim
    )


def load_raw_claims(out_dir: Path) -> dict[str, dict | None]:
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
        pa = canonical_value.get("part_a") or {}
        pb = canonical_value.get("part_b") or {}
        if (
            pa.get("degree") in (None, "")
            and not pa.get("compass_name")
            and not pb.get("water_body")
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
                    "max_score": BENCHMARK["max_score"],
                    "dimensions": {
                        "A_direction": _dim_to_dict(r.A),
                        "B_water": _dim_to_dict(r.B),
                    },
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
                "max_score": BENCHMARK["max_score"],
                "benchmark_version": BENCHMARK["version"],
                "scoring_rule": "A direction: 5 full / 2 partial / 0; B water: 5 full / 0",
                "results": {
                    r.model: {
                        "total_score": r.total,
                        "A_direction": r.A.score,
                        "B_water": r.B.score,
                        "claim": r.raw_claim,
                    }
                    for r in results
                },
                "ranking": [
                    {"rank": i + 1, "model": r.model, "score": r.total,
                     "A": r.A.score, "B": r.B.score}
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
        "> 评分：A 方向 5(full)/2(partial)/0 · B 水域 5/0 · 满分 10  ",
        "> 标准答案：**A: 310° / NW · B: Hudson River**",
        "",
        "| Rank | Model | Total | A direction | B water |",
        "|---:|---|---:|---:|---:|",
    ]
    for i, r in enumerate(ranked, 1):
        lines.append(
            f"| {i} | {r.model} | {r.total}/{BENCHMARK['max_score']} "
            f"| {r.A.score} | {r.B.score} |"
        )
    lines.append("")
    lines.append("## 各模型主张")
    lines.append("")
    lines.append("| Model | Degree | Compass | Water |")
    lines.append("|---|---:|---|---|")
    for r in ranked:
        rc = r.raw_claim or {}
        pa = rc.get("part_a") or {}
        pb = rc.get("part_b") or {}
        lines.append(
            f"| {r.model} "
            f"| {pa.get('degree') if pa.get('degree') is not None else '—'} "
            f"| {pa.get('compass_name') or '—'} "
            f"| {pb.get('water_body') or '—'} |"
        )
    lines.append("")
    lines.append("## 维度判定理由")
    lines.append("")
    for r in ranked:
        lines.append(f"### {r.model}  (total={r.total}/{BENCHMARK['max_score']})")
        lines.append(f"- A (direction): **{r.A.score}** — {r.A.reason}")
        lines.append(f"- B (water): **{r.B.score}** — {r.B.reason}")
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
        mark = "✅" if r.total >= 8 else ("🟡" if r.total > 0 else "❌")
        print(
            f"  {mark} {r.model:30s} {r.total}/{BENCHMARK['max_score']}  "
            f"A{r.A.score} B{r.B.score}"
        )


if __name__ == "__main__":
    main()
