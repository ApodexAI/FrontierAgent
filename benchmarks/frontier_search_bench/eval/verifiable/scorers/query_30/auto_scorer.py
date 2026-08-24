"""Query 30 — Yuille genealogy auto-scorer.

Reference (locked 2026-05-05; source: Mathematics Genealogy Project):
  Core front-segment (4 links, all models should agree):
    Alan Yuille → Stephen Hawking → Dennis Sciama → Paul Dirac → Ralph Fowler
  Above Fowler — two valid MGP branches:
    - Rutherford branch: Fowler ← Rutherford ← Thomson ← Rayleigh ← Routh ← …
      → 18C: Cotes / R. Smith / Taylor / Whisson / Postlethwaite / T. Jones
    - Hill branch: Fowler ← A.V. Hill ← Fletcher ← Langley ← … ← Boerhaave
      → 18C: van Swieten / Störck / Barth / Beer / Boerhaave

Common error: Rutherford branch with Thomson directly to Routh, skipping
Rayleigh — penalised by `rayleigh_skip_penalty`.

Scoring rule (max 10):
  +1 per core link (4 links, max +4)
  +2 if Fowler-upstream is reached (Rutherford or Hill named)
  +3 if at least one 18th-century canonical figure is mentioned
  +1 bonus if ≥2 different 18th-century figures are mentioned
  -2 if Rutherford branch is taken AND Thomson/Routh both present AND
     Rayleigh absent (the Rayleigh-skip error)
  Final score is clipped to [0, 10] is NOT applied — score can be negative.
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
    "core_chain": [
        {"student": "Alan Yuille", "advisor": "Stephen Hawking"},
        {"student": "Stephen Hawking", "advisor": "Dennis Sciama"},
        {"student": "Dennis Sciama", "advisor": "Paul Dirac"},
        {"student": "Paul Dirac", "advisor": "Ralph Fowler"},
    ],
    "core_aliases": {
        "Alan Yuille": ["Yuille", "艾伦·尤勒"],
        "Stephen Hawking": ["Hawking", "霍金", "斯蒂芬·霍金"],
        "Dennis Sciama": ["Sciama", "夏玛", "丹尼斯·夏玛", "席阿玛"],
        "Paul Dirac": ["Dirac", "狄拉克", "保罗·狄拉克"],
        "Ralph Fowler": ["Fowler", "福勒", "拉尔夫·福勒"],
    },
    "fowler_upstream_names": [
        "Ernest Rutherford", "Rutherford", "卢瑟福",
        "A.V. Hill", "Hill", "希尔",
    ],
    "century_18_figures": [
        {"name": "Gerard van Swieten", "aliases": ["van Swieten", "范斯维滕"]},
        {"name": "Anton von Störck", "aliases": ["Störck", "Stoerck", "von Storck"]},
        {"name": "Joseph Barth", "aliases": ["Barth"]},
        {"name": "Georg Joseph Beer", "aliases": ["G.J. Beer", "Georg Beer"]},
        {"name": "Roger Cotes", "aliases": ["Cotes", "科茨"]},
        {"name": "Robert Smith", "aliases": ["R. Smith"]},
        {"name": "Walter Taylor", "aliases": ["W. Taylor"]},
        {"name": "Stephen Whisson", "aliases": ["Whisson"]},
        {"name": "Thomas Postlethwaite", "aliases": ["Postlethwaite"]},
        {"name": "Thomas Jones", "aliases": ["T. Jones"]},
        {"name": "Herman Boerhaave", "aliases": ["Boerhaave", "布尔哈弗"]},
    ],
    "rayleigh_check": {
        # Trigger Rayleigh-skip penalty when the model uses the Rutherford
        # branch (mentions Rutherford+Thomson+Routh) but does NOT mention
        # Rayleigh — a known wrong shortcut.
        "rutherford_markers": ["Rutherford", "卢瑟福"],
        "thomson_markers": ["Thomson", "汤姆逊", "汤姆生", "汤森"],
        "routh_markers": ["Routh", "劳思", "劳斯"],
        "rayleigh_markers": ["Rayleigh", "瑞利", "瑞利勋爵"],
    },
    "scoring": {
        "per_core_link": 1,
        "fowler_upstream": 2,
        "reach_18th_century": 3,
        "multiple_18th_figures": 1,
        "rayleigh_skip_penalty": -2,
        "max_score": 10,
    },
}


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════


def _norm(s) -> str:
    if s is None:
        return ""
    return re.sub(r"\s+", " ", str(s).strip().lower())


def _contains(text: str, name: str, aliases: list[str] | None = None) -> bool:
    t = _norm(text)
    if not t:
        return False
    if _norm(name) in t:
        return True
    if aliases:
        return any(_norm(a) in t for a in aliases)
    return False


def _haystack_from_claim(claim: dict) -> str:
    """Concatenate all string-ish content in the claim into one searchable
    text. Used for tolerance against extractor variation (chain may be
    list-of-strings or list-of-dicts depending on extractor)."""
    parts: list[str] = []
    chain = claim.get("advisor_chain") or []
    for item in chain:
        if isinstance(item, str):
            parts.append(item)
        elif isinstance(item, dict):
            for v in item.values():
                if v:
                    parts.append(str(v))
    for n in claim.get("century_18_names") or []:
        if isinstance(n, str):
            parts.append(n)
        elif isinstance(n, dict) and n.get("name"):
            parts.append(str(n["name"]))
    fb = claim.get("fowler_branch")
    if fb:
        parts.append(str(fb))
    return " ".join(parts)


# ═══════════════════════════════════════════════════════════════════════════
# Per-dimension scoring
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class DimScore:
    score: int
    reason: str


def score_core_links(claim: dict) -> tuple[DimScore, list[str], list[str]]:
    """+1 per core link present (both endpoints mentioned)."""
    sc = BENCHMARK["scoring"]
    haystack = _haystack_from_claim(claim)
    hit_links: list[str] = []
    missed: list[str] = []
    for link in BENCHMARK["core_chain"]:
        stu = link["student"]
        adv = link["advisor"]
        stu_aliases = BENCHMARK["core_aliases"].get(stu, [])
        adv_aliases = BENCHMARK["core_aliases"].get(adv, [])
        if _contains(haystack, stu, stu_aliases) and _contains(
            haystack, adv, adv_aliases
        ):
            hit_links.append(f"{stu}→{adv}")
        else:
            missed.append(f"{stu}→{adv}")
    pts = len(hit_links) * sc["per_core_link"]
    return (
        DimScore(pts, f"{len(hit_links)}/{len(BENCHMARK['core_chain'])} core links → +{pts}"),
        hit_links,
        missed,
    )


def score_fowler_upstream(claim: dict) -> DimScore:
    """+2 if Rutherford or Hill is mentioned as Fowler's upstream."""
    sc = BENCHMARK["scoring"]
    haystack = _haystack_from_claim(claim)
    branch = (claim.get("fowler_branch") or "").lower()
    if branch in ("rutherford", "hill", "both"):
        return DimScore(sc["fowler_upstream"], f"fowler_branch='{branch}' → +{sc['fowler_upstream']}")
    if any(_contains(haystack, n) for n in BENCHMARK["fowler_upstream_names"]):
        return DimScore(sc["fowler_upstream"], "Rutherford/Hill named → +{}".format(sc["fowler_upstream"]))
    return DimScore(0, "no Fowler upstream (Rutherford/Hill) named")


def score_18th_century(claim: dict) -> tuple[DimScore, list[str]]:
    """+3 if at least one 18C figure mentioned, +1 bonus if ≥2."""
    sc = BENCHMARK["scoring"]
    haystack = _haystack_from_claim(claim)
    matched: list[str] = []
    for fig in BENCHMARK["century_18_figures"]:
        if _contains(haystack, fig["name"], fig.get("aliases", [])):
            matched.append(fig["name"])
    n = len(matched)
    pts = 0
    reasons = []
    if n >= 1:
        pts += sc["reach_18th_century"]
        reasons.append(f"reach 18C ({n} figures: {', '.join(matched[:4])}) → +{sc['reach_18th_century']}")
    else:
        reasons.append("no 18C figures mentioned")
    if n >= 2:
        pts += sc["multiple_18th_figures"]
        reasons.append(f"multiple 18C → +{sc['multiple_18th_figures']}")
    return DimScore(pts, "; ".join(reasons)), matched


def score_rayleigh_skip(claim: dict) -> DimScore:
    """-2 if Rutherford branch used (Rutherford+Thomson+Routh present) but
    Rayleigh absent."""
    sc = BENCHMARK["scoring"]
    rc = BENCHMARK["rayleigh_check"]
    haystack = _haystack_from_claim(claim)
    has_ruth = any(_contains(haystack, m) for m in rc["rutherford_markers"])
    has_thomson = any(_contains(haystack, m) for m in rc["thomson_markers"])
    has_routh = any(_contains(haystack, m) for m in rc["routh_markers"])
    has_rayleigh = any(_contains(haystack, m) for m in rc["rayleigh_markers"])
    if has_ruth and has_thomson and has_routh and not has_rayleigh:
        return DimScore(
            sc["rayleigh_skip_penalty"],
            f"Rutherford path missing Rayleigh (Thomson→Routh shortcut) → {sc['rayleigh_skip_penalty']}",
        )
    return DimScore(0, "no Rayleigh-skip detected")


# ═══════════════════════════════════════════════════════════════════════════
# Per-model aggregator
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class ModelScore:
    model: str
    A: DimScore  # core_links
    B: DimScore  # fowler_upstream
    C: DimScore  # 18th century
    D: DimScore  # rayleigh penalty
    total: int
    hit_links: list[str]
    missed_links: list[str]
    matched_18c: list[str]
    raw_claim: dict
    notes: str = ""


def _score_one(model_name: str, claim: dict | None) -> ModelScore:
    if not claim:
        empty = DimScore(0, "no extraction")
        return ModelScore(
            model=model_name,
            A=empty, B=empty, C=empty, D=empty,
            total=0,
            hit_links=[], missed_links=[],
            matched_18c=[], raw_claim={},
            notes="no extraction (empty / not_mentioned)",
        )
    a, hits, missed = score_core_links(claim)
    b = score_fowler_upstream(claim)
    c, c18 = score_18th_century(claim)
    d = score_rayleigh_skip(claim)
    total = a.score + b.score + c.score + d.score
    return ModelScore(
        model=model_name,
        A=a, B=b, C=c, D=d,
        total=total,
        hit_links=hits, missed_links=missed,
        matched_18c=c18, raw_claim=claim,
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
        if (
            not canonical_value.get("advisor_chain")
            and not canonical_value.get("century_18_names")
            and not canonical_value.get("fowler_branch")
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
                        "A_core_links": _dim_to_dict(r.A),
                        "B_fowler_upstream": _dim_to_dict(r.B),
                        "C_18th_century": _dim_to_dict(r.C),
                        "D_rayleigh_penalty": _dim_to_dict(r.D),
                    },
                    "hit_links": r.hit_links,
                    "missed_links": r.missed_links,
                    "matched_18c_figures": r.matched_18c,
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
                "scoring_rule": (
                    "A core links +1×N (max +4), B Fowler upstream +2, "
                    "C 18C +3 (+1 if ≥2), D Rayleigh-skip -2"
                ),
                "results": {
                    r.model: {
                        "total_score": r.total,
                        "A_core_links": r.A.score,
                        "B_fowler_upstream": r.B.score,
                        "C_18th_century": r.C.score,
                        "D_rayleigh_penalty": r.D.score,
                        "hit_links": r.hit_links,
                        "matched_18c_figures": r.matched_18c,
                        "claim": r.raw_claim,
                    }
                    for r in results
                },
                "ranking": [
                    {"rank": i + 1, "model": r.model, "score": r.total,
                     "A": r.A.score, "B": r.B.score, "C": r.C.score, "D": r.D.score}
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
        "> 评分：A 核心链 +1×N · B Fowler 上游 +2 · C 18 世纪 +3(+1) · D Rayleigh-skip -2 · 满分 10  ",
        "> 标准答案：核心链 Yuille→Hawking→Sciama→Dirac→Fowler ；上游 Rutherford 或 Hill；至少 1 位 18 世纪人物",
        "",
        "| Rank | Model | Total | A links | B Fowler | C 18C | D Rayleigh |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for i, r in enumerate(ranked, 1):
        lines.append(
            f"| {i} | {r.model} | {r.total}/{BENCHMARK['scoring']['max_score']} "
            f"| {r.A.score} | {r.B.score} | {r.C.score} | {r.D.score} |"
        )
    lines.append("")
    lines.append("## 维度判定理由")
    lines.append("")
    for r in ranked:
        lines.append(f"### {r.model}  (total={r.total}/{BENCHMARK['scoring']['max_score']})")
        lines.append(f"- A (core links): **{r.A.score}** — {r.A.reason}")
        if r.missed_links:
            lines.append(f"  · missed: {', '.join(r.missed_links)}")
        lines.append(f"- B (Fowler upstream): **{r.B.score}** — {r.B.reason}")
        lines.append(f"- C (18th century): **{r.C.score}** — {r.C.reason}")
        lines.append(f"- D (Rayleigh-skip): **{r.D.score}** — {r.D.reason}")
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
            f"  {mark} {r.model:30s} {r.total}/{BENCHMARK['scoring']['max_score']}  "
            f"A{r.A.score} B{r.B.score} C{r.C.score} D{r.D.score}"
        )


if __name__ == "__main__":
    main()
