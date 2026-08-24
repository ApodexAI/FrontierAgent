"""
Query 23 — Auto scorer (binary, single-fact)

Question: slime 在 2025 年出现的一个重大 megatron 问题，对 moe 训练有影响，是哪个

Scoring rule:
  Single binary dimension —
    1 if extracted issue_number ∈ {958}
       AND repo / title / link / supporting_span 任一字段含 'slime'（防止误命中其它仓库 #958）
    0 otherwise.
  Auxiliary fields (title / link / description / affects_moe) recorded for
  audit only, not included in total.

Flow:
  1) Run the shared extraction pipeline (primary + secondary; phase-4 on
     disagreements). Canonical extraction lands in
     auto_scores/{model}/extraction.json.
  2) Load each model's canonical value, judge issue number + slime context,
     write auto_scores/{model}/score.json and a top-level ranking_report.md.
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
from pipeline.extraction_pipeline import run_pipeline  # noqa: E402

# ═══════════════════════════════════════════════════════════════════════════
# Ground truth
# ═══════════════════════════════════════════════════════════════════════════

BENCHMARK: dict = {
    "version": "1.0",
    "locked_at": "2026-05-05",
    "accepted_issue_numbers": {958},
    "required_repo_marker": "slime",  # case-insensitive substring check
    "reference": {
        "repo": "THUDM/slime",
        "issue_number": 958,
        "title": (
            "[bug] while train moe model with mcore, it seems only return "
            "moe optimizer in setup_model_and_optimizer"
        ),
        "link": "https://github.com/THUDM/slime/issues/958",
        "affects_moe": True,
    },
}


# ═══════════════════════════════════════════════════════════════════════════
# Issue-number parsing
# ═══════════════════════════════════════════════════════════════════════════

_NUM_RE = re.compile(r"\b\d{1,6}\b")
_HASH_NUM_RE = re.compile(r"#(\d{1,6})")


def _coerce_issue_numbers(v) -> list[int]:
    """Extract candidate issue/PR numbers from a value. A value may be:
    - int → single number
    - str → all 1-6 digit numbers found (prefer #-prefixed)
    - list → flatten
    """
    if v is None:
        return []
    if isinstance(v, bool):
        return []
    if isinstance(v, int):
        return [v]
    if isinstance(v, float):
        return [int(v)]
    if isinstance(v, (list, tuple)):
        out: list[int] = []
        for item in v:
            for n in _coerce_issue_numbers(item):
                if n not in out:
                    out.append(n)
        return out
    s = str(v)
    out: list[int] = []
    # Prefer #-prefixed numbers first (more unambiguous)
    for m in _HASH_NUM_RE.finditer(s):
        n = int(m.group(1))
        if n not in out:
            out.append(n)
    for m in _NUM_RE.finditer(s):
        n = int(m.group())
        if n not in out:
            out.append(n)
    return out


def _collect_candidate_issue_numbers(canonical_value) -> list[int]:
    """Pick issue numbers from canonical value + alternative_candidates list."""
    nums: list[int] = []
    if not isinstance(canonical_value, dict):
        for n in _coerce_issue_numbers(canonical_value):
            if n not in nums:
                nums.append(n)
        return nums

    for n in _coerce_issue_numbers(canonical_value.get("issue_number")):
        if n not in nums:
            nums.append(n)

    for alt in canonical_value.get("alternative_candidates") or []:
        if isinstance(alt, dict):
            for an in _coerce_issue_numbers(alt.get("issue_number")):
                if an not in nums:
                    nums.append(an)
    return nums


def _has_slime_marker(canonical_value, supporting_span: str | None) -> bool:
    """True if any field (repo, title, link, supporting_span, plus alt entries)
    contains 'slime' substring case-insensitive."""
    marker = BENCHMARK["required_repo_marker"].lower()
    fields_to_check: list[str] = []
    if supporting_span:
        fields_to_check.append(str(supporting_span))
    if isinstance(canonical_value, dict):
        for k in ("repo", "title", "link", "description"):
            v = canonical_value.get(k)
            if v:
                fields_to_check.append(str(v))
        for alt in canonical_value.get("alternative_candidates") or []:
            if isinstance(alt, dict):
                for k in ("repo", "title", "link", "description"):
                    v = alt.get(k)
                    if v:
                        fields_to_check.append(str(v))
    elif canonical_value:
        fields_to_check.append(str(canonical_value))
    return any(marker in s.lower() for s in fields_to_check)


# ═══════════════════════════════════════════════════════════════════════════
# Per-model scoring
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class ScoreResult:
    model: str
    score: int
    max_score: int
    claimed_issue_numbers: list[int]
    matched_issue_number: int | None
    slime_marker_present: bool
    resolution: str
    reference_snapshot: dict = field(default_factory=dict)


def _score_model(model_name: str, extraction_payload: dict) -> ScoreResult:
    entities = extraction_payload.get("entities", [])
    if not entities:
        return ScoreResult(
            model=model_name,
            score=0,
            max_score=1,
            claimed_issue_numbers=[],
            matched_issue_number=None,
            slime_marker_present=False,
            resolution="empty_extraction",
            reference_snapshot={},
        )
    ent = entities[0]
    canonical = (ent.get("canonical") or {}).get("value")
    span = (ent.get("canonical") or {}).get("supporting_span")
    nums = _collect_candidate_issue_numbers(canonical)
    has_slime = _has_slime_marker(canonical, span)

    matched: int | None = None
    for n in nums:
        if n in BENCHMARK["accepted_issue_numbers"]:
            matched = n
            break

    score = 1 if (matched is not None and has_slime) else 0

    ref_snap: dict = {}
    if isinstance(canonical, dict):
        for k in ("repo", "issue_number", "title", "link", "affects_moe", "description"):
            ref_snap[k] = canonical.get(k)

    return ScoreResult(
        model=model_name,
        score=score,
        max_score=1,
        claimed_issue_numbers=nums,
        matched_issue_number=matched,
        slime_marker_present=has_slime,
        resolution=ent.get("resolution", "unknown"),
        reference_snapshot=ref_snap,
    )


# ═══════════════════════════════════════════════════════════════════════════
# I/O
# ═══════════════════════════════════════════════════════════════════════════


def _write_model_score(output_dir: Path, r: ScoreResult) -> None:
    path = output_dir / r.model / "score.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "query_id": QUERY_ID,
                "model": r.model,
                "score": r.score,
                "max_score": r.max_score,
                "claimed_issue_numbers": r.claimed_issue_numbers,
                "matched_issue_number": r.matched_issue_number,
                "slime_marker_present": r.slime_marker_present,
                "extraction_resolution": r.resolution,
                "reference_snapshot": r.reference_snapshot,
                "benchmark_version": BENCHMARK["version"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def _write_ranking_report(output_dir: Path, results: list[ScoreResult]) -> None:
    lines: list[str] = []
    lines.append("# Query 23 Ranking Report (v1)")
    lines.append("")
    lines.append(
        f"> Benchmark version: {BENCHMARK['version']} (locked {BENCHMARK['locked_at']})"
    )
    lines.append(
        f"> Accepted issue numbers: {sorted(BENCHMARK['accepted_issue_numbers'])} "
        f"(must co-occur with '{BENCHMARK['required_repo_marker']}' marker)"
    )
    lines.append("")
    lines.append("## 排名")
    lines.append("")
    lines.append(
        "| Rank | Model | Score | Claimed Issue(s) | Matched | slime? | Resolution |"
    )
    lines.append("|---:|---|---:|---|---|:---:|---|")
    ranked = sorted(results, key=lambda r: (-r.score, r.model))
    for i, r in enumerate(ranked, start=1):
        nums = ", ".join(f"#{n}" for n in r.claimed_issue_numbers) or "—"
        matched = f"#{r.matched_issue_number}" if r.matched_issue_number is not None else "—"
        marker = "✅" if r.slime_marker_present else "❌"
        lines.append(
            f"| {i} | {r.model} | {r.score}/{r.max_score} | "
            f"{nums} | {matched} | {marker} | {r.resolution} |"
        )
    lines.append("")
    lines.append("## 参考项（不计分）")
    lines.append("")
    lines.append("| Model | Repo | Issue# | Title | Link | affects_moe |")
    lines.append("|---|---|---:|---|---|:---:|")
    for r in ranked:
        s = r.reference_snapshot
        title = (s.get("title") or "—")
        if len(title) > 80:
            title = title[:77] + "…"
        am = s.get("affects_moe")
        am_str = "—" if am is None else ("✅" if am else "❌")
        lines.append(
            f"| {r.model} | {s.get('repo') or '—'} | "
            f"{s.get('issue_number') if s.get('issue_number') is not None else '—'} | "
            f"{title} | {s.get('link') or '—'} | {am_str} |"
        )
    (output_dir / "ranking_report.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


# ═══════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════


def main() -> None:
    ap = argparse.ArgumentParser(description="Query 23 auto-scorer (v1)")
    ap.add_argument(
        "--models",
        nargs="+",
        required=True,
        help="name=path list of model answer JSON files.",
    )
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--primary-model", default=None)
    ap.add_argument("--secondary-model", default=None)
    ap.add_argument("--parallel-models", nargs="+", default=None)
    ap.add_argument("--analyzer-model", default=None)
    ap.add_argument("--concurrency", type=int, default=None)
    ap.add_argument(
        "--skip-extract",
        action="store_true",
        help="Skip extraction; reuse existing extraction.json.",
    )
    args = ap.parse_args()

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

    results: list[ScoreResult] = []
    for model_name, payload in all_results.items():
        r = _score_model(model_name, payload)
        _write_model_score(out_dir, r)
        results.append(r)

    _write_ranking_report(out_dir, results)
    print("\n" + "─" * 60)
    print("Query 23 scoring done.")
    for r in sorted(results, key=lambda x: (-x.score, x.model)):
        mark = "✅" if r.score else "❌"
        nums = ", ".join(f"#{n}" for n in r.claimed_issue_numbers) or "—"
        print(
            f"  {mark} {r.model}: {r.score}/{r.max_score}  "
            f"matched={r.matched_issue_number}  slime={'Y' if r.slime_marker_present else 'N'}  "
            f"all={nums}"
        )


if __name__ == "__main__":
    main()
