"""Query 29 — FlashAttention CUDA kernel origin auto-scorer.

Reference answer (ground truth inline):
  - 论文标题             : CudaDMA: Optimizing GPU Memory Bandwidth via Warp Specialization
  - 技巧名称             : warp specialization
  - 首次应用版本         : FlashAttention-3
  - 首次应用 kernel 文件 : hopper/mainloop_fwd_sm90_tma_gmma_ws.hpp

Scoring rule (4 independent dimensions, 0/1 each, max 4, min 0, no penalty):
  A_paper          : value contains both 'cudadma' AND 'warpspecialization'
                     (compact-normalized)
  B_technique      : value contains 'warpspecialization' (compact-normalized)
  C_version        : value contains any of 'flashattention3' / 'flashattentionv3' / 'fa3'
                     (compact-normalized)
  D_kernel_file    : value contains 'mainloop_fwd_sm90_tma_gmma_ws.hpp'
                     (with or without 'hopper/' prefix)

Matching is fully deterministic — no LLM judge required.
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
# Benchmark (inline GT)
# ═══════════════════════════════════════════════════════════════════════════

BENCHMARK: dict = {
    "version": "1.0",
    "locked_at": "2026-05-08",
    "standard_answer": {
        "paper_title": "CudaDMA: Optimizing GPU Memory Bandwidth via Warp Specialization",
        "technique_name": "warp specialization",
        "first_applied_version": "FlashAttention-3",
        "first_applied_kernel_file": "hopper/mainloop_fwd_sm90_tma_gmma_ws.hpp",
    },
    "paper_compact_keys": ["cudadma", "warpspecialization"],  # all required
    "technique_compact_keys": ["warpspecialization"],
    "version_compact_keys": [
        "flashattention3",
        "flashattentionv3",
        "fa3",
    ],
    "kernel_file_substrings": [
        "hopper/mainloop_fwd_sm90_tma_gmma_ws.hpp",
        "mainloop_fwd_sm90_tma_gmma_ws.hpp",
    ],
}


# ═══════════════════════════════════════════════════════════════════════════
# Normalization
# ═══════════════════════════════════════════════════════════════════════════


def _norm(value) -> str:
    """lower + unify hyphens + collapse whitespace."""
    if value is None:
        return ""
    v = str(value).strip().lower()
    v = v.replace("‑", "-").replace("–", "-").replace("—", "-")
    v = re.sub(r"\s+", " ", v)
    return v


def _compact(value) -> str:
    """_norm + drop punctuation, hyphens, spaces (for tolerant token search)."""
    v = _norm(value)
    v = re.sub(r"[`\"'“”‘’（）()，。；：:]", "", v)
    v = v.replace("-", "")
    v = v.replace(" ", "")
    return v


# ═══════════════════════════════════════════════════════════════════════════
# Per-dimension matching
# ═══════════════════════════════════════════════════════════════════════════


def _check_paper(value) -> tuple[int, str]:
    c = _compact(value)
    if not c:
        return (0, "paper empty")
    missing = [k for k in BENCHMARK["paper_compact_keys"] if k not in c]
    if not missing:
        return (1, "compact form contains both 'cudadma' and 'warpspecialization'")
    return (0, f"compact form missing {missing}")


def _check_technique(value) -> tuple[int, str]:
    c = _compact(value)
    if not c:
        return (0, "technique empty")
    if any(k in c for k in BENCHMARK["technique_compact_keys"]):
        return (1, "contains 'warpspecialization'")
    return (0, f"compact='{c}' does not contain 'warpspecialization'")


def _check_version(value) -> tuple[int, str]:
    c = _compact(value)
    if not c:
        return (0, "version empty")
    hit = next((k for k in BENCHMARK["version_compact_keys"] if k in c), None)
    if hit:
        return (1, f"compact contains '{hit}'")
    return (0, f"compact='{c}' does not contain any of fa3 / flashattention3 / flashattentionv3")


def _check_kernel_file(value) -> tuple[int, str]:
    n = _norm(value)
    if not n:
        return (0, "kernel_file empty")
    hit = next((s for s in BENCHMARK["kernel_file_substrings"] if s in n), None)
    if hit:
        return (1, f"normalized form contains '{hit}'")
    return (0, f"normalized='{n}' does not contain mainloop_fwd_sm90_tma_gmma_ws.hpp")


# ═══════════════════════════════════════════════════════════════════════════
# Per-model scoring
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class DimScore:
    score: int
    reason: str


@dataclass
class ModelScore:
    model: str
    A: DimScore
    B: DimScore
    C: DimScore
    D: DimScore
    total: int
    raw_claim: dict
    alternative_candidates: list[dict] = field(default_factory=list)
    resolution: str = "unknown"
    notes: str = ""


def _score_one_model(model_name: str, payload: dict) -> ModelScore:
    entities = payload.get("entities", [])
    empty_dim = DimScore(0, "no extraction entities")
    if not entities:
        return ModelScore(
            model=model_name,
            A=empty_dim,
            B=empty_dim,
            C=empty_dim,
            D=empty_dim,
            total=0,
            raw_claim={},
            resolution="empty",
            notes="no entities in extraction.json",
        )

    ent = entities[0]
    canonical = (ent.get("canonical") or {}).get("value") or {}
    if not isinstance(canonical, dict):
        canonical = {}

    paper = canonical.get("paper_title")
    technique = canonical.get("technique_name")
    version = canonical.get("first_applied_version")
    kernel = canonical.get("first_applied_kernel_file")
    alternatives = canonical.get("alternative_candidates") or []
    if not isinstance(alternatives, list):
        alternatives = []

    a = DimScore(*_check_paper(paper))
    b = DimScore(*_check_technique(technique))
    c = DimScore(*_check_version(version))
    d = DimScore(*_check_kernel_file(kernel))

    total = a.score + b.score + c.score + d.score
    if total == 4:
        notes = "四个字段都与标准答案一致或可接受地等价。"
    elif total == 0:
        notes = "四个字段均未命中标准答案。"
    else:
        correct = [
            n
            for n, s in zip(["paper", "technique", "version", "kernel_file"],
                            [a.score, b.score, c.score, d.score], strict=False)
            if s == 1
        ]
        wrong = [
            n
            for n, s in zip(["paper", "technique", "version", "kernel_file"],
                            [a.score, b.score, c.score, d.score], strict=False)
            if s == 0
        ]
        notes = f"命中：{'、'.join(correct)}；未命中：{'、'.join(wrong)}。"

    return ModelScore(
        model=model_name,
        A=a,
        B=b,
        C=c,
        D=d,
        total=total,
        raw_claim={
            "paper_title": paper,
            "technique_name": technique,
            "first_applied_version": version,
            "first_applied_kernel_file": kernel,
        },
        alternative_candidates=alternatives,
        resolution=ent.get("resolution", "unknown"),
        notes=notes,
    )


# ═══════════════════════════════════════════════════════════════════════════
# I/O
# ═══════════════════════════════════════════════════════════════════════════


def _dim_to_dict(d: DimScore) -> dict:
    return {"score": d.score, "reason": d.reason}


def _write_model_score(out_dir: Path, r: ModelScore) -> None:
    path = out_dir / r.model / "score.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "query_id": QUERY_ID,
                "model": r.model,
                "score": r.total,
                "max_score": 4,
                "min_score": 0,
                "total_rate": r.total / 4.0,
                "dimensions": {
                    "A_paper_title": _dim_to_dict(r.A),
                    "B_technique_name": _dim_to_dict(r.B),
                    "C_first_applied_version": _dim_to_dict(r.C),
                    "D_first_applied_kernel_file": _dim_to_dict(r.D),
                },
                "raw_claim": r.raw_claim,
                "alternative_candidates": r.alternative_candidates,
                "standard_answer": BENCHMARK["standard_answer"],
                "scoring_rule": (
                    "A paper ±1 (cudadma + warp specialization), "
                    "B technique ±1 (warp specialization), "
                    "C version ±1 (FA3 / FlashAttention-3), "
                    "D kernel file ±1 (mainloop_fwd_sm90_tma_gmma_ws.hpp). "
                    "No penalty, no LLM judge."
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
                "max_score": 4,
                "min_score": 0,
                "scoring_rule": (
                    "A paper ±1 (cudadma + warp specialization), "
                    "B technique ±1 (warp specialization), "
                    "C version ±1 (FA3 / FlashAttention-3), "
                    "D kernel file ±1 (mainloop_fwd_sm90_tma_gmma_ws.hpp). "
                    "No penalty, no LLM judge."
                ),
                "benchmark_version": BENCHMARK["version"],
                "standard_answer": BENCHMARK["standard_answer"],
                "results": {
                    r.model: {
                        "total_score": r.total,
                        "total_rate": r.total / 4.0,
                        "A_paper": r.A.score,
                        "B_technique": r.B.score,
                        "C_version": r.C.score,
                        "D_kernel_file": r.D.score,
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
                        "D": r.D.score,
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
        "> Scoring: A paper ±1 · B technique ±1 · C version ±1 · D kernel_file ±1 — independent, no penalty.  ",
        "> Reference answer:",
        f">   - 论文标题: **{BENCHMARK['standard_answer']['paper_title']}**",
        f">   - 技巧名称: **{BENCHMARK['standard_answer']['technique_name']}**",
        f">   - 首次应用版本: **{BENCHMARK['standard_answer']['first_applied_version']}**",
        f">   - 首次应用 kernel 文件: `{BENCHMARK['standard_answer']['first_applied_kernel_file']}`",
        "",
        "| Rank | Model | Total | A paper | B technique | C version | D kernel |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for i, r in enumerate(ranked, start=1):
        lines.append(
            f"| {i} | {r.model} | {r.total}/4 | {r.A.score} | {r.B.score} | {r.C.score} | {r.D.score} |"
        )
    lines.append("")
    lines.append("## 各模型主张")
    lines.append("")
    lines.append("| Model | paper_title | technique_name | first_applied_version | first_applied_kernel_file |")
    lines.append("|---|---|---|---|---|")
    for r in ranked:
        rc = r.raw_claim or {}
        lines.append(
            f"| {r.model} "
            f"| {str(rc.get('paper_title') or '—')[:50]} "
            f"| {str(rc.get('technique_name') or '—')[:30]} "
            f"| {str(rc.get('first_applied_version') or '—')[:25]} "
            f"| {str(rc.get('first_applied_kernel_file') or '—')[:60]} |"
        )
    lines.append("")
    lines.append("## 维度判定理由")
    lines.append("")
    for r in ranked:
        lines.append(f"### {r.model}  (total={r.total}/4)")
        lines.append(f"- A_paper ({r.A.score}): {r.A.reason}")
        lines.append(f"- B_technique ({r.B.score}): {r.B.reason}")
        lines.append(f"- C_version ({r.C.score}): {r.C.reason}")
        lines.append(f"- D_kernel_file ({r.D.score}): {r.D.reason}")
        lines.append("")
    (out_dir / "ranking_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


# ═══════════════════════════════════════════════════════════════════════════
# Entry
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

    results: list[ModelScore] = []
    for model_name, payload in all_results.items():
        r = _score_one_model(model_name, payload)
        _write_model_score(out_dir, r)
        results.append(r)

    _write_ranking_report(out_dir, results)
    _write_scores_json(out_dir, results)

    print("\n" + "─" * 64)
    print(f"Query {QUERY_ID} scoring done.")
    for r in sorted(results, key=lambda x: (-x.total, x.model)):
        mark = "✅" if r.total == 4 else ("🟡" if r.total > 0 else "❌")
        print(
            f"  {mark} {r.model:28s} {r.total}/4  "
            f"A={r.A.score} B={r.B.score} C={r.C.score} D={r.D.score}"
        )


if __name__ == "__main__":
    main()
