"""Query 08 auto-scorer — Kobe's 2nd-highest scoring game vs LeBron's
Cavaliers + that game's top Cavalier scorer.

Pipeline (deviates from the standard 4-stage flow — see §1 below):
  Stage 1  Extraction (LLM, multi-vote)  — uses
           pipeline.extraction_pipeline.run_pipeline().
  Stage 2  Direct field comparison (PYTHON, no LLM alignment) — the answer
           space is a single closed 4-field tuple, so deterministic
           normalisation + comparison is more accurate and faster than the
           generic alignment.py vote+judge layer used by open-set queries (e.g. Q06/Q10).
  Stage 3  (skipped) — no null verification needed; closed answer space.
  Stage 4  Scoring — each of the 4 fields independently scores +1 / 0 / -1.

Scoring rule per answer tuple:
  game_date            == "2007-02-11"  → +1; wrong → -1; missing → 0
  kobe_points          == 36            → +1; wrong → -1; missing → 0
  cavs_top_scorer_name matches Pavlović → +1; wrong → -1; missing → 0
  cavs_top_scorer_pts  == 21            → +1; wrong → -1; missing → 0

Per-model aggregation:
  - If a model returns N answer tuples:
      base_score = max(tuple_scores)            # take the best tuple
      penalty    = -1 * (N - 1)  if base==4 ELSE -1 * count(wrong_tuples)
      total      = base_score + penalty
  - This penalises models that "shotgun" multiple answers to hedge.
  - MAX_SCORE = 4, total can be negative.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from datetime import datetime
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
# Section 1 — Ground truth
# ═══════════════════════════════════════════════════════════════════════════

SNAPSHOT_DATE = "2026-04-30"

GROUND_TRUTH = {
    "game_date": "2007-02-11",
    "kobe_points": 36,
    "top_scorer_canonical": "Sasha Pavlović",
    "top_scorer_points": 21,
    # Aliases used for fuzzy match. Each alias is checked as a substring of the
    # normalised (lowercase, accent-folded, punctuation-stripped) candidate.
    # All aliases here are themselves already in normalised form.
    "top_scorer_aliases": [
        "pavlovic",  # English (accent-folded)
        "sashapavlovic",
        "帕夫洛维奇",  # Chinese surname
        "萨沙帕夫洛维奇",  # Chinese full
        "萨沙",  # Chinese given (specific enough — no other "萨沙" on Cavs)
    ],
}

MAX_SCORE = 4  # 4 fields, each contributes +1 max


# ═══════════════════════════════════════════════════════════════════════════
# Section 2 — Field normalisation (Python, no LLM)
# ═══════════════════════════════════════════════════════════════════════════


def _normalise_date(raw) -> str | None:
    """Coerce common date formats to 'YYYY-MM-DD'. Return None on failure."""
    if raw is None:
        return None
    s = str(raw).strip()
    if not s:
        return None

    # Replace Chinese 年/月/日 with separators
    s_zh = s.replace("年", "-").replace("月", "-").replace("日", "")
    s_zh = s_zh.rstrip("-").strip()

    candidates = [s, s_zh]
    formats = [
        "%Y-%m-%d",
        "%Y/%m/%d",
        "%Y.%m.%d",
        "%Y-%-m-%-d",
        "%Y/%-m/%-d",
        "%b %d, %Y",
        "%B %d, %Y",
        "%b %d %Y",
        "%B %d %Y",
        "%m/%d/%Y",
        "%m-%d-%Y",
        "%d %b %Y",
        "%d %B %Y",
    ]
    for cand in candidates:
        for fmt in formats:
            try:
                dt = datetime.strptime(cand, fmt)
                return dt.strftime("%Y-%m-%d")
            except ValueError:
                continue

    # Fallback: regex pull out YYYY, MM, DD
    m = re.search(r"(\d{4})\D+(\d{1,2})\D+(\d{1,2})", s)
    if m:
        try:
            dt = datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)))
            return dt.strftime("%Y-%m-%d")
        except ValueError:
            pass

    return None


def _normalise_int(raw) -> int | None:
    """Pull a (signed) integer from raw. Tolerates '36', '36分', '36 pts',
    '36.0', 36, 36.0, etc. Returns None when no digit found."""
    if raw is None:
        return None
    if isinstance(raw, bool):  # exclude True/False misread as int
        return None
    if isinstance(raw, (int, float)):
        try:
            return int(raw)
        except (ValueError, TypeError):
            return None
    s = str(raw).strip()
    if not s:
        return None
    m = re.search(r"-?\d+", s)
    if not m:
        return None
    return int(m.group())


def _normalise_name(raw) -> str:
    """Lowercase + NFKD-fold (drop combining marks like ć's diacritic) +
    strip punctuation/whitespace. 'Sasha Pavlović (萨沙·帕夫洛维奇)' →
    'sashapavlovic萨沙帕夫洛维奇'."""
    if raw is None:
        return ""
    s = str(raw).lower()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"[\s\-\.,;:'\"()\[\]（）「」【】·••·]+", "", s)
    return s


# ═══════════════════════════════════════════════════════════════════════════
# Section 3 — Per-field scoring
# ═══════════════════════════════════════════════════════════════════════════


def _score_date(raw) -> tuple[int, str]:
    norm = _normalise_date(raw)
    if norm is None:
        return 0, f"missing/unparseable: {raw!r}"
    if norm == GROUND_TRUTH["game_date"]:
        return 1, f"correct ({norm})"
    return -1, f"wrong: {norm} (gt={GROUND_TRUTH['game_date']})"


def _score_int(raw, gt_val: int, label: str) -> tuple[int, str]:
    norm = _normalise_int(raw)
    if norm is None:
        return 0, f"missing/unparseable: {raw!r}"
    if norm == gt_val:
        return 1, f"correct ({norm})"
    return -1, f"wrong {label}: {norm} (gt={gt_val})"


def _score_top_scorer_name(raw) -> tuple[int, str]:
    norm = _normalise_name(raw)
    if not norm:
        return 0, f"missing: {raw!r}"
    matched = [a for a in GROUND_TRUTH["top_scorer_aliases"] if a in norm]
    if matched:
        return 1, f"correct (matched alias {matched[0]!r}; raw={raw!r})"
    return (
        -1,
        f"wrong: {raw!r} (normalised={norm!r}; gt={GROUND_TRUTH['top_scorer_canonical']!r})",
    )


def score_tuple(claim: dict) -> dict:
    """Score one answer tuple. Returns per-field results + sum."""
    g_score, g_reason = _score_date(claim.get("game_date"))
    k_score, k_reason = _score_int(
        claim.get("kobe_points"), GROUND_TRUTH["kobe_points"], "kobe_points"
    )
    n_score, n_reason = _score_top_scorer_name(claim.get("cavs_top_scorer_name"))
    p_score, p_reason = _score_int(
        claim.get("cavs_top_scorer_points"),
        GROUND_TRUTH["top_scorer_points"],
        "top_scorer_points",
    )
    total = g_score + k_score + n_score + p_score
    return {
        "raw_claim": claim,
        "field_scores": {
            "game_date": {"score": g_score, "reason": g_reason},
            "kobe_points": {"score": k_score, "reason": k_reason},
            "cavs_top_scorer_name": {"score": n_score, "reason": n_reason},
            "cavs_top_scorer_points": {"score": p_score, "reason": p_reason},
        },
        "tuple_total": total,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Section 4 — Per-model aggregation
# ═══════════════════════════════════════════════════════════════════════════


def score_model(model_name: str, claims: list[dict]) -> dict:
    """Aggregate per-tuple scores for one model.

    Rule:
      - If 0 tuples: total = 0, status = 'no_answer'.
      - If N tuples: pick the best tuple as the base score.
        - If best == MAX_SCORE (=4), every other tuple is "extra" (-1 each).
        - Else, every tuple with total < best counts as -1 each (avoids
          double-penalising the best tuple but punishes filler).
    """
    if not claims:
        return {
            "model": model_name,
            "n_tuples": 0,
            "tuple_results": [],
            "best_tuple_score": 0,
            "extra_tuple_penalty": 0,
            "total_score": 0.0,
            "max_score": MAX_SCORE,
            "status": "no_answer",
        }

    tuple_results = [score_tuple(c) for c in claims]
    best = max(t["tuple_total"] for t in tuple_results)
    n = len(tuple_results)

    if n == 1:
        penalty = 0
    elif best == MAX_SCORE:
        # Best tuple is fully correct; any extra tuple is hedge-noise.
        penalty = -1 * (n - 1)
    else:
        # No tuple fully correct — each non-best tuple still counts as filler.
        # Count how many tuples scored strictly below `best`; the best one
        # itself doesn't add penalty.
        penalty = -1 * sum(1 for t in tuple_results if t["tuple_total"] < best)

    total = best + penalty
    return {
        "model": model_name,
        "n_tuples": n,
        "tuple_results": tuple_results,
        "best_tuple_score": best,
        "extra_tuple_penalty": penalty,
        "total_score": float(total),
        "max_score": MAX_SCORE,
        "status": "answered",
    }


# ═══════════════════════════════════════════════════════════════════════════
# Section 5 — Load extraction output
# ═══════════════════════════════════════════════════════════════════════════


def load_raw_claims(out_dir: Path) -> dict[str, list[dict]]:
    """Read each model's `{dir}/{model}/extraction.json` and return the raw
    `entities[0].canonical.value` list."""
    out: dict[str, list[dict]] = {}
    for d in sorted(Path(out_dir).iterdir()):
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
# Section 6 — Output builders
# ═══════════════════════════════════════════════════════════════════════════


def build_scores_json(model_results: list[dict]) -> dict:
    ranked = sorted(model_results, key=lambda x: -x["total_score"])
    return {
        "query_id": QUERY_ID,
        "query_text": QUERY_TEXT,
        "snapshot_date": SNAPSHOT_DATE,
        "max_score": MAX_SCORE,
        "ground_truth": {
            "game_date": GROUND_TRUTH["game_date"],
            "kobe_points": GROUND_TRUTH["kobe_points"],
            "top_scorer_canonical": GROUND_TRUTH["top_scorer_canonical"],
            "top_scorer_points": GROUND_TRUTH["top_scorer_points"],
        },
        "scoring_pipeline": (
            "v1 — Stage1 LLM extraction (run_pipeline) + Stage2 deterministic "
            "Python field comparison (no alignment LLM)."
        ),
        "results": {r["model"]: r for r in model_results},
        "ranking": [
            {"rank": i + 1, "model": r["model"], "score": r["total_score"]}
            for i, r in enumerate(ranked)
        ],
    }


def build_ranking_md(model_results: list[dict]) -> str:
    ranked = sorted(model_results, key=lambda x: -x["total_score"])
    lines = [
        f"# Query {QUERY_ID} 排名报告",
        "",
        f"> Snapshot: {SNAPSHOT_DATE}  Max: {MAX_SCORE}",
        f"> Ground truth: {GROUND_TRUTH['game_date']} / "
        f"科比 {GROUND_TRUTH['kobe_points']}分 / "
        f"骑士得分王 {GROUND_TRUTH['top_scorer_canonical']} "
        f"{GROUND_TRUTH['top_scorer_points']}分",
        "> 评分: 4 字段独立 ±1，每模型取最佳 tuple + 多余 tuple 罚分",
        "",
        "| Rank | Model | Score | Best Tuple | #Tuples | Penalty | Status |",
        "|---:|---|---:|---:|---:|---:|---|",
    ]
    for i, r in enumerate(ranked, 1):
        lines.append(
            f"| {i} | {r['model']} | "
            f"{r['total_score']}/{r['max_score']} | "
            f"{r['best_tuple_score']}/{MAX_SCORE} | "
            f"{r['n_tuples']} | "
            f"{r['extra_tuple_penalty']} | "
            f"{r['status']} |"
        )

    # Per-model field detail (best tuple only, for readability)
    lines += ["", "## 每模型最佳 tuple 的字段详情", ""]
    for r in ranked:
        lines.append(f"### {r['model']} (score = {r['total_score']}/{MAX_SCORE})")
        if r["status"] == "no_answer":
            lines.append("- 未给出答案")
            lines.append("")
            continue
        # Find the best tuple
        best_idx = max(
            range(len(r["tuple_results"])),
            key=lambda i: r["tuple_results"][i]["tuple_total"],
        )
        best = r["tuple_results"][best_idx]
        raw = best["raw_claim"]
        lines.append(
            f"- 抽取: date={raw.get('game_date')!r}, "
            f"kobe={raw.get('kobe_points')!r}, "
            f"scorer={raw.get('cavs_top_scorer_name')!r}, "
            f"scorer_pts={raw.get('cavs_top_scorer_points')!r}"
        )
        for fname, fres in best["field_scores"].items():
            sym = "✓" if fres["score"] == 1 else ("✗" if fres["score"] == -1 else "○")
            lines.append(f"  - `{fname}` {sym} {fres['score']:+d} — {fres['reason']}")
        if r["n_tuples"] > 1:
            lines.append(
                f"- 共 {r['n_tuples']} 条 tuple，本表仅展示最佳；多余 tuple 罚分 {r['extra_tuple_penalty']}"
            )
        lines.append("")

    return "\n".join(lines) + "\n"


# ═══════════════════════════════════════════════════════════════════════════
# Section 7 — Entry point
# ═══════════════════════════════════════════════════════════════════════════


def main() -> None:
    ap = argparse.ArgumentParser(description=f"Query {QUERY_ID} auto-scorer")
    ap.add_argument(
        "--models",
        nargs="+",
        required=True,
        help="name=path list of model answer JSON files (e.g. claude=<crawl-output>/claude.json)",
    )
    ap.add_argument("--output-dir", default=str(THIS / "auto_scores"))
    ap.add_argument("--primary-model", default=None)
    ap.add_argument("--secondary-model", default=None)
    ap.add_argument("--parallel-models", nargs="+", default=None)
    ap.add_argument("--analyzer-model", default=None)
    ap.add_argument("--concurrency", type=int, default=None)
    ap.add_argument(
        "--skip-extract",
        action="store_true",
        help="Skip extraction; reuse existing extraction.json from output dir.",
    )
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Stage 1: LLM extraction
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

    # Stage 2: deterministic Python scoring (no alignment LLM)
    raw_claims = load_raw_claims(out_dir)
    model_results = [score_model(name, claims) for name, claims in raw_claims.items()]

    # Persist outputs
    (out_dir / "scores.json").write_text(
        json.dumps(build_scores_json(model_results), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (out_dir / "ranking_report.md").write_text(
        build_ranking_md(model_results),
        encoding="utf-8",
    )

    print("\n" + "─" * 62)
    print(f"Query {QUERY_ID} scoring done.")
    print(
        f"GT: {GROUND_TRUTH['game_date']} / Kobe {GROUND_TRUTH['kobe_points']} pts / "
        f"{GROUND_TRUTH['top_scorer_canonical']} {GROUND_TRUTH['top_scorer_points']} pts"
    )
    print()
    for i, r in enumerate(sorted(model_results, key=lambda x: -x["total_score"]), 1):
        print(
            f"  {i:2d}. {r['model']:30s} "
            f"{r['total_score']:>5.1f}/{r['max_score']:.1f}  "
            f"best={r['best_tuple_score']}/{MAX_SCORE}  "
            f"n_tuples={r['n_tuples']}  "
            f"penalty={r['extra_tuple_penalty']}"
        )


if __name__ == "__main__":
    main()
