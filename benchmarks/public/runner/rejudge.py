#!/usr/bin/env python3
"""Re-judge existing trial result.json files in place."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from benchmarks.public.core.question import BenchmarkQuestion
from benchmarks.public.core.registry import load_questions
from benchmarks.public.judges import score_answer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s — %(message)s",
)
logger = logging.getLogger("rejudge")


def _collect_results(out_dir: Path) -> list[Path]:
    """Find result.json files under both single-run and multi-run layouts."""
    return sorted(out_dir.glob("**/trials/*/result.json"))


async def _rejudge_one(
    path: Path,
    benchmark: str,
    questions: dict[str, BenchmarkQuestion],
    sem: asyncio.Semaphore,
    force: bool,
) -> str:
    r = json.loads(path.read_text(encoding="utf-8"))
    if not force and "rubric_score" in r:
        return "skip"
    pred = (r.get("predicted_answer") or "").strip()
    if not pred:
        r["rubric_score"] = None
        _atomic_write(path, r)
        return "empty"
    # result.json's ``question`` is truncated to 200 chars for display; reload
    # the full question + ground truth from the dataset so the judge sees the
    # same payload that the original run did.
    q = questions.get(r["question_id"])
    if q is None:
        logger.error("[%s] question_id not in dataset — skipping", path.parent.name)
        return "fail"

    async with sem:
        try:
            verdict, score = await score_answer(
                benchmark, q.question, q.ground_truth, pred,
            )
        except Exception as exc:
            logger.error("[%s] judge failed: %s", path.parent.name, exc)
            return "fail"

    r["rubric_score"] = score
    r["is_correct"] = verdict == "CORRECT"
    r["reward"] = 1 if verdict == "CORRECT" else 0
    r["judge_method"] = "llm"
    r["scoring_error"] = None
    _atomic_write(path, r)
    return "ok"


def _atomic_write(path: Path, obj: dict) -> None:
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


async def _amain(args: argparse.Namespace) -> int:
    out_dir = Path(args.out_dir).resolve()
    cfg = json.loads((out_dir / "config.json").read_text(encoding="utf-8"))
    benchmark = cfg["benchmark"]
    results = _collect_results(out_dir)
    questions = {q.id: q for q in load_questions(benchmark)}
    logger.info(
        "Rejudging %d trial(s) under %s (benchmark=%s, concurrency=%d, force=%s)",
        len(results), out_dir, benchmark, args.concurrency, args.force,
    )

    sem = asyncio.Semaphore(args.concurrency)
    outcomes = await asyncio.gather(
        *(_rejudge_one(p, benchmark, questions, sem, args.force) for p in results),
    )
    counts = {k: outcomes.count(k) for k in ("ok", "skip", "empty", "fail")}
    logger.info(
        "Done: %(ok)d updated, %(skip)d skipped, %(empty)d empty-pred, %(fail)d failed",
        counts,
    )
    return 0 if counts["fail"] == 0 else 1


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("out_dir", help="The --out dir passed to run_subprocess")
    p.add_argument("--concurrency", type=int, default=10,
                   help="Parallel judge calls (default: 10)")
    p.add_argument("--force", action="store_true",
                   help="Re-judge even if rubric_score key already present")
    return asyncio.run(_amain(p.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
