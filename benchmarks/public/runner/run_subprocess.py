#!/usr/bin/env python3
"""Run a benchmark via subprocess-per-question isolation."""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s — %(message)s",
)
logger = logging.getLogger("run_harbor_subprocess")

# Per-question ceiling. Must stay ABOVE the largest inner budget a question can
# legitimately consume — chiefly ``SUB_AGENT_TIMEOUT_S`` (see .env.example),
# which is why the two are documented together: an outer kill below the inner
# budget truncates work the agent still believed it had time for.
_QUESTION_TIMEOUT = 10800  # 3h

# A question that has produced no output for this long is hung, not slow. Two
# agent-team questions once sat silent for two hours after a cloud sandbox
# keepalive, and only the per-question ceiling would ever have noticed — at
# which point most of a day is gone. Progress is measured by writes to the
# trial's own log, which every tool result and LLM turn touches.
_NO_PROGRESS_TIMEOUT = 1800  # 30min of total silence
_WORKER_SCRIPT = Path(__file__).resolve().parent / "run_one_task.py"


# ── Subprocess runner ────────────────────────────────────────────────

async def _tee(reader, log, mirror, prefix=""):
    while line := await reader.readline():
        s = line.decode("utf-8", errors="replace")
        log.write(s)
        log.flush()
        if mirror:
            sys.stdout.write(prefix + s if prefix else s)
            sys.stdout.flush()


async def _wait_with_progress(proc, log_path: Path, qid: str) -> int:
    """Wait for *proc*, killing it on either ceiling.

    Raises ``asyncio.TimeoutError`` whose message names which limit tripped, so
    the synthesized result says *why* rather than just "timeout".
    """
    t0 = time.monotonic()
    last_size = -1
    last_change = t0
    # Poll often enough that the silence check has usable granularity even if a
    # caller shrinks the threshold (tests do), but never busier than every 30s
    # for the real 30-minute one.
    poll = max(1.0, min(30.0, _NO_PROGRESS_TIMEOUT / 4))
    while True:
        try:
            return await asyncio.wait_for(proc.wait(), timeout=poll)
        except TimeoutError:
            pass
        now = time.monotonic()
        if now - t0 >= _QUESTION_TIMEOUT:
            raise TimeoutError(f"subprocess_timeout_{_QUESTION_TIMEOUT}s")
        try:
            size = log_path.stat().st_size
        except OSError:
            size = last_size
        if size != last_size:
            last_size, last_change = size, now
        elif now - last_change >= _NO_PROGRESS_TIMEOUT:
            logger.error(
                "[%s] no output for %ds — treating as hung",
                qid, int(now - last_change),
            )
            raise TimeoutError(
                f"no_progress_{_NO_PROGRESS_TIMEOUT}s",
            )


def _failure_result(
    qid: str,
    *,
    error: str,
    duration: float,
    judge_method: str,
    external_scoring: bool,
) -> dict:
    """Synthetic result for a trial that never produced an answer.

    Under external scoring there is no verdict to report — the trial has no
    answer to hand to the batch scorer, and calling it WRONG would make
    ``check_progress`` compute an accuracy for a collection-only run. ``None``
    keeps it in the unscored bucket, matching what ``run_one_task`` writes.
    """
    return {
        "question_id": qid,
        "is_correct": None if external_scoring else False,
        "reward": None if external_scoring else 0,
        "error": error,
        "duration_seconds": round(duration, 2),
        "judge_method": judge_method,
    }


async def run_task_subprocess(
    qid: str,
    out_dir: Path,
    benchmark: str,
    sem: asyncio.Semaphore,
    mirror: bool = False,
    external_scoring: bool = False,
) -> dict:
    """Spawn a worker subprocess for one question. Returns the result dict
    from result.json, or a synthetic error dict on timeout/crash."""
    trial_dir = out_dir / "trials" / qid
    result_path = trial_dir / "result.json"

    # Resume: skip if already scored.
    if result_path.exists():
        try:
            return json.loads(result_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass  # corrupted, re-run

    async with sem:
        trial_dir.mkdir(parents=True, exist_ok=True)
        log_path = trial_dir / "agent.log"

        # Deliberately not a context manager: the handle outlives this
        # statement, streaming subprocess output via the _tee task, and is
        # closed in the matching finally below.
        log_file = open(log_path, "w", encoding="utf-8")  # noqa: SIM115
        tee = None
        try:
            t0 = time.monotonic()
            proc = await asyncio.create_subprocess_exec(
                sys.executable, str(_WORKER_SCRIPT),
                qid, str(out_dir), benchmark,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
            tee = asyncio.create_task(_tee(proc.stdout, log_file, mirror, prefix=f"[{qid}] "))
            exit_code: int | None = None
            reason = ""
            try:
                exit_code = await _wait_with_progress(proc, log_path, qid)
            except TimeoutError as exc:
                reason = str(exc) or f"subprocess_timeout_{_QUESTION_TIMEOUT}s"
                proc.kill()
                await proc.wait()
                duration = time.monotonic() - t0
                logger.error("[%s] subprocess SIGKILLed: %s", qid, reason)
                synth = _failure_result(
                    qid,
                    error=reason,
                    duration=duration,
                    judge_method="subprocess_timeout",
                    external_scoring=external_scoring,
                )
                result_path.write_text(
                    json.dumps(synth, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
                return synth

            duration = time.monotonic() - t0
        finally:
            if tee is not None:
                await tee
            log_file.close()

    if result_path.exists():
        try:
            return json.loads(result_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            logger.error("[%s] result.json unreadable: %s", qid, exc)

    logger.error(
        "[%s] worker exit=%s but no result.json (duration=%.1fs)",
        qid, exit_code, duration,
    )
    synth = _failure_result(
        qid,
        error=f"worker_exit_{exit_code}_no_result",
        duration=duration,
        judge_method="worker_crash",
        external_scoring=external_scoring,
    )
    result_path.write_text(
        json.dumps(synth, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return synth


# ── One-run evaluator ────────────────────────────────────────────────

async def run_eval(
    args: argparse.Namespace,
    *,
    out_dir: Path,
    seed: int,
) -> dict:
    """Generate tasks + dispatch subprocesses for one run pass."""
    from benchmarks.public.core.harbor_task_generator import generate_task_dirs
    from benchmarks.public.core.registry import get_config, load_questions

    cfg = get_config(args.benchmark)
    pipeline = args.pipeline or cfg.default_pipeline or "stateful-react-agent"
    external_scoring = cfg.scoring_mode == "external"

    # Book policy, decided once per run and inherited by every worker (they are
    # spawned without an explicit env, so the parent's environment is theirs).
    # Both names are set because the toggle is read per workflow — stateful
    # reads REACT_NO_WEB, agent-team reads SWARM_NO_WEB — and a benchmark should
    # get the same policy whichever workflow runs it.
    from benchmarks.public.sandbox_profiles import resolve_closed_book
    closed = resolve_closed_book(args.benchmark, getattr(args, "web", None))
    os.environ["REACT_NO_WEB"] = "1" if closed else "0"
    os.environ["SWARM_NO_WEB"] = "1" if closed else "0"
    source = "--web/--no-web" if getattr(args, "web", None) is not None else "benchmark default"
    logger.info(
        "Book policy: %s (%s) — web tools %s",
        "closed-book" if closed else "open-book", source,
        "unbound" if closed else "available",
    )

    # Prove the judge works before spending the run on it. An unreachable judge
    # model returns NOT_ATTEMPTED for everything, so the run finishes and
    # reports 0% — indistinguishable from a model that got everything wrong,
    # after however long the run took.
    if external_scoring:
        logger.info(
            "Scoring mode: external batch scorer (collecting answers only)"
        )
    else:
        from benchmarks.public.judges import preflight_judge
        judge_ok, judge_detail = await preflight_judge(args.benchmark)
        if not judge_ok:
            logger.error(
                "Judge preflight FAILED for %s: %s\n"
                "        Refusing to run: every question would score WRONG. Set "
                "JUDGE_MODEL to a model your JUDGE_BASE_URL actually serves, or "
                "fix JUDGE_API_KEY / JUDGE_BASE_URL.",
                args.benchmark, judge_detail,
            )
            raise SystemExit(2)
        logger.info("Judge preflight: %s", judge_detail)

    questions = load_questions(
        args.benchmark,
        limit=args.limit,
        offset=args.offset,
        answer_type=args.answer_type,
        category=args.category,
    )

    if not args.no_shuffle:
        import random
        random.Random(seed).shuffle(questions)
        logger.info("Shuffled questions with seed=%d", seed)

    if args.limit:
        questions = questions[:args.limit]

    total = len(questions)
    logger.info("Loaded %d %s questions", total, cfg.name)
    if total == 0:
        logger.error("No questions to run")
        return {}

    tasks_dir = out_dir / "tasks"
    q_dicts = [
        {
            "id": q.id,
            "question": q.question,
            "answer": q.ground_truth,
            "answer_type": q.answer_type,
            "category": q.metadata.get("category", ""),
            "image_path": q.image_path,
            "file_path": q.file_path,
            "file_name": q.file_name,
            # Inject workflow-profile so it lands in task.toml's [metadata]
            # section. Read back by FrontierAgentAgent._read_task_metadata
            # and forwarded into the pipeline state as ``state.metadata.profile``.
            "metadata": {
                **q.metadata,
                "benchmark": args.benchmark,
                "profile": args.profile,
                **({"fs_mode": "true"} if args.fs_mode else {}),
            },
        }
        for q in questions
    ]
    generate_task_dirs(q_dicts, tasks_dir, pipeline_id=pipeline)

    (out_dir / "trials").mkdir(parents=True, exist_ok=True)

    # One judge session per run, inherited by every worker subprocess (they get
    # os.environ). All judge calls share a long grading prompt, so a stable
    # session id is what lets the upstream prompt cache hit across the batch.
    # Derived from the out dir so it is reproducible and greppable in logs.
    if "JUDGE_SESSION" not in os.environ:
        digest = hashlib.sha256(str(out_dir.resolve()).encode()).hexdigest()[:12]
        os.environ["JUDGE_SESSION"] = f"judge-{args.benchmark}-{digest}"
    logger.info("Judge session: %s", os.environ["JUDGE_SESSION"])

    sem = asyncio.Semaphore(args.concurrency)
    mirror = sys.stdout.isatty()
    t_start = time.monotonic()
    completed = 0
    correct_count = 0
    collected_count = 0

    async def wrap(q):
        nonlocal completed, correct_count, collected_count
        result = await run_task_subprocess(
            q.id, out_dir, args.benchmark, sem, mirror,
            external_scoring=external_scoring,
        )
        completed += 1
        if result.get("predicted_answer"):
            collected_count += 1
        if result.get("is_correct"):
            correct_count += 1
        if completed % 10 == 0 or completed == total:
            elapsed = time.monotonic() - t_start
            if external_scoring:
                logger.info(
                    "PROGRESS: %d/%d done, %d answers collected (%.0fs elapsed)",
                    completed, total, collected_count, elapsed,
                )
            else:
                logger.info(
                    "PROGRESS: %d/%d done, %d correct (%.1f%%) (%.0fs elapsed)",
                    completed, total, correct_count,
                    correct_count * 100 / completed, elapsed,
                )
        return result

    results = await asyncio.gather(
        *(wrap(q) for q in questions),
        return_exceptions=True,
    )

    final: list[dict] = []
    for q, r in zip(questions, results, strict=False):
        if isinstance(r, BaseException):
            logger.error("[%s] orchestrator error: %s", q.id, r)
            final.append(_failure_result(
                q.id,
                error=f"orchestrator_{type(r).__name__}: {r}",
                duration=0.0,
                judge_method="orchestrator_exception",
                external_scoring=external_scoring,
            ))
        else:
            final.append(r)

    total_duration = time.monotonic() - t_start

    attempted = sum(1 for r in final if r.get("predicted_answer"))
    # Paired: both stay None under external scoring, and computing them in one
    # branch is what lets the reporting below narrow them together.
    correct: int | None = None
    accuracy: float | None = None
    if not external_scoring:
        correct = sum(1 for r in final if r.get("is_correct"))
        accuracy = (correct / total) if total else 0.0

    by_method: dict[str, int] = {}
    for r in final:
        m = r.get("judge_method", "unknown")
        by_method[m] = by_method.get(m, 0) + 1

    summary = (
        f"\n{'=' * 60}\n"
        f"  {cfg.name} Results\n"
        f"{'=' * 60}\n"
        f"  Pipeline:     {pipeline}\n"
        f"  Profile:      {args.profile}\n"
        f"  Total:        {total}\n"
        f"  Attempted:    {attempted}\n"
    )
    if external_scoring:
        summary += "  Scoring:      external batch pending\n"
    else:
        summary += f"  Correct:      {correct}\n  Accuracy:     {accuracy:.2%}\n"
    summary += (
        f"  Duration:     {total_duration:.1f}s\n"
        f"  Concurrency:  {args.concurrency}\n"
        f"\n  Scoring method breakdown:\n"
    )
    for m, cnt in sorted(by_method.items()):
        summary += f"    {m}: {cnt}\n"
    summary += f"{'=' * 60}\n"
    logger.info(summary)

    report = {
        "benchmark": args.benchmark,
        "pipeline": pipeline,
        "profile": args.profile,
        "total": total,
        "attempted": attempted,
        "correct": correct,
        "accuracy": accuracy,
        "scoring_mode": cfg.scoring_mode,
        "by_method": by_method,
        "duration_seconds": round(total_duration, 2),
        "config": {
            "limit": args.limit,
            "seed": seed,
            "concurrency": args.concurrency,
            "offset": args.offset,
        },
        "started_at": datetime.now(UTC).isoformat(),
        "results": final,
    }
    (out_dir / "results.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (out_dir / "summary.txt").write_text(summary, encoding="utf-8")
    if external_scoring:
        logger.info(
            "Run finished — %d/%d answers collected; external scoring pending",
            attempted, total,
        )
    else:
        logger.info(
            "Run finished — %d/%d correct (%.1f%%)",
            correct or 0, total, (accuracy or 0.0) * 100,
        )
    return report


# ── CLI ──────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--benchmark", required=True,
                   help="Dataset key registered in benchmarks.public.core.registry.REGISTRY")
    p.add_argument("--pipeline", default=None,
                   help="FrontierAgent pipeline (default: dataset's default_pipeline)")
    p.add_argument("--profile", default="default",
                   help="Workflow profile name (default: 'default'). Written into "
                        "task.toml metadata; read back as state.metadata.profile.")
    p.add_argument(
        "--fs-mode",
        action="store_true",
        help="Enable the workflows' file/problem-solving prompt mode.",
    )
    p.add_argument("--runs", type=int, default=1,
                   help="Number of independent runs over the question set. "
                        "Each lands in <out>/run_<i>/. Default: 1.")
    p.add_argument("--concurrency", type=int, default=1,
                   help="Per-run concurrent subprocess workers (default: 1). When --runs N, all N runs launch concurrently; total in-flight = runs × concurrency.")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--offset", type=int, default=0)
    p.add_argument("--seed", type=int, default=None,
                   help="Shuffle seed (default: 42). For --runs N, each "
                        "run uses seed + run_index - 1.")
    p.add_argument("--no-shuffle", action="store_true",
                   help="Disable question shuffling (run in dataset order)")
    web = p.add_mutually_exclusive_group()
    web.add_argument("--no-web", dest="web", action="store_false", default=None,
                     help="Force closed-book: unbind web_search/web_fetch/download_file. "
                          "Default comes from the benchmark (corpus benchmarks are "
                          "closed-book already), so you rarely need this.")
    web.add_argument("--web", dest="web", action="store_true", default=None,
                     help="Force open-book even for a corpus benchmark. The score is then "
                          "not comparable with closed-book reports of the same benchmark.")
    p.add_argument("--answer-type", default=None)
    p.add_argument("--category", default=None)
    p.add_argument("--out", default=None,
                   help="Output directory (default: benchmarks/public/results/<benchmark>/<ts>)")
    args = p.parse_args()

    if args.out is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.out = f"benchmarks/public/results/{args.benchmark}/{ts}"

    base_out = Path(args.out).resolve()
    base_out.mkdir(parents=True, exist_ok=True)
    (base_out / "config.json").write_text(
        json.dumps(vars(args), indent=2, default=str),
        encoding="utf-8",
    )

    base_seed = args.seed if args.seed is not None else 42

    if args.runs == 1:
        asyncio.run(run_eval(args, out_dir=base_out, seed=base_seed))
    else:
        # All N runs concurrently. Each run has its own ``asyncio.Semaphore``
        # of ``args.concurrency`` — total subprocesses in-flight =
        # args.runs × args.concurrency.
        async def _run_all() -> None:
            coros = []
            for run_idx in range(1, args.runs + 1):
                run_out = base_out / f"run_{run_idx}"
                run_out.mkdir(parents=True, exist_ok=True)
                seed = base_seed + run_idx - 1
                logger.info("queue run %d / %d  (seed=%d, out=%s)",
                            run_idx, args.runs, seed, run_out)
                coros.append(run_eval(args, out_dir=run_out, seed=seed))
            await asyncio.gather(*coros)

        asyncio.run(_run_all())


if __name__ == "__main__":
    main()
