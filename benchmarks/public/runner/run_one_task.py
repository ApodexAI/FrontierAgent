#!/usr/bin/env python3
"""Single-question worker — spawned by run_harbor_subprocess.py."""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import logging
import os
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scorer"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s — %(message)s",
)
logger = logging.getLogger("run_one_task")


# ── Scoring ───────────────────────────────────────────────────────────


from benchmarks.public.judges import has_judge, score_answer  # noqa: E402

_QUESTION_TIMEOUT = 28800  # 8h per question (pipeline + scoring)


async def _score_answer(
    predicted: str,
    ground_truth: str,
    question: str,
    answer_type: str,
    q_id: str,
    benchmark: str,
    trial_dir: Path | None = None,
) -> tuple[int | None, str, str | None, float | None]:
    """Score an answer or mark it pending for an external batch scorer."""
    from benchmarks.public.core.registry import get_config

    if get_config(benchmark).scoring_mode == "external":
        return None, "external_pending", None, None

    if not predicted or not predicted.strip():
        logger.info("[%s] SCORE: empty prediction → WRONG", q_id)
        return 0, "empty", None, None

    if benchmark == "gdpval":
        from benchmarks.public.judges.gdpval import score_gdpval_outputs

        if trial_dir is None:
            return 0, "gdpval_artifact", "trial directory unavailable", 0.0
        outputs_dir = trial_dir / "sandbox" / "outputs"
        reward, rubric_score = score_gdpval_outputs(outputs_dir, ground_truth)
        return reward, "gdpval_artifact", None, rubric_score

    if not has_judge(benchmark):
        msg = f"no judge registered for benchmark {benchmark!r}"
        logger.error("[%s] SCORE: %s → WRONG", q_id, msg)
        return 0, "no_judge", msg, None

    logger.info(
        "[%s] SCORE: calling %s judge | pred=%r gt=%r",
        q_id, benchmark, predicted[:50], ground_truth[:50],
    )
    try:
        verdict, rubric_score = await score_answer(
            benchmark, question, ground_truth, predicted,
        )
    except Exception as exc:
        detail = f"{type(exc).__name__}: {exc}"
        logger.error(
            "[%s] SCORE: judge raised — WRONG (%s). NEEDS MANUAL REVIEW.",
            q_id, detail,
        )
        return 0, "scoring_failed", detail, None

    score_tag = f" rubric_score={rubric_score:.2f}" if rubric_score is not None else ""
    judge_method = "officeqa_reward" if benchmark in {"officeqa", "officeqa_full"} else "llm"
    if verdict == "CORRECT":
        logger.info("[%s] SCORE: %s=CORRECT%s", q_id, judge_method, score_tag)
        return 1, judge_method, None, rubric_score
    if verdict == "INCORRECT":
        logger.info("[%s] SCORE: %s=WRONG%s", q_id, judge_method, score_tag)
        return 0, judge_method, None, rubric_score

    logger.error(
        "[%s] SCORE: judge NOT_ATTEMPTED after retries → WRONG. "
        "NEEDS MANUAL REVIEW.", q_id,
    )
    return 0, "scoring_failed", "judge NOT_ATTEMPTED after retries", rubric_score


# ── Single question runner ────────────────────────────────────────────

async def run_one_question(
    agent,
    question: dict,
    trial_dir: Path,
    task_src_dir: Path,
    benchmark: str,
) -> dict:
    """Run one question through FrontierAgentAgent and score it.

    Returns a result dict with question_id, predicted, ground_truth,
    is_correct, duration, etc.
    """
    from harbor.models.agent.context import AgentContext

    q_id = question["id"]
    trial_dir.mkdir(parents=True, exist_ok=True)

    # ── Resume: skip if already scored ──────────────────────────
    existing_result = trial_dir / "result.json"
    if existing_result.exists():
        try:
            cached = json.loads(existing_result.read_text(encoding="utf-8"))
            logger.debug("[%s] Skipping — result.json exists", q_id)
            return cached
        except (json.JSONDecodeError, OSError):
            pass  # corrupted, re-run

    # Write config.json so agent can find task.toml metadata
    (trial_dir / "config.json").write_text(
        json.dumps({"task": {"path": str(task_src_dir)}}),
        encoding="utf-8",
    )

    # Create a fresh agent wrapper pointing to this trial
    # We can't change agent.logs_dir after init, so we set it directly
    agent.logs_dir = trial_dir

    instruction_path = task_src_dir / "instruction.md"
    instruction = instruction_path.read_text(encoding="utf-8")

    context = AgentContext()
    fake_env = MagicMock()

    t0 = time.monotonic()
    error = None
    try:
        await asyncio.wait_for(
            agent.run(instruction, fake_env, context),
            timeout=_QUESTION_TIMEOUT,
        )
    except TimeoutError:
        error = f"timeout after {_QUESTION_TIMEOUT}s"
        logger.error("[%s] Pipeline TIMEOUT after %ds", q_id, _QUESTION_TIMEOUT)
    except Exception as exc:
        error = str(exc)
        logger.error("[%s] Pipeline error: %s", q_id, exc)
    duration = time.monotonic() - t0

    # Read answer
    answer_file = trial_dir / "agent" / "final_answer.txt"
    predicted = ""
    if answer_file.exists():
        predicted = answer_file.read_text(encoding="utf-8").strip()

    # Score: rule-based first, LLM judge if rule says wrong
    ground_truth = question["ground_truth"]
    answer_type = question["answer_type"]
    reward, judge_method, scoring_error, rubric_score = await _score_answer(
        predicted, ground_truth, question["question"], answer_type, q_id,
        benchmark, trial_dir,
    )

    result = {
        "question_id": q_id,
        "question": question["question"][:200],
        "answer_type": answer_type,
        "ground_truth": ground_truth,
        "predicted_answer": predicted,
        "is_correct": None if reward is None else reward == 1,
        "reward": reward,
        "rubric_score": rubric_score,  # None = binary judge; float = rubric value
        "judge_method": judge_method,
        "scoring_error": scoring_error,  # None = OK, string = needs manual review
        "duration_seconds": round(duration, 2),
        "error": error,
        "category": question.get("category", ""),
    }

    # Write per-question result
    (trial_dir / "result.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    if reward is None:
        status = "COLLECTED" if predicted else "NO ANSWER"
    else:
        status = "CORRECT" if reward == 1 else "WRONG"
    method_tag = f" [{judge_method}]" if judge_method != "rule" else ""
    logger.info(
        "[%s] %s%s | predicted=%r truth=%r (%.1fs)",
        q_id, status, method_tag, predicted[:40], ground_truth[:40], duration,
    )
    return result


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("qid", help="Question id (matches trials/<qid>/ dir)")
    p.add_argument("out_dir", help="Run output directory (contains trials/, tasks/)")
    p.add_argument("benchmark", nargs="?", default="browsecomp")
    return p.parse_args()


async def _run_async(args: argparse.Namespace) -> int:
    from benchmarks.public.core.registry import load_questions
    from benchmarks.public.harbor_agent.agent import FrontierAgentAgent

    out_dir = Path(args.out_dir).resolve()
    trial_dir = out_dir / "trials" / args.qid
    task_src_dir = out_dir / "tasks" / args.qid

    # Resume: if already scored, no-op (master should have skipped us, but be safe).
    if (trial_dir / "result.json").exists():
        logger.info("[%s] result.json already present — nothing to do", args.qid)
        return 0

    if not task_src_dir.exists():
        logger.error("[%s] task source dir not found: %s", args.qid, task_src_dir)
        return 1

    # Locate the single question by qid.
    all_questions = load_questions(args.benchmark)
    match = next((q for q in all_questions if q.id == args.qid), None)
    if match is None:
        logger.error("[%s] question id not found in %s dataset", args.qid, args.benchmark)
        return 2

    question = {
        "id": match.id,
        "question": match.question,
        "ground_truth": match.ground_truth,
        "answer_type": match.answer_type,
        "category": match.metadata.get("category", ""),
        "image_path": match.image_path,
        "file_path": match.file_path,
        "file_name": match.file_name,
        "metadata": match.metadata,
    }

    # Bootstrap agent (one per subprocess — own :memory: DB, own kernel).
    agent = FrontierAgentAgent(logs_dir=trial_dir.parent, model_name=None)
    run_db = str(out_dir / f".worker_kernel_{args.qid}.db")
    setup_env = MagicMock(spec=[])  # no _pipeline_hint — react_base is the only workflow
    await agent.setup(setup_env, db_path=run_db)

    await run_one_question(agent, question, trial_dir, task_src_dir, args.benchmark)

    # Cleanup the per-worker kernel db file (best-effort).
    with contextlib.suppress(FileNotFoundError):
        Path(run_db).unlink()

    return 0


def main() -> None:
    args = _parse_args()
    exit_code = asyncio.run(_run_async(args))
    # Force-exit: frontier_agent kernel keeps a non-daemon aiosqlite worker
    # thread alive, which blocks threading._shutdown() forever. result.json
    # has already been flushed to disk by run_one_question; nothing else
    # in this one-shot worker process is worth waiting to clean up.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(exit_code)


if __name__ == "__main__":
    main()
