"""Harbor task directory generator."""

from __future__ import annotations

import argparse
import logging
import random
import shutil
import tempfile
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

TEST_SH = "#!/bin/bash\nset -euo pipefail\npython /opt/score.py\n"

_DEFAULT_PIPELINE = "stateful-react-agent"
_DEFAULT_VERIFIER_IMAGE = "frontier-agent/verifier:v1"


# ── TOML helpers ──────────────────────────────────────────────────────────────


def _escape_toml_string(s: str) -> str:
    """Escape special characters for TOML double-quoted string values.

    Handles backslash, double-quote, and common control characters.
    """
    return (
        s.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", "\\n")
        .replace("\r", "\\r")
        .replace("\t", "\\t")
    )


def _render_task_toml(
    question_id: str,
    ground_truth: str,
    answer_type: str,
    pipeline_id: str,
    verifier_image: str,
    image_path: str | None,
    category: str,
    extra_metadata: dict[str, str] | None = None,
) -> str:
    """Render a task.toml string for a single Harbor task.

    extra_metadata is appended to the [metadata] section verbatim — used for
    pipeline-specific fields (e.g. SWE's instance_id, repo_name, base_commit).
    """
    escaped_gt = _escape_toml_string(ground_truth)
    escaped_answer_type = _escape_toml_string(answer_type)
    escaped_pipeline = _escape_toml_string(pipeline_id)
    escaped_category = _escape_toml_string(category)
    escaped_qid = _escape_toml_string(question_id)

    lines = [
        'schema_version = "1.1"',
        "",
        "[task]",
        f'name = "{escaped_pipeline}/{escaped_qid}"',
        f'description = "Benchmark task {escaped_qid}"',
        'authors = [{name = "FrontierAgent"}]',
        f'keywords = ["{escaped_pipeline}", "benchmark"]',
        "",
        "[environment]",
        f'docker_image = "{_escape_toml_string(verifier_image)}"',
        "cpus = 1",
        "memory_mb = 1024",
        "",
        "[agent]",
        "timeout_sec = 7200",
        "",
        "[verifier]",
        "timeout_sec = 120",
        "",
        "[verifier.env]",
        f'GROUND_TRUTH = "{escaped_gt}"',
        f'ANSWER_TYPE = "{escaped_answer_type}"',
        "",
        "[metadata]",
        f'question_id = "{escaped_qid}"',
        f'answer_type = "{escaped_answer_type}"',
        f'pipeline_id = "{escaped_pipeline}"',
        f'category = "{escaped_category}"',
    ]

    if image_path is not None:
        lines.append(f'image_path = "{_escape_toml_string(image_path)}"')

    if extra_metadata:
        for key, value in extra_metadata.items():
            lines.append(f'{key} = "{_escape_toml_string(str(value))}"')

    lines.append("")
    return "\n".join(lines)


def _render_instruction(question: str, image_path: str | None) -> str:
    """Render instruction.md for a single Harbor task."""
    lines = ["# Task", "", question, ""]
    if image_path is not None:
        lines += [
            "## Attached Image",
            "",
            f"Image path: `{image_path}`",
            "",
        ]
    return "\n".join(lines)


# ── Main generator ─────────────────────────────────────────────────────────────


def generate_task_dirs(
    questions: list[dict[str, Any]],
    output_dir: Path,
    *,
    verifier_image: str = _DEFAULT_VERIFIER_IMAGE,
    pipeline_id: str = _DEFAULT_PIPELINE,
) -> None:
    """Write Harbor task directories for a list of questions.

    Uses atomic write: builds all files in a temporary directory, then
    replaces output_dir in a single rename operation.

    Args:
        questions: List of question dicts with keys: id, question, answer,
                   answer_type, category, image_path.
        output_dir: Destination directory (will be replaced atomically).
        verifier_image: Docker image tag for the verifier container.
        pipeline_id: FrontierAgent pipeline to use for evaluation.
    """
    output_dir = Path(output_dir)
    parent = output_dir.parent
    parent.mkdir(parents=True, exist_ok=True)

    # Write to a temp dir in the same filesystem for atomic rename.
    with tempfile.TemporaryDirectory(dir=parent, prefix=".harbor_tmp_") as tmp_str:
        tmp_dir = Path(tmp_str)

        for q in questions:
            qid = q["id"]
            question_text = q["question"]
            ground_truth = str(q.get("answer", ""))
            answer_type = q.get("answer_type", "exactMatch")
            category = q.get("category", "")
            image_path: str | None = q.get("image_path")

            task_dir = tmp_dir / qid
            tests_dir = task_dir / "tests"
            tests_dir.mkdir(parents=True)

            # instruction.md
            (task_dir / "instruction.md").write_text(
                _render_instruction(question_text, image_path), encoding="utf-8"
            )

            # task.toml — extra_metadata flows through pipeline-specific fields
            # (e.g. SWE: instance_id, repo_name, base_commit, hints_text, swe_profile)
            extra_meta = {
                k: v for k, v in (q.get("metadata") or {}).items()
                if k != "category" and v
            }
            (task_dir / "task.toml").write_text(
                _render_task_toml(
                    question_id=qid,
                    ground_truth=ground_truth,
                    answer_type=answer_type,
                    pipeline_id=pipeline_id,
                    verifier_image=verifier_image,
                    image_path=image_path,
                    category=category,
                    extra_metadata=extra_meta,
                ),
                encoding="utf-8",
            )

            # tests/test.sh
            test_sh_path = tests_dir / "test.sh"
            test_sh_path.write_text(TEST_SH, encoding="utf-8")
            test_sh_path.chmod(0o755)

        # Copy to staging dir outside the temp context before it gets deleted.
        staging = parent / f".harbor_stage_{output_dir.name}"
        shutil.copytree(tmp_str, str(staging))

    # Now tmp context is closed; rename staging to final destination.
    if output_dir.exists():
        shutil.rmtree(output_dir)
    staging.rename(output_dir)

    logger.info("Generated %d task dirs in %s", len(questions), output_dir)


# ── CLI ────────────────────────────────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Convert HLE/GAIA JSONL dataset into Harbor task directories.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    from benchmarks.public.core.registry import REGISTRY

    p.add_argument(
        "--benchmark",
        choices=sorted(REGISTRY),
        required=True,
        help="Which benchmark dataset to use.",
    )
    p.add_argument(
        "--out",
        default="benchmarks/public/tasks-generated",
        help="Output directory for generated task dirs.",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of questions to convert.",
    )
    p.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Number of questions to skip at the start of the dataset.",
    )
    p.add_argument(
        "--answer-type",
        dest="answer_type",
        default=None,
        choices=["exactMatch", "multipleChoice"],
        help="Filter questions by answer type.",
    )
    p.add_argument(
        "--category",
        default=None,
        help="Filter questions by category.",
    )
    p.add_argument(
        "--pipeline",
        default=None,
        help="FrontierAgent pipeline ID (default: from dataset config).",
    )
    p.add_argument(
        "--verifier-image",
        dest="verifier_image",
        default=_DEFAULT_VERIFIER_IMAGE,
        help="Docker image tag for the Harbor verifier container.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (used for reproducible shuffles if added later).",
    )
    return p


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = _build_parser()
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    from benchmarks.public.core.registry import get_config, load_questions

    cfg = get_config(args.benchmark)
    pipeline = args.pipeline or cfg.default_pipeline

    questions_obj = load_questions(
        args.benchmark,
        limit=args.limit,
        offset=args.offset,
        answer_type=args.answer_type,
        category=args.category,
    )

    # Convert BenchmarkQuestion dataclass instances to plain dicts.
    questions = [
        {
            "id": q.id,
            "question": q.question,
            "answer": q.ground_truth,
            "answer_type": q.answer_type,
            "category": q.metadata.get("category", ""),
            "image_path": q.image_path,
            "file_path": q.file_path,
            "file_name": q.file_name,
        }
        for q in questions_obj
    ]

    logger.info(
        "Loaded %d questions from %s benchmark", len(questions), cfg.name,
    )

    out = Path(args.out)
    generate_task_dirs(
        questions,
        out,
        verifier_image=args.verifier_image,
        pipeline_id=pipeline,
    )

    print(f"Generated {len(questions)} task dirs → {out.resolve()}")


if __name__ == "__main__":
    main()
