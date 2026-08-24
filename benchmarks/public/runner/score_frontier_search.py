#!/usr/bin/env python3
"""Run FrontierSearchBench's official scorer without writing into its source tree."""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

_SOURCE_ROOT = Path(__file__).resolve().parents[2] / "frontier_search_bench"
_DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"


def _scorer_env() -> dict[str, str]:
    """Resolve the scorer's OPENROUTER_* credentials from one provider.

    Key and base URL are taken from the same source. Mixing them — an
    OPENROUTER_API_KEY already in the environment paired with a JUDGE_BASE_URL
    pointing at an unrelated gateway — sends the key to the wrong host and
    every extraction call 401s, long after answer collection was paid for.
    """
    env = dict(os.environ)
    if env.get("OPENROUTER_API_KEY"):
        # Explicit OPENROUTER_* wins; only fill in the URL from its own family.
        env.setdefault("OPENROUTER_BASE_URL", _DEFAULT_BASE_URL)
        if not env["OPENROUTER_BASE_URL"]:
            env["OPENROUTER_BASE_URL"] = _DEFAULT_BASE_URL
        return env

    for key_var, url_var in (("JUDGE_API_KEY", "JUDGE_BASE_URL"),
                             ("OPENAI_API_KEY", "OPENAI_BASE_URL")):
        if env.get(key_var):
            env["OPENROUTER_API_KEY"] = env[key_var]
            env["OPENROUTER_BASE_URL"] = env.get(url_var) or _DEFAULT_BASE_URL
            return env

    # No credentials anywhere. Leave the key empty so the upstream scorer's
    # own "OPENROUTER_API_KEY not set" check reports it.
    env["OPENROUTER_API_KEY"] = ""
    env.setdefault("OPENROUTER_BASE_URL", _DEFAULT_BASE_URL)
    return env


def _seed_previous_scores(output_dir: Path, eval_root: Path) -> int:
    """Copy previously exported per-query scores back into the working tree.

    The scorer skips a query whose ``auto_scores/scores.json`` already covers
    every target model. Without this the temp tree is always pristine, so a
    rerun after a mid-run failure re-pays for every query that already
    succeeded and ``--force-rerun`` has nothing to override.
    """
    per_query_root = output_dir / "per_query"
    if not per_query_root.is_dir():
        return 0
    seeded = 0
    for source in sorted(per_query_root.glob("query_*")):
        destination = eval_root / "scorers" / source.name / "auto_scores"
        if not destination.parent.is_dir():
            continue  # per_query entry with no matching scorer — ignore
        if destination.exists():
            shutil.rmtree(destination)
        shutil.copytree(source, destination)
        seeded += 1
    return seeded


def run_official_scorer(args: argparse.Namespace) -> int:
    output_dir = args.out.resolve()
    model_specs: list[str] = []
    for spec in args.models:
        if "=" not in spec:
            raise ValueError(f"Model input must be name=path, got {spec!r}")
        name, raw_path = spec.split("=", 1)
        if not name or not raw_path:
            raise ValueError(f"Model input must be name=path, got {spec!r}")
        model_specs.append(f"{name}={Path(raw_path).expanduser().resolve()}")

    if os.environ.get("JUDGE_MODEL"):
        # The bundled scorer pins its own model slate (primary / secondary /
        # parallel / analyzer) in extraction_pipeline.DEFAULTS and reads no
        # model name from the environment. Say so rather than let the operator
        # assume JUDGE_MODEL took effect.
        logger.warning(
            "JUDGE_MODEL is set but FrontierSearchBench's scorer pins its own "
            "model slate; only JUDGE_API_KEY / JUDGE_BASE_URL are used here."
        )

    with tempfile.TemporaryDirectory(prefix="frontier-search-scorer-") as tmp:
        work_root = Path(tmp) / "frontier_search_bench"
        shutil.copytree(_SOURCE_ROOT, work_root)
        eval_root = work_root / "eval" / "verifiable"
        if not args.dry_run and not args.force_rerun:
            seeded = _seed_previous_scores(output_dir, eval_root)
            if seeded:
                logger.info(
                    "Reusing %d previously scored quer%s from %s/per_query",
                    seeded, "y" if seeded == 1 else "ies", output_dir,
                )
        command = [
            sys.executable,
            str(eval_root / "run_all.py"),
            "--models",
            *model_specs,
            "--out",
            str(output_dir),
            "--per-query-timeout",
            str(args.per_query_timeout),
        ]
        if args.only:
            command.extend(["--only", args.only])
        if args.force:
            command.append("--force")
        if args.force_rerun:
            command.append("--force-rerun")
        if args.allow_query_mismatch:
            command.append("--allow-query-mismatch")
        if args.dry_run:
            command.append("--dry-run")

        completed = subprocess.run(command, cwd=eval_root, env=_scorer_env(), check=False)

        if not args.dry_run:
            per_query_root = output_dir / "per_query"
            for scores_dir in sorted(eval_root.glob("scorers/query_*/auto_scores")):
                destination = per_query_root / scores_dir.parent.name
                if destination.exists():
                    shutil.rmtree(destination)
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copytree(scores_dir, destination)
        return completed.returncode


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", required=True, metavar="NAME=JSON")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--only", default="")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--force-rerun", action="store_true")
    parser.add_argument("--allow-query-mismatch", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--per-query-timeout", type=int, default=3600)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")
    raise SystemExit(run_official_scorer(args))


if __name__ == "__main__":
    main()
