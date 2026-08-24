#!/usr/bin/env python3
"""Export one FrontierSearchBench run to the official unified JSON contract."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_BENCH_ROOT = Path(__file__).resolve().parents[2] / "frontier_search_bench"
_QUERIES = _BENCH_ROOT / "queries" / "verifiable.json"


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_trials_dir(run_dir: Path) -> Path:
    if (run_dir / "trials").is_dir():
        return run_dir / "trials"
    runs = sorted(path for path in run_dir.glob("run_*") if (path / "trials").is_dir())
    if len(runs) == 1:
        return runs[0] / "trials"
    if len(runs) > 1:
        names = ", ".join(path.name for path in runs)
        raise ValueError(
            f"{run_dir} contains multiple runs ({names}); export one run_N directory at a time"
        )
    raise FileNotFoundError(f"No trials/ directory found under {run_dir}")


def export_run(run_dir: Path, output_path: Path) -> dict[str, Any]:
    """Write answered trials in canonical query order and return a summary."""
    run_dir = run_dir.resolve()
    trials_dir = _resolve_trials_dir(run_dir)
    canonical = _load_json(_QUERIES)
    if not isinstance(canonical, list):
        raise ValueError(f"Canonical query bank is not a JSON list: {_QUERIES}")

    results: dict[int, dict[str, Any]] = {}
    invalid_results: list[str] = []
    for result_path in sorted(trials_dir.glob("*/result.json")):
        try:
            result = _load_json(result_path)
            qid = int(result.get("question_id", result_path.parent.name))
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            invalid_results.append(str(result_path))
            continue
        if qid in results:
            raise ValueError(f"Duplicate FrontierSearchBench result id: {qid}")
        results[qid] = result

    exported: list[dict[str, Any]] = []
    missing_ids: list[int] = []
    empty_ids: list[int] = []
    canonical_ids = {int(item["id"]) for item in canonical}
    unknown_ids = sorted(set(results) - canonical_ids)

    for item in canonical:
        qid = int(item["id"])
        result = results.get(qid)
        if result is None:
            missing_ids.append(qid)
            continue
        answer = str(result.get("predicted_answer") or "").strip()
        if not answer:
            empty_ids.append(qid)
        exported.append(
            {
                "id": qid,
                "query": item["query"],
                "report_content": answer,
                "response": "",
            }
        )

    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(exported, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return {
        "output": str(output_path),
        "exported": len(exported),
        "canonical": len(canonical),
        "missing_ids": missing_ids,
        "empty_ids": empty_ids,
        "unknown_ids": unknown_ids,
        "invalid_results": invalid_results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="Run directory containing trials/")
    parser.add_argument("--out", type=Path, required=True, help="Unified JSON output path")
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Exit 0 even when the run is missing canonical queries. The "
             "official scorer averages only over answered queries, so a "
             "partial export produces a score that is not comparable.",
    )
    args = parser.parse_args()

    summary = export_run(args.run_dir, args.out)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if summary["invalid_results"] or summary["unknown_ids"]:
        raise SystemExit(2)
    if summary["missing_ids"] and not args.allow_partial:
        # run_all.build_ranking computes mean(total_rate) over the queries the
        # model actually answered, so exporting 12 of 41 yields an inflated
        # headline score. Fail loudly; --allow-partial opts in deliberately.
        print(
            f"ERROR: {len(summary['missing_ids'])} of {summary['canonical']} "
            f"canonical queries have no result; the official score would be "
            f"averaged over answered queries only. Finish the run, or pass "
            f"--allow-partial to export anyway.",
            file=sys.stderr,
        )
        raise SystemExit(3)


if __name__ == "__main__":
    main()
