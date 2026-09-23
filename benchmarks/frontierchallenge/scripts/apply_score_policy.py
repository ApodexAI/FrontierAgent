"""Set the sole pass rule in an evaluator-owned, freshly unsealed task.

The encrypted reference and partial-credit rubric are not modified. Apply this
after archive verification/unsealing and before Harbor reads the verifier.
Fail closed if a future reference changes the known reward adapter contract.
"""
from __future__ import annotations

import argparse
from pathlib import Path

OLD_RULE = '"passed": 1.0 if passed is True else 0.0,'
FULL_SCORE_RULE = (
    '"passed": 1.0 if complete and float(score or 0.0) / 100.0 == 1.0 else 0.0,'
)


def apply_policy(task_dir: Path) -> None:
    adapter = task_dir / "tests" / "run_frontier_verifier.py"
    text = adapter.read_text(encoding="utf-8")
    if text.count(OLD_RULE) != 1 or FULL_SCORE_RULE in text:
        raise ValueError(f"unsupported or already modified reward adapter: {adapter}")
    adapter.write_text(text.replace(OLD_RULE, FULL_SCORE_RULE), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task_dir", type=Path)
    args = parser.parse_args()
    try:
        apply_policy(args.task_dir)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
