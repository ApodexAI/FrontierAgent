"""Set the sole pass rule in an evaluator-owned, freshly unsealed task.

The encrypted reference and partial-credit rubric are not modified. Apply this
after archive verification/unsealing and before Harbor reads the verifier.
Fail closed if a future reference changes the known reward adapter contract.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

OLD_RULE = '"passed": 1.0 if passed is True else 0.0,'
PASS_RULE = (
    '"passed": 1.0 if complete and 0.999 < float(score or 0.0) / 100.0 <= 1.0 else 0.0,'
)
ENTRYPOINT = '\nif __name__ == "__main__":'
LOG_WRAPPER = '''
def main():
    # Native decisions exist only in a private temporary directory. Publish
    # sanitized diagnostics on success and failure, never a second pass metric.
    import tempfile
    from verifier_log_policy import publish_logs
    global LOGS
    public_logs = LOGS
    with tempfile.TemporaryDirectory(prefix="frontier-native-verifier-") as scratch:
        LOGS = Path(scratch)
        try:
            _run_native_verifier()
        finally:
            try:
                publish_logs(LOGS, public_logs)
            finally:
                LOGS = public_logs

'''


def apply_policy(task_dir: Path) -> None:
    adapter = task_dir / "tests" / "run_frontier_verifier.py"
    text = adapter.read_text(encoding="utf-8")
    if (text.count(OLD_RULE) != 1 or PASS_RULE in text
            or text.count("def main():") != 1 or text.count(ENTRYPOINT) != 1):
        raise ValueError(f"unsupported or already modified reward adapter: {adapter}")
    updated = text.replace(OLD_RULE, PASS_RULE).replace("def main():", "def _run_native_verifier():")
    updated = updated.replace(ENTRYPOINT, LOG_WRAPPER + ENTRYPOINT)
    compile(updated, str(adapter), "exec")
    shutil.copyfile(Path(__file__).with_name("verifier_log_policy.py"),
                    adapter.with_name("verifier_log_policy.py"))
    adapter.write_text(updated, encoding="utf-8")


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
