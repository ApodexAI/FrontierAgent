"""Publish grader diagnostics without retaining alternate pass decisions."""
from __future__ import annotations

import json
import re
from pathlib import Path

PASS_KEYS = {"pass", "passed", "native_passed", "pass_threshold"}
DECISION_WORD = re.compile(r"\b(?:pass|passed|fail|native_passed|pass_threshold)\b", re.IGNORECASE)


def clean_json(value):
    if isinstance(value, dict):
        return {key: clean_json(item) for key, item in value.items()
                if key.casefold() not in PASS_KEYS}
    if isinstance(value, list):
        return [clean_json(item) for item in value]
    if isinstance(value, str) and value.strip().casefold() in {"pass", "passed", "fail"}:
        return "[decision omitted]"
    return value


def clean_text(text: str) -> str:
    # Structured grader stdout retains scores and diagnostics, minus decisions.
    try:
        return json.dumps(clean_json(json.loads(text)), indent=2) + "\n"
    except json.JSONDecodeError:
        return "".join(
            "[decision omitted]\n" if DECISION_WORD.search(line) else line
            for line in text.splitlines(keepends=True)
        )


def publish_logs(private: Path, public: Path) -> None:
    """Raw grader files never enter the Harbor log directory.

    reward.json already uses the sole benchmark pass rule and is preserved.
    Other JSON/text files are diagnostic, not an additional scoring interface.
    """
    public.mkdir(parents=True, exist_ok=True)
    for source in sorted(private.iterdir()):
        if not source.is_file() or source.is_symlink():
            raise ValueError(f"unexpected verifier log entry: {source.name}")
        text = source.read_text(encoding="utf-8")
        if source.name == "reward.json":
            cleaned = text
        elif source.suffix == ".json":
            cleaned = json.dumps(clean_json(json.loads(text)), indent=2) + "\n"
        elif source.suffix == ".txt":
            cleaned = clean_text(text)
        else:
            raise ValueError(f"unsupported verifier log type: {source.name}")
        (public / source.name).write_text(cleaned, encoding="utf-8")
