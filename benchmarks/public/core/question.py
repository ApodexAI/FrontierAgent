"""Single benchmark question data structure."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class BenchmarkQuestion:
    """A single benchmark question."""

    id: str
    question: str
    ground_truth: str
    answer_type: str  # e.g. "exactMatch" | "multipleChoice" | "report"
    image_path: str | None = None
    file_path: str | None = None
    file_name: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
