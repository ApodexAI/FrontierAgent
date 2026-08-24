"""HLE-text — text-only subset of Humanity's Last Exam (no image questions)."""

from __future__ import annotations

from benchmarks.public.core.registry import DatasetConfig
from benchmarks.public.families._schema import STD_SCHEMA

CONFIGS: list[tuple[str, DatasetConfig]] = [
    ("hle_text", DatasetConfig(
        name="HLE-text", key="HLE-text",
        default_pipeline="stateful-react-agent", **STD_SCHEMA,
    )),
]
