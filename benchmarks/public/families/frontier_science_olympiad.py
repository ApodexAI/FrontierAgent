"""FrontierScience-Olympiad (answer-match science questions)."""

from __future__ import annotations

from benchmarks.public.core.registry import DatasetConfig
from benchmarks.public.families._schema import STD_SCHEMA

CONFIGS: list[tuple[str, DatasetConfig]] = [
    ("frontier_science_olympiad", DatasetConfig(
        name="FrontierScience-Olympiad", key="FrontierScience-Olympiad",
        default_pipeline="stateful-react-agent", **STD_SCHEMA,
    )),
]
