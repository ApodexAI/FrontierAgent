"""xbench-DeepResearch (2025-10 snapshot)."""

from __future__ import annotations

from benchmarks.public.core.registry import DatasetConfig
from benchmarks.public.families._schema import STD_SCHEMA

CONFIGS: list[tuple[str, DatasetConfig]] = [
    ("xbench_dr_202510", DatasetConfig(
        name="XBench-DeepResearch-202510", key="XBench-DeepResearch-202510",
        default_pipeline="stateful-react-agent", **STD_SCHEMA,
    )),
]
