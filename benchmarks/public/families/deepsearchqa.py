"""DeepSearchQA — answer_type column carries DSQA prompt_type ("Single Answer" / "Set Answer")."""

from __future__ import annotations

from benchmarks.public.core.registry import DatasetConfig
from benchmarks.public.families._schema import STD_SCHEMA

CONFIGS: list[tuple[str, DatasetConfig]] = [
    ("deepsearchqa", DatasetConfig(
        name="DeepSearchQA", key="DeepSearchQA",
        default_pipeline="stateful-react-agent", **STD_SCHEMA,
    )),
]
