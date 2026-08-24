"""OneMillion-Bench weighted-rubric benchmark."""

from benchmarks.public.core.registry import DatasetConfig
from benchmarks.public.families._schema import STD_SCHEMA

CONFIGS = [
    ("onemillion_bench", DatasetConfig(
        name="OneMillion-Bench", key="OneMillion-Bench", default_pipeline="agent_team",
        extra_metadata_fields=("language", "economic_value"), **STD_SCHEMA,
    )),
]
