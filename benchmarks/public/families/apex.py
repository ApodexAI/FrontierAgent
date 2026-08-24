"""APEX file-and-app-state benchmark."""

from benchmarks.public.core.registry import DatasetConfig
from benchmarks.public.families._schema import STD_SCHEMA

CONFIGS = [
    ("apex", DatasetConfig(
        name="APEX", key="APEX", default_pipeline="stateful-react-agent",
        extra_metadata_fields=("world_id", "domain"), **STD_SCHEMA,
    )),
]
