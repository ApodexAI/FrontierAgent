"""GDPval deliverable benchmark."""

from benchmarks.public.core.registry import DatasetConfig
from benchmarks.public.families._schema import STD_SCHEMA

CONFIGS = [
    ("gdpval", DatasetConfig(
        name="GDPval", key="GDPval", default_pipeline="stateful-react-agent",
        extra_metadata_fields=("reference_files",), **STD_SCHEMA,
    )),
]
