"""OfficeQA Treasury-document benchmarks."""

from benchmarks.public.core.registry import DatasetConfig
from benchmarks.public.families._schema import STD_SCHEMA

CONFIGS = [
    ("officeqa", DatasetConfig(
        name="OfficeQA", key="OfficeQA", default_pipeline="stateful-react-agent",
        extra_metadata_fields=("source_files", "difficulty"), **STD_SCHEMA,
    )),
    ("officeqa_full", DatasetConfig(
        name="OfficeQA-Full", key="OfficeQA", jsonl="standardized_full.jsonl",
        default_pipeline="stateful-react-agent",
        extra_metadata_fields=("source_files", "difficulty"), **STD_SCHEMA,
    )),
]
