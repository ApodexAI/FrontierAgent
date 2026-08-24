"""WideSearch — structural F1 scorer (not LLM judge).

``ground_truth`` is a JSON blob carrying the eval_spec + gold_table; the
structural scorer parses it at scoring time.
"""

from __future__ import annotations

from benchmarks.public.core.registry import DatasetConfig
from benchmarks.public.families._schema import STD_SCHEMA

CONFIGS: list[tuple[str, DatasetConfig]] = [
    ("widesearch", DatasetConfig(
        name="WideSearch", key="WideSearch",
        default_pipeline="stateful-react-agent", **STD_SCHEMA,
    )),
]
