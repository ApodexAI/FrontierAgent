"""SUPERChem-Text — single-letter MCQ (A-J), graded with the HLE judge schema."""

from __future__ import annotations

from benchmarks.public.core.registry import DatasetConfig
from benchmarks.public.families._schema import STD_SCHEMA

CONFIGS: list[tuple[str, DatasetConfig]] = [
    ("superchem_text", DatasetConfig(
        name="SUPERChem-Text", key="SUPERChem-Text",
        default_pipeline="stateful-react-agent", **STD_SCHEMA,
    )),
]
