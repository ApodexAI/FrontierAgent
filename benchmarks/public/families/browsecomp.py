"""BrowseComp (English)."""

from __future__ import annotations

from benchmarks.public.core.registry import DatasetConfig
from benchmarks.public.families._schema import STD_SCHEMA

CONFIGS: list[tuple[str, DatasetConfig]] = [
    ("browsecomp", DatasetConfig(
        name="BrowseComp", key="BrowseComp",
        default_pipeline="stateful-react-agent", **STD_SCHEMA,
    )),
]
