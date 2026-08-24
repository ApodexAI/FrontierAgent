"""BrowseComp-ZH (Chinese)."""

from __future__ import annotations

from benchmarks.public.core.registry import DatasetConfig
from benchmarks.public.families._schema import STD_SCHEMA

CONFIGS: list[tuple[str, DatasetConfig]] = [
    ("browsecomp_zh", DatasetConfig(
        name="BrowseComp-ZH", key="BrowseComp-ZH",
        default_pipeline="stateful-react-agent", **STD_SCHEMA,
    )),
]
