"""FrontierSearchBench — bundled queries with an external batch scorer."""

from __future__ import annotations

from pathlib import Path

from benchmarks.public.core.registry import DatasetConfig

_FRONTIER_SEARCH_ROOT = Path(__file__).resolve().parents[2] / "frontier_search_bench"

CONFIGS: list[tuple[str, DatasetConfig]] = [
    (
        "frontier_search",
        DatasetConfig(
            name="FrontierSearchBench",
            key="queries",
            jsonl="verifiable.json",
            data_root=str(_FRONTIER_SEARCH_ROOT),
            id_field="id",
            question_field="query",
            answer_field="",
            source_format="json",
            scoring_mode="external",
            default_answer_type="report",
            default_pipeline="stateful-react-agent",
        ),
    ),
]
