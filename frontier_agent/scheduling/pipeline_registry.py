"""PipelineRegistry — stores and retrieves PipelineSpec objects."""

from __future__ import annotations

import logging

from frontier_agent.models.pipeline_spec import PipelineSpec

logger = logging.getLogger(__name__)


class PipelineRegistry:
    """Central registry for pipeline specifications."""

    def __init__(self) -> None:
        self._pipelines: dict[str, PipelineSpec] = {}

    def register(self, spec: PipelineSpec) -> None:
        self._pipelines[spec.pipeline_id] = spec
        logger.info("Registered pipeline: %s (%s)", spec.pipeline_id, spec.name)

    def get(self, pipeline_id: str) -> PipelineSpec:
        if pipeline_id not in self._pipelines:
            available = list(self._pipelines.keys())
            raise KeyError(f"Pipeline '{pipeline_id}' not registered. Available: {available}")
        return self._pipelines[pipeline_id]

    def list_all(self) -> list[PipelineSpec]:
        return list(self._pipelines.values())

    def has(self, pipeline_id: str) -> bool:
        return pipeline_id in self._pipelines
