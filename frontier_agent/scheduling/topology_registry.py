"""Topology factory registry — workflow contributes, kernel dispatches."""

from __future__ import annotations

import logging
from collections.abc import Callable

from frontier_agent.models.pipeline_spec import PipelineSpec

logger = logging.getLogger(__name__)


TopologyFactory = Callable[[dict, dict], PipelineSpec]


class TopologyRegistry:
    """Name → topology factory lookup.

    Factory signature: ``(options: dict, role_tiers: dict) -> PipelineSpec``.
    ``options`` carries topology-specific inputs (e.g., ``role_id`` for
    solo, ``n_rounds`` / ``n_agents`` for debate); ``role_tiers`` maps
    role names to model tier strings pulled from the active budget.
    """

    def __init__(self) -> None:
        self._factories: dict[str, TopologyFactory] = {}

    def register(self, name: str, factory: TopologyFactory) -> None:
        if name in self._factories:
            logger.warning("Topology factory %r already registered — overwriting", name)
        self._factories[name] = factory

    def has(self, name: str) -> bool:
        return name in self._factories

    def build(
        self,
        name: str,
        options: dict | None = None,
        role_tiers: dict | None = None,
    ) -> PipelineSpec:
        try:
            factory = self._factories[name]
        except KeyError as exc:
            raise KeyError(
                f"No topology factory registered for {name!r}. "
                "Did the owning workflow register it?"
            ) from exc
        return factory(options or {}, role_tiers or {})
