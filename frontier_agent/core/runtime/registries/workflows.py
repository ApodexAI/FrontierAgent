"""WorkflowContext — registration interface for external workflow plugins."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from frontier_agent.core.runtime.registries.agents import AgentRegistry
    from frontier_agent.models.agent_definition import AgentDefinition
    from frontier_agent.models.pipeline_spec import PipelineSpec
    from frontier_agent.scheduling.pipeline_registry import PipelineRegistry
    from frontier_agent.scheduling.topology_registry import (
        TopologyFactory,
        TopologyRegistry,
    )

logger = logging.getLogger(__name__)


class WorkflowContext:
    """Safe registration facade passed to workflow ``register()`` hooks."""

    def __init__(
        self,
        pipeline_registry: PipelineRegistry,
        agent_registry: AgentRegistry,
        topology_registry: TopologyRegistry | None = None,
    ) -> None:
        self._pipelines = pipeline_registry
        self._agents = agent_registry
        self._topologies = topology_registry

    # -- Pipeline registration ------------------------------------------------

    def register_pipeline(self, spec: PipelineSpec) -> None:
        """Register a PipelineSpec so it can be selected at runtime."""
        if self._pipelines.has(spec.pipeline_id):
            raise ValueError(
                f"Pipeline '{spec.pipeline_id}' is already registered; "
                "plugin registration cannot override existing pipelines"
            )
        self._pipelines.register(spec)
        logger.info(
            "Workflow plugin registered pipeline: %s", spec.pipeline_id,
        )

    # -- Agent registration ---------------------------------------------------

    def register_agent(self, definition: AgentDefinition) -> None:
        """Register a custom AgentDefinition (role, prompt, tools)."""
        self._agents.register(definition)

    def register_agents(self, definitions: list[AgentDefinition]) -> None:
        """Convenience: register multiple AgentDefinitions at once."""
        for defn in definitions:
            self.register_agent(defn)

    # -- Topology registration -----------------------------------------------

    def register_topology(self, name: str, factory: TopologyFactory) -> None:
        """Register a topology factory — ``(options, role_tiers) -> PipelineSpec``.

        Noops if no ``TopologyRegistry`` was provided (unit-test paths that
        don't wire the full runtime). Workflows relying on dynamic topology
        dispatch should pass one in from bootstrap.
        """
        if self._topologies is None:
            logger.debug(
                "register_topology(%s) skipped — no TopologyRegistry wired",
                name,
            )
            return
        self._topologies.register(name, factory)
        logger.info("Workflow plugin registered topology factory: %s", name)

    # -- Introspection (read-only) --------------------------------------------

    def has_pipeline(self, pipeline_id: str) -> bool:
        return self._pipelines.has(pipeline_id)

    def has_agent(self, role_id: str) -> bool:
        return self._agents.has(role_id)
