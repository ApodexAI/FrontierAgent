"""AgentRegistry — dynamic agent role registration and lookup.

Replaces the hardcoded AgentRole enum + _TOOL_PERMISSIONS dict.
New agent roles can be registered at any time without modifying kernel code.
"""

from __future__ import annotations

import logging

from frontier_agent.models.agent_definition import AgentDefinition

logger = logging.getLogger(__name__)


class AgentRegistry:
    """Central registry for all agent definitions.

    Usage:
        registry = AgentRegistry()
        registry.register(AgentDefinition(role_id="researcher", ...))
        defn = registry.get("researcher")
        prompt = registry.get_prompt_for("researcher")
        tools = registry.get_tools_for("researcher")
    """

    def __init__(self) -> None:
        self._agents: dict[str, AgentDefinition] = {}

    def register(self, definition: AgentDefinition) -> None:
        """Register a new agent definition. Overwrites if role_id exists."""
        self._agents[definition.role_id] = definition
        logger.info("Registered agent: %s (%s)", definition.role_id, definition.display_name)

    def get(self, role_id: str) -> AgentDefinition:
        """Get agent definition by role_id. Raises KeyError if not found."""
        if role_id not in self._agents:
            raise KeyError(f"Agent role '{role_id}' not registered. Available: {list(self._agents.keys())}")
        return self._agents[role_id]

    def list_all(self) -> list[AgentDefinition]:
        """List all registered agent definitions."""
        return list(self._agents.values())

    def get_tools_for(self, role_id: str) -> list[str]:
        """Get allowed tool names for a role."""
        return self.get(role_id).allowed_tools

    def get_prompt_for(self, role_id: str) -> str:
        """Get system prompt for a role."""
        return self.get(role_id).system_prompt

    def has(self, role_id: str) -> bool:
        """Check if a role is registered."""
        return role_id in self._agents
