"""Tool-permission helpers for ResourceManager."""

from __future__ import annotations

import logging

from frontier_agent.core.errors import ServiceNotRegistered
from frontier_agent.core.runtime.registries import services as service_registry
from frontier_agent.core.runtime.resources.tool_permission import ToolPermissionContext

logger = logging.getLogger(__name__)


def get_allowed_tools_for_role(role_id: str) -> set[str]:
    """Return the role's allowed tool names from AgentRegistry.

    Fail-closed: unknown role or unavailable registry returns an empty set.
    """
    from frontier_agent.core.runtime.registries.agents import AgentRegistry

    try:
        agent_reg = service_registry.get(AgentRegistry)
        return set(agent_reg.get_tools_for(role_id))
    except (KeyError, RuntimeError, ServiceNotRegistered):
        return set()


def get_effective_tools_for_role(
    *,
    role_id: str,
    permission_context: ToolPermissionContext | None = None,
) -> set[str]:
    """Return the effective tool set for a role under the active policy."""
    role_tools = get_allowed_tools_for_role(role_id)
    if permission_context is None:
        return role_tools
    effective = permission_context.filter(role_tools)
    if effective != role_tools:
        blocked = role_tools - effective
        if blocked:
            logger.debug(
                "Tool policy blocked tools for role '%s': %s",
                role_id, blocked,
            )
    return effective
