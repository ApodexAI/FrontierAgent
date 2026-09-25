"""Registries — service DI container, agent definitions, workflow context."""

from frontier_agent.core.runtime.registries.agents import AgentRegistry
from frontier_agent.core.runtime.registries.scope import (
    ServiceScope,
    current_scope,
    use_scope,
)
from frontier_agent.core.runtime.registries.services import (
    clear,
    get,
    get_local,
    get_optional,
    is_registered,
    is_registered_local,
    register,
)
from frontier_agent.core.runtime.registries.workflows import WorkflowContext

__all__ = [
    "AgentRegistry",
    "ServiceScope",
    "WorkflowContext",
    "clear",
    "current_scope",
    "get",
    "get_local",
    "get_optional",
    "is_registered",
    "is_registered_local",
    "register",
    "use_scope",
]
