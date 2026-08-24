"""Registries — service DI container, agent definitions, workflow context."""

from frontier_agent.core.runtime.registries.agents import AgentRegistry
from frontier_agent.core.runtime.registries.services import (
    clear,
    get,
    get_optional,
    is_registered,
    register,
)
from frontier_agent.core.runtime.registries.workflows import WorkflowContext

__all__ = [
    "AgentRegistry",
    "WorkflowContext",
    "clear",
    "get",
    "get_optional",
    "is_registered",
    "register",
]
