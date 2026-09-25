"""Product composition adapter for AgentCore's ResourceManager."""

from __future__ import annotations

from collections.abc import Callable

from agent_core.llm import LLMClient
from agent_core.models.agent_definition import AgentDefinition
from agent_core.runtime.resources.manager import ResourceManager as _ResourceManager
from agent_core.tool import Tool

from frontier_agent.core.runtime.resources.llm import create_role_llm


class ResourceManager(_ResourceManager):
    """Default per-role LLMs to FrontierAgent's provider configuration."""

    def __init__(
        self,
        llm: LLMClient,
        tools: dict[str, Tool],
        *,
        role_llm_factory: Callable[[AgentDefinition], LLMClient] | None = None,
    ) -> None:
        super().__init__(llm, tools, role_llm_factory=role_llm_factory or create_role_llm)


__all__ = ["ResourceManager"]
