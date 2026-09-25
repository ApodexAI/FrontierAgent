"""Product LLM construction adapter for AgentCore resources."""

from __future__ import annotations

from agent_core.llm import LLMClient
from agent_core.models.agent_definition import AgentDefinition
from agent_core.runtime.resources.llm import (
    resolve_base_llm_for_role as _resolve_base_llm_for_role,
)
from agent_core.runtime.resources.llm import (
    wrap_llm_with_middleware,
)


def create_role_llm(defn: AgentDefinition) -> LLMClient:
    from frontier_agent.infra.config import get_config
    from frontier_agent.infra.llm_adapter import create_llm_with_overrides

    return create_llm_with_overrides(
        get_config(),
        model=defn.model,
        temperature=defn.temperature,
        max_tokens=defn.max_tokens,
    )


def resolve_base_llm_for_role(
    *,
    default_llm: LLMClient,
    role_id: str | None,
    cache: dict[str, LLMClient],
) -> LLMClient:
    return _resolve_base_llm_for_role(
        default_llm=default_llm,
        role_id=role_id,
        cache=cache,
        role_llm_factory=create_role_llm,
    )


__all__ = [
    "create_role_llm",
    "resolve_base_llm_for_role",
    "wrap_llm_with_middleware",
]
