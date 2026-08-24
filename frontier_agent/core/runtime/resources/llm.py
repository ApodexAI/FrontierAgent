"""LLM-resolution helpers for ResourceManager."""

from __future__ import annotations

from frontier_agent.core.llm import LLMClient
from frontier_agent.core.protocols import LLMWrapper
from frontier_agent.core.runtime.registries import services as service_registry


def resolve_base_llm_for_role(
    *,
    default_llm: LLMClient,
    role_id: str | None,
    cache: dict[str, LLMClient],
) -> LLMClient:
    """Resolve the base LLM for a role without middleware wrapping."""
    if role_id is None:
        return default_llm

    if role_id in cache:
        return cache[role_id]

    from frontier_agent.core.runtime.registries.agents import AgentRegistry

    try:
        agent_reg = service_registry.get(AgentRegistry)
        defn = agent_reg.get(role_id)
        if defn.model:
            from frontier_agent.infra.config import get_config
            from frontier_agent.infra.llm_adapter import create_llm_with_overrides

            role_llm = create_llm_with_overrides(
                get_config(),
                model=defn.model,
                temperature=defn.temperature,
                max_tokens=defn.max_tokens,
            )
            cache[role_id] = role_llm
            return role_llm
    except (KeyError, RuntimeError, ImportError):
        pass

    return default_llm


def wrap_llm_with_middleware(
    llm: LLMClient,
    *,
    role_id: str,
) -> LLMClient:
    """Wrap the LLM with an optional LLM wrapper service."""
    try:
        wrapper = service_registry.get_optional(LLMWrapper)
        if wrapper is not None:
            return wrapper.wrap_llm(llm, role_id=role_id)
    except (ImportError, RuntimeError):
        pass
    return llm
