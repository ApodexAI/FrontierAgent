"""FrontierAgent adapter for AgentCore LLM binding helpers."""

from __future__ import annotations

from typing import Any

from agent_core.runtime.loop._bind import (
    _BoundLLM as _BoundLLM,
)
from agent_core.runtime.loop._bind import (
    bind_max_tokens,
    bind_temperature,
    bind_tools,
)
from agent_core.runtime.loop._bind import (
    bind_session_id as _bind_session_id,
)

from frontier_agent.infra.session_context import sticky_session_enabled


def bind_session_id(llm: Any, task_id: str) -> Any:
    """Bind product session affinity using AgentCore's portable helper."""

    return _bind_session_id(
        llm,
        task_id,
        sticky_session_enabled=sticky_session_enabled,
    )


__all__ = [
    "_BoundLLM",
    "bind_max_tokens",
    "bind_session_id",
    "bind_temperature",
    "bind_tools",
]
