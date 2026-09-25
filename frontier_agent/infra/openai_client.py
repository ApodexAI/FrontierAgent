"""FrontierAgent composition of AgentCore's OpenAI Chat Completions client.

Importing this module wires FrontierAgent's session-affinity policy into the
shared provider. Every ``OpenAIClient`` in the product must be constructed
through this module; building ``agent_core.providers.openai_chat.OpenAIClient``
directly before this import silently drops the session id.
"""

from __future__ import annotations

from typing import Any

import agent_core.providers.openai_chat as _implementation

from frontier_agent.infra.session_context import get_task_session_id, mirror_session_query

_implementation.configure_session_query_resolver(mirror_session_query)
_implementation.configure_session_scope_resolver(get_task_session_id)


class OpenAIClient(_implementation.OpenAIClient):
    """Treat an empty ``api_key`` as unspecified.

    The OpenAI SDK only consults ``OPENAI_API_KEY`` when ``api_key`` is None;
    a cached config can legitimately hold the pre-environment empty string.
    """

    def __init__(self, model: str, *, api_key: str | None = None, **kwargs: Any) -> None:
        super().__init__(model, api_key=api_key or None, **kwargs)


__all__ = ["OpenAIClient"]
