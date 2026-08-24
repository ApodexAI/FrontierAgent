"""Anthropic prompt-cache adapter — opt-in cache_control injection."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from typing import Any

from frontier_agent.core.llm import LLMResponse, StreamDelta
from frontier_agent.core.messages import Message

logger = logging.getLogger(__name__)


class AnthropicPromptCacheAdapter:
    """:class:`~frontier_agent.core.llm.LLMClient` wrapper that converts the
    leading ``system`` message ``{"content": str}`` into a single
    content-block with ``cache_control: ephemeral`` before delegating to
    the inner Claude client. See module docstring.
    """

    def __init__(
        self, inner: Any, *, bound_tools: list[dict[str, Any]] | None = None,
    ) -> None:
        self.inner = inner
        self._bound_tools = bound_tools

    @property
    def model(self) -> str:
        return getattr(self.inner, "model", "") or ""

    def bind_tools(self, tools: Any) -> AnthropicPromptCacheAdapter:
        """Bind a default ``tools=`` payload, keeping the cache adapter on
        the *outside* so the tool-augmented call still goes out with the
        cache-marked system prompt.

        Native ``LLMClient`` carries no langchain ``RunnableBinding`` —
        tools are simply threaded through the per-call ``tools=`` kwarg.
        We therefore carry the bound payload here and merge it into
        :meth:`chat` / :meth:`stream` (caller-supplied ``tools`` win, to
        match native ``chat(..., tools=...)`` precedence). Returns a fresh
        adapter; the shared inner client is never mutated.
        """
        schemas = [
            t.to_openai_schema() if hasattr(t, "to_openai_schema") else t
            for t in tools
        ]
        return AnthropicPromptCacheAdapter(inner=self.inner, bound_tools=schemas)

    async def chat(
        self,
        messages: list[Message],
        *,
        tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        extra_headers: dict[str, str] | None = None,
        timeout: float | None = None,
    ) -> LLMResponse:
        mutated = _inject_cache_control(messages)
        return await self.inner.chat(
            mutated,
            tools=tools if tools is not None else self._bound_tools,
            temperature=temperature,
            max_tokens=max_tokens,
            extra_headers=extra_headers,
            timeout=timeout,
        )

    def stream(
        self,
        messages: list[Message],
        *,
        tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        extra_headers: dict[str, str] | None = None,
        timeout: float | None = None,
    ) -> AsyncIterator[StreamDelta]:
        mutated = _inject_cache_control(messages)
        return self.inner.stream(
            mutated,
            tools=tools if tools is not None else self._bound_tools,
            temperature=temperature,
            max_tokens=max_tokens,
            extra_headers=extra_headers,
            timeout=timeout,
        )


def _inject_cache_control(
    messages: list[Message],
) -> list[Message]:
    """Mutate the first ``system`` message so its content carries
    ``cache_control: ephemeral``. Returns a NEW list — original
    messages are untouched (callers may reuse the list for retries
    where the unmutated form needs to remain intact)."""
    out: list[Message] = []
    sys_marked = False
    for msg in messages:
        content = msg.get("content")
        if (
            not sys_marked
            and msg.get("role") == "system"
            and isinstance(content, str)
            and content
        ):
            new_msg: Message = {
                "content": [{
                    "type": "text",
                    "text": content,
                    "cache_control": {"type": "ephemeral"},
                }],
                "role": "system",
            }
            out.append(new_msg)
            sys_marked = True
        else:
            out.append(msg)
    return out


def _is_claude_family(provider: str, model: str) -> bool:
    """Return True when the upstream is Anthropic-Messages-API-shaped
    (Anthropic direct, OpenRouter forwarding to Anthropic, or any
    proxy that preserves ``cache_control`` passthrough)."""
    model_lower = (model or "").lower()
    provider_lower = (provider or "").lower()
    if provider_lower == "anthropic":
        return True
    # Claude via openrouter / new_api / any proxy that keeps the
    # Anthropic message shape. We gate on model name rather than just
    # provider so a non-Anthropic provider serving Claude (e.g. a
    # custom relay) still benefits.
    return "claude" in model_lower


def maybe_wrap_for_prompt_cache(
    client: Any, *, provider: str, model: str,
) -> Any:
    """Wrap ``client`` with :class:`AnthropicPromptCacheAdapter` when
    the upstream is Claude-family. Returns ``client`` unchanged
    otherwise — callers can use this in a single line without gating."""
    if _is_claude_family(provider, model):
        return AnthropicPromptCacheAdapter(inner=client)
    return client


__all__ = [
    "AnthropicPromptCacheAdapter",
    "maybe_wrap_for_prompt_cache",
]
