"""LLM client contracts — provider-agnostic chat completion interface."""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from frontier_agent.core.messages import Message, ToolCall


@dataclass
class LLMResponse:
    """One non-streaming completion result."""

    content: Any = ""                                # str | list[dict]
    tool_calls: list[ToolCall] = field(default_factory=list)
    reasoning_content: str = ""
    finish_reason: str = ""
    model: str = ""
    usage: dict[str, int] = field(default_factory=dict)
    response_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class StreamDelta:
    """Incremental update during a streaming completion."""

    content: str = ""
    reasoning_content: str = ""
    tool_call_deltas: list[dict[str, Any]] = field(default_factory=list)
    # Terminal metadata. Providers send these late in the stream — usage on a
    # separate ``choices=[]`` chunk (OpenAI ``include_usage``), finish_reason on
    # the last content chunk. Carried here so the stream assembler can put them
    # on the final ``LLMResponse`` (else streaming usage/billing reads 0 and
    # ``finish_reason="length"`` is invisible to truncation/rollback observers).
    usage: dict[str, int] = field(default_factory=dict)
    finish_reason: str = ""
    model: str = ""
    # Vendor label of the leg serving this stream, stamped by
    # ``LLMFallbackChain.stream`` (constant once the chain commits to an
    # entry — failover only fires before the first yield). The stream
    # assembler folds it into ``LLMResponse.response_metadata`` so per-call
    # billing attribution works for streamed calls too — without this the
    # streaming path had no channel for the provider and every billing
    # consumer read an empty vendor, which split one model's usage across
    # a ``provider=""`` bucket and a named bucket.
    provider: str = ""


@runtime_checkable
class LLMClient(Protocol):
    """Minimal async chat completion client."""

    # Stays a settable attribute: LLMClient is not purely structural — concrete
    # clients such as OpenAIClient subclass it and assign ``self.model`` in
    # __init__, so a read-only property here would break them at runtime.
    model: str

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
        """Send one non-streaming completion request. ``tools`` is a list
        of OpenAI function-schema dicts; ``None`` runs without tools."""
        ...

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
        """Stream a completion as a sequence of incremental ``StreamDelta``s.

        The terminal ``LLMResponse`` (with assembled content + finalised
        tool_calls + usage) is accessible via :meth:`last_response` after the
        stream is exhausted.
        """
        ...


__all__ = ["LLMClient", "LLMResponse", "StreamDelta"]
