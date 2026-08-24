"""OpenAI-compatible async LLM client.

Wire-shape choices preserve Chat Completions compatibility across OpenAI,
vLLM, SGLang, OpenRouter, and similar endpoints.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from typing import Any

from openai import AsyncOpenAI, BadRequestError

from frontier_agent.core.llm import LLMClient, LLMResponse, StreamDelta
from frontier_agent.core.messages import Message, ToolCall, for_wire
from frontier_agent.infra.session_context import (
    get_task_session_id,
    mirror_session_query,
)

logger = logging.getLogger(__name__)


class OpenAIClient(LLMClient):
    """Thin async wrapper. Honors the standard OpenAI knobs (``temperature``,
    ``max_completion_tokens``, ``tools``, ``stream``); per-call ``extra_headers``
    are forwarded for proxies needing custom auth or session affinity."""

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float | None = None,
        max_completion_tokens: int | None = None,
        timeout: float | None = None,
        default_headers: dict[str, str] | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> None:
        self.model = model
        self.default_temperature = temperature
        self.default_max_tokens = max_completion_tokens
        self.default_timeout = timeout
        self.extra_body = extra_body or {}
        # Some OpenAI-compatible gateways reject ``stream_options`` with a 400.
        # We probe optimistically and flip this off on first rejection so the
        # stream still runs (streaming usage then reads 0 for that gateway).
        self._stream_options_supported = True
        # EAS UCH affinity hashes the URL query parameter rather than the
        # header, so a session id has to be mirrored onto the URL. It is NOT
        # handed to the SDK as ``default_query``: this client is cached per
        # profile and reused across tasks (``_llm_cache`` in
        # ``workflows/agent_team/nodes/main_agent.py``), and a frozen query
        # param would keep routing every later task to the first task's
        # worker. Resolved per request by ``_session_query`` instead.
        self._default_session_query = mirror_session_query(default_headers)
        self._session_query_task = get_task_session_id()
        # Per-call timeout is enforced by the agent loop's
        # ``asyncio.wait_for`` wrapper, not by the SDK.
        self._client = AsyncOpenAI(
            # OpenAI 2.54 rejects an explicitly supplied empty key before it
            # consults OPENAI_API_KEY. A cached config can legitimately hold
            # the pre-environment empty value, so treat it as unspecified and
            # preserve the SDK's normal environment fallback.
            api_key=api_key or None,
            base_url=base_url or None,
            timeout=timeout,
            default_headers=default_headers,
            max_retries=0,           # retries are owned by the runtime loop
        )

    def _session_query(
        self, extra_headers: dict[str, str] | None,
    ) -> dict[str, str]:
        """Session-affinity URL query for one request, or ``{}`` for none.

        Per-call headers win: ``bind_session_id`` stamps the *current* task's
        id onto every agent-loop request, so that is the authoritative value.

        The construction-time mirror is only the fallback, for clients whose
        session id is fixed at build time and never bound per call (aux LLMs,
        rebuilt inside each task). It is dropped once the task that built it
        is no longer the current one, so a cached client can never pin a later
        task to the earlier task's worker — the failure mode that made the
        ``FRONTIER_AGENT_LLM_STICKY_SESSION`` kill switch inert. A mirror that
        did not come from a task context (a statically configured header) has
        no task to go stale against and always applies.
        """
        per_call = mirror_session_query(extra_headers)
        if per_call:
            return per_call
        fallback = self._default_session_query
        built_for = self._session_query_task
        if not fallback or not built_for:
            return fallback
        if built_for != get_task_session_id():
            logger.debug(
                "Dropping stale session-affinity query built for task %r "
                "on a client now serving task %r",
                built_for, get_task_session_id(),
            )
            return {}
        return fallback

    # ── Non-streaming ────────────────────────────────────────────────────

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
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": list(for_wire(messages)),
            "stream": False,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["parallel_tool_calls"] = True
        eff_temp = temperature if temperature is not None else self.default_temperature
        if eff_temp is not None:
            kwargs["temperature"] = eff_temp
        eff_mt = max_tokens if max_tokens is not None else self.default_max_tokens
        if eff_mt is not None:
            kwargs["max_completion_tokens"] = eff_mt
        if extra_headers:
            kwargs["extra_headers"] = extra_headers
        session_query = self._session_query(extra_headers)
        if session_query:
            kwargs["extra_query"] = session_query
        if self.extra_body:
            kwargs["extra_body"] = self.extra_body
        # Per-call ``timeout`` triggers ``x-stainless-read-timeout``
        # header; only set when the caller explicitly passed one.
        if timeout is not None:
            kwargs["timeout"] = timeout

        raw_response = await self._client.chat.completions.with_raw_response.create(
            **kwargs,
        )
        raw = raw_response.parse()
        return _to_llm_response(raw)

    # ── Streaming ────────────────────────────────────────────────────────

    async def stream(
        self,
        messages: list[Message],
        *,
        tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        extra_headers: dict[str, str] | None = None,
        timeout: float | None = None,
    ) -> AsyncIterator[StreamDelta]:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": list(for_wire(messages)),
            "stream": True,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["parallel_tool_calls"] = True
        eff_temp = temperature if temperature is not None else self.default_temperature
        if eff_temp is not None:
            kwargs["temperature"] = eff_temp
        eff_mt = max_tokens if max_tokens is not None else self.default_max_tokens
        if eff_mt is not None:
            kwargs["max_completion_tokens"] = eff_mt
        if extra_headers:
            kwargs["extra_headers"] = extra_headers
        session_query = self._session_query(extra_headers)
        if session_query:
            kwargs["extra_query"] = session_query
        if self.extra_body:
            kwargs["extra_body"] = self.extra_body
        if timeout is not None:
            kwargs["timeout"] = timeout
        elif self.default_timeout is not None:
            kwargs["timeout"] = self.default_timeout

        async for chunk in await self._open_stream(kwargs):
            chunk_usage = _usage_dict(getattr(chunk, "usage", None))
            chunk_model = getattr(chunk, "model", "") or ""
            if not chunk.choices:
                # Terminal ``include_usage`` chunk: empty ``choices`` but carries
                # the final token usage. Forward it (was previously dropped, so
                # streaming usage/billing read 0).
                if chunk_usage or chunk_model:
                    yield StreamDelta(usage=chunk_usage, model=chunk_model)
                continue
            choice = chunk.choices[0]
            delta = choice.delta
            yield StreamDelta(
                content=getattr(delta, "content", None) or "",
                reasoning_content=_reasoning_text(delta),
                tool_call_deltas=_tool_call_deltas(getattr(delta, "tool_calls", None)),
                finish_reason=getattr(choice, "finish_reason", None) or "",
                model=chunk_model,
                usage=chunk_usage,
            )

    async def _open_stream(self, kwargs: dict[str, Any]) -> Any:
        """Open the streaming completion, requesting ``stream_options.include_usage``
        so the terminal usage chunk arrives. Some OpenAI-compatible gateways
        reject ``stream_options`` with a 400 — on first rejection we disable it
        for this client and retry once without it (the stream still runs;
        streaming usage just reads 0 for that gateway, as it did before usage
        forwarding existed)."""
        if self._stream_options_supported:
            try:
                return await self._client.chat.completions.create(
                    stream_options={"include_usage": True}, **kwargs,
                )
            except BadRequestError as exc:
                if "stream_options" not in str(exc).lower():
                    raise
                self._stream_options_supported = False
                logger.warning(
                    "Gateway rejected stream_options.include_usage; disabling "
                    "it for this client (streaming token usage will read 0). %s",
                    exc,
                )
        return await self._client.chat.completions.create(**kwargs)


# ── Adapters ─────────────────────────────────────────────────────────────


def _reasoning_text(obj: Any) -> str:
    """Pull a model's thinking-channel text off a streamed ``delta`` or a
    completed ``message``.

    OpenAI-compatible endpoints disagree on the field name: SGLang / DeepSeek
    use ``reasoning_content``, while some proxies surface it as ``reasoning``.
    We accept either so the
    thinking channel survives whichever gateway is in front of the model.
    """
    return (
        getattr(obj, "reasoning_content", None)
        or getattr(obj, "reasoning", None)
        or ""
    )


def _usage_dict(usage: Any) -> dict[str, int]:
    """Normalise an OpenAI ``usage`` object into the wire-shape token dict.

    Shared by the non-streaming ``_to_llm_response`` and the streaming
    assembler (the terminal ``include_usage`` chunk carries the same shape).
    """
    out: dict[str, int] = {}
    if not usage:
        return out
    for k in ("prompt_tokens", "completion_tokens", "total_tokens"):
        v = getattr(usage, k, None)
        if v is not None:
            out[k] = int(v)
    # Prompt-caching surfaced under ``usage.prompt_tokens_details.cached_tokens``
    # on modern OpenAI / OpenRouter responses.
    ptd = getattr(usage, "prompt_tokens_details", None)
    cached = getattr(ptd, "cached_tokens", None) if ptd is not None else None
    if cached is not None:
        out["cached_tokens"] = int(cached)
    return out


def _to_llm_response(raw: Any) -> LLMResponse:
    """Convert an OpenAI ``ChatCompletion`` into an :class:`LLMResponse`."""
    choices = getattr(raw, "choices", None)
    if not choices:
        raise ValueError(
            "OpenAI-compatible response has no choices "
            f"(response_id={getattr(raw, 'id', '')!r}, "
            f"model={getattr(raw, 'model', '')!r})",
        )
    choice = choices[0]
    msg = choice.message
    tool_calls: list[ToolCall] = []
    for tc in (getattr(msg, "tool_calls", None) or []):
        tool_calls.append({
            "type": "function",
            "id": tc.id,
            "function": {
                "name": tc.function.name,
                "arguments": tc.function.arguments or "{}",
            },
        })
    usage_dict = _usage_dict(getattr(raw, "usage", None))
    content = getattr(msg, "content", "") or ""
    # Mirror the streaming path (llm_client._stream_llm_response): a Qwen
    # ``</think>\n\n`` separator remnant is either the whole of ``content``
    # (whitespace-only → drop) or leads the real answer (``\n\nAnswer…`` →
    # lstrip). It carries no user-visible meaning and doubles the separator
    # when ``thinking_in_history`` reconstructs the turn.
    if isinstance(content, str) and content:
        content = content.lstrip() if content.strip() else ""
    return LLMResponse(
        content=content,
        tool_calls=tool_calls,
        reasoning_content=_reasoning_text(msg),
        finish_reason=getattr(choice, "finish_reason", "") or "",
        model=getattr(raw, "model", "") or "",
        usage=usage_dict,
        response_metadata={"id": getattr(raw, "id", "")},
    )


def _tool_call_deltas(raw: Any) -> list[dict[str, Any]]:
    if not raw:
        return []
    out: list[dict[str, Any]] = []
    for tc in raw:
        out.append({
            "index": getattr(tc, "index", 0),
            "id": getattr(tc, "id", None),
            "name": getattr(getattr(tc, "function", None), "name", None),
            "arguments": getattr(getattr(tc, "function", None), "arguments", None),
        })
    return out


__all__ = ["OpenAIClient"]
