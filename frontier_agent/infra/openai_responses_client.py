"""OpenAI **Responses API** LLMClient — wraps :class:`openai.AsyncOpenAI`."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from typing import Any

from openai import AsyncOpenAI

from frontier_agent.core.llm import LLMClient, LLMResponse, StreamDelta
from frontier_agent.core.messages import Message, ToolCall, text_of

logger = logging.getLogger(__name__)


class OpenAIResponsesClient(LLMClient):
    """OpenAI Responses API adapter with encrypted-reasoning round-trip."""

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
        timeout: float | None = None,
        default_headers: dict[str, str] | None = None,
        reasoning: dict[str, Any] | None = None,
        store: bool = False,
    ) -> None:
        self.model = model
        self.default_temperature = temperature
        self.default_max_tokens = max_output_tokens
        self.default_timeout = timeout
        # ``reasoning`` = {effort?, summary?}. ``summary="auto"`` makes the
        # response carry a readable reasoning summary; without it the reasoning
        # item's summary array is empty (encrypted_content only).
        self._reasoning = reasoning or None
        self._store = store
        self._client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url or None,
            timeout=timeout,
            default_headers=default_headers,
            max_retries=0,
        )

    def _build_kwargs(
        self,
        messages: list[Message],
        *,
        tools: list[dict[str, Any]] | None,
        temperature: float | None,
        max_tokens: int | None,
        extra_headers: dict[str, str] | None,
        timeout: float | None,
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "input": _to_responses_input(messages),
            # Client-side reasoning round-trip: ask for encrypted_content and
            # keep the exchange stateless (no server-side response storage).
            "include": ["reasoning.encrypted_content"],
            "store": self._store,
        }
        if tools:
            kwargs["tools"] = _to_responses_tools(tools)
        if self._reasoning:
            kwargs["reasoning"] = self._reasoning
        eff_temp = temperature if temperature is not None else self.default_temperature
        if eff_temp is not None:
            kwargs["temperature"] = eff_temp
        eff_mt = max_tokens if max_tokens is not None else self.default_max_tokens
        if eff_mt is not None:
            kwargs["max_output_tokens"] = eff_mt
        if extra_headers:
            kwargs["extra_headers"] = extra_headers
        if timeout is not None:
            kwargs["timeout"] = timeout
        elif self.default_timeout is not None:
            kwargs["timeout"] = self.default_timeout
        return kwargs

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
        kwargs = self._build_kwargs(
            messages, tools=tools, temperature=temperature,
            max_tokens=max_tokens, extra_headers=extra_headers, timeout=timeout,
        )
        raw = await self._client.responses.create(**kwargs)
        return _parse_responses_output(raw)

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
        # Provided for Protocol conformance. The agent loop forces this client
        # non-streaming (reasoning blocks / encrypted_content only survive on
        # the non-streaming response), so this path is not used to preserve
        # verbatim reasoning — it maps the Responses semantic events to
        # ``StreamDelta`` for any caller that streams anyway.
        kwargs = self._build_kwargs(
            messages, tools=tools, temperature=temperature,
            max_tokens=max_tokens, extra_headers=extra_headers, timeout=timeout,
        )
        kwargs["stream"] = True
        stream = await self._client.responses.create(**kwargs)
        async for event in stream:
            etype = getattr(event, "type", "")
            if etype == "response.output_text.delta":
                yield StreamDelta(content=getattr(event, "delta", "") or "")
            elif etype in (
                "response.reasoning_summary_text.delta",
                "response.reasoning_text.delta",
            ):
                yield StreamDelta(reasoning_content=getattr(event, "delta", "") or "")
            elif etype == "response.completed":
                resp = getattr(event, "response", None)
                usage = _responses_usage_dict(getattr(resp, "usage", None))
                yield StreamDelta(
                    usage=usage,
                    model=getattr(resp, "model", "") or "",
                    finish_reason="stop",
                )


# ── Conversion helpers (pure — unit-tested) ────────────────────────────────


def _get(obj: Any, key: str, default: Any = None) -> Any:
    """Read ``key`` off an SDK object (attr) or a plain dict."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _to_responses_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Chat-Completions ``{type:function, function:{name,description,parameters}}``
    → Responses flat ``{type:function, name, description, parameters}``."""
    out: list[dict[str, Any]] = []
    for t in tools:
        fn = t.get("function") or t
        out.append({
            "type": "function",
            "name": fn.get("name", ""),
            "description": fn.get("description", ""),
            "parameters": fn.get("parameters", {}),
        })
    return out


def _to_responses_input(messages: list[Message]) -> list[dict[str, Any]]:
    """Convert Chat-Completions ``Message``s into a Responses ``input`` list.

    Assistant turns saved verbatim (``content`` is a block list of
    ``reasoning`` / ``text`` items) re-emit the reasoning items FIRST — incl.
    ``encrypted_content`` — so the model continues the prior reasoning state,
    then the assistant text, then any ``function_call`` items. Tool results
    become ``function_call_output`` items keyed by ``call_id``.
    """
    items: list[dict[str, Any]] = []
    for m in messages:
        role = m.get("role")
        if role == "tool":
            items.append({
                "type": "function_call_output",
                "call_id": m.get("tool_call_id", ""),
                "output": text_of(m.get("content", "")),
            })
            continue
        if role == "assistant":
            raw = m.get("content")
            text_parts: list[str] = []
            if isinstance(raw, list):
                for block in raw:
                    if not isinstance(block, dict):
                        continue
                    bt = block.get("type")
                    if bt == "reasoning":
                        item: dict[str, Any] = {"type": "reasoning"}
                        if block.get("id"):
                            item["id"] = block["id"]
                        item["summary"] = block.get("summary") or []
                        if block.get("encrypted_content"):
                            item["encrypted_content"] = block["encrypted_content"]
                        items.append(item)
                    elif bt == "text":
                        text_parts.append(block.get("text", "") or "")
            else:
                text_parts.append(text_of(raw or ""))
            body = "\n".join(p for p in text_parts if p)
            if body:
                items.append({
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": body}],
                })
            for tc in m.get("tool_calls", []) or []:
                items.append({
                    "type": "function_call",
                    "call_id": tc["id"],
                    "name": tc["function"]["name"],
                    "arguments": tc["function"].get("arguments") or "{}",
                })
            continue
        # system / user
        items.append({"role": role or "user", "content": text_of(m.get("content", ""))})
    return items


def _parse_responses_output(raw: Any) -> LLMResponse:
    """Parse a Responses API result into an :class:`LLMResponse`.

    ``content`` is kept as a verbatim block list (reasoning items incl.
    ``encrypted_content`` + text blocks) so the ``content_block`` thinking
    parser preserves them as ``raw_content_blocks`` for faithful replay.
    """
    blocks_out: list[dict[str, Any]] = []
    text_parts: list[str] = []
    summary_parts: list[str] = []
    tool_calls: list[ToolCall] = []

    for item in (_get(raw, "output", None) or []):
        itype = _get(item, "type", "")
        if itype == "reasoning":
            summary = _get(item, "summary", None) or []
            norm_summary: list[Any] = []
            for s in summary:
                stext = _get(s, "text", None)
                if isinstance(s, str):
                    norm_summary.append({"type": "summary_text", "text": s})
                    summary_parts.append(s)
                elif stext is not None:
                    norm_summary.append({"type": "summary_text", "text": stext})
                    summary_parts.append(stext)
            block: dict[str, Any] = {"type": "reasoning", "summary": norm_summary}
            if _get(item, "id", None):
                block["id"] = _get(item, "id")
            ec = _get(item, "encrypted_content", None)
            if ec:
                block["encrypted_content"] = ec
            blocks_out.append(block)
        elif itype == "message":
            for part in (_get(item, "content", None) or []):
                if _get(part, "type", "") == "output_text":
                    txt = _get(part, "text", "") or ""
                    text_parts.append(txt)
                    blocks_out.append({"type": "text", "text": txt})
        elif itype == "function_call":
            tool_calls.append({
                "id": _get(item, "call_id", "") or _get(item, "id", "") or "",
                "type": "function",
                "function": {
                    "name": _get(item, "name", "") or "",
                    "arguments": _get(item, "arguments", "") or "{}",
                },
            })

    # Prefer the top-level ``output_text`` convenience string when the SDK
    # provides it (it is the concatenation of message output_text parts).
    flat_text = _get(raw, "output_text", None)
    if flat_text and not text_parts:
        text_parts.append(flat_text)
        blocks_out.append({"type": "text", "text": flat_text})

    content: Any = blocks_out if blocks_out else "\n".join(text_parts)
    return LLMResponse(
        content=content,
        tool_calls=tool_calls,
        reasoning_content="\n".join(summary_parts),
        finish_reason=_get(raw, "status", "") or "",
        model=_get(raw, "model", "") or "",
        usage=_responses_usage_dict(_get(raw, "usage", None)),
        response_metadata={"id": _get(raw, "id", "")},
    )


def _responses_usage_dict(usage: Any) -> dict[str, int]:
    """Normalise a Responses ``usage`` object into the wire-shape token dict.

    Responses reports ``input_tokens`` / ``output_tokens`` (not prompt/
    completion), reasoning tokens under ``output_tokens_details.reasoning_tokens``,
    and cache reads under ``input_tokens_details.cached_tokens``."""
    out: dict[str, int] = {}
    if not usage:
        return out
    inp = _get(usage, "input_tokens", None)
    outp = _get(usage, "output_tokens", None)
    total = _get(usage, "total_tokens", None)
    if inp is not None:
        out["prompt_tokens"] = int(inp)
    if outp is not None:
        out["completion_tokens"] = int(outp)
    if total is not None:
        out["total_tokens"] = int(total)
    itd = _get(usage, "input_tokens_details", None)
    cached = _get(itd, "cached_tokens", None) if itd is not None else None
    if cached is not None:
        out["cached_tokens"] = int(cached)
    otd = _get(usage, "output_tokens_details", None)
    reasoning = _get(otd, "reasoning_tokens", None) if otd is not None else None
    if reasoning:
        out["reasoning_tokens"] = int(reasoning)
    return out


__all__ = ["OpenAIResponsesClient"]
