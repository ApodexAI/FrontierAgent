"""Anthropic LLMClient — wraps :class:`anthropic.AsyncAnthropic`."""

from __future__ import annotations

import json
import logging
import os
from collections.abc import AsyncIterator
from typing import Any

from frontier_agent.core.llm import LLMClient, LLMResponse, StreamDelta
from frontier_agent.core.messages import Message, ToolCall, text_of

logger = logging.getLogger(__name__)


class AnthropicClient(LLMClient):
    """Non-streaming-first Anthropic adapter."""

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = 4096,
        timeout: float | None = 300.0,
        thinking: dict[str, Any] | None = None,
        effort: str = "",
        bedrock: bool = False,
        default_headers: dict[str, str] | None = None,
    ) -> None:
        self.model = model
        self.default_temperature = temperature
        self.default_max_tokens = max_tokens
        self.default_timeout = timeout
        # Extended thinking: when set (e.g. ``{"type": "adaptive", "display":
        # "summarized"}``) the request carries ``thinking=`` so responses return
        # thinking + signature blocks; the response parser keeps them verbatim
        # (content_block) for faithful multi-turn replay. ``temperature`` is
        # dropped when thinking is on (Anthropic 400s on the combo). ``effort``
        # (low|medium|high|xhigh|max) → ``output_config.effort`` via extra_body.
        self._thinking = thinking or None
        self._effort = (effort or "").strip()
        # Transport: ``bedrock`` swaps AsyncAnthropic (``/v1/messages`` +
        # ``x-api-key``) for the AWS Bedrock runtime (``/model/{id}/invoke`` +
        # ``anthropic_version`` body stamp) authenticated with a Bedrock API Key
        # (``Authorization: Bearer``) instead of IAM SigV4. Everything downstream
        # (_build_kwargs / _to_llm_response / thinking replay) is transport-
        # agnostic and reused unchanged.
        if bedrock:
            self._client = _build_bedrock_client(
                api_key,
                base_url,
                timeout,
                default_headers,
            )
        else:
            from anthropic import AsyncAnthropic
            self._client = AsyncAnthropic(
                api_key=api_key, base_url=base_url, timeout=timeout, max_retries=0,
                default_headers=default_headers,
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
        """Shared request-shape builder for :meth:`chat` and :meth:`stream`."""
        system, msgs = _split_system(messages)
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": [_to_anthropic_msg(m) for m in msgs],
            "max_tokens": max_tokens or self.default_max_tokens or 4096,
        }
        if system:
            kwargs["system"] = system
        if self._thinking:
            # Anthropic rejects ``temperature`` together with thinking, so it is
            # OMITTED here regardless of the configured default. ``effort`` rides
            # on ``extra_body.output_config`` so any value (incl. ``xhigh``)
            # reaches ``messages.create`` without the SDK's stricter validation.
            kwargs["thinking"] = self._thinking
            if self._effort:
                kwargs["extra_body"] = {"output_config": {"effort": self._effort}}
        else:
            eff_temp = temperature if temperature is not None else self.default_temperature
            if eff_temp is not None:
                kwargs["temperature"] = eff_temp
        if tools:
            kwargs["tools"] = [_to_anthropic_tool(t) for t in tools]
        if extra_headers:
            kwargs["extra_headers"] = extra_headers
        if timeout is not None:
            kwargs["timeout"] = timeout
        elif self.default_timeout is not None:
            kwargs["timeout"] = self.default_timeout
        _add_prompt_cache(kwargs)
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
        raw = await self._client.messages.create(**kwargs)
        return _to_llm_response(raw)

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
        # Real token-by-token streaming over Anthropic's raw event stream.
        # Each event maps to the same ``StreamDelta`` shape the kernel
        # assembler consumes for OpenAI (content / reasoning_content /
        # tool_call_deltas), and the terminal delta carries usage/finish/model
        # just like the OpenAI ``include_usage`` chunk.
        kwargs = self._build_kwargs(
            messages, tools=tools, temperature=temperature,
            max_tokens=max_tokens, extra_headers=extra_headers, timeout=timeout,
        )
        kwargs["stream"] = True
        input_tokens: int | None = None
        output_tokens: int | None = None
        cache_read: int | None = None
        cache_write: int | None = None
        model = ""
        stop_reason = ""
        stream = await self._client.messages.create(**kwargs)
        async for event in stream:
            etype = getattr(event, "type", "")
            if etype == "message_start":
                msg = getattr(event, "message", None)
                model = getattr(msg, "model", "") or model
                u = getattr(msg, "usage", None)
                if u is not None:
                    input_tokens = getattr(u, "input_tokens", input_tokens)
                    cr = getattr(u, "cache_read_input_tokens", None)
                    if cr is not None:
                        cache_read = cr
                    cw = _anthropic_cache_write_tokens(u)
                    if cw is not None:
                        cache_write = cw
            elif etype == "content_block_start":
                cb = getattr(event, "content_block", None)
                if getattr(cb, "type", "") == "tool_use":
                    # Open a tool-call slot: id + name set once; arguments
                    # arrive as ``input_json_delta`` partial-JSON fragments.
                    yield StreamDelta(tool_call_deltas=[{
                        "index": getattr(event, "index", 0),
                        "id": getattr(cb, "id", "") or "",
                        "name": getattr(cb, "name", "") or "",
                        "arguments": "",
                    }])
            elif etype == "content_block_delta":
                d = getattr(event, "delta", None)
                dtype = getattr(d, "type", "")
                if dtype == "text_delta":
                    yield StreamDelta(content=getattr(d, "text", "") or "")
                elif dtype == "thinking_delta":
                    yield StreamDelta(
                        reasoning_content=getattr(d, "thinking", "") or "",
                    )
                elif dtype == "input_json_delta":
                    yield StreamDelta(tool_call_deltas=[{
                        "index": getattr(event, "index", 0),
                        "id": None,
                        "name": None,
                        "arguments": getattr(d, "partial_json", "") or "",
                    }])
            elif etype == "message_delta":
                d = getattr(event, "delta", None)
                stop_reason = getattr(d, "stop_reason", "") or stop_reason
                u = getattr(event, "usage", None)
                if u is not None:
                    ot = getattr(u, "output_tokens", None)
                    if ot is not None:
                        output_tokens = ot
        # Terminal delta: fold the accumulated usage/finish/model onto the
        # assembled ``LLMResponse`` (mirrors OpenAI's empty-choices chunk).
        yield StreamDelta(
            usage=_anthropic_usage_dict(
                input_tokens,
                output_tokens,
                cache_read,
                cache_write,
            ),
            finish_reason=stop_reason,
            model=model,
        )


# ── Bedrock transport ────────────────────────────────────────────────────


def _bedrock_region_from_url(base_url: str | None) -> str:
    """Best-effort region from a bedrock-runtime base_url.

    ``https://bedrock-runtime.us-east-1.amazonaws.com`` → ``us-east-1``;
    defaults to ``us-east-1`` when it can't be parsed (the region only labels
    the SDK client — the endpoint is ``base_url`` verbatim)."""
    host = (base_url or "").split("//", 1)[-1].split("/", 1)[0]
    parts = host.split(".")
    if len(parts) >= 3 and parts[0].startswith("bedrock-runtime"):
        return parts[1]
    return "us-east-1"


def _build_bedrock_client(
    api_key: str | None,
    base_url: str | None,
    timeout: float | None,
    default_headers: dict[str, str] | None = None,
) -> Any:
    """AsyncAnthropicBedrock that authenticates with a Bedrock API Key
    (``Authorization: Bearer``) instead of IAM SigV4.

    Mirrors the proven reporter pattern (``report_llm._build_bedrock_raw``):
    the stock ``AsyncAnthropicBedrock._prepare_request`` SigV4-signs via boto3
    (needs AWS creds); we override it to inject the Bearer header. Everything
    else the Bedrock client gives for free is what we want — the
    ``/v1/messages`` → ``/model/{id}/invoke`` URL rewrite and the
    ``anthropic_version: bedrock-2023-05-31`` body stamp."""
    import httpx

    # ``anthropic.AsyncAnthropicBedrock`` is the SDK's documented entry point,
    # but it is missing from the package's ``__all__``, so the checker suggests
    # importing from ``anthropic._client`` instead. Keep the public path — a
    # private module is far likelier to move between SDK releases — and suppress
    # just this rule.
    from anthropic import AsyncAnthropicBedrock  # pyright: ignore[reportPrivateImportUsage]

    bearer = api_key or ""

    class _BearerBedrock(AsyncAnthropicBedrock):
        async def _prepare_request(self, request: httpx.Request) -> None:
            request.headers["Authorization"] = f"Bearer {bearer}"

    return _BearerBedrock(
        aws_region=_bedrock_region_from_url(base_url),
        base_url=base_url or None,
        timeout=timeout,
        max_retries=0,
        default_headers=default_headers,
    )


# ── Conversion helpers ───────────────────────────────────────────────────


def _split_system(messages: list[Message]) -> tuple[str, list[Message]]:
    """Pull out the (single) leading system message; Anthropic takes it
    as a top-level kwarg, not as a message."""
    if messages and messages[0].get("role") == "system":
        return text_of(messages[0].get("content", "")), messages[1:]
    return "", list(messages)


def _add_prompt_cache(kwargs: dict[str, Any]) -> None:
    """Set Anthropic prompt-cache breakpoints on ``kwargs`` in place.

    Anthropic caching is opt-in per content block (unlike OpenAI's automatic
    caching), so without breakpoints the full growing prompt is re-billed every
    turn (``cached_tokens=0``). Place two ``ephemeral`` breakpoints — the system
    prefix (static across the run) and the last message's final block (a rolling
    breakpoint that caches the growing conversation prefix). Anthropic allows up
    to 4 and serves the longest matching cached prefix, so these two cover the
    static head and the moving tail. This lives inside ``AnthropicClient`` so it
    only ever touches Anthropic requests. Disable with ``ANTHROPIC_PROMPT_CACHE=0``.
    """
    if os.getenv("ANTHROPIC_PROMPT_CACHE", "1") == "0":
        return
    # System prefix (a plain string) -> one cache-controlled text block.
    system = kwargs.get("system")
    if isinstance(system, str) and system:
        kwargs["system"] = [{
            "type": "text",
            "text": system,
            "cache_control": {"type": "ephemeral"},
        }]
    # Rolling tail: mark the last message's final content block.
    msgs = kwargs.get("messages")
    if not msgs:
        return
    last = msgs[-1]
    content = last.get("content")
    if isinstance(content, str):
        if content:
            last["content"] = [{
                "type": "text",
                "text": content,
                "cache_control": {"type": "ephemeral"},
            }]
    elif isinstance(content, list) and content and isinstance(content[-1], dict):
        content[-1] = {**content[-1], "cache_control": {"type": "ephemeral"}}


def _to_anthropic_msg(m: Message) -> dict[str, Any]:
    role = m.get("role")
    if role == "tool":
        return {
            "role": "user",
            "content": [{
                "type": "tool_result",
                "tool_use_id": m.get("tool_call_id", ""),
                "content": text_of(m.get("content", "")),
            }],
        }
    if role == "assistant":
        blocks: list[dict[str, Any]] = []
        raw = m.get("content")
        if isinstance(raw, list):
            # Extended-thinking continuation: history kept the VERBATIM block
            # list (via model_profile.to_history for content_block). Re-send the
            # signed ``thinking`` / ``redacted_thinking`` blocks UNMODIFIED so
            # Anthropic can validate the signature server-side and continue the
            # signed reasoning state, then append the visible text + tool_use.
            for block in raw:
                if not isinstance(block, dict):
                    continue
                bt = block.get("type")
                if bt == "thinking":
                    tb: dict[str, Any] = {
                        "type": "thinking",
                        "thinking": block.get("thinking", "") or "",
                    }
                    sig = block.get("signature")
                    if sig:
                        tb["signature"] = sig
                    blocks.append(tb)
                elif bt == "redacted_thinking":
                    blocks.append({
                        "type": "redacted_thinking",
                        "data": block.get("data", "") or "",
                    })
                elif bt == "text":
                    txt = block.get("text", "") or ""
                    if txt:
                        blocks.append({"type": "text", "text": txt})
        else:
            body = text_of(raw or "")
            if body:
                blocks.append({"type": "text", "text": body})
        for tc in m.get("tool_calls", []) or []:
            blocks.append({
                "type": "tool_use",
                "id": tc["id"],
                "name": tc["function"]["name"],
                "input": json.loads(tc["function"].get("arguments") or "{}"),
            })
        return {"role": "assistant", "content": blocks or [{"type": "text", "text": ""}]}
    return {"role": "user", "content": text_of(m.get("content", ""))}


def _to_anthropic_tool(t: dict[str, Any]) -> dict[str, Any]:
    """OpenAI ``{type:function, function:{name,description,parameters}}`` →
    Anthropic ``{name, description, input_schema}``."""
    fn = t.get("function") or t
    return {
        "name": fn.get("name", ""),
        "description": fn.get("description", ""),
        "input_schema": fn.get("parameters", {}),
    }


def _anthropic_usage_dict(
    input_tokens: int | None,
    output_tokens: int | None,
    cache_read: int | None,
    cache_write: int | None,
    reasoning: int | None = None,
) -> dict[str, int]:
    """Normalise Anthropic token counts into the wire-shape usage dict shared
    by the non-streaming ``_to_llm_response`` and the streaming assembler.
    Cache reads and writes are kept separate for billing and also summed into
    the backward-compatible ``cached_tokens`` field. ``reasoning``
    (extended-thinking tokens, part of ``output_tokens``) is surfaced
    separately when present."""
    out: dict[str, int] = {}
    if input_tokens is not None:
        out["prompt_tokens"] = int(input_tokens)
    if output_tokens is not None:
        out["completion_tokens"] = int(output_tokens)
    if cache_read is not None or cache_write is not None:
        read = int(cache_read or 0)
        write = int(cache_write or 0)
        out["cache_read_tokens"] = read
        out["cache_write_tokens"] = write
        out["cached_tokens"] = read + write
        out["cache_creation_tokens"] = write
    if reasoning:
        out["reasoning_tokens"] = int(reasoning)
    if out.get("prompt_tokens") or out.get("completion_tokens"):
        out["total_tokens"] = (
            out.get("prompt_tokens", 0) + out.get("completion_tokens", 0)
        )
    return out


def _anthropic_cache_write_tokens(usage: Any) -> int | None:
    """Return Anthropic cache-creation tokens, including the 1-hour extension."""
    if usage is None:
        return None
    raw = getattr(usage, "cache_creation_input_tokens", None)
    if raw is None and isinstance(usage, dict):
        raw = usage.get("cache_creation_input_tokens")
    nested = getattr(usage, "cache_creation", None)
    if nested is None and isinstance(usage, dict):
        nested = usage.get("cache_creation")
    extension = getattr(nested, "ephemeral_1h_input_tokens", None)
    if extension is None and isinstance(nested, dict):
        extension = nested.get("ephemeral_1h_input_tokens")
    if raw is None and extension is None:
        return None
    return max(0, int(raw or 0)) + max(0, int(extension or 0))


def _anthropic_reasoning_tokens(usage: Any) -> int:
    """Best-effort extended-thinking token count off an Anthropic usage object.

    Newer usage payloads may expose ``output_tokens_details.thinking_tokens``;
    absent that the count is folded into ``output_tokens`` and unrecoverable, so
    we return 0 (the ``reasoning_tokens`` key is then omitted)."""
    if usage is None:
        return 0
    otd = getattr(usage, "output_tokens_details", None)
    if otd is None:
        return 0
    val = getattr(otd, "thinking_tokens", None)
    if val is None and isinstance(otd, dict):
        val = otd.get("thinking_tokens")
    return int(val or 0)


def _to_llm_response(raw: Any) -> LLMResponse:
    text_parts: list[str] = []
    thinking_parts: list[str] = []
    has_redacted = False
    blocks_out: list[dict[str, Any]] = []
    tool_calls: list[ToolCall] = []
    for block in (getattr(raw, "content", None) or []):
        btype = getattr(block, "type", None)
        if btype == "text":
            text = getattr(block, "text", "") or ""
            text_parts.append(text)
            blocks_out.append({"type": "text", "text": text})
        elif btype == "thinking":
            thinking = getattr(block, "thinking", "") or ""
            thinking_parts.append(thinking)
            blocks_out.append({
                "type": "thinking",
                "thinking": thinking,
                # ``signature`` is the cryptographic token Anthropic returns
                # with each thinking block; resending it on the next turn
                # lets the model continue from the same reasoning state.
                "signature": getattr(block, "signature", "") or "",
            })
        elif btype == "redacted_thinking":
            # Encrypted thinking Anthropic chose not to surface. It carries no
            # readable text but MUST be preserved verbatim (raw_content_blocks)
            # and replayed unmodified — the outbound ``_to_anthropic_msg`` echoes
            # it, and dropping it here would break signature/replay continuity.
            has_redacted = True
            blocks_out.append({
                "type": "redacted_thinking",
                "data": getattr(block, "data", "") or "",
            })
        elif btype == "tool_use":
            tool_calls.append({
                "id": getattr(block, "id", ""),
                "type": "function",
                "function": {
                    "name": getattr(block, "name", ""),
                    "arguments": json.dumps(getattr(block, "input", {}) or {}, ensure_ascii=False),
                },
            })

    # When thinking is present (readable or redacted), keep the structured block
    # list so the ``content_block`` parser picks out reasoning vs visible text
    # AND the verbatim signed/redacted blocks survive for replay. Otherwise
    # flatten to a string for the simpler downstream path.
    if thinking_parts or has_redacted:
        content: Any = blocks_out
    else:
        content = "\n".join(text_parts)

    usage = getattr(raw, "usage", None)
    usage_dict = _anthropic_usage_dict(
        getattr(usage, "input_tokens", None) if usage else None,
        getattr(usage, "output_tokens", None) if usage else None,
        getattr(usage, "cache_read_input_tokens", None) if usage else None,
        _anthropic_cache_write_tokens(usage),
        _anthropic_reasoning_tokens(usage),
    )

    return LLMResponse(
        content=content,
        tool_calls=tool_calls,
        reasoning_content="\n".join(thinking_parts),
        finish_reason=getattr(raw, "stop_reason", "") or "",
        model=getattr(raw, "model", "") or "",
        usage=usage_dict,
        response_metadata={"id": getattr(raw, "id", "")},
    )


__all__ = ["AnthropicClient"]
