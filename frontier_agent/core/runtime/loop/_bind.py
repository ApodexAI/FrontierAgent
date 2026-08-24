from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from typing import Any

from frontier_agent.core.llm import LLMResponse
from frontier_agent.core.messages import Message

logger = logging.getLogger(__name__)

@dataclass
class _BoundLLM:
    """Lightweight binding wrapper around a native :class:`LLMClient`.

    The agent loop builds its per-turn LLM by chaining
    ``bind_tools(bind_session_id(llm, task_id), tools)``. Native clients
    have no langchain ``Runnable.bind`` to attach kwargs to, so the bound
    knobs are carried here and threaded into :meth:`LLMClient.chat` /
    :meth:`LLMClient.stream` per call. Each ``bind_*`` returns a fresh
    wrapper via :func:`dataclasses.replace` — never a mutation of the
    shared long-lived client.
    """

    client: Any
    tools: list[dict[str, Any]] | None = None
    temperature: float | None = None
    extra_headers: dict[str, str] | None = None
    max_tokens: int | None = None

    @property
    def model(self) -> str:
        return getattr(self.client, "model", "") or ""

    def _call_kwargs(
        self,
        timeout: float | None,
        *,
        tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        extra_headers: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Merge the bound fields with any kwarg-style overrides.

        ``_BoundLLM`` is bind-style (knobs live on the dataclass), but it
        gets nested *underneath* kwarg-style callers — ``LLMProxy`` and the
        fallback-chain wrappers unconditionally forward
        ``tools=/temperature=/max_tokens=/extra_headers=`` to their inner
        client. When that inner client is a ``_BoundLLM`` (e.g.
        ``LLMProxy(inner=_BoundLLM(...))`` built by
        ``create_subagent._bind_sub_agent_llm``), a bind-only signature
        raised ``TypeError: stream() got an unexpected keyword argument
        'tools'`` and killed every sub-agent on turn 1. Accepting + merging
        these kwargs makes ``_BoundLLM`` a tolerant drop-in.

        Precedence: an explicit non-``None`` kwarg overrides the bound
        field (the caller asked for it this call); otherwise the bound
        field is used. ``extra_headers`` is merged (bound base, kwarg wins
        per key) so a forwarded header never drops the session-affinity
        header bound earlier.
        """
        kw: dict[str, Any] = {}
        eff_tools = tools if tools is not None else self.tools
        if eff_tools:
            kw["tools"] = eff_tools
        eff_temperature = (
            temperature if temperature is not None else self.temperature
        )
        if eff_temperature is not None:
            kw["temperature"] = eff_temperature
        merged_headers = {**(self.extra_headers or {}), **(extra_headers or {})}
        if merged_headers:
            kw["extra_headers"] = merged_headers
        eff_max_tokens = (
            max_tokens if max_tokens is not None else self.max_tokens
        )
        if eff_max_tokens is not None:
            kw["max_tokens"] = eff_max_tokens
        if timeout is not None:
            kw["timeout"] = timeout
        return kw

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
        return await self.client.chat(
            messages,
            **self._call_kwargs(
                timeout,
                tools=tools,
                temperature=temperature,
                max_tokens=max_tokens,
                extra_headers=extra_headers,
            ),
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
    ) -> Any:
        return self.client.stream(
            messages,
            **self._call_kwargs(
                timeout,
                tools=tools,
                temperature=temperature,
                max_tokens=max_tokens,
                extra_headers=extra_headers,
            ),
        )


def _ensure_bound(llm: Any) -> _BoundLLM:
    """Wrap a raw ``LLMClient`` in a ``_BoundLLM``; pass an existing one through."""
    return llm if isinstance(llm, _BoundLLM) else _BoundLLM(client=llm)


def bind_tools(llm: Any, tools: list[Any]) -> Any:
    """Bind native :class:`frontier_agent.core.tool.Tool` objects (or pre-built
    OpenAI function-schema dicts) so the model can emit multiple tool_calls
    per turn. ``parallel_tool_calls=True`` is applied by the client adapter.
    No-op on an empty tool list.
    """
    if not tools:
        return llm
    schemas = [
        t.to_openai_schema() if hasattr(t, "to_openai_schema") else t
        for t in tools
    ]
    return replace(_ensure_bound(llm), tools=schemas)


def _sticky_session_enabled() -> bool:
    """Whether to inject the sticky-session header (default: yes).

    Thin alias over ``session_context.sticky_session_enabled`` — the kill
    switch has to gate the construction-time header injection as well
    (``workflows/agent_team/profile.py``, ``infra/llm/aux_builder.py``), so
    the env parsing lives there and both carriers read the same answer.
    """
    from frontier_agent.infra.session_context import sticky_session_enabled

    return sticky_session_enabled()


def bind_session_id(llm: Any, task_id: str) -> Any:
    """Attach ``x-upstream-session-id: <task_id>`` to every LLM request.

    Pinning it at client-construction time is the obvious approach, but
    our LLM is per-profile-cached (one client shared across tasks), so we
    bind the header per-call via LangChain's ``.bind(extra_headers=...)``
    which flows through to the OpenAI SDK ``extra_headers`` kwarg.

    Why it matters: EAS-backed gateways use this header for **session
    affinity** — the same session-id consistently
    routes to the same backend worker, preserving KV-cache across a task's
    turns.

    No-op when ``task_id`` is empty (standalone debug) or when
    ``FRONTIER_AGENT_LLM_STICKY_SESSION`` is falsey. This header is set ONLY
    here — ``create_swarm_llm`` does not use it. Sticky routing was once
    suspected of amplifying a high-concurrency stampede and disabled for it;
    that attribution was retracted when the real cause turned out to be silent
    mid-stream stalls, now handled by the stall watchdog.
    """
    if not task_id or not _sticky_session_enabled():
        return llm
    bound = _ensure_bound(llm)
    headers = dict(bound.extra_headers or {})
    headers["x-upstream-session-id"] = task_id
    return replace(bound, extra_headers=headers)


def bind_temperature(llm: Any, temperature: float) -> Any:
    """Bind ``temperature`` for a single invocation.

    Used by retry observers to escalate sampling on a retry turn without
    mutating the long-lived LLM (a fresh ``_BoundLLM`` is returned).
    """
    return replace(_ensure_bound(llm), temperature=temperature)
