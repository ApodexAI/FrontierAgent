"""FrontierAgent execution-context adapter for AgentCore LLM calls."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from agent_core.llm import LLMResponse
from agent_core.messages import Message
from agent_core.runtime.loop import _call as _shared

from frontier_agent.core.execution_context import chain_fallback_active
from frontier_agent.core.loop_types import wall_deadline_remaining_s


async def call_llm(
    llm: Any,
    messages: list[Message],
    timeout: int,
    max_retries: int,
    turn: int,
    on_delta: Callable[..., Awaitable[None]] | None = None,
    retry_wait_fixed: int | None = None,
    runaway_state: dict[str, Any] | None = None,
    first_chunk_s: float | None = None,
    on_attempt: Callable[[dict[str, Any]], Awaitable[None]] | None = None,
    reasoning_only_timeout_s: float | None = None,
    reasoning_only_max_tokens: int | None = None,
    logical_call_timeout_s: float | None = None,
    max_completion_tokens_hint: int | None = None,
    context_token_limit_hint: int | None = None,
) -> LLMResponse | None:
    """Call AgentCore with the active FrontierAgent runtime decisions."""

    return await _shared.call_llm(
        llm,
        messages,
        timeout,
        max_retries,
        turn,
        on_delta=on_delta,
        retry_wait_fixed=retry_wait_fixed,
        runaway_state=runaway_state,
        first_chunk_s=first_chunk_s,
        on_attempt=on_attempt,
        reasoning_only_timeout_s=reasoning_only_timeout_s,
        reasoning_only_max_tokens=reasoning_only_max_tokens,
        logical_call_timeout_s=logical_call_timeout_s,
        max_completion_tokens_hint=max_completion_tokens_hint,
        context_token_limit_hint=context_token_limit_hint,
        wall_deadline_remaining=wall_deadline_remaining_s,
        chain_fallback_active=chain_fallback_active,
    )


def __getattr__(name: str) -> Any:
    """Forward private migration-era test hooks to the shared module.

    Read-through only. Rebinding a shared module-level knob through this
    facade (``monkeypatch.setattr(<this module>, "_WALL_DEADLINE_FLOOR_S",
    ...)``) sets a *new* attribute here that shadows nothing the shared
    ``call_llm`` reads, so it silently has no effect. Patch
    ``agent_core.runtime.loop._call`` directly instead.
    """

    return getattr(_shared, name)


__all__ = ["call_llm"]
