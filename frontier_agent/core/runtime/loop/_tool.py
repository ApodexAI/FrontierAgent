"""Compatibility facade for shared streamed tool-call recovery checks."""

from typing import Any

from agent_core.runtime.loop.tool_call_recovery import (
    stream_tool_calls_missing_required_arguments as _shared_missing_required_arguments,
)

from frontier_agent.core.llm import LLMResponse


def stream_tool_calls_missing_required_arguments(
    response: LLMResponse,
    llm: Any,
) -> list[str]:
    """Preserve the product boundary's legacy nil-safe response handling."""
    if not getattr(response, "tool_calls", None):
        return []
    return _shared_missing_required_arguments(response, llm)


__all__ = ["stream_tool_calls_missing_required_arguments"]
