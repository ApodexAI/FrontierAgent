"""Token estimation for messages.

The estimate itself lives in
:func:`frontier_agent.core.runtime.loop.context_budget.estimate_tokens`; this
module adds the message-shaped wrapper the loop and the finalization recovery
path import (both via ``loop.llm_client``, which re-exports them).
"""

from __future__ import annotations

from typing import Any


def estimate_text_tokens(text: str) -> int:
    """Estimate token count for a text string."""
    from frontier_agent.core.runtime.loop.context_budget import estimate_tokens
    return estimate_tokens(text)


def estimate_message_tokens(message: Any) -> int:
    """Estimate total tokens for a Message dict or object."""
    if not message:
        return 0
    if isinstance(message, dict):
        content = message.get("content") or ""
        tool_calls = message.get("tool_calls")
    else:
        content = getattr(message, "content", "") or ""
        tool_calls = getattr(message, "tool_calls", None)

    content_tokens = estimate_text_tokens(str(content))
    tool_tokens = 0
    if tool_calls:
        for tc in tool_calls:
            if isinstance(tc, dict):
                name = tc.get("name", "")
                args = str(tc.get("args", ""))
            else:
                name = getattr(tc, "name", "")
                args = str(getattr(tc, "args", ""))
            tool_tokens += estimate_text_tokens(str(name)) + estimate_text_tokens(str(args)) + 4
    return content_tokens + tool_tokens + 4
