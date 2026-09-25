"""FrontierAgent policy over AgentCore's compaction prompts.

AgentCore owns the prompt text and the ``auto`` dispatch; this module supplies
the product inputs: the configured style and the plugin tool categories.
"""

from __future__ import annotations

from agent_core.messages import Message
from agent_core.runtime.loop.summary_prompt import (
    COMPACTION_PROMPT,
    HANDOFF_COMPACTION_PROMPT,
    RESEARCH_COMPACTION_PROMPT,
    format_conversation_for_summary,
)
from agent_core.runtime.loop.summary_prompt import compaction_prompt as _compaction_prompt


def _tool_category(name: str) -> str:
    from plugins.tools.meta import get_tool_meta

    return get_tool_meta(name).category


def compaction_prompt(messages: list[Message] | None = None) -> str:
    """The compaction prompt for the configured style.

    Read per call so an A/B run can switch arms through
    ``COMPACTION_PROMPT_STYLE`` without a rebuild. An unknown or unreadable
    value falls back to ``research``.
    """
    from frontier_agent.infra.config import get_config

    try:
        style = str(get_config().compaction_prompt_style)
    except Exception:
        return RESEARCH_COMPACTION_PROMPT
    return _compaction_prompt(messages, style=style, tool_category=_tool_category)


__all__ = [
    "COMPACTION_PROMPT",
    "HANDOFF_COMPACTION_PROMPT",
    "RESEARCH_COMPACTION_PROMPT",
    "compaction_prompt",
    "format_conversation_for_summary",
]


def __getattr__(name: str) -> object:
    """Read-through to the shared prompt module for private helpers."""
    import agent_core.runtime.loop.summary_prompt as _shared

    return getattr(_shared, name)
