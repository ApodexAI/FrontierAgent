"""Bind, invoke, and normalize LLM clients used by the agent loop.

Provider adaptation stays here so the loop remains a readable sequence of
turn-level operations.
"""

from __future__ import annotations

import logging

from frontier_agent.core.errors import (
    LLMCallExhausted as LLMCallExhausted,
)
from frontier_agent.core.errors import (
    LLMReasoningRunaway,
    LLMStreamStalled,
)
from frontier_agent.core.runtime.loop._bind import (
    _ensure_bound as _ensure_bound,
)
from frontier_agent.core.runtime.loop._bind import (
    bind_session_id,
    bind_temperature,
    bind_tools,
)
from frontier_agent.core.runtime.loop._call import call_llm
from frontier_agent.core.runtime.loop._response import (
    extract_final_content,
    extract_leaked_reasoning,
    extract_model_name,
    extract_usage,
)
from frontier_agent.core.runtime.loop._runaway import (
    RUNAWAY_STATE_KEY,
    TRUNCATION_CONTINUATION_GUIDANCE,
    is_truncated_with_text,
)
from frontier_agent.utils.tokens import (
    estimate_message_tokens,
    estimate_text_tokens,
)

logger = logging.getLogger(__name__)

__all__ = [
    "RUNAWAY_STATE_KEY",
    "TRUNCATION_CONTINUATION_GUIDANCE",
    "LLMCallExhausted",
    "LLMReasoningRunaway",
    "LLMStreamStalled",
    "bind_session_id",
    "bind_temperature",
    "bind_tools",
    "call_llm",
    "estimate_message_tokens",
    "estimate_text_tokens",
    "extract_final_content",
    "extract_leaked_reasoning",
    "extract_model_name",
    "extract_usage",
    "is_truncated_with_text",
]
