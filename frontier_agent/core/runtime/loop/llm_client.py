"""Stable FrontierAgent facade over the shared AgentCore LLM runtime."""

from agent_core.errors import (
    LLMCallExhausted,
    LLMDeadlineExceeded,
    LLMReasoningRunaway,
    LLMStreamStalled,
)
from agent_core.runtime.loop._response import (
    extract_final_content,
    extract_leaked_reasoning,
    extract_model_name,
    extract_usage,
)
from agent_core.runtime.loop._runaway import (
    RUNAWAY_STATE_KEY,
    TRUNCATION_CONTINUATION_GUIDANCE,
    is_truncated_with_text,
)
from agent_core.runtime.loop._streaming import ThinkTagSplitter
from agent_core.tokens import estimate_message_tokens, estimate_text_tokens

from frontier_agent.core.runtime.loop._bind import (
    bind_max_tokens,
    bind_session_id,
    bind_temperature,
    bind_tools,
)
from frontier_agent.core.runtime.loop._call import call_llm

__all__ = [
    "RUNAWAY_STATE_KEY",
    "TRUNCATION_CONTINUATION_GUIDANCE",
    "LLMCallExhausted",
    "LLMDeadlineExceeded",
    "LLMReasoningRunaway",
    "LLMStreamStalled",
    "ThinkTagSplitter",
    "bind_max_tokens",
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
