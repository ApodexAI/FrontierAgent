"""FrontierAgent adapter for AgentCore's LLM summary compactor.

The only product override left here is the prompt. AgentCore's compactor
already pins the original user task outside the lossy summary on both the
message path (``partition_for_compaction``) and the string-slice fallback, so
the ``_partition`` / ``_string_slice`` overrides this module used to carry are
now verbatim-equivalent to upstream and have been dropped rather than
maintained in two places.
"""

from __future__ import annotations

from typing import Any

from agent_core.runtime.loop.compact_llm import (
    CompactionEventEmitter,
    SummaryPromptBuilder,
    is_transient_summary_error,
)
from agent_core.runtime.loop.compact_llm import LLMSummaryCompactor as _CoreCompactor

from frontier_agent.core.messages import Message
from frontier_agent.infra.llm.summary_prompt import compaction_prompt

__all__ = [
    "CompactionEventEmitter",
    "LLMSummaryCompactor",
    "SummaryPromptBuilder",
    "is_transient_summary_error",
]


def _apodex_prompt(messages: list[Message]) -> str:
    """Select the product's enriched research prompt.

    Routed through ``compaction_prompt`` rather than pinning the constant so
    that wiring a tool-category callback is the only step left to enable
    AgentCore's handoff shape. With no callback the dispatch falls back to
    research, which is the prompt this product has always sent.
    """
    return compaction_prompt(messages)


class LLMSummaryCompactor(_CoreCompactor):
    """Keep the FrontierAgent prompt while sharing compaction mechanics."""

    def __init__(
        self,
        *,
        summary_llm: Any = None,
        emit_event: CompactionEventEmitter | None = None,
        failure_fallback: str = "panic",
        max_transient_retries: int = 0,
        retry_total_timeout_s: float | None = None,
        prompt_builder: SummaryPromptBuilder = _apodex_prompt,
    ) -> None:
        super().__init__(
            summary_llm=summary_llm,
            emit_event=emit_event,
            failure_fallback=failure_fallback,
            max_transient_retries=max_transient_retries,
            retry_total_timeout_s=retry_total_timeout_s,
            prompt_builder=prompt_builder,
        )


def __getattr__(name: str) -> Any:
    """Read-through to the shared module. Patch ``agent_core`` to rebind knobs."""
    import agent_core.runtime.loop.compact_llm as _shared

    return getattr(_shared, name)
