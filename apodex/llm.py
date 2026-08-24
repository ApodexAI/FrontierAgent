"""Build the chat model for the coding agent.

Uses FrontierAgent's native :class:`frontier_agent.infra.openai_client.OpenAIClient`
— an OpenAI-Chat-Completions client that preserves per-chunk
``reasoning_content`` (so the TUI can stream a model's thinking channel,
matching apodex's live thinking blocks). Streaming is a per-call concern
(``client.stream(...)``), so there is no client-construction streaming flag.
"""

from __future__ import annotations

from typing import Any

from apodex.config import ModelConfig
from frontier_agent.core.llm import LLMClient


def build_llm(cfg: ModelConfig) -> LLMClient:
    """Construct a streaming-capable chat client from :class:`ModelConfig`."""
    # Imported lazily so ``--help`` / config errors don't pull the LLM stack.
    from frontier_agent.infra.openai_client import OpenAIClient

    kwargs: dict[str, Any] = {
        "model": cfg.model,
        "temperature": cfg.temperature,
        # langchain ``max_tokens`` maps onto OpenAI ``max_completion_tokens``.
        "max_completion_tokens": cfg.max_tokens,
    }
    # Neither is an ``OpenAIClient`` parameter, and neither is a
    # Chat-Completions field, so both ride ``extra_body`` — the same route the
    # workflow profile loaders use. Omitted entirely when unset, so a profile
    # that says nothing keeps the server's own defaults.
    extra_body: dict[str, Any] = {}
    if cfg.top_p is not None:
        extra_body["top_p"] = cfg.top_p
    if cfg.top_k is not None:
        extra_body["top_k"] = cfg.top_k
    if extra_body:
        kwargs["extra_body"] = extra_body
    if cfg.api_key:
        kwargs["api_key"] = cfg.api_key
    if cfg.base_url:
        kwargs["base_url"] = cfg.base_url
    return OpenAIClient(**kwargs)
