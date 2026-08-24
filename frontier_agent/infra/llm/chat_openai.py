"""``ChatOpenAI`` — native OpenAI-compatible LLM client."""

from __future__ import annotations

from frontier_agent.infra.openai_client import OpenAIClient

# Back-compat alias — ``ChatOpenAI`` is now just the native OpenAI client.
#
# origin/main carried a langchain ``ChatOpenAI`` subclass whose only job was
# to rescue ``delta.reasoning_content`` / ``delta.reasoning`` from streamed
# chunks (and surface it via ``additional_kwargs``). That behavior is now
# native: ``OpenAIClient.stream`` reads the thinking channel off the wire via
# ``_reasoning_text`` (accepting either field name) and emits it as
# ``StreamDelta.reasoning_content``; the non-streaming path does the same off
# the completed message. So no langchain subclass is needed — the alias is the
# whole story.
ChatOpenAI = OpenAIClient


__all__ = ["ChatOpenAI"]
