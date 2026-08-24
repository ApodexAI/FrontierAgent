"""Per-profile ``protocol`` → native LLM client selection."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from frontier_agent.core.runtime.loop.model_profile import (
        ThinkingFormat,
        WireProtocol,
    )

from frontier_agent.core.llm import LLMClient


def protocol_of(cfg: dict[str, Any]) -> WireProtocol:
    """Normalised ``llm.protocol`` (default ``chat_completions``).

    The value comes from YAML, so an unrecognised protocol falls back to
    ``chat_completions`` rather than propagating a bare str that every
    ModelProfile construction would then reject.
    """
    from frontier_agent.core.runtime.loop.model_profile import is_wire_protocol

    # ``.lower()`` is reached before any narrowing, so a non-string value from
    # YAML (a list, dict, or number) would raise AttributeError here rather than
    # falling back — the isinstance check has to come first.
    raw = cfg.get("protocol")
    if not isinstance(raw, str):
        return "chat_completions"
    lowered = raw.lower()
    return lowered if is_wire_protocol(lowered) else "chat_completions"


def provider_label(cfg: dict[str, Any]) -> str:
    """Usage-attribution provider label for a profile ``llm`` block.

    Prefers the explicit ``_provider_label`` / ``provider``; otherwise DERIVES
    it from ``protocol`` so native reasoning profiles that omit ``provider``
    aren't mislabelled ``openai`` in billing/usage traces: ``anthropic`` →
    ``anthropic``, ``bedrock`` → ``bedrock``, ``responses`` / ``chat_completions``
    → ``openai`` (both are OpenAI-wire)."""
    explicit = cfg.get("_provider_label") or cfg.get("provider")
    if explicit:
        return str(explicit)
    proto = protocol_of(cfg)
    if proto == "anthropic":
        return "anthropic"
    if proto == "bedrock":
        return "bedrock"
    return "openai"


def thinking_format_for_protocol(protocol: str) -> ThinkingFormat | None:
    """``content_block`` for the reasoning protocols, else ``None`` (caller
    falls back to explicit YAML / model-id inference)."""
    return (
        "content_block"
        if protocol in ("anthropic", "responses", "bedrock")
        else None
    )


def _effort_str(cfg: dict[str, Any]) -> str:
    effort = cfg.get("effort")
    return effort.strip() if isinstance(effort, str) else ""


def _build_anthropic(cfg: dict[str, Any], *, bedrock: bool = False) -> LLMClient:
    """Native Anthropic Messages API with extended thinking.

    Responses carry thinking + signature blocks, kept verbatim
    (raw_content_blocks) and replayed unmodified. ``temperature`` is OMITTED
    (the client drops it when thinking is on). ``base_url`` posts to
    ``{base_url}/v1/messages`` (direct) or ``{base_url}/model/{id}/invoke``
    (``bedrock=True``, AWS Bedrock runtime, Bearer API-key auth + the
    ``anthropic_version`` body stamp). Optional ``effort`` →
    ``output_config.effort``.

    ``thinking_type`` selects the request shape (default ``adaptive``). Live-
    verified against api.anthropic.com + Bedrock 2026-07-09 (see
    ``temp/2026-07-09_reasoning-protocol-live-verification.md``); matches the
    official matrix at platform.claude.com/docs/en/build-with-claude/adaptive-thinking:

    - ``adaptive`` (DEFAULT) — the RECOMMENDED mode for all current Claude
      (Opus 4.6/4.7/4.8, Sonnet 4.6/5, Fable/Mythos), and the ONLY mode on the
      newest (Opus 4.7/4.8, Sonnet 5) — ``enabled`` is rejected there with 400.
      Emits ``thinking={"type":"adaptive"}`` + ``thinking_display`` (default
      ``summarized`` so the readable thinking text is captured; the newest
      models default ``display`` to ``omitted`` = empty ``thinking`` field with
      the ``signature`` still present for replay). ``effort`` is forwarded only
      in this mode (it is an adaptive-only knob; the oldest models 400 on
      ``enabled``+effort).
    - ``enabled`` — LEGACY opt-in for models older than Opus 4.6 / Sonnet 4.6
      (Sonnet 4.5, Opus 4.5, …), which reject ``adaptive`` with 400. Emits
      ``thinking={"type":"enabled","budget_tokens":N}`` (``N`` from
      ``thinking_budget_tokens``, default 8192, clamped to ``[1024, max_tokens-1]``
      since Anthropic requires ``budget_tokens < max_tokens``). ``effort`` is
      NOT sent (budget_tokens is the control knob here; oldest models 400 on it).
      Deprecated on Opus 4.6 / Sonnet 4.6 per Anthropic.
    """
    from frontier_agent.infra.anthropic_client import AnthropicClient

    max_tokens = int(cfg.get("max_tokens", 32768))
    ttype = str(cfg.get("thinking_type", "adaptive")).strip().lower()
    if ttype == "enabled":
        budget = int(cfg.get("thinking_budget_tokens", 8192))
        budget = max(1024, min(budget, max_tokens - 1))
        thinking: dict[str, Any] = {"type": "enabled", "budget_tokens": budget}
        effort = ""
    else:
        thinking = {"type": "adaptive"}
        display = cfg.get("thinking_display", "summarized")
        if isinstance(display, str) and display.strip():
            thinking["display"] = display.strip()
        effort = _effort_str(cfg)
    return AnthropicClient(
        model=cfg["model"],
        api_key=cfg.get("api_key", "dummy"),
        base_url=cfg.get("base_url") or None,
        max_tokens=max_tokens,
        thinking=thinking,
        effort=effort,
        bedrock=bedrock,
    )


def _build_responses(cfg: dict[str, Any], title: str) -> LLMClient:
    """OpenAI Responses API with encrypted reasoning.

    ``reasoning`` is built from ``effort`` + ``reasoning_summary`` (default
    ``auto`` → the response carries a readable reasoning summary; opt out with
    ``reasoning_summary: ""`` or set ``reasoning: {...}`` verbatim). The client
    always sends ``include=['reasoning.encrypted_content']`` + ``store=False``.
    ``temperature`` is only sent when the profile sets it (reasoning models
    reject non-default values).
    """
    from frontier_agent.infra.openai_responses_client import OpenAIResponsesClient

    reasoning = cfg.get("reasoning")
    if reasoning is None:
        reasoning = {}
        if cfg.get("effort"):
            reasoning["effort"] = cfg["effort"]
        summary = cfg.get("reasoning_summary", "auto")
        if isinstance(summary, str) and summary.strip():
            reasoning["summary"] = summary.strip()
    return OpenAIResponsesClient(
        model=cfg["model"],
        api_key=cfg.get("api_key", "dummy"),
        base_url=cfg.get("base_url"),
        temperature=cfg.get("temperature"),
        max_output_tokens=int(cfg.get("max_tokens") or 32768),
        default_headers={"HTTP-Referer": "frontier_agent", "X-Title": title},
        reasoning=reasoning or None,
        store=False,
    )


def build_protocol_client(cfg: dict[str, Any], *, title: str) -> LLMClient | None:
    """Build the native client for ``cfg['protocol']``.

    Returns ``None`` for ``chat_completions`` (the caller builds its usual
    ``OpenAIClient``); an :class:`AnthropicClient` / :class:`OpenAIResponsesClient`
    otherwise. ``title`` is the ``X-Title`` header stamped on Responses calls.
    """
    protocol = protocol_of(cfg)
    if protocol == "anthropic":
        return _build_anthropic(cfg)
    if protocol == "bedrock":
        return _build_anthropic(cfg, bedrock=True)
    if protocol == "responses":
        return _build_responses(cfg, title)
    return None


__all__ = [
    "build_protocol_client",
    "protocol_of",
    "provider_label",
    "thinking_format_for_protocol",
]
