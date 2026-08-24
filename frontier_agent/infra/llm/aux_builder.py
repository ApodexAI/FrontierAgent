"""``aux_builder`` — provider-type-dispatched aux LLM factory."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Memoise the "dummy key" warning per (provider, model) so a profile
# that builds the same aux LLM repeatedly (once per ReAct turn, per
# chain attempt, etc.) doesn't flood the logs. Set entries are
# ``(provider, model)`` tuples — small, no need to bound.
_DUMMY_KEY_WARNED: set[tuple[str, str]] = set()


def _resolve_api_key(section: dict[str, Any], provider: str) -> str:
    """Return the section's ``api_key``, or fall back to ``"dummy"``.

    Behavioural fallback exists because some test paths construct minimal
    section dicts without going through ``_inject_provider_creds`` —
    requiring a non-empty key here would force every such caller to
    plumb a placeholder. Keeping the legacy default preserves that
    contract.

    But: when the fallback fires in PRODUCTION (env var unset or
    typoed, profile loaded via ``load_swarm_profile``), the request
    sends the literal string ``"dummy"`` as the Bearer token and the
    provider returns ``401 Missing Authentication header`` (or similar)
    — a confusing signal that doesn't point at the real cause. So we
    emit a once-per-(provider, model) WARN as a second line of defence
    on top of the workflow profile loader's own credential warning
    (which fires at config-load time).
    """
    key = section.get("api_key") or ""
    if key and str(key).strip():
        return str(key)

    model = str(section.get("model") or "")
    cache_key = (provider, model)
    if cache_key not in _DUMMY_KEY_WARNED:
        _DUMMY_KEY_WARNED.add(cache_key)
        logger.warning(
            "build_aux_llm: section for provider=%r model=%r has no "
            "api_key — falling back to the literal 'dummy' Bearer "
            "token. If this is a real run, set $%s_API_KEY in the "
            "environment; the upstream will otherwise return '401 "
            "Missing Authentication header'.",
            provider or "<unknown>", model or "<unknown>",
            (provider or "PROVIDER").upper(),
        )
    return "dummy"


def build_aux_llm(section: dict[str, Any]) -> Any:
    """Construct a chat model for a profile aux LLM section, dispatching
    on the resolved provider's ``type:`` in ``config/providers.yaml``.

    Aux LLMs (outline / report / decision / synth / summary) don't
    need the main agent's thinking-format mixin — they're called
    once per window/turn and return JSON or markdown. Two transport
    shapes:

    - ``openai_compat`` (default) — ChatOpenAI against an OpenAI-style
      ``/v1/chat/completions`` endpoint.
    - ``anthropic`` — AnthropicClient against the Messages API
      (``/v1/messages``).

    The section's ``_provider_label`` (an optional stamp a caller sets after
    resolving a ``model_chains`` rotation) wins over its literal
    ``provider:`` value — that lets an outer fallback chain flip
    ``openai_compat`` ↔ ``anthropic`` mid-chain when it advances to the
    next leg (anthropic→openrouter is the common case).

    Recognised optional knobs on ``section`` (openai_compat path only;
    Anthropic ignores ``extra_body`` and the session header):

    - ``enable_thinking`` (bool) — forwards
      ``extra_body.chat_template_kwargs.enable_thinking=True`` to SGLang
      / vLLM. qwen3 reasoning models need this for the chain-of-thought
      head.
    - ``max_tokens`` / ``max_completion_tokens`` (int) — output cap.
    - ``thinking_budget`` / ``thinking_budget_tokens`` (int) — forwarded
      into ``extra_body.chat_template_kwargs.thinking_budget``. Also
      accepted under ``thinking: {budget|budget_tokens|max_tokens: ...}``.
    """
    provider = str(
        section.get("_provider_label")
        or section.get("provider")
        or "",
    )

    provider_type = _resolve_provider_type(provider)
    model = str(section.get("model") or "")
    if provider_type in {"anthropic", "bedrock"}:
        if not _is_claude_model(model):
            raise ValueError(
                f"provider {provider!r} uses {provider_type!r} transport, "
                f"which requires a Claude model; got {model!r}",
            )
        client = _build_anthropic_aux_llm(
            section,
            bedrock=provider_type == "bedrock",
        )
    else:
        client = _build_openai_compat_aux_llm(section, provider)

    from frontier_agent.infra.prompt_cache import maybe_wrap_for_prompt_cache
    client = maybe_wrap_for_prompt_cache(
        client, provider=provider, model=str(section.get("model", "")),
    )

    from frontier_agent.infra.llm import with_provider_stamp
    return with_provider_stamp(client, provider) if provider else client


def _resolve_provider_type(provider_name: str) -> str:
    """Look up ``type:`` for a provider in ``config/providers.yaml``.

    Defaults to ``openai_compat`` when the provider isn't registered —
    legacy profile sections that use raw ``api_key`` / ``base_url``
    interpolation without going through the registry land here, and
    those were openai-compat by construction.
    """
    if not provider_name:
        return "openai_compat"
    try:
        from frontier_agent.infra.providers import load_providers
        return str(
            load_providers().get(provider_name, {}).get("type")
            or "openai_compat",
        ).lower()
    except Exception:
        return "openai_compat"


def _is_claude_model(model: str) -> bool:
    """Whether a resolved model id belongs to the Claude family.

    The check runs against the *attempt's* model id, not the profile's
    canonical name. It therefore accepts direct Anthropic ids
    (``claude-sonnet-4-6``), OpenRouter's Anthropic skin
    (``anthropic/claude-sonnet-4.6``), and Bedrock inference profiles
    (``global.anthropic.claude-sonnet-4-6``), while keeping
    self-hosted models on the OpenAI-compatible path.
    """
    return "claude" in (model or "").strip().lower()


def _anthropic_thinking_config(section: dict[str, Any]) -> dict[str, Any] | None:
    """Translate an aux section's optional native-thinking block."""
    raw = section.get("thinking")
    if not isinstance(raw, dict):
        return None
    thinking_type = str(raw.get("type") or "").strip().lower()
    if thinking_type in {"", "disabled", "off", "none", "false"}:
        return None
    return dict(raw)


def _build_anthropic_aux_llm(
    section: dict[str, Any],
    *,
    bedrock: bool = False,
) -> Any:
    """Build an ``AnthropicClient`` for sections whose resolved provider has
    ``type: anthropic`` or ``type: bedrock`` in providers.yaml.

    Anthropic Messages API lives at ``/v1/messages``, not
    ``/v1/chat/completions`` — using ``OpenAIClient`` against
    ``api.anthropic.com`` returns 404. Bedrock uses the same message shape but
    changes endpoint/auth through ``AnthropicClient(..., bedrock=True)``.
    OpenAI-only ``extra_body`` / ``chat_template_kwargs`` knobs remain ignored.
    """
    from frontier_agent.infra.anthropic_client import AnthropicClient

    provider = str(
        section.get("_provider_label") or section.get("provider") or "",
    )
    kwargs: dict[str, Any] = {
        "model": section["model"],
        "api_key": _resolve_api_key(section, provider),
        "temperature": section.get("temperature", 0.0),
        "timeout": float(section.get("llm_timeout_s", 120)),
        "thinking": _anthropic_thinking_config(section),
        "effort": str(section.get("effort") or ""),
        "bedrock": bedrock,
    }
    base_url = section.get("base_url")
    if base_url:
        kwargs["base_url"] = base_url
    extra_headers = section.get("extra_headers")
    if isinstance(extra_headers, dict) and extra_headers:
        kwargs["default_headers"] = {
            str(key): str(value)
            for key, value in extra_headers.items()
            if value is not None
        }
    max_tokens = (
        section.get("max_completion_tokens") or section.get("max_tokens")
    )
    if max_tokens is not None:
        kwargs["max_tokens"] = int(max_tokens)
    return AnthropicClient(**kwargs)


def _build_openai_compat_aux_llm(
    section: dict[str, Any], provider: str,
) -> Any:
    """Build a ``ChatOpenAI`` for OpenAI-compatible aux LLM sections."""
    from frontier_agent.infra.llm.chat_openai import ChatOpenAI

    kwargs: dict[str, Any] = {
        "model": section["model"],
        "api_key": _resolve_api_key(section, provider),
        "base_url": section.get("base_url") or None,
        "temperature": section.get("temperature", 0.0),
        "timeout": float(section.get("llm_timeout_s", 120)),
    }

    max_tokens = section.get("max_completion_tokens") or section.get("max_tokens")
    if max_tokens is not None:
        kwargs["max_completion_tokens"] = int(max_tokens)

    extra_body = dict(section.get("extra_body") or {})
    chat_template_kwargs = dict(extra_body.get("chat_template_kwargs") or {})
    if section.get("enable_thinking"):
        chat_template_kwargs["enable_thinking"] = True
        chat_template_kwargs.setdefault("preserve_thinking", False)
    thinking_budget = _resolve_thinking_budget(section)
    if thinking_budget is not None:
        chat_template_kwargs["thinking_budget"] = int(thinking_budget)
    if chat_template_kwargs:
        extra_body["chat_template_kwargs"] = chat_template_kwargs
    if extra_body:
        kwargs["extra_body"] = extra_body

    # Aliyun EAS-backed providers (apodex / aliyun_pai_eas{,_35b}) use
    # ``x-upstream-session-id`` for sticky
    # upstream routing — pin to the current task's session id so every
    # LLM call in this task hits the same KV-cached node. Header is
    # empty (no-op) for other providers + outside a task context. Must
    # be set before constructing ChatOpenAI since default_headers
    # can't be patched after.
    from frontier_agent.infra.session_context import (
        eas_session_headers,
        fold_extra_headers,
        provider_needs_session_header,
    )
    if provider_needs_session_header(provider):
        session_headers = eas_session_headers()
        if session_headers:
            # ``session_suffix`` steers this workload to a DIFFERENT sticky
            # worker than the main task. ``x-upstream-session-id`` is HARD
            # sticky on the EAS gateway (one session-id → one pinned worker),
            # so appending e.g. ":dag" to the task id routes these calls off
            # the react worker — where the analyzer's prompt prefix would
            # otherwise evict react's radix cache and its long-transcript
            # prefills would starve react's batching slots. Empty/absent ⇒
            # share the task session (unchanged behaviour).
            suffix = str(section.get("session_suffix") or "")
            if suffix:
                base_sid = session_headers.get("x-upstream-session-id", "")
                if base_sid:
                    session_headers = {
                        **session_headers,
                        "x-upstream-session-id": f"{base_sid}{suffix}",
                    }
            kwargs["default_headers"] = {
                **(kwargs.get("default_headers") or {}),
                **session_headers,
            }

    # Provider-declared headers (e.g. aliyun X-DashScope-DataInspection)
    # injected onto the section by ``_inject_provider_creds`` — fold into
    # default_headers so they ride every aux-LLM request too.
    section_headers = section.get("extra_headers")
    if isinstance(section_headers, dict) and section_headers:
        kwargs["default_headers"] = fold_extra_headers(
            kwargs.get("default_headers"), section_headers,
        )

    return ChatOpenAI(**kwargs)


def _resolve_thinking_budget(section: dict[str, Any]) -> Any | None:
    for key in ("thinking_budget", "thinking_budget_tokens"):
        value = section.get(key)
        if value is not None:
            return value
    thinking = section.get("thinking")
    if isinstance(thinking, dict):
        for key in ("budget", "budget_tokens", "max_tokens"):
            value = thinking.get(key)
            if value is not None:
                return value
    return None


__all__ = ["build_aux_llm"]
