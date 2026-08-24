"""LLM provider factory — configurable OpenAI / Anthropic / Qwen / DeepSeek."""

from __future__ import annotations

import asyncio
import logging
import random
import time
from collections.abc import AsyncIterator
from typing import Any, TypeGuard

from frontier_agent.core.execution_context import ensure_trace_metadata, get_current_execution_scope
from frontier_agent.core.llm import LLMClient, LLMResponse, StreamDelta
from frontier_agent.core.messages import Message
from frontier_agent.infra.config import FrontierAgentConfig

logger = logging.getLogger(__name__)


_MODEL_TIERS = {"light", "medium", "strong"}


def _resolve_model_name(config: FrontierAgentConfig, provider: str, model: str | None) -> str:
    """Resolve a model override.

    Accepts either a concrete model name or a logical tier label
    (`light`/`medium`/`strong`). Tier labels resolve through optional
    config overrides and fall back to the provider default model.
    """
    default_model = getattr(config, f"{provider}_model", "gpt-4o")
    if not model:
        return default_model

    model_key = model.strip()
    if model_key not in _MODEL_TIERS:
        return model_key

    configured = getattr(config, f"model_{model_key}", "").strip()
    return configured or default_model


_RETRYABLE_KEYWORDS = frozenset({
    "timeout", "timed out", "429", "500", "502", "503", "504", "529",
    "overloaded", "rate limit", "rate_limit", "server error",
    "connection reset", "connection error", "econnreset",
    "gateway timeout",
    "model_dump",  # OpenRouter proxy returns malformed response → AttributeError
    "model_not_found",  # Proxy distributor group doesn't have the model
})


def _is_retryable(error: Exception) -> bool:
    """Check if the error is transient and worth retrying.

    Also treats AttributeError from malformed proxy responses as retryable,
    since these are typically caused by proxy-side issues (e.g., model_dump
    on a non-Pydantic response from OpenRouter-compatible gateways).
    """
    if isinstance(error, AttributeError):
        return True
    msg = str(error).lower()
    return any(kw in msg for kw in _RETRYABLE_KEYWORDS)


class FallbackLLM(LLMClient):
    """Transparent :class:`LLMClient` wrapper: retry primary, then degrade to fallback model.

    When ``llm_fallback_model`` is set, ``create_llm()`` wraps the primary LLM
    in this class. All callers (ResourceManager, LLMProxy, pipeline nodes)
    see a normal ``LLMClient`` — zero upstream changes.
    """

    def __init__(
        self,
        primary: Any,
        fallback: Any,
        max_retries: int = 2,
        cooldown_seconds: int = 60,
        trace_sink: Any | None = None,
    ) -> None:
        self.primary = primary
        self.fallback = fallback
        self.max_retries = max_retries
        self.cooldown_seconds = cooldown_seconds
        self.trace_sink = trace_sink
        self._cooldown_until: float = 0.0
        # A plain attribute, not a property: LLMClient declares ``model`` as a
        # settable attribute (concrete clients assign it in __init__), and a
        # read-only property here does not satisfy that. ``primary`` is fixed at
        # construction, so the label is computed once rather than per access.
        self.model: str = self._model_label(primary)

    @property
    def model_name(self) -> str:
        return f"fallback({self._model_label(self.primary)})"

    @staticmethod
    def _model_label(llm: Any) -> str:
        return (
            getattr(llm, "model_name", None)
            or getattr(llm, "model", None)
            or llm.__class__.__name__
        )

    async def _trace(
        self,
        method_name: str,
        *,
        metadata: dict[str, Any] | None = None,
        retry_count: int = 0,
        degrade_from: str | None = None,
        degrade_to: str | None = None,
        error: str | None = None,
        reason: str | None = None,
    ) -> None:
        try:
            scope = get_current_execution_scope()
            if not scope:
                return
            ensure_trace_metadata(scope.metadata, refresh_prompt_id=False)
            tracer = self.trace_sink
            if not tracer:
                return

            common = dict(
                task_id=scope.task_id or "unknown",
                agent_role_id=scope.role_id or "default",
                session_id=scope.metadata.get("session_id"),
                prompt_id=scope.metadata.get("prompt_id"),
                step_id=scope.metadata.get("step_id"),
            )
            payload = metadata or {}
            if method_name == "log_api_request":
                await tracer.log_api_request(
                    provider=str(payload.get("provider", "llm")),
                    model=str(payload.get("model", "unknown")),
                    metadata=payload,
                    **common,
                )
            elif method_name == "log_api_error":
                await tracer.log_api_error(
                    error=error or "LLM error",
                    metadata=payload,
                    **common,
                )
            elif method_name == "log_retry":
                await tracer.log_retry(
                    action=str(payload.get("action", "llm_retry")),
                    retry_count=retry_count,
                    metadata=payload,
                    **common,
                )
            elif method_name == "log_degrade":
                await tracer.log_degrade(
                    reason=reason or "llm_degrade",
                    degrade_from=degrade_from or "unknown",
                    degrade_to=degrade_to or "unknown",
                    metadata=payload,
                    **common,
                )
        except Exception:
            logger.debug("FallbackLLM trace emission failed", exc_info=True)

    # ── core async path ─────────────────────────────────────────────

    async def chat(
        self,
        messages: list[Message],
        *,
        tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        extra_headers: dict[str, str] | None = None,
        timeout: float | None = None,
    ) -> LLMResponse:
        call_kwargs: dict[str, Any] = dict(
            tools=tools,
            temperature=temperature,
            max_tokens=max_tokens,
            extra_headers=extra_headers,
            timeout=timeout,
        )

        # While in cooldown, go straight to fallback
        if time.time() < self._cooldown_until:
            logger.debug("FallbackLLM: primary in cooldown, using fallback")
            await self._trace(
                "log_degrade",
                reason="primary_cooldown",
                degrade_from=self._model_label(self.primary),
                degrade_to=self._model_label(self.fallback),
            )
            await self._trace(
                "log_api_request",
                metadata={"provider": "fallback", "model": self._model_label(self.fallback), "mode": "cooldown"},
            )
            return await self.fallback.chat(messages, **call_kwargs)

        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                await self._trace(
                    "log_api_request",
                    metadata={"provider": "primary", "model": self._model_label(self.primary), "attempt": attempt + 1},
                )
                return await self.primary.chat(messages, **call_kwargs)
            except Exception as e:
                last_error = e
                await self._trace(
                    "log_api_error",
                    error=str(e),
                    metadata={"provider": "primary", "model": self._model_label(self.primary), "attempt": attempt + 1},
                )
                if not _is_retryable(e):
                    break
                delay = min(0.5 * (2 ** attempt), 8) + random.random() * 0.25
                logger.warning(
                    "FallbackLLM: primary attempt %d/%d failed (%s), retrying in %.1fs",
                    attempt + 1, self.max_retries, type(e).__name__, delay,
                )
                await self._trace(
                    "log_retry",
                    retry_count=attempt + 1,
                    metadata={
                        "action": "primary_llm_retry",
                        "delay_s": round(delay, 3),
                        "provider": "primary",
                        "model": self._model_label(self.primary),
                    },
                )
                await asyncio.sleep(delay)

        # Primary exhausted — switch to fallback
        logger.warning(
            "FallbackLLM: primary exhausted after %d attempts (%s), switching to fallback for %ds",
            self.max_retries, last_error, self.cooldown_seconds,
        )
        self._cooldown_until = time.time() + self.cooldown_seconds
        try:
            await self._trace(
                "log_degrade",
                reason="primary_exhausted",
                degrade_from=self._model_label(self.primary),
                degrade_to=self._model_label(self.fallback),
                metadata={"cooldown_seconds": self.cooldown_seconds},
            )
            await self._trace(
                "log_api_request",
                metadata={"provider": "fallback", "model": self._model_label(self.fallback), "mode": "degraded"},
            )
            return await self.fallback.chat(messages, **call_kwargs)
        except Exception as fb_err:
            logger.error("FallbackLLM: fallback also failed: %s", fb_err)
            await self._trace(
                "log_api_error",
                error=str(fb_err),
                metadata={"provider": "fallback", "model": self._model_label(self.fallback)},
            )
            # ``last_error`` is None when the retry loop never ran
            # (max_retries <= 0): there is no original error to surface and
            # the fallback failure is the whole story. A bare ``raise``
            # re-raises fb_err; ``raise None`` would become a TypeError that
            # hid the real cause.
            if last_error is None:
                raise
            raise last_error from fb_err  # raise original error, chain fallback error

    # ── streaming ───────────────────────────────────────────────────

    async def stream(
        self,
        messages: list[Message],
        *,
        tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        extra_headers: dict[str, str] | None = None,
        timeout: float | None = None,
    ) -> AsyncIterator[StreamDelta]:
        call_kwargs: dict[str, Any] = dict(
            tools=tools,
            temperature=temperature,
            max_tokens=max_tokens,
            extra_headers=extra_headers,
            timeout=timeout,
        )

        # While in cooldown, go straight to fallback
        if time.time() < self._cooldown_until:
            logger.debug("FallbackLLM: primary in cooldown, streaming from fallback")
            await self._trace(
                "log_degrade",
                reason="primary_stream_cooldown",
                degrade_from=self._model_label(self.primary),
                degrade_to=self._model_label(self.fallback),
            )
            async for delta in self.fallback.stream(messages, **call_kwargs):
                yield delta
            return

        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                await self._trace(
                    "log_api_request",
                    metadata={"provider": "primary", "model": self._model_label(self.primary), "attempt": attempt + 1, "streaming": True},
                )
                async for delta in self.primary.stream(messages, **call_kwargs):
                    yield delta
                return  # stream completed successfully
            except Exception as e:
                last_error = e
                await self._trace(
                    "log_api_error",
                    error=str(e),
                    metadata={"provider": "primary", "model": self._model_label(self.primary), "attempt": attempt + 1, "streaming": True},
                )
                if not _is_retryable(e):
                    break
                delay = min(0.5 * (2 ** attempt), 8) + random.random() * 0.25
                logger.warning(
                    "FallbackLLM: primary stream attempt %d/%d failed (%s), retrying in %.1fs",
                    attempt + 1, self.max_retries, type(e).__name__, delay,
                )
                await self._trace(
                    "log_retry",
                    retry_count=attempt + 1,
                    metadata={
                        "action": "primary_stream_retry",
                        "delay_s": round(delay, 3),
                        "provider": "primary",
                        "model": self._model_label(self.primary),
                    },
                )
                await asyncio.sleep(delay)

        # Primary exhausted — switch to fallback stream
        logger.warning(
            "FallbackLLM: primary stream exhausted, switching to fallback for %ds",
            self.cooldown_seconds,
        )
        self._cooldown_until = time.time() + self.cooldown_seconds
        try:
            await self._trace(
                "log_degrade",
                reason="primary_stream_exhausted",
                degrade_from=self._model_label(self.primary),
                degrade_to=self._model_label(self.fallback),
                metadata={"cooldown_seconds": self.cooldown_seconds, "streaming": True},
            )
            async for delta in self.fallback.stream(messages, **call_kwargs):
                yield delta
        except Exception as fb_err:
            logger.error("FallbackLLM: fallback stream also failed: %s", fb_err)
            await self._trace(
                "log_api_error",
                error=str(fb_err),
                metadata={"provider": "fallback", "model": self._model_label(self.fallback), "streaming": True},
            )
            # See the non-streaming path above: with max_retries <= 0 the retry
            # loop never ran, so there is no original error and a bare ``raise``
            # re-raises fb_err rather than raising None.
            if last_error is None:
                raise
            raise last_error from fb_err


def _anthropic_thinking_kwarg(config: Any, max_tokens: int) -> dict[str, Any] | None:
    """Anthropic extended-thinking request kwarg, or ``None`` when off.

    ``ANTHROPIC_THINKING`` on → the shape selected by ``anthropic_thinking_type``
    (default ``adaptive``); matches the official model matrix (see
    ``protocol_client._build_anthropic``):

    - ``adaptive`` (default) — ``{"type":"adaptive","display":<display>}`` with
      ``display`` from ``anthropic_thinking_display`` (default ``summarized``).
      The RECOMMENDED / only mode on current Claude (Opus 4.6+, Sonnet 4.6+).
    - ``enabled`` — ``{"type":"enabled","budget_tokens":N}`` (``N`` from
      ``anthropic_thinking_budget`` clamped to ``[1024, max_tokens-1]``). LEGACY
      opt-in for models older than Opus 4.6 / Sonnet 4.6, which reject adaptive.
    """
    if not getattr(config, "anthropic_thinking", False):
        return None
    ttype = str(getattr(config, "anthropic_thinking_type", "adaptive") or "adaptive").strip().lower()
    if ttype == "enabled":
        budget = int(getattr(config, "anthropic_thinking_budget", 8192) or 8192)
        budget = max(1024, min(budget, int(max_tokens) - 1))
        return {"type": "enabled", "budget_tokens": budget}
    thinking: dict[str, Any] = {"type": "adaptive"}
    display = str(getattr(config, "anthropic_thinking_display", "summarized") or "").strip()
    if display:
        thinking["display"] = display
    return thinking


def _anthropic_effort_kwarg(config: Any, thinking: dict[str, Any] | None) -> str:
    """Reasoning effort to forward — only with ADAPTIVE thinking.

    ``effort`` is an adaptive-only knob (the oldest models 400 on
    ``enabled``+effort; ``budget_tokens`` is the control knob for ``enabled``)."""
    if not thinking or thinking.get("type") != "adaptive":
        return ""
    return str(getattr(config, "anthropic_effort", "") or "")


def _create_single_provider(config: FrontierAgentConfig, model_override: str | None = None) -> LLMClient:
    """Create a single LLM instance for the configured provider, optionally overriding the model name."""
    provider = config.llm_provider.lower()

    max_tokens = config.llm_max_tokens

    if provider == "openai":
        from frontier_agent.infra.openai_client import OpenAIClient
        kwargs: dict[str, Any] = dict(
            model=model_override or config.openai_model,
            api_key=config.openai_api_key,
            base_url=config.openai_base_url if config.openai_base_url != "https://api.openai.com/v1" else None,
            temperature=0.3,
            max_completion_tokens=max_tokens,
            timeout=300,
        )
        # OpenRouter-compatible proxies need these headers for correct routing.
        # Use default_headers so they apply to all internal SDK clients.
        if config.openai_base_url and config.openai_base_url != "https://api.openai.com/v1":
            kwargs["default_headers"] = {
                "HTTP-Referer": "frontier_agent",
                "X-Title": "FrontierAgent",
            }
        return OpenAIClient(**kwargs)

    elif provider == "anthropic":
        from frontier_agent.infra.anthropic_client import AnthropicClient
        # Opt-in extended thinking (ANTHROPIC_THINKING): standard enabled+budget
        # thinking + drop temperature (Anthropic 400s on the combo). Flag off →
        # prior kwargs preserved exactly (temperature=0.3, no thinking).
        thinking = _anthropic_thinking_kwarg(config, max_tokens)
        return AnthropicClient(
            model=model_override or config.anthropic_model,
            api_key=config.anthropic_api_key,
            base_url=config.anthropic_base_url or None,
            temperature=None if thinking else 0.3,
            max_tokens=max_tokens,
            thinking=thinking,
            effort=_anthropic_effort_kwarg(config, thinking),
        )

    elif provider == "qwen":
        from frontier_agent.infra.openai_client import OpenAIClient
        return OpenAIClient(
            model=model_override or config.qwen_model,
            api_key=config.qwen_api_key,
            base_url=config.qwen_base_url,
            temperature=0.3,
            max_completion_tokens=max_tokens,
        )

    elif provider == "deepseek":
        from frontier_agent.infra.openai_client import OpenAIClient
        return OpenAIClient(
            model=model_override or config.deepseek_model,
            api_key=config.deepseek_api_key,
            base_url=config.deepseek_base_url,
            temperature=0.3,
            max_completion_tokens=max_tokens,
        )

    else:
        raise ValueError(f"Unknown LLM provider: {provider}")


def create_llm(config: FrontierAgentConfig) -> LLMClient:
    """Create an LLMClient based on the configured provider.

    Fallback selection (in priority order):
    1. ``llm_fallback_chain`` (list of entries) — wraps the primary in an
       ``LLMFallbackChain``. Each entry can target a different provider /
       model / endpoint.
    2. ``llm_fallback_model`` (single string, legacy) — wraps in the older
       ``FallbackLLM``. Kept for back-compat.
    3. Neither set — return the primary LLM unwrapped.
    """
    primary = _create_single_provider(config)

    if config.llm_fallback_chain:
        return _build_fallback_chain(config, primary)

    if config.llm_fallback_model:
        fallback = _create_single_provider(config, model_override=config.llm_fallback_model)
        logger.info(
            "FallbackLLM enabled: primary=%s, fallback=%s",
            getattr(primary, "model", "?"), config.llm_fallback_model,
        )
        return FallbackLLM(
            primary=primary,
            fallback=fallback,
            max_retries=config.llm_fallback_max_retries,
            cooldown_seconds=config.llm_fallback_cooldown,
        )

    return primary


def _build_fallback_chain(
    config: FrontierAgentConfig, primary: LLMClient,
) -> LLMClient:
    """Build an ``LLMFallbackChain`` from ``config.llm_fallback_chain``.

    The primary LLM is the chain's first entry (its triggers default to
    ``("any_error",)`` since the user hasn't named it explicitly). Each
    config entry becomes one additional ``FallbackEntry``.
    """
    from typing import get_args

    from frontier_agent.infra.llm import (
        FallbackEntry,
        FallbackTrigger,
        LLMFallbackChain,
    )

    # Source of truth lives on the Literal — no drift if a new trigger is added.
    valid_triggers = set(get_args(FallbackTrigger))

    def _is_trigger(value: object) -> TypeGuard[FallbackTrigger]:
        return value in valid_triggers

    primary_provider = (config.llm_provider or "").lower()
    entries: list[FallbackEntry] = [
        FallbackEntry(model=primary, provider=primary_provider),
    ]
    for raw in config.llm_fallback_chain:
        if not isinstance(raw, dict):
            logger.warning("llm_fallback_chain entry is not a dict: %r — skipping", raw)
            continue
        model = _build_chain_entry_llm(config, raw)
        triggers_raw = raw.get("triggers") or ["any_error"]
        triggers: tuple[FallbackTrigger, ...] = tuple(
            t for t in triggers_raw if _is_trigger(t)
        ) or ("any_error",)
        entry_provider = (
            str(raw.get("provider") or config.llm_provider or "").lower()
        )
        entries.append(FallbackEntry(
            model=model, triggers=triggers, provider=entry_provider,
        ))

    primary_id = getattr(primary, "model", None) or type(primary).__name__
    fallback_ids = [
        getattr(e.model, "model", None) or type(e.model).__name__
        for e in entries[1:]
    ]
    logger.info(
        "LLMFallbackChain enabled: primary=%s fallbacks=%s",
        primary_id, fallback_ids,
    )
    return LLMFallbackChain(entries=entries)


def _create_overridden_provider(
    config: FrontierAgentConfig,
    model: str | None = None,
    temperature: float = 0.3,
    max_tokens: int | None = None,
    *,
    provider: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
) -> LLMClient:
    """Create a single LLM instance with per-role / per-fallback-entry overrides.

    Without explicit overrides this resolves provider / api_key / base_url
    from ``config.llm_provider`` and the matching ``<provider>_*`` fields,
    which is the per-role override path used by
    ``create_llm_with_overrides``. Passing ``provider`` / ``api_key`` /
    ``base_url`` (used by ``_build_chain_entry_llm`` for fallback-chain
    entries) lets a single entry target a different vendor.
    """
    if max_tokens is None:
        max_tokens = config.llm_max_tokens
    resolved_provider = (provider or config.llm_provider).lower()

    if resolved_provider in ("openai", "qwen", "deepseek"):
        from frontier_agent.infra.openai_client import OpenAIClient

        resolved_key = (
            api_key
            if api_key is not None
            else getattr(config, f"{resolved_provider}_api_key", "")
        )
        resolved_base_url = (
            base_url
            if base_url is not None
            else getattr(config, f"{resolved_provider}_base_url", None)
        )
        resolved_model = model if (provider and model) else _resolve_model_name(
            config, resolved_provider, model,
        )

        extra_kwargs: dict[str, Any] = {}
        # Skip ``base_url`` when it's the OpenAI default so the SDK uses
        # its own built-in default.
        if resolved_base_url and resolved_base_url != "https://api.openai.com/v1":
            extra_kwargs["base_url"] = resolved_base_url
            extra_kwargs["default_headers"] = {
                "HTTP-Referer": "frontier_agent",
                "X-Title": "FrontierAgent",
            }
        return OpenAIClient(
            model=resolved_model,
            api_key=resolved_key,
            temperature=temperature,
            max_completion_tokens=max_tokens,
            timeout=300,
            **extra_kwargs,
        )

    if resolved_provider == "anthropic":
        from frontier_agent.infra.anthropic_client import AnthropicClient

        # Opt-in extended thinking (see sibling branch above): standard
        # enabled+budget thinking + drop temperature. Flag off → prior kwargs
        # preserved exactly.
        thinking = _anthropic_thinking_kwarg(config, max_tokens)
        return AnthropicClient(
            model=model if (provider and model) else _resolve_model_name(
                config, resolved_provider, model,
            ),
            api_key=api_key if api_key is not None else config.anthropic_api_key,
            # Honour a per-entry base_url (fallback-chain / override) — fall back
            # to the configured default only when the caller didn't pass one.
            base_url=(base_url if base_url is not None else config.anthropic_base_url) or None,
            temperature=None if thinking else temperature,
            max_tokens=max_tokens,
            thinking=thinking,
            effort=_anthropic_effort_kwarg(config, thinking),
        )

    raise ValueError(f"Unknown LLM provider: {resolved_provider}")


def _build_chain_entry_llm(
    config: FrontierAgentConfig, entry: dict[str, Any],
) -> LLMClient:
    """Build a single ``LLMClient`` from one ``fallback_chain`` entry.

    Each field falls back to the matching ``config`` default when omitted,
    so a minimal entry like ``{"model": "gpt-4o"}`` reuses the primary
    provider's key + base_url. Cross-provider entries just specify
    ``provider`` + ``model`` (+ optional ``api_key`` / ``base_url``).
    """
    model = entry.get("model")
    if not model:
        raise ValueError(
            f"llm_fallback_chain entry missing 'model': {entry!r}",
        )
    return _create_overridden_provider(
        config,
        model=model,
        provider=entry.get("provider") or config.llm_provider or "openai",
        api_key=entry.get("api_key") or None,
        base_url=entry.get("base_url") or None,
    )


def create_llm_with_overrides(
    config: FrontierAgentConfig,
    model: str | None = None,
    temperature: float = 0.3,
    max_tokens: int | None = None,
) -> LLMClient:
    """Create an LLMClient with per-role overrides (model, temperature, max_tokens).

    Used by ResourceManager.get_llm(role_id) for roles with custom model config.
    Wraps in ``LLMFallbackChain`` (preferred) or legacy ``FallbackLLM``
    when fallback is configured.
    """
    primary = _create_overridden_provider(config, model, temperature, max_tokens)

    if config.llm_fallback_chain:
        return _build_fallback_chain(config, primary)

    if config.llm_fallback_model:
        fallback = _create_overridden_provider(
            config, model=config.llm_fallback_model,
            temperature=temperature, max_tokens=max_tokens,
        )
        return FallbackLLM(
            primary=primary,
            fallback=fallback,
            max_retries=config.llm_fallback_max_retries,
            cooldown_seconds=config.llm_fallback_cooldown,
        )

    return primary
