"""LLM Middleware framework — context, protocol, chain, and proxy."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from frontier_agent.core.llm import LLMResponse, StreamDelta
from frontier_agent.core.messages import Message

logger = logging.getLogger(__name__)


# ── Helpers ──────────────────────────────────────────────────────────────


def unwrap_runnable_binding(
    model: Any, kwargs: dict[str, Any],
) -> tuple[Any, dict[str, Any]]:
    """Identity passthrough retained for backward-compatible imports.

    In the langchain era this unwrapped a ``RunnableBinding`` (the object
    ``BaseChatModel.bind_tools()`` returned) and merged its stored kwargs.
    The native :class:`~frontier_agent.core.llm.LLMClient` substrate has no
    such wrapper — tools / temperature / headers are passed per call to
    :meth:`LLMClient.chat` — so there is nothing to unwrap and this returns
    ``(model, kwargs)`` unchanged. Kept (and re-exported by the package
    ``__init__``) only so any lingering import site keeps resolving.
    """
    return model, kwargs


# Keywords that indicate a transient error worth retrying.
# Shared with FallbackLLM in llm_adapter.py — keep in sync.
_RETRYABLE_KEYWORDS = frozenset({
    "timeout", "timed out", "429", "500", "502", "503", "504", "529",
    "overloaded", "rate limit", "rate_limit", "server error",
    "connection reset", "connection error", "econnreset",
    "gateway timeout",
    "model_dump",
    "model_not_found",
})


def _is_retryable(error: Exception) -> bool:
    """Return True if *error* looks transient and worth retrying."""
    if isinstance(error, AttributeError):
        return True
    msg = str(error).lower()
    return any(kw in msg for kw in _RETRYABLE_KEYWORDS)


# ── Context ──────────────────────────────────────────────────────────────


@dataclass
class LLMCallContext:
    """Context for a single LLM invocation."""

    task_id: str = ""
    role_id: str = ""
    phase_id: str = ""
    call_index: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


# ── Protocol ─────────────────────────────────────────────────────────────


class LLMMiddleware(ABC):
    """Base class for LLM-call-level middlewares.

    Subclasses override before_llm / after_llm as needed.
    Default implementations are no-ops (pass-through).
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique middleware name for config/logging."""
        ...

    @property
    def enabled(self) -> bool:
        """Override to make middleware dynamically disableable."""
        return True

    async def before_llm(
        self, ctx: LLMCallContext, messages: list[Message],
    ) -> list[Message]:
        """Called before each LLM invocation. Can modify messages."""
        return messages

    async def after_llm(
        self, ctx: LLMCallContext, response: LLMResponse,
    ) -> LLMResponse:
        """Called after each LLM invocation. Can modify response."""
        return response

    async def on_llm_error(
        self, ctx: LLMCallContext, error: Exception, attempt: int,
    ) -> bool:
        """Called when an LLM invocation raises an exception.

        Returns True to retry the call, False to propagate the error.
        """
        return False

    async def on_chunk(
        self,
        ctx: LLMCallContext,
        delta: StreamDelta,
        full_text: str,
    ) -> bool:
        """Called once per streamed delta, between ``inner.stream`` and
        ``yield`` in :meth:`LLMProxy.stream`.

        ``full_text`` is the cumulative concatenation of all
        ``delta.content`` seen so far on this stream — saves middlewares
        from each maintaining their own accumulator.

        Returns ``True`` to **abort** the stream: the proxy stops
        consuming from the inner LLM, runs ``after_llm`` on the
        partial assembled message, and exits the stream cleanly (no
        exception raised). Used by ``StreamRepetitionDetector`` to
        kill degenerate loops mid-generation.

        Returns ``False`` (the default) to continue streaming
        normally.

        Per-chunk hooks must be **fast** — they run on every token of
        every streamed call. Use early-exit checks (e.g. only inspect
        ``full_text`` when ``len(full_text) >= threshold``) and avoid
        per-chunk regex on large strings.
        """
        return False


# ── Chain ────────────────────────────────────────────────────────────────


class LLMMiddlewareChain:
    """Ordered collection of LLM middlewares. Onion model for after."""

    def __init__(self) -> None:
        self._middlewares: list[LLMMiddleware] = []

    def add(self, mw: LLMMiddleware) -> None:
        self._middlewares.append(mw)

    def remove_by_name(self, name: str) -> None:
        self._middlewares = [
            m for m in self._middlewares if m.name != name
        ]

    def get(self, name: str) -> LLMMiddleware | None:
        for m in self._middlewares:
            if m.name == name:
                return m
        return None

    @property
    def middlewares(self) -> list[LLMMiddleware]:
        return list(self._middlewares)

    def wrap_llm(self, llm: Any, *, role_id: str) -> Any:
        """Return an LLM proxy for this chain.

        ``core/runtime`` depends only on the structural
        ``core.protocols.LLMWrapper`` contract; the concrete proxy stays in
        components and is imported here at the component boundary.
        """
        from frontier_agent.components.middleware.llm.proxy import LLMProxy

        return LLMProxy(inner=llm, chain=self, role_id=role_id)

    async def run_before(
        self, ctx: LLMCallContext, messages: list[Message],
    ) -> list[Message]:
        for mw in self._middlewares:
            if mw.enabled:
                try:
                    messages = await mw.before_llm(ctx, messages)
                except Exception:
                    logger.exception(
                        "LLMMiddleware %s.before_llm failed", mw.name,
                    )
        return messages

    async def run_after(
        self, ctx: LLMCallContext, response: LLMResponse,
    ) -> LLMResponse:
        for mw in reversed(self._middlewares):
            if mw.enabled:
                try:
                    response = await mw.after_llm(ctx, response)
                except Exception:
                    logger.exception(
                        "LLMMiddleware %s.after_llm failed", mw.name,
                    )
        return response

    async def run_on_llm_error(
        self, ctx: LLMCallContext, error: Exception, attempt: int,
    ) -> bool:
        for mw in self._middlewares:
            if mw.enabled:
                try:
                    if await mw.on_llm_error(ctx, error, attempt):
                        return True
                except Exception:
                    logger.exception(
                        "LLMMiddleware %s.on_llm_error failed",
                        mw.name,
                    )
        return False

    async def run_on_chunk(
        self,
        ctx: LLMCallContext,
        delta: StreamDelta,
        full_text: str,
    ) -> bool:
        """Fan out to every enabled middleware's ``on_chunk``.

        Returns ``True`` if **any** middleware asks to abort the stream
        (short-circuiting on the first True — later middlewares are
        skipped because the stream is going to terminate anyway). A
        middleware raising inside ``on_chunk`` does NOT abort the
        stream — we log + continue. The stream-control hook should
        never crash a perfectly good generation.
        """
        for mw in self._middlewares:
            if not mw.enabled:
                continue
            try:
                if await mw.on_chunk(ctx, delta, full_text):
                    return True
            except Exception:
                logger.exception(
                    "LLMMiddleware %s.on_chunk failed", mw.name,
                )
        return False
