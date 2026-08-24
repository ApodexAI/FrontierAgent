"""``LLMProxy`` — transparent :class:`LLMClient` wrapping a middleware chain."""

from __future__ import annotations

import itertools
import logging
import time
from collections.abc import AsyncIterator
from typing import Any

from frontier_agent.components.middleware.llm.base import (
    LLMCallContext,
)
from frontier_agent.core.execution_context import (
    ensure_trace_metadata,
    get_current_execution_scope,
)
from frontier_agent.core.llm import LLMResponse, StreamDelta
from frontier_agent.core.messages import Message

logger = logging.getLogger(__name__)

__all__ = ["LLMProxy"]


async def _log_llm_exception(
    ctx: LLMCallContext, error: Exception,
) -> None:
    """Best-effort trace emission for LLM failures.

    Resolves the tracer through the structural ``core.protocols.TraceSink``
    Protocol — the proxy never imports a concrete implementation. A caller
    registers whichever sink it wants (a file-backed logger, a no-op, a JSONL
    writer) against ``TraceSink`` in the service registry, or skips
    registration entirely, in which case this method is a no-op.
    """
    try:
        from frontier_agent.core.protocols import TraceSink
        from frontier_agent.core.runtime.registries import services as registry

        tracer: TraceSink | None = registry.get_optional(TraceSink)
        if not tracer:
            return
        await tracer.log_api_error(
            task_id=ctx.task_id or "unknown",
            agent_role_id=ctx.role_id or "default",
            error=str(error),
            session_id=ctx.metadata.get("session_id"),
            prompt_id=ctx.metadata.get("prompt_id"),
            step_id=ctx.metadata.get("step_id"),
            metadata={
                "phase_id": ctx.phase_id,
                "call_index": ctx.call_index,
            },
        )
    except Exception:
        logger.debug("Failed to log LLM exception", exc_info=True)


class LLMProxy:
    """Transparent :class:`LLMClient` wrapper that applies LLM middleware.

    Returned by ``ResourceManager.get_llm()`` when a chain is registered.
    All callers (pipeline nodes, the agent loop) use it exactly like a
    normal ``LLMClient``.
    """

    def __init__(
        self,
        inner: Any,
        chain: Any,
        role_id: str = "default",
    ) -> None:
        self.inner = inner
        self.chain = chain
        self.role_id = role_id
        self.model = getattr(inner, "model", "") or ""
        self._counter = itertools.count(1)

    @property
    def call_counter(self) -> int:
        """Last-issued call index (peek without advancing).

        Exposed for tests/diagnostics only; hot paths use
        ``_next_call_index``.
        """
        import re
        m = re.search(r"count\((\d+)\)", repr(self._counter))
        return int(m.group(1)) - 1 if m else 0

    def _next_call_index(self) -> int:
        """Atomically reserve the next call index.

        ``itertools.count.__next__`` is GIL-atomic under CPython, so no
        explicit lock is needed to keep indices unique across concurrent
        ``chat`` / ``stream`` invocations sharing this proxy.
        """
        return next(self._counter)

    def _make_ctx(self, call_index: int) -> LLMCallContext:
        scope = get_current_execution_scope()
        metadata = dict(scope.metadata) if scope else {}
        default_step_id = ""
        if scope:
            default_step_id = (
                f"{scope.phase_id or 'phase'}:llm:{call_index}"
            )
            metadata = ensure_trace_metadata(
                metadata,
                default_step_id=default_step_id,
                refresh_prompt_id=True,
            )
            metadata["step_id"] = default_step_id
            scope.metadata.update(metadata)
        # Stash the inner model id so a cost-accounting middleware can record
        # spend even when the provider's response_metadata doesn't echo
        # model_name (some models behind an OpenAI-compat gateway don't).
        inner_model = (
            getattr(self.inner, "model", None)
            or getattr(self.inner, "model_name", None)
        )
        if inner_model:
            metadata.setdefault("model_id", str(inner_model))
        return LLMCallContext(
            task_id=scope.task_id if scope else "",
            role_id=self.role_id,
            phase_id=scope.phase_id if scope else "",
            call_index=call_index,
            metadata=metadata,
        )

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
        ctx = self._make_ctx(self._next_call_index())

        messages = await self.chain.run_before(ctx, messages)

        attempt = 0
        while True:
            try:
                response = await self.inner.chat(
                    messages,
                    tools=tools,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    extra_headers=extra_headers,
                    timeout=timeout,
                )
                break
            except Exception as exc:
                await _log_llm_exception(ctx, exc)
                should_retry = await self.chain.run_on_llm_error(
                    ctx, exc, attempt,
                )
                if not should_retry:
                    raise
                attempt += 1
                logger.info(
                    "LLMProxy: retrying after attempt %d (%s)",
                    attempt, type(exc).__name__,
                )

        return await self.chain.run_after(ctx, response)

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
        """Stream with before/after hooks and retry logic."""
        ctx = self._make_ctx(self._next_call_index())
        messages = await self.chain.run_before(ctx, messages)

        attempt = 0
        while True:
            start_time = time.time()
            full_content = ""
            full_reasoning = ""
            stream_error: Exception | None = None
            any_chunk_yielded = False

            try:
                async for delta in self.inner.stream(
                    messages,
                    tools=tools,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    extra_headers=extra_headers,
                    timeout=timeout,
                ):
                    full_content += delta.content or ""
                    full_reasoning += delta.reasoning_content or ""
                    any_chunk_yielded = True
                    # Per-chunk middleware hook. A middleware returning
                    # True (e.g. StreamRepetitionDetector noticing a
                    # degenerate loop) tells us to stop consuming the
                    # inner stream and exit cleanly — the partial
                    # response still flows through ``after_llm`` in the
                    # finally block so observers see the truncated
                    # content rather than nothing. The delta that
                    # triggered the abort IS still yielded so the
                    # consumer's accumulator stays consistent with the
                    # LLMResponse we'll synthesise.
                    yield delta
                    if await self.chain.run_on_chunk(
                        ctx, delta, full_content,
                    ):
                        ctx.metadata["stream_aborted_by_middleware"] = True
                        logger.info(
                            "LLMProxy: stream aborted by middleware "
                            "after %d chars", len(full_content),
                        )
                        break
                break
            except Exception as e:
                stream_error = e
                await _log_llm_exception(ctx, e)
                if not any_chunk_yielded:
                    should_retry = (
                        await self.chain.run_on_llm_error(
                            ctx, e, attempt,
                        )
                    )
                    if should_retry:
                        attempt += 1
                        logger.info(
                            "LLMProxy: retrying stream after "
                            "attempt %d (%s)",
                            attempt, type(e).__name__,
                        )
                        continue
                raise
            finally:
                duration_ms = int(
                    (time.time() - start_time) * 1000,
                )
                ctx.metadata["duration_ms"] = duration_ms
                if stream_error:
                    ctx.metadata["error"] = str(stream_error)

                await self.chain.run_after(
                    ctx,
                    LLMResponse(
                        content=full_content,
                        reasoning_content=full_reasoning,
                    ),
                )

    def __getattr__(self, name: str) -> Any:
        # Defer unknown attributes to the inner client so callers that
        # read provider-specific fields (e.g. ``base_url``) keep working.
        # Underscore-prefixed names are never proxied — they're internal
        # state set in ``__init__`` and must raise cleanly when missing.
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self.inner, name)
