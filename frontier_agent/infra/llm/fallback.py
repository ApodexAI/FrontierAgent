"""``LLMFallbackChain`` — provider-failover :class:`LLMClient` wrapper."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from frontier_agent.core.llm import LLMResponse, StreamDelta
from frontier_agent.core.messages import Message

logger = logging.getLogger(__name__)

__all__ = [
    "FallbackEntry",
    "FallbackTrigger",
    "LLMFallbackChain",
    "with_provider_stamp",
]


FallbackTrigger = Literal["timeout", "rate_limit", "5xx", "any_error"]


@dataclass
class FallbackEntry:
    """A single tier in the fallback chain.

    Args:
        model: any :class:`LLMClient` instance.
        triggers: which failure modes count as "fall through to the
            next entry". Empty tuple ``()`` means "never fall through" —
            useful as a hard barrier on the last entry. ``("any_error",)``
            means "always fall through".
        provider: vendor label (``openai`` / ``anthropic`` / ``qwen`` /
            ``deepseek``) — stamped onto ``response_metadata`` as
            ``provider_actually_used`` so per-call ``usage`` events can
            tell downstream billing which vendor served the request.
            Empty string when the construction site doesn't know.
    """
    model: Any  # LLMClient
    triggers: tuple[FallbackTrigger, ...] = ("any_error",)
    provider: str = ""

    def matches(self, exc: BaseException) -> bool:
        """Whether ``exc`` should trigger fall-through past this entry."""
        return any(_trigger_matches(trig, exc) for trig in self.triggers)


def _model_id(model: Any) -> str:
    """Best-effort short label for a model. Used for telemetry."""
    return (
        getattr(model, "model_name", None)
        or getattr(model, "model", None)
        or type(model).__name__
    )


def _trigger_matches(trigger: FallbackTrigger, exc: BaseException) -> bool:
    """Decide whether ``exc`` matches the given trigger keyword."""
    if trigger == "any_error":
        return True

    name = type(exc).__name__.lower()
    msg = str(exc).lower()

    if trigger == "timeout":
        if isinstance(exc, asyncio.TimeoutError):
            return True
        return "timeout" in name or "timeout" in msg or "timed out" in msg

    if trigger == "rate_limit":
        status = getattr(exc, "status_code", None)
        if status == 429:
            return True
        return any(
            phrase in msg
            for phrase in (
                "rate limit",
                "rate_limit",
                "quota",
                "too many requests",
            )
        )

    if trigger == "5xx":
        status = getattr(exc, "status_code", None)
        if isinstance(status, int) and 500 <= status <= 599:
            return True
        # Many SDKs encode the code in the message.
        return any(
            phrase in msg
            for phrase in (
                "internal server error",
                "bad gateway",
                "service unavailable",
                "gateway timeout",
                "503",
                "502",
                "500",
                "504",
            )
        )

    return False


@dataclass
class LLMFallbackChain:
    """Ordered list of :class:`LLMClient` entries with per-entry triggers.

    ``LLMFallbackChain([primary, fallback], default_triggers=("timeout","5xx"))``
    is the typical 2-tier setup. The chain's call surface is identical to
    a single :class:`LLMClient`: ``chat`` / ``stream`` delegate to whichever
    entry succeeds.

    Args:
        entries: ordered list of ``FallbackEntry``. The first entry is
            the primary; subsequent entries are tried in order on
            matching failure. Must be non-empty.
        default_triggers: applied to entries that don't carry an
            explicit ``triggers`` field. Default is ``("any_error",)``.

    Notes:
        - Streaming (``stream``) only fails over BEFORE the first chunk
          is yielded. Once the consumer has seen output we cannot rewind,
          so a mid-stream error propagates as-is. Token batching strategies
          (yield-once-complete) sidestep this; for true streaming
          robustness use a ``rate_limit`` / ``timeout`` trigger only and
          accept the mid-stream failure mode.
    """

    entries: list[FallbackEntry] = field(default_factory=list)
    default_triggers: tuple[FallbackTrigger, ...] = ("any_error",)
    # Best-effort model id of the primary entry (the ``LLMClient`` contract,
    # which declares ``model`` as a settable attribute — a read-only property
    # here would not satisfy it). ``init=False`` keeps the constructor
    # signature unchanged; the value is filled in by __post_init__ once the
    # entries have been validated.
    model: str = field(init=False, default="")

    def __post_init__(self) -> None:
        if not self.entries:
            raise ValueError("LLMFallbackChain requires at least one entry")
        # Normalise default triggers onto entries that left it empty.
        for entry in self.entries:
            if not entry.triggers:
                entry.triggers = self.default_triggers
        self.model = _model_id(self.entries[0].model)

    @classmethod
    def from_models(
        cls,
        models: Sequence[Any],
        *,
        triggers: tuple[FallbackTrigger, ...] = ("any_error",),
    ) -> LLMFallbackChain:
        """Convenience: build a chain where every entry uses the same triggers."""
        entries = [FallbackEntry(model=m, triggers=triggers) for m in models]
        return cls(entries=entries, default_triggers=triggers)

    @property
    def model_name(self) -> str:
        names = ",".join(_model_id(e.model) for e in self.entries)
        return f"llm_fallback_chain[{names}]"

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
        last_exc: BaseException | None = None
        for idx, entry in enumerate(self.entries):
            try:
                result = await entry.model.chat(
                    messages,
                    tools=tools,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    extra_headers=extra_headers,
                    timeout=timeout,
                )
                _stamp_metadata(result, idx, entry.model, entry.provider)
                return result
            except Exception as exc:
                last_exc = exc
                if idx == len(self.entries) - 1 or not entry.matches(exc):
                    raise
                logger.info(
                    "LLMFallbackChain: entry %d (%s) failed (%s); "
                    "falling through to entry %d",
                    idx, _model_id(entry.model), type(exc).__name__, idx + 1,
                )
        raise RuntimeError("LLMFallbackChain exhausted") from last_exc

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
        # Try entries in order. We can only fail over BEFORE any chunk
        # has been forwarded to the caller — once we yield, the consumer
        # has committed to that entry.
        last_exc: BaseException | None = None
        for idx, entry in enumerate(self.entries):
            yielded_any = False
            try:
                async for delta in entry.model.stream(
                    messages,
                    tools=tools,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    extra_headers=extra_headers,
                    timeout=timeout,
                ):
                    # ``StreamDelta`` has no ``response_metadata`` channel, but
                    # it does carry a ``provider`` slot: stamp the serving leg's
                    # vendor so the stream assembler can fold it into
                    # ``LLMResponse.response_metadata["provider_actually_used"]``
                    # (matching the non-streaming ``chat`` path's
                    # ``_stamp_metadata``). Constant for the whole stream —
                    # failover only fires before the first yield, so every
                    # delta below comes from this single committed entry.
                    # ``fallback_used`` / ``model_actually_used`` still have no
                    # streaming landing spot; only the billing-critical provider
                    # is carried here.
                    if entry.provider:
                        delta.provider = entry.provider
                    yielded_any = True
                    yield delta
                return
            except Exception as exc:
                last_exc = exc
                if yielded_any:
                    raise
                if idx == len(self.entries) - 1 or not entry.matches(exc):
                    raise
                logger.info(
                    "LLMFallbackChain stream: entry %d (%s) failed "
                    "before any chunk (%s); falling through",
                    idx, _model_id(entry.model), type(exc).__name__,
                )
        raise RuntimeError("LLMFallbackChain exhausted") from last_exc


def _stamp_metadata(
    result: LLMResponse, idx: int, model: Any, provider: str = "",
) -> None:
    """Record which entry served the call on the response metadata.

    Mutates ``LLMResponse.response_metadata`` in place — the proxy /
    tracing middleware reads ``fallback_used`` / ``model_actually_used``
    from there and forwards them into the ``llm_call_finished`` event.

    ``provider`` is stamped as ``provider_actually_used`` so downstream
    billing (``extract_usage`` → per-call ``usage`` event) can tell which
    vendor served a given call. Empty string is allowed when the chain
    construction site didn't know the vendor.
    """
    if not isinstance(result, LLMResponse):
        return
    result.response_metadata["fallback_used"] = idx
    result.response_metadata["model_actually_used"] = _model_id(model)
    if provider:
        result.response_metadata["provider_actually_used"] = provider


def with_provider_stamp(llm: Any, provider: str) -> Any:
    """Wrap any :class:`LLMClient` so responses carry ``provider_actually_used``.

    Returns ``llm`` unchanged when ``provider`` is empty (no-op fast path).
    Otherwise builds a 1-entry ``LLMFallbackChain`` with ``triggers=()``
    (never falls through), making the wrap semantically transparent:

    - Successful response → ``response_metadata.provider_actually_used``
      gets stamped with ``provider`` (and ``model_actually_used`` /
      ``fallback_used=0`` come along — the chain's standard stamping).
    - Any exception → propagates as-is, since the single entry's empty
      trigger tuple matches nothing.

    Used by workflows that construct raw ``ChatOpenAI`` /
    ``ChatAnthropic`` outside the kernel's ``llm_fallback_chain`` config
    (workflow profiles, provider-chain attempts, benchmark bases).
    Without this wrap the per-call
    ``usage`` event's ``provider`` field stays empty, blocking
    cross-vendor billing attribution.

    No-op for duck-typed test stubs that don't implement the native
    ``LLMClient`` dispatch surface — the chain routes through ``chat`` /
    ``stream``, so a stub lacking those would crash the call. The quiet
    skip keeps tests stable while still wrapping every real LLM instance
    in production.
    """
    if not provider:
        return llm
    if not (hasattr(llm, "chat") and hasattr(llm, "stream")):
        return llm
    return LLMFallbackChain(
        entries=[
            FallbackEntry(model=llm, triggers=(), provider=provider),
        ],
    )
