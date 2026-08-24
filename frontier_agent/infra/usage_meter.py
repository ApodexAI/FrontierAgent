"""Process-wide external API and tool-call meter."""

from __future__ import annotations

import contextlib
import contextvars
import threading
import time
from collections.abc import Callable
from typing import Any

# Annotated tuple[str, ...] rather than a tuple of literals: dict.fromkeys
# otherwise yields dict[Literal[...], int], which is not assignable to the
# dict[str, float] provider slots (both dict type params are invariant).
_BASE_FIELDS: tuple[str, ...] = ("requests", "cache_hits", "retries", "errors")


class ExternalAPIMeter:
    """Accumulates external-API request counts + tool-call counts."""

    def __init__(
        self,
        *,
        llm_recorder: Callable[..., None] | None = None,
    ) -> None:
        self._lock = threading.Lock()
        self._providers: dict[str, dict[str, float]] = {}
        self._tool_counts: dict[str, int] = {}
        # Open wall-clock spans (E2B sandbox lifetimes): (provider, key)
        # → (monotonic_start, field). Closed spans fold into the field;
        # still-open spans contribute elapsed-so-far at snapshot time so
        # a killed-before-cleanup task still reports the burn.
        self._open_spans: dict[tuple[str, str], tuple[float, str]] = {}
        # Gauges: last-known config values folded with max() (e.g.
        # ``sandbox_ttl_seconds``) — unlike counters these must NOT sum
        # across calls. Surfaced verbatim on the provider slot so
        # consumers can bound the unobservable tail:
        # ``true_bill ≤ sandbox_seconds + spans_open × sandbox_ttl_seconds``.
        self._gauges: dict[tuple[str, str], float] = {}
        # Forwarded LLM usage (keyword-compatible with the external usage
        # aggregator's ``record_llm_call``). Kept as a callback so
        # this module stays free of upper-layer imports (Layer 0 must not
        # depend on Layer 6).
        self._llm_recorder = llm_recorder

    # ── recording ────────────────────────────────────────────────────

    def record_api_request(
        self,
        provider: str,
        *,
        requests: int = 1,
        cache_hits: int = 0,
        retries: int = 0,
        errors: int = 0,
        **extra: float,
    ) -> None:
        """Fold one (or several) wire-level events into ``provider``'s slot.

        ``extra`` accepts additive numeric provider-specific fields
        (e.g. ``sandbox_seconds=12.5``, ``sandboxes_created=1``).
        """
        with self._lock:
            slot = self._slot_locked(provider)
            slot["requests"] += requests
            slot["cache_hits"] += cache_hits
            slot["retries"] += retries
            slot["errors"] += errors
            for field, value in extra.items():
                slot[field] = slot.get(field, 0) + float(value)

    def record_tool_call(self, name: str) -> None:
        with self._lock:
            self._tool_counts[name] = self._tool_counts.get(name, 0) + 1

    def set_gauge(self, provider: str, field: str, value: float) -> None:
        """Record a non-additive config value (max-wins across calls).

        Used for ``sandbox_ttl_seconds``: different create paths can carry
        different TTLs (a pooled lease's 1800s vs a per-call 300s) — max is
        the conservative choice for the billing upper bound.
        """
        with self._lock:
            key = (provider, field)
            self._gauges[key] = max(self._gauges.get(key, 0), float(value))

    def record_llm_usage(self, **kwargs: Any) -> None:
        """Forward raw-client LLM usage to the wired aggregator.

        Keyword-compatible with the external usage aggregator's
        ``record_llm_call`` (``model=``, ``prompt_tokens=``, ``completion_tokens=``,
        ``cache_read_tokens=``, ``cache_write_tokens=``, ``provider=``,
        ``scene=``). No-op when no recorder is wired.
        """
        if self._llm_recorder is None:
            return
        # pragma: no cover - accounting must never break the call it measures
        with contextlib.suppress(Exception):
            self._llm_recorder(**kwargs)

    # ── wall-clock spans (sandbox lifetimes) ─────────────────────────

    def open_span(
        self, provider: str, key: str, *, field: str = "sandbox_seconds",
    ) -> None:
        """Start a wall-clock span; idempotent per (provider, key)."""
        with self._lock:
            self._open_spans.setdefault(
                (provider, key), (time.monotonic(), field),
            )

    def close_span(self, provider: str, key: str) -> None:
        """Close a span, folding its elapsed seconds into the provider slot."""
        with self._lock:
            span = self._open_spans.pop((provider, key), None)
            if span is None:
                return
            started, field = span
            slot = self._slot_locked(provider)
            slot[field] = slot.get(field, 0) + (time.monotonic() - started)

    # ── snapshot ─────────────────────────────────────────────────────

    def snapshot(self) -> dict[str, Any]:
        """Return ``{"external_apis": {...}, "tools": {...}}`` (deep copy).

        Still-open spans contribute elapsed-so-far without being closed,
        so repeated snapshots stay monotonic and the final flush after
        ``close_span`` doesn't double-count.
        """
        with self._lock:
            now = time.monotonic()
            apis: dict[str, dict[str, Any]] = {}
            for provider, slot in self._providers.items():
                apis[provider] = {
                    k: (round(v, 2) if isinstance(v, float) else v)
                    for k, v in slot.items()
                }
            for (provider, _key), (started, field) in self._open_spans.items():
                slot_view = apis.setdefault(
                    provider, dict.fromkeys(_BASE_FIELDS, 0.0),
                )
                slot_view[field] = round(
                    float(slot_view.get(field, 0)) + (now - started), 2,
                )
                # Floor-flag: spans still open at snapshot time mean the
                # measured seconds are a lower bound — the remote resource
                # keeps billing past this snapshot (until kill or TTL).
                slot_view["spans_open"] = int(slot_view.get("spans_open", 0)) + 1
            for (provider, field), value in self._gauges.items():
                slot_view = apis.setdefault(
                    provider, dict.fromkeys(_BASE_FIELDS, 0.0),
                )
                slot_view[field] = (
                    round(value, 2) if isinstance(value, float) else value
                )
            return {
                "external_apis": apis,
                "tools": dict(self._tool_counts),
            }

    # ── internals ────────────────────────────────────────────────────

    def _slot_locked(self, provider: str) -> dict[str, float]:
        slot = self._providers.get(provider)
        if slot is None:
            slot = dict.fromkeys(_BASE_FIELDS, 0.0)
            self._providers[provider] = slot
        return slot


# ── contextvar binding ───────────────────────────────────────────────

_CURRENT_METER: contextvars.ContextVar[ExternalAPIMeter | None] = (
    contextvars.ContextVar("frontier_agent_usage_meter", default=None)
)


def bind_usage_meter(meter: ExternalAPIMeter) -> contextvars.Token:
    """Bind ``meter`` to the current context; returns the reset token."""
    return _CURRENT_METER.set(meter)


def reset_usage_meter(token: contextvars.Token) -> None:
    _CURRENT_METER.reset(token)


def get_usage_meter() -> ExternalAPIMeter | None:
    return _CURRENT_METER.get()


# ── module-level no-op-safe helpers (the API plugin tools use) ───────


def record_api_request(provider: str, **kwargs: Any) -> None:
    meter = _CURRENT_METER.get()
    if meter is not None:
        meter.record_api_request(provider, **kwargs)


def record_tool_call(name: str) -> None:
    meter = _CURRENT_METER.get()
    if meter is not None:
        meter.record_tool_call(name)


def record_llm_usage(**kwargs: Any) -> None:
    meter = _CURRENT_METER.get()
    if meter is not None:
        meter.record_llm_usage(**kwargs)


def open_meter_span(provider: str, key: str, *, field: str = "sandbox_seconds") -> None:
    meter = _CURRENT_METER.get()
    if meter is not None:
        meter.open_span(provider, key, field=field)


def set_meter_gauge(provider: str, field: str, value: float) -> None:
    meter = _CURRENT_METER.get()
    if meter is not None:
        meter.set_gauge(provider, field, value)


def close_meter_span(provider: str, key: str) -> None:
    meter = _CURRENT_METER.get()
    if meter is not None:
        meter.close_span(provider, key)


__all__ = [
    "ExternalAPIMeter",
    "bind_usage_meter",
    "close_meter_span",
    "get_usage_meter",
    "open_meter_span",
    "record_api_request",
    "record_llm_usage",
    "record_tool_call",
    "reset_usage_meter",
    "set_meter_gauge",
]
