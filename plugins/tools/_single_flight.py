"""Process-local single-flight coalescer for idempotent upstream calls."""

from __future__ import annotations

import asyncio
import os
from collections.abc import Awaitable, Callable
from typing import TypeVar

from frontier_agent.infra.usage_meter import record_api_request

T = TypeVar("T")

# Follower wait cap before falling back to an independent call. Bounds the
# "leader stalled → everyone blocks" failure mode.
_DEFAULT_WAIT_TIMEOUT_S = 90.0


def _enabled() -> bool:
    """On by default; set ``TOOL_CALL_COALESCE=0`` (false/off/no) to disable."""
    raw = (os.environ.get("TOOL_CALL_COALESCE") or "").strip().lower()
    return raw not in {"0", "false", "off", "no"}


class SingleFlightCoalescer:
    """Coalesce concurrent identical in-flight calls into one upstream call.

    ``run(key, fn)`` runs ``fn`` once per ``key`` while that call is in flight;
    concurrent callers for the same ``key`` (*followers*) await the *leader*'s
    result instead of duplicating the call. Nothing is stored after the call
    completes — the next caller for the same ``key`` is a fresh leader (and
    hits the proxy's own cache). Followers fall back to an independent ``fn``
    call if the leader raises or the wait times out, so coalescing never turns
    one transient failure into a correlated N-caller failure.
    """

    def __init__(self, name: str = "", *, meter_provider: str = "") -> None:
        self.name = name
        # Billing provider this coalescer fronts (e.g.
        # ``"serper"``). When set, every coalesced follower counts one
        # ``external_apis.{provider}.cache_hits`` — an upstream call saved.
        self.meter_provider = meter_provider
        self._inflight: dict[str, asyncio.Future] = {}
        self._lock = asyncio.Lock()
        self.leaders = 0
        self.coalesced = 0

    async def run(
        self,
        key: str,
        fn: Callable[[], Awaitable[T]],
        *,
        wait_timeout: float = _DEFAULT_WAIT_TIMEOUT_S,
    ) -> T:
        if not key or not _enabled():
            return await fn()

        async with self._lock:
            fut = self._inflight.get(key)
            leader = fut is None
            if leader:
                fut = asyncio.get_event_loop().create_future()
                self._inflight[key] = fut
                self.leaders += 1
            else:
                self.coalesced += 1
                # A coalesced follower is one upstream call
                # saved — surface it as a cache hit on the meter.
                if self.meter_provider:
                    record_api_request(
                        self.meter_provider, requests=0, cache_hits=1,
                    )

        if not leader:
            try:
                # ``shield`` so a follower's timeout can't cancel shared work.
                return await asyncio.wait_for(
                    asyncio.shield(fut), timeout=wait_timeout,
                )
            except Exception:
                # Leader failed / timed out → independent call (resilience).
                return await fn()

        # Leader path — run once, hand the result to followers, store nothing.
        try:
            result = await fn()
        except BaseException as exc:
            async with self._lock:
                self._inflight.pop(key, None)
            if not fut.done():
                fut.set_exception(exc)
                # Retrieve eagerly so a leader failure with no waiting follower
                # doesn't log "Future exception was never retrieved".
                fut.exception()
            raise
        async with self._lock:
            self._inflight.pop(key, None)
        if not fut.done():
            fut.set_result(result)
        return result

    def clear(self) -> None:
        """Drop in-flight map + counters. Primarily for tests."""
        self._inflight.clear()
        self.leaders = self.coalesced = 0

    def stats(self) -> dict[str, int]:
        return {
            "leaders": self.leaders,
            "coalesced": self.coalesced,
            "inflight": len(self._inflight),
        }
