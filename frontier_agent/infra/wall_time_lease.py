"""Renewable wall-time lease shared by live intervention and loop guards."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from datetime import UTC, datetime

WALL_TIME_LEASE_SCOPE_KEY = "_renewable_wall_time_lease"


@dataclass(frozen=True)
class WallTimeRenewal:
    """Immutable description of one accepted wall-time renewal."""

    sequence: int
    monotonic_s: float
    unix_ms: int

    def ack_fields(self) -> dict[str, int | bool | str]:
        """Protocol fields attached to a successful ``queued`` ack."""
        timestamp = datetime.fromtimestamp(
            self.unix_ms / 1000, tz=UTC,
        ).isoformat()
        return {
            "walltime_reset": True,
            "walltime_reset_seq": self.sequence,
            "walltime_reset_unix_ms": self.unix_ms,
            "walltime_reset_at": timestamp,
        }


class RenewableWallTimeLease:
    """Thread-safe stream of accepted wall-time renewals.

    Each consumer binds its own :class:`RenewableWallTimeDeadline`, so a loop's
    soft duration cannot overwrite another loop or the serve-level hard guard.
    The lease owns only renewal ordering and timestamps.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._sequence = 0
        self._created_monotonic = time.monotonic()
        self._latest_renewal_monotonic: float | None = None

    def renew(self) -> WallTimeRenewal:
        """Start a fresh wall-time window and return its wire description."""
        with self._lock:
            # Sample while holding the same lock that orders sequence updates.
            # Sampling before the lock lets an earlier caller stall, then
            # overwrite a later caller's newer anchor after it finally enters
            # the critical section — moving the renewable deadline backwards.
            monotonic_s = time.monotonic()
            unix_ms = int(time.time() * 1000)
            self._sequence += 1
            self._latest_renewal_monotonic = monotonic_s
            return WallTimeRenewal(
                sequence=self._sequence,
                monotonic_s=monotonic_s,
                unix_ms=unix_ms,
            )

    def bind_duration(self, duration_s: float) -> RenewableWallTimeDeadline:
        """Create an independent renewable deadline starting now.

        Binding never mutates shared timing state. Multiple concurrent loops
        may therefore use different soft durations with the same lease.
        """
        return RenewableWallTimeDeadline(
            lease=self,
            duration_s=max(0.0, float(duration_s)),
            started_monotonic=time.monotonic(),
        )

    def remaining_s_for(
        self,
        duration_s: float,
        *,
        started_monotonic: float | None = None,
    ) -> float:
        """Seconds left for one consumer's renewable window.

        ``started_monotonic`` is owned by the consumer. Only a later accepted
        renewal can move its anchor, so another consumer binding a deadline
        cannot slide this window.
        """
        start = (
            self._created_monotonic
            if started_monotonic is None
            else started_monotonic
        )
        with self._lock:
            renewal = self._latest_renewal_monotonic
            anchor = max(start, renewal) if renewal is not None else start
            now = time.monotonic()
        return max(0.0, float(duration_s)) - (now - anchor)

    def elapsed_s(self, *, started_monotonic: float | None = None) -> float:
        """Seconds elapsed in one consumer's current renewable window."""
        start = (
            self._created_monotonic
            if started_monotonic is None
            else started_monotonic
        )
        with self._lock:
            renewal = self._latest_renewal_monotonic
            anchor = max(start, renewal) if renewal is not None else start
            now = time.monotonic()
        return max(0.0, now - anchor)

    @property
    def sequence(self) -> int:
        with self._lock:
            return self._sequence


@dataclass(frozen=True)
class RenewableWallTimeDeadline:
    """One consumer's deadline view over a shared renewal lease."""

    lease: RenewableWallTimeLease
    duration_s: float
    started_monotonic: float

    def remaining_s(self) -> float:
        """Seconds left in this view's current renewable window."""
        return self.lease.remaining_s_for(
            self.duration_s,
            started_monotonic=self.started_monotonic,
        )

    def elapsed_s(self) -> float:
        """Seconds elapsed in this view's current renewable window."""
        return self.lease.elapsed_s(started_monotonic=self.started_monotonic)


__all__ = [
    "WALL_TIME_LEASE_SCOPE_KEY",
    "RenewableWallTimeDeadline",
    "RenewableWallTimeLease",
    "WallTimeRenewal",
]
