"""Guard sub-agent spawning by depth, concurrency, budget, and wall time.

Reservations are released on both completion and failure.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

from frontier_agent.models.task_budget import TaskBudget

logger = logging.getLogger(__name__)

# Ceiling the long-running callers (a worker bootstrap, the benchmark
# ``kernel_adapter``) pin explicitly — 5 hours, deliberately far above
# ``SpawnGuard.DEFAULT_SUB_AGENT_TIMEOUT_S``. See
# :func:`resolve_sub_agent_timeout_s` for why it is safe to be this loose and
# ``benchmarks/kernel_adapter.py`` for the measurements.
PINNED_SUB_AGENT_TIMEOUT_S = 18_000

_SUB_AGENT_TIMEOUT_ENV = "SUB_AGENT_TIMEOUT_S"


def resolve_sub_agent_timeout_s(
    default: int = PINNED_SUB_AGENT_TIMEOUT_S,
) -> int:
    """Resolve the per-sub-agent wall-clock ceiling from the environment.

    Single source of truth for the ``SUB_AGENT_TIMEOUT_S`` override so the
    literal is not duplicated across call sites, and so a malformed value
    cannot take down the process it is read in. A worker bootstrap reads
    this at start-up, where a bare ``int(os.environ[...])`` would turn
    a templated-but-empty ``SUB_AGENT_TIMEOUT_S=`` into a boot failure.

    Non-positive values are rejected rather than honoured: ``0`` would disable
    the hard ``asyncio.wait_for`` while also skipping ``WallClockGuard``, and a
    negative would make ``wait_for`` fail every sub-agent instantly.
    """
    raw = os.environ.get(_SUB_AGENT_TIMEOUT_ENV, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        logger.warning(
            "Invalid %s=%r (not an integer); using %d",
            _SUB_AGENT_TIMEOUT_ENV, raw, default,
        )
        return default
    if value <= 0:
        logger.warning(
            "Ignoring %s=%d (must be > 0); using %d",
            _SUB_AGENT_TIMEOUT_ENV, value, default,
        )
        return default
    return value

# ── Exceptions ──────────────────────────────────────────────────────────


class SpawnDepthExceeded(RuntimeError):
    """Depth limit reached — cannot spawn deeper."""


class BudgetExhausted(RuntimeError):
    """Token/cost budget insufficient for this spawn."""


# ── SpawnReservation ────────────────────────────────────────────────────


@dataclass
class SpawnReservation:
    """RAII handle for a single spawn slot.

    Holds a semaphore slot + pre-charged tokens.
    Must be released via SpawnGuard.release() or the async context manager.

    ``acquired`` flips to ``True`` only after :meth:`SpawnGuard.acquire_slot`
    returns with the semaphore in hand. The bus.py ``finally`` path may
    call ``release`` after ``pre_check`` succeeded but before
    ``acquire_slot`` finished (cancellation / timeout); without this
    flag, ``release`` would hand back a slot it never owned and inflate
    ``max_parallel`` over time.
    """

    job_id: str
    depth: int
    estimated_tokens: int
    acquired_at: float = field(default_factory=time.monotonic)
    acquired: bool = False


# ── SpawnGuard ──────────────────────────────────────────────────────────


class SpawnGuard:
    """Budget-aware spawn controller for AgentBus.

    Enforces five layers of protection:
    1. Depth limit
    2. Concurrency limit (semaphore — queues, never rejects)
    3. Token budget (rejects if insufficient)
    4. Timeout (enforced by caller via asyncio.wait_for)
    5. RAII reservation (auto-release on exception)

    Thread-safe within a single asyncio event loop.
    """

    # Default sub-agent timeout: 90 minutes. Deliberately generous — slow
    # models (e.g. Qwen 397B, R1, Sonnet thinking) routinely spend 400-800s
    # per turn on real research, so a tight default silently masks legitimate
    # work as ``(agent failed: timeout)`` and triggers cascading respawns.
    # Callers that want a tighter ceiling should pass ``sub_agent_timeout_s``
    # explicitly.
    #
    # Set above the usual one-hour ceiling because long research calls may
    # opus5-mix3 (4417 sub-agent trajectories). **52.8% of sub-agents were
    # killed by this ceiling** and returned ``(empty report)``, discarding
    # ~30 turns of work each; the main agent then respawned from scratch
    # (the "cascading respawns" this comment already warned about).
    #
    # The wall clock was NOT spent in tools — median ``bash`` 0.8s,
    # ``web_search`` 1.6s, ``web_fetch`` 6.3s; tools were only 12-15% of
    # the 58 min. **85-88% was waiting on the LLM**: ~100s per turn × 30
    # turns. And that latency was queueing, not compute — single-stream
    # output throughput measured 12.9 tok/s under CONCURRENCY=64 vs
    # 26.1 tok/s on a lighter run, and latency *fell* as prompts grew
    # (turn 1-5: 14k prompt → 112s; turn 41+: 71k prompt → 60s).
    #
    # So the real fix for that run was cutting eval concurrency; this raise
    # is headroom for genuinely long sub-tasks, deliberately kept at +50%
    # rather than 2× so a wedged sub-agent still cannot squat a slot for
    # hours. The clean-exit half of the fix is
    # :class:`~frontier_agent.components.observers.wall_clock_guard.\
    # WallClockGuard`, which ``AgentBus`` attaches from this same
    # ``timeout_s`` so the loop stops itself before the hard
    # ``asyncio.wait_for`` can cancel it.
    DEFAULT_SUB_AGENT_TIMEOUT_S = 5400

    def __init__(
        self,
        budget: TaskBudget | None = None,
        sub_agent_timeout_s: int | None = None,
    ) -> None:
        b = budget or TaskBudget()
        self._max_depth = b.max_depth
        self._max_tokens = b.max_tokens
        self._max_parallel = b.max_parallel
        # Sub-agent timeout is fully decoupled from ``TaskBudget.max_wall_time_s``
        # (the root task's wall budget): they answer different questions
        # and the previous ``b.max_wall_time_s or DEFAULT`` fallback turned
        # the root default (300s) into a silent sub-agent kill switch.
        # Use the caller-supplied value or the class-wide default; never
        # inherit from the root budget.
        self._timeout_s = (
            sub_agent_timeout_s
            if sub_agent_timeout_s is not None
            else self.DEFAULT_SUB_AGENT_TIMEOUT_S
        )
        self._semaphore = asyncio.Semaphore(max(b.max_parallel, 1))
        self._tokens_reserved: int = 0
        self._tokens_actual: int = 0
        self._active: dict[str, SpawnReservation] = {}
        self._total_spawns: int = 0
        self._lock = asyncio.Lock()

    # ── Properties ──────────────────────────────────────────────────────

    @property
    def max_depth(self) -> int:
        return self._max_depth

    @property
    def max_parallel(self) -> int:
        return self._max_parallel

    @property
    def timeout_s(self) -> int:
        return self._timeout_s

    @property
    def tokens_reserved(self) -> int:
        return self._tokens_reserved

    @property
    def tokens_actual(self) -> int:
        return self._tokens_actual

    @property
    def active_count(self) -> int:
        return len(self._active)

    @property
    def total_spawns(self) -> int:
        return self._total_spawns

    @property
    def remaining_tokens(self) -> int:
        return max(0, self._max_tokens - self._tokens_reserved)

    # ── pre_check + acquire_slot + acquire (combined) ─────────────────

    async def pre_check(
        self,
        job_id: str,
        depth: int,
        estimated_tokens: int = 0,
    ) -> None:
        """Non-blocking pre-check: depth + budget. Called at submit time.

        Raises:
            SpawnDepthExceeded: if depth >= max_depth
            BudgetExhausted: if estimated_tokens > remaining budget
        """
        # Layer 5: Depth check
        if depth >= self._max_depth:
            raise SpawnDepthExceeded(
                f"Spawn depth {depth} >= max {self._max_depth}"
            )

        # Layer 3: Token budget check (under lock)
        async with self._lock:
            remaining = self._max_tokens - self._tokens_reserved
            if estimated_tokens > 0 and estimated_tokens > remaining:
                raise BudgetExhausted(
                    f"Need {estimated_tokens} tokens, "
                    f"only {remaining} remaining "
                    f"(reserved {self._tokens_reserved} "
                    f"of {self._max_tokens})"
                )
            self._tokens_reserved += estimated_tokens

        # Register reservation (without semaphore yet)
        reservation = SpawnReservation(
            job_id=job_id,
            depth=depth,
            estimated_tokens=estimated_tokens,
        )
        self._active[job_id] = reservation
        self._total_spawns += 1

        logger.debug(
            "SpawnGuard.pre_check(%s): depth=%d, est_tokens=%d, "
            "reserved=%d/%d",
            job_id, depth, estimated_tokens,
            self._tokens_reserved, self._max_tokens,
        )

    async def acquire_slot(self, job_id: str) -> None:
        """Acquire concurrency slot. May block (queue). Called at run time.

        Layer 4: Concurrency limit (semaphore — queues, never rejects).
        """
        await self._semaphore.acquire()
        reservation = self._active.get(job_id)
        if reservation is not None:
            reservation.acquired = True
        logger.debug(
            "SpawnGuard.acquire_slot(%s): active=%d/%d",
            job_id, len(self._active), self._max_parallel,
        )

    async def acquire(
        self,
        job_id: str,
        depth: int,
        estimated_tokens: int = 0,
    ) -> SpawnReservation:
        """Combined pre_check + acquire_slot. Blocks on concurrency.

        Use pre_check + acquire_slot separately for non-blocking submit.
        This combined method is for the RAII context manager and tests.
        """
        await self.pre_check(job_id, depth, estimated_tokens)
        await self.acquire_slot(job_id)
        return self._active[job_id]

    def release(
        self,
        job_id: str,
        actual_tokens: int = 0,
    ) -> None:
        """Release a spawn slot. Corrects token estimate with actual usage.

        Safe to call multiple times (idempotent).
        """
        reservation = self._active.pop(job_id, None)
        if reservation is None:
            return

        self._tokens_reserved -= reservation.estimated_tokens
        self._tokens_actual += actual_tokens

        if reservation.acquired:
            self._semaphore.release()

        logger.debug(
            "SpawnGuard.release(%s): est=%d, actual=%d, "
            "active=%d/%d, slot_returned=%s",
            job_id, reservation.estimated_tokens, actual_tokens,
            len(self._active), self._max_parallel, reservation.acquired,
        )

    @asynccontextmanager
    async def reservation(
        self,
        job_id: str,
        depth: int,
        estimated_tokens: int = 0,
    ) -> AsyncIterator[SpawnReservation]:
        """RAII context manager for acquire/release.

        Guarantees release even if the spawned task raises.
        """
        res = await self.acquire(job_id, depth, estimated_tokens)
        try:
            yield res
        finally:
            self.release(job_id)

    # ── Stats ───────────────────────────────────────────────────────────

    def stats(self) -> dict:
        """Return current guard state for debugging/SSE."""
        return {
            "active": len(self._active),
            "max_parallel": self._max_parallel,
            "max_depth": self._max_depth,
            "tokens_reserved": self._tokens_reserved,
            "tokens_actual": self._tokens_actual,
            "max_tokens": self._max_tokens,
            "total_spawns": self._total_spawns,
        }
