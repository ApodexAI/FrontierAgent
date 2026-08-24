"""Loop type contracts for the agent-loop engine."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Literal

logger = logging.getLogger(__name__)

# Absolute monotonic soft deadline stored in execution-scope metadata.
WALL_DEADLINE_MONOTONIC_KEY = "wall_deadline_monotonic"


def wall_deadline_remaining_s() -> float | None:
    """Return seconds to the soft deadline, or ``None`` when unset."""
    from frontier_agent.core.execution_context import get_current_execution_scope
    from frontier_agent.infra.wall_time_lease import RenewableWallTimeDeadline

    scope = get_current_execution_scope()
    if scope is None:
        return None
    deadline = (scope.metadata or {}).get(WALL_DEADLINE_MONOTONIC_KEY)
    if isinstance(deadline, RenewableWallTimeDeadline):
        try:
            return float(deadline.remaining_s())
        except Exception:
            return None
    if not isinstance(deadline, (int, float)):
        return None
    return float(deadline) - time.monotonic()


@dataclass(frozen=True)
class LoopPolicy:
    """Workflow-specific behavior injected into the generic loop."""

    phase_id: str = ""
    no_tool_behavior: Literal["stop", "nudge"] = "nudge"
    no_tool_nudge_message: str = ""
    terminal_tool_names: tuple[str, ...] = ()


@dataclass
class LoopConfig:
    max_turns: int = 50
    max_tool_calls_per_turn: int = 5
    tool_timeout: int = 120
    llm_timeout: int = 180
    # First streamed chunk timeout; None defers to environment configuration.
    first_chunk_timeout: float | None = None
    # Abort reasoning-only streams after either enabled bound.
    reasoning_only_timeout_s: float | None = None
    reasoning_only_max_tokens: int | None = None
    # Total budget across admission, attempts, backoff, and recovery.
    logical_call_timeout_s: float | None = None
    context_token_limit: int = 120_000
    compact_after_turns: int = 12
    keep_recent: int = 16
    no_tool_max_retries: int = 2
    # Continuations offered to a reply the output cap cut off mid-sentence.
    # Separate from ``no_tool_max_retries`` because the two are opposite signals:
    # a tool-less turn is the model choosing to stop, a truncated one is the
    # model being stopped, so a truncation must not spend the nudge budget.
    truncation_max_continuations: int = 2
    max_llm_retries: int = 5
    # Fixed retry delay; None uses exponential backoff.
    retry_wait_fixed: int | None = None
    task_id: str = ""
    # Optional gateway affinity key; task_id remains the runtime scope.
    llm_session_id: str = field(default="", kw_only=True)
    role_id: str = ""
    loop_policy: LoopPolicy = field(default_factory=LoopPolicy)
    # ToolMessage character cap; None preserves full output.
    tool_result_max_chars: int | None = None
    # Any avoids importing runtime compaction interfaces into this type layer.
    compactor: Any = None
    compaction_policy: Any = None
    tool_result_post_processor: Any = None

    # Stop before tool output makes the next LLM plus summary request overflow.
    context_overflow_guard: bool = False
    max_context_length: int = 262_144
    max_completion_tokens: int = 32_768
    summary_prompt: str = ""

    # Per-call reminder added to a copy of history, never persisted.
    system_addendum_per_call: str = ""
    system_addendum_min_turn: int = 0


@dataclass
class TurnContext:
    turn: int
    max_turns: int
    task_id: str
    role_id: str
    ai_text: str
    thinking: str
    tool_calls: list[dict]
    messages: list
    usage: dict | None
    metadata: dict
    # Reasoning recovered from tags leaked into visible content.
    leaked_reasoning: str = ""
    # Native content blocks retained for signed/encrypted replay.
    thinking_blocks: list = field(default_factory=list)


@dataclass
class LLMDeltaContext:
    turn: int
    max_turns: int
    task_id: str
    role_id: str
    delta: str
    accumulated_text: str
    delta_index: int
    metadata: dict
    # Provider-native reasoning, kept separate from visible content.
    thinking_delta: str = ""
    # Partial JSON args keyed by call id, or index before an id arrives.
    tool_call_args_chunks: list[dict] = field(default_factory=list)
    # Identifies deltas from attempts that may later be discarded.
    attempt_id: str = ""
    attempt_index: int = 1
    call_id: str = ""


# Attempt outcome describes delivery; health details live in reason fields.
ATTEMPT_ACCEPTED = "accepted"
ATTEMPT_ACCEPTED_DEGRADED = "accepted_degraded"
ATTEMPT_DISCARDED = "discarded"
ATTEMPT_FAILED = "failed"

# Both outcomes deliver bytes to the loop and must retain streamed state.
DELIVERED_ATTEMPT_OUTCOMES = frozenset({
    ATTEMPT_ACCEPTED,
    ATTEMPT_ACCEPTED_DEGRADED,
})


@dataclass
class LLMAttemptContext:
    """Summary-only lifecycle snapshot for one provider attempt."""

    turn: int
    max_turns: int
    task_id: str
    role_id: str
    call_id: str
    attempt_id: str
    attempt_index: int
    phase: str
    outcome: str = ""
    reason: str = ""
    recovery_action: str = ""
    duration_ms: int = 0
    ttft_ms: int | None = None
    usage: dict | None = None
    finish_reason: str = ""
    visible_chars: int = 0
    reasoning_chars: int = 0
    tool_calls_count: int = 0
    max_tokens: int | None = None
    error_type: str = ""
    metadata: dict = field(default_factory=dict)


@dataclass
class ToolResult:
    name: str
    args: dict
    result: str
    duration_ms: int
    tool_call_id: str
    is_error: bool
    # Interrupted results remain in history to preserve tool-call pairing.
    interrupted: bool = False


@dataclass
class Intervention:
    inject_messages: list[str] | None = None
    stop_reason: str | None = None
    skip_tool_execution: bool = False
    # Applied after message injection and before continuing the turn.
    pop_last_message: bool = False
    continue_to_next_turn: bool = False


@dataclass
class ToolCallIntervention:
    """Tool-call rewrite, short-circuit result, and metadata updates."""

    rewrite_args: dict | None = None
    skip_with_result: str | None = None
    metadata_updates: dict | None = None


@dataclass
class AgentLoopResult:
    messages: list
    final_content: str = ""
    turns_used: int = 0
    tool_calls_count: int = 0
    stopped_by: str = ""
    metadata: dict = field(default_factory=dict)


@dataclass
class CompactionEvent:
    """What one compaction did, for the durable record.

    Compaction is the one history rewrite that leaves no trace: it replaces
    messages in place, so a trajectory reading only the post-compaction history
    shows the rollup with nothing to compare it against, and the replaced turns
    are simply gone. ``selected`` and the token pair say how much was freed;
    ``summary`` is the only field that says what survived.

    Three outcomes have to stay distinguishable, because two of them produce an
    empty ``summary``:

    * a tier that does not summarise at all (Tier 1 blanking, the
      ``tool_compression_*`` fallbacks) — ``summary`` and ``rollback_reason``
      both empty;
    * a summariser that ran and produced text — ``summary`` set;
    * a summariser that ran and **failed**, whose deterministic slice can still
      win — ``summary`` empty but ``rollback_reason`` set.

    Without the third, a failed summariser is indistinguishable from one that
    never ran, which is precisely the confusion this record exists to remove.
    """

    turn: int
    seq: int
    selected: str
    tokens_before: int
    tokens_after: int
    relief_met: bool
    spill_refs: int
    #: Number of summariser calls made by the selected tier. Zero means the
    #: selected compaction path did not run the summariser.
    attempts: int = 0
    summary: str = ""
    #: Why the summariser rolled back (``llm_error`` /
    #: ``llm_error_permanent`` / ``empty_summary``), or empty when it did not
    #: run or did not fail.
    rollback_reason: str = ""


class BaseObserver:
    """No-op observer base; override only required hooks."""

    critical: bool = False

    async def on_loop_start(self, config: LoopConfig) -> None:
        pass

    async def on_llm_delta(self, ctx: LLMDeltaContext) -> Intervention | None:
        return None

    async def on_llm_attempt(
        self, ctx: LLMAttemptContext,
    ) -> Intervention | None:
        return None

    async def on_llm_response(self, ctx: TurnContext) -> Intervention | None:
        return None

    async def on_tool_call(
        self, ctx: TurnContext, tool_call: dict,
    ) -> ToolCallIntervention | None:
        return None

    async def on_tool_result(
        self, ctx: TurnContext, result: ToolResult,
    ) -> ToolResult | None:
        return None

    async def on_turn_end(self, ctx: TurnContext) -> Intervention | None:
        return None

    async def on_compaction(self, event: CompactionEvent) -> None:
        """History was rewritten. Passive: compaction has already happened by
        the time this runs, so there is no intervention to return."""

    async def on_loop_end(self, result: AgentLoopResult) -> None:
        pass

    async def on_loop_cancelled(self) -> None:
        """Release resources when cancellation bypasses ``on_loop_end``."""


# Prevent GC of fire-and-forget observer tasks.
_background_tasks: set[asyncio.Task] = set()


# Log each observer-hook failure once at warning level.
_warned_observer_errors: set[tuple[str, str]] = set()


def _handle_observer_error(
    observer: Any, method: str, exc: BaseException,
) -> None:
    """Log an observer crash without propagating it into the loop."""
    obs_class = type(observer).__name__
    key = (obs_class, method)
    if key in _warned_observer_errors:
        logger.debug(
            "Observer %s.%s raised (suppressed)", obs_class, method,
            exc_info=True,
        )
        return
    _warned_observer_errors.add(key)
    logger.warning(
        "Observer %s.%s raised: %s — subsequent failures DEBUG only",
        obs_class, method, exc, exc_info=True,
    )


async def notify_observers(
    observers: list[Any],
    method: str,
    *args: Any,
    **kwargs: Any,
) -> list[Intervention]:
    """Run hooks, awaiting critical observers and isolating hook errors.

    ``on_loop_end`` drains passive hooks so their side effects are visible on return.
    """
    interventions: list[Intervention] = []

    for obs in observers:
        fn = getattr(obs, method, None)
        if fn is None:
            continue

        if getattr(obs, "critical", False):
            try:
                rv = await fn(*args, **kwargs)
                if isinstance(rv, Intervention):
                    interventions.append(rv)
            except Exception as exc:
                _handle_observer_error(obs, method, exc)
        else:
            async def _run(
                observer: Any = obs,
                m: str = method,
                f: Callable[..., Awaitable[Any]] = fn,
                a: tuple[Any, ...] = args,
                kw: dict[str, Any] = kwargs,
            ) -> None:
                try:
                    await f(*a, **kw)
                except Exception as exc:
                    _handle_observer_error(observer, m, exc)

            task = asyncio.create_task(_run())
            _background_tasks.add(task)
            task.add_done_callback(_background_tasks.discard)

    if method == "on_loop_end":
        await drain_background_observers()

    return interventions


async def drain_background_observers() -> None:
    """Drain outstanding passive observer tasks."""
    pending = [task for task in _background_tasks if not task.done()]
    if pending:
        await asyncio.gather(*pending, return_exceptions=True)


def merge_interventions(interventions: list[Intervention]) -> Intervention:
    """Merge messages, take the first stop reason, and OR boolean controls."""
    all_messages: list[str] = []
    stop_reason: str | None = None
    skip: bool = False
    pop_last: bool = False
    continue_turn: bool = False

    for iv in interventions:
        if iv.inject_messages:
            all_messages.extend(iv.inject_messages)
        if stop_reason is None and iv.stop_reason is not None:
            stop_reason = iv.stop_reason
        if iv.skip_tool_execution:
            skip = True
        if iv.pop_last_message:
            pop_last = True
        if iv.continue_to_next_turn:
            continue_turn = True

    return Intervention(
        inject_messages=all_messages if all_messages else None,
        stop_reason=stop_reason,
        skip_tool_execution=skip,
        pop_last_message=pop_last,
        continue_to_next_turn=continue_turn,
    )


async def notify_tool_call(
    observers: list[Any], ctx: TurnContext, tool_call: dict,
) -> ToolCallIntervention:
    """Merge tool-call hooks; last rewrite and first skip win."""
    rewrite: dict | None = None
    skip: str | None = None
    meta_updates: dict = {}

    for obs in observers:
        fn = getattr(obs, "on_tool_call", None)
        if fn is None:
            continue
        try:
            rv = await fn(ctx, tool_call)
        except Exception as exc:
            _handle_observer_error(obs, "on_tool_call", exc)
            continue
        if rv is None:
            continue
        if rv.rewrite_args is not None:
            rewrite = rv.rewrite_args
        if skip is None and rv.skip_with_result is not None:
            skip = rv.skip_with_result
        if rv.metadata_updates:
            meta_updates.update(rv.metadata_updates)

    return ToolCallIntervention(
        rewrite_args=rewrite,
        skip_with_result=skip,
        metadata_updates=meta_updates or None,
    )


async def notify_tool_result(
    observers: list[Any], ctx: TurnContext, result: ToolResult,
) -> ToolResult:
    """Chain tool-result hooks with last-mutation-wins semantics."""
    current = result
    for obs in observers:
        fn = getattr(obs, "on_tool_result", None)
        if fn is None:
            continue
        try:
            rv = await fn(ctx, current)
        except Exception as exc:
            _handle_observer_error(obs, "on_tool_result", exc)
            continue
        if rv is not None:
            current = rv
    return current


__all__ = [
    "ATTEMPT_ACCEPTED",
    "ATTEMPT_ACCEPTED_DEGRADED",
    "ATTEMPT_DISCARDED",
    "ATTEMPT_FAILED",
    "DELIVERED_ATTEMPT_OUTCOMES",
    "WALL_DEADLINE_MONOTONIC_KEY",
    "AgentLoopResult",
    "BaseObserver",
    "CompactionEvent",
    "Intervention",
    "LLMAttemptContext",
    "LLMDeltaContext",
    "LoopConfig",
    "LoopPolicy",
    "ToolCallIntervention",
    "ToolResult",
    "TurnContext",
    "merge_interventions",
    "notify_observers",
    "wall_deadline_remaining_s",
]
