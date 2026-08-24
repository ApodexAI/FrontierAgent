from __future__ import annotations

import asyncio
import logging
import random
import time
from collections.abc import Awaitable, Callable
from typing import Any

from frontier_agent.core.errors import (
    LLMCallExhausted,
    LLMReasoningRunaway,
    LLMStreamStalled,
)
from frontier_agent.core.llm import LLMResponse
from frontier_agent.core.loop_types import (
    ATTEMPT_ACCEPTED,
    ATTEMPT_ACCEPTED_DEGRADED,
    ATTEMPT_DISCARDED,
    ATTEMPT_FAILED,
    wall_deadline_remaining_s,
)
from frontier_agent.core.messages import Message, user_msg
from frontier_agent.infra.retriable import is_transient_network

from ._bind import _ensure_bound
from ._response import _visible_response_text, extract_usage
from ._runaway import (
    _RUNAWAY_BACKOFF_S,
    _RUNAWAY_MAX_RETRIES,
    _RUNAWAY_RECOVERY_GUIDANCE,
    _bind_reduced_max_tokens,
    _env_int,
    _is_runaway_response,
)
from ._streaming import (
    _accepts_tool_call_arg_chunks,
    _stream_llm_response,
    _stream_stall_max_before_advance,
)
from ._tool import _stream_tool_calls_missing_required_arguments

logger = logging.getLogger(__name__)


def _get_status_code(exc: Exception) -> int | None:
    """Extract an HTTP status code from an exception, if available."""
    for attr in ("status_code", "status", "code"):
        val = getattr(exc, attr, None)
        if isinstance(val, int):
            return val
    return None


def _get_retry_after(exc: Exception) -> float | None:
    """Extract a Retry-After header value (seconds) from a 429 exception."""
    for attr in ("response", "headers"):
        obj = getattr(exc, attr, None)
        if obj is None:
            continue
        headers = getattr(obj, "headers", obj) if attr == "response" else obj
        if not hasattr(headers, "get"):
            continue
        val = headers.get("retry-after") or headers.get("Retry-After")
        if val:
            try:
                return float(val)
            except (ValueError, TypeError):
                pass
    return None


# ±25% jitter on the exponential schedules. Without it, parallel runs
# that failed together retry together: 5 attempts timing out inside one
# 3-minute window had their retries re-collide 20 minutes later — a
# synchronised stampede against an already struggling gateway. ``retry_wait_fixed`` and literal ``Retry-After``
# values are intentionally NOT jittered (explicit caller contracts).
_BACKOFF_JITTER = 0.25


def _jittered(base: float) -> float:
    return base * random.uniform(1 - _BACKOFF_JITTER, 1 + _BACKOFF_JITTER)


def _default_backoff(attempt: int) -> float:
    """Exponential schedule for timeouts / transient errors:
    2/4/8/16/32/60s base, ±25% jitter."""
    return _jittered(min(2 * (2 ** attempt), 60))


def _default_rate_limit_backoff(attempt: int) -> float:
    """Exponential schedule for 429 fallback (no Retry-After):
    30/60/120/240/300s base, ±25% jitter."""
    return _jittered(min(30 * (2 ** attempt), 300))


# ── Global LLM concurrency gate (opt-in, default OFF) ─────────────────
# Client-side admission control: caps in-flight LLM attempts process-
# wide so a heavy fan-out (8 runs × 2-4 sub-agents) queues at the
# client — observable, cancellable — instead of inside the gateway,
# where the thundering herd was observed to correlate with silent-stream
# black holes. 0 / unset disables; deploys size it to the endpoint's
# decode slots (e.g. 12-16 for a large mixture-of-experts gateway).
#
# The semaphore is loop-affine: rebuilt whenever the running loop
# changes (worker = one loop for life; tests get one per case).
_LLM_GATE_ENV = "FRONTIER_AGENT_LLM_MAX_CONCURRENT"
_llm_gate_state: tuple[Any, asyncio.Semaphore] | None = None


def _llm_gate() -> asyncio.Semaphore | None:
    limit = _env_int(_LLM_GATE_ENV, 0)
    if limit <= 0:
        return None
    global _llm_gate_state
    loop = asyncio.get_running_loop()
    if _llm_gate_state is None or _llm_gate_state[0] is not loop:
        _llm_gate_state = (loop, asyncio.Semaphore(limit))
    return _llm_gate_state[1]


# ── Wall-deadline budget closure ──────────────────────────────────────
# ``WallClockDeadlineObserver`` stamps the loop's absolute monotonic
# soft deadline into scope metadata; ``collect_reports`` already clamps
# its sub-agent wait to it. ``call_llm`` was the remaining leak: each
# attempt got the full configured timeout regardless of remaining wall
# — an attempt with a fresh 1200 s budget could launch with 90 s of wall
# left. Each attempt now clamps its timeout to the
# remaining budget, refuses to start under the floor, and abandons
# backoff sleeps that would cross the deadline.
#
# Floor: below this many remaining seconds an LLM attempt can't return
# anything useful — fail fast with reason="wall_deadline" so the loop
# stops cleanly and the post-loop salvage (force_final_answer) gets the
# reserve instead. The remaining-budget read itself is the shared
# ``loop_types.wall_deadline_remaining_s`` (same helper collect_reports
# clamps with).
_WALL_DEADLINE_FLOOR_S = 20.0

# Floor for the non-streaming replay that recovers a tool call whose streamed
# arguments came back empty. The replay is opportunistic: below this many
# seconds of remaining attempt budget it would almost certainly time out, and
# the streamed response we already hold is a better outcome than burning the
# rest of the turn on a doomed second request.
_STREAM_RECOVERY_MIN_TIMEOUT_S = 10.0


def _stream_recovery_budget_too_small(
    remaining_s: float,
    attempt_budget_s: float,
) -> bool:
    """Whether the empty-arguments replay should be skipped for lack of time.

    Skipped only when the remaining budget is under the absolute floor *and*
    under half of what the attempt started with. The second term matters: a
    deployment that configures a short ``timeout`` would otherwise never get
    the recovery at all, since the post-stream remainder is always slightly
    below a floor set at or above ``timeout``. Being wrong here is cheap —
    a failed replay falls back to the streamed response.
    """
    return (
        remaining_s < _STREAM_RECOVERY_MIN_TIMEOUT_S
        and remaining_s < attempt_budget_s / 2
    )

async def call_llm(
    llm: Any,
    messages: list[Message],
    timeout: int,
    max_retries: int,
    turn: int,
    # Two accepted shapes, picked apart at runtime by
    # ``_accepts_tool_call_arg_chunks``: with or without the keyword-only
    # ``tool_call_args_chunks``. Hence Callable[...] rather than a fixed
    # parameter list.
    on_delta: Callable[..., Awaitable[None]] | None = None,
    retry_wait_fixed: int | None = None,
    runaway_state: dict[str, Any] | None = None,
    first_chunk_s: float | None = None,
    on_attempt: Callable[[dict[str, Any]], Awaitable[None]] | None = None,
    reasoning_only_timeout_s: float | None = None,
    reasoning_only_max_tokens: int | None = None,
    logical_call_timeout_s: float | None = None,
    max_completion_tokens_hint: int | None = None,
) -> LLMResponse | None:
    """Call ``llm.chat`` (``llm.stream`` when ``on_delta`` is set) with
    exponential backoff on transient errors.

    Default retry schedule (when ``retry_wait_fixed`` is ``None``):
    exponential 2/4/8/16/32s capped at 60 for timeouts and generic errors;
    429 honours ``Retry-After`` (clamped at 300s) when present, else
    exponential 30/60/120/240s capped at 300.

    Failure modes:

    - **Non-transient HTTP (400 schema, 401/403 auth, 404 route)**:
      raises :class:`LLMCallExhausted` immediately (reason=``non_transient``)
      — retrying won't help and burning fallback keys/chain legs is just
      slower failure.
    - **Chain-aware fallback signal** (``model_not_found`` / overload /
      credit-exhausted / safety-filter — anything ``is_retriable_with_fallback``
      flags): raises immediately (reason=``chain_advance``). Same-key retry
      can't change the outcome; the next chain leg may.
    - **Retries exhausted on transient errors** (timeout / stream stall
      / 5xx / 429 / proxy-wrap): raises (reason=``exhausted``) carrying
      the last transient exception.

    Streaming calls additionally run under the inter-chunk stall
    watchdog (:class:`LLMStreamStalled`,
    ``FRONTIER_AGENT_LLM_STREAM_STALL_S``, default 180 s): a stream that
    goes silent mid-flight is aborted and retried like a timeout
    instead of pinning the attempt for the full ``timeout``. After
    ``FRONTIER_AGENT_LLM_STREAM_STALL_MAX`` such stalls in one call (default
    2) — and only when an outer provider chain is active — the call
    raises ``LLMCallExhausted(reason="chain_advance")`` instead of
    retrying the same black-holed endpoint. What the caller does with it
    is turn-dependent (see ``agent_loop``): a turn-1 exhaustion is
    re-raised to ``run_with_chain`` to rotate to the next leg; past turn 1
    the loop stops with ``llm_error`` and hands off to salvage. Either way
    the retry budget is no longer burned on a dead gateway.

    When ``retry_wait_fixed`` is set (int seconds), all transient retries
    use that fixed wait regardless of error class — used by noisy-endpoint
    workflows — a self-hosted endpoint at high concurrency, where stream
    timeouts stampede and the server-side worker recovery cycle is longer
    than the exponential schedule's early attempts. A fixed
    60-90s wait gives the worker pool one full recovery cycle between
    attempts.

    ``runaway_state`` (optional, mutable dict the caller keeps alive
    across turns) enables the reasoning-runaway recovery documented at
    :data:`RUNAWAY_STATE_KEY`: a successful-but-empty capped completion
    is resampled same-key at a reduced ``max_tokens`` with transient guidance
    instead of being returned for the loop to nudge blind. The runaway
    retries share the ``max_retries`` attempt budget; an unrecovered
    runaway is returned as-is (never raised) so existing nudge handling
    remains the floor.

    Streaming reasoning can be stopped before it reaches the provider cap by
    setting ``reasoning_only_timeout_s`` and/or
    ``reasoning_only_max_tokens``. The semantic guard remains armed only while
    the stream has reasoning but no non-whitespace visible text or tool-call
    delta. It shares the same reduced-cap recovery path as a completed
    capped-empty response.

    ``logical_call_timeout_s`` is a separate opt-in deadline spanning gate
    wait, all physical attempts, and retry backoff. It never extends an earlier
    run-level wall deadline.

    Callers wrap with try/except :class:`LLMCallExhausted` and decide
    whether to surface (chain advance) or degrade (partial content).
    """
    from frontier_agent.infra.retriable import (
        is_empty_completion,
        is_overloaded_error,
        is_retriable_with_fallback,
    )

    def _transient_backoff(attempt: int) -> float:
        """Backoff for transient errors: caller-fixed wait, else the
        jittered exponential default. (429 has its own schedule.)"""
        return (
            retry_wait_fixed
            if retry_wait_fixed is not None
            else _default_backoff(attempt)
        )

    logical_timeout_s = max(float(logical_call_timeout_s or 0), 0.0)
    logical_deadline = (
        time.monotonic() + logical_timeout_s if logical_timeout_s else None
    )

    def _nearest_deadline() -> tuple[float | None, str]:
        candidates: list[tuple[float, str]] = []
        wall_remaining = wall_deadline_remaining_s()
        if wall_remaining is not None:
            candidates.append((wall_remaining, "wall_deadline"))
        if logical_deadline is not None:
            candidates.append((
                logical_deadline - time.monotonic(),
                "logical_call_deadline",
            ))
        if not candidates:
            return None, ""
        return min(candidates, key=lambda item: item[0])

    def _effective_timeout_or_deadline_exhausted(
        *, attempt: int, reason: str,
    ) -> float:
        effective_timeout = timeout
        deadline_remaining, deadline_reason = _nearest_deadline()
        if deadline_remaining is not None and deadline_remaining < timeout:
            if deadline_remaining < _WALL_DEADLINE_FLOOR_S:
                deadline_exc = TimeoutError(
                    f"{deadline_reason} reached ({deadline_remaining:.0f}s left "
                    f"< {_WALL_DEADLINE_FLOOR_S:.0f}s floor)",
                )
                logger.warning(
                    "LLM call refused: %.0fs to %s (< %.0fs "
                    "floor; turn=%d, attempt=%d/%d, reason=%s) — surfacing "
                    "deadline for clean loop exit",
                    deadline_remaining, deadline_reason,
                    _WALL_DEADLINE_FLOOR_S,
                    turn, attempt + 1, max_retries, reason,
                )
                raise LLMCallExhausted(
                    last_exc or deadline_exc, deadline_reason,
                ) from deadline_exc
            effective_timeout = deadline_remaining
            logger.info(
                "LLM call timeout clamped to %s remaining: %ds → %.0fs "
                "(turn=%d, attempt=%d/%d, reason=%s)",
                deadline_reason,
                timeout, effective_timeout, turn, attempt + 1, max_retries,
                reason,
            )
        return effective_timeout

    def _deadline_allows_retry_after(delay_s: float) -> bool:
        deadline_remaining, _deadline_reason = _nearest_deadline()
        return (
            deadline_remaining is None
            or delay_s + _WALL_DEADLINE_FLOOR_S < deadline_remaining
        )

    async def _emit_attempt(event: dict[str, Any]) -> None:
        if on_attempt is None:
            return
        try:
            await on_attempt(event)
        except Exception:
            # Attempt observability is passive. A broken consumer must not
            # turn a valid provider response into an LLM failure.
            logger.warning("LLM attempt callback failed", exc_info=True)

    def _response_attempt_fields(response: LLMResponse) -> dict[str, Any]:
        return {
            "usage": extract_usage(response),
            "finish_reason": response.finish_reason or "",
            "visible_chars": len(_visible_response_text(response)),
            "reasoning_chars": len(response.reasoning_content or ""),
            "tool_calls_count": len(response.tool_calls or []),
        }

    last_exc: BaseException | None = None
    runaway_retries = 0
    last_runaway_reason = ""
    stream_stall_count = 0
    if runaway_state is not None:
        # Per-call diagnostics. A protocol stream observer surfaces these so
        # a consumer can distinguish "content was clipped" from earlier
        # reasoning-only attempts that were streamed and then resampled.
        runaway_state["last_call_runaway_responses"] = 0
        runaway_state["last_call_runaway_reasoning_chars"] = 0
        runaway_state["last_call_recovered"] = False
        runaway_state["last_call_reason"] = ""
    # ``llm_active`` may be re-bound with a reduced max_tokens after a
    # repeated reasoning runaway — error retries then reuse the bound
    # variant too, which is fine (the cap only applies post-runaway).
    llm_active = _ensure_bound(llm)
    messages_active = messages
    # One PHYSICAL request per index, not one retry per index. They only
    # diverge when a stream is discarded and replayed inside a single retry
    # (empty tool arguments, below): ``agent_loop`` derives ``attempt_id``
    # from this number, so reusing it would emit two ``finished`` events
    # under one id and double-count that attempt's usage downstream.
    physical_attempt_index = 0
    for attempt in range(max_retries):
        # Budget closure: clamp this attempt to the remaining wall (when
        # a deadline is stamped) so a retry chain can never outlive the
        # loop's own budget. Under the floor, refuse to start at all.
        effective_timeout = _effective_timeout_or_deadline_exhausted(
            attempt=attempt, reason="pre_gate",
        )
        physical_attempt_index += 1
        attempt_index = physical_attempt_index
        attempt_started = time.monotonic()
        attempt_first_delta: float | None = None
        active_cap = (
            getattr(llm_active, "max_tokens", None)
            or max_completion_tokens_hint
        )
        await _emit_attempt({
            "phase": "started",
            "attempt_index": attempt_index,
            "max_tokens": active_cap,
        })

        attempt_delta = on_delta
        if on_delta is not None:
            downstream_accepts_tool_chunks = _accepts_tool_call_arg_chunks(
                on_delta,
            )

            async def _attempt_delta(
                delta: str,
                accumulated: str,
                delta_index: int,
                thinking_delta: str = "",
                *,
                tool_call_args_chunks: list[dict] | None = None,
            ) -> None:
                nonlocal attempt_first_delta
                if (
                    attempt_first_delta is None
                    and (delta or thinking_delta or tool_call_args_chunks)
                ):
                    attempt_first_delta = time.monotonic()
                if downstream_accepts_tool_chunks:
                    await on_delta(
                        delta,
                        accumulated,
                        delta_index,
                        thinking_delta,
                        tool_call_args_chunks=tool_call_args_chunks or [],
                    )
                else:
                    await on_delta(
                        delta, accumulated, delta_index, thinking_delta,
                    )

            attempt_delta = _attempt_delta

        async def _finish_attempt(
            *,
            outcome: str,
            reason: str,
            recovery_action: str,
            response: LLMResponse | None = None,
            error: BaseException | None = None,
            ended_at: float | None = None,
        ) -> None:
            # ``ended_at`` back-dates the close for an attempt that finished
            # earlier than this call — the discarded stream below is reported
            # only once its replacement is known, and must not be charged for
            # the replay's wall time.
            now = time.monotonic() if ended_at is None else ended_at
            event: dict[str, Any] = {
                "phase": "finished",
                "attempt_index": attempt_index,
                "outcome": outcome,
                "reason": reason,
                "recovery_action": recovery_action,
                "duration_ms": int((now - attempt_started) * 1000),
                "ttft_ms": (
                    int((attempt_first_delta - attempt_started) * 1000)
                    if attempt_first_delta is not None
                    else None
                ),
                "max_tokens": active_cap,
                "error_type": type(error).__name__ if error is not None else "",
            }
            if response is not None:
                event.update(_response_attempt_fields(response))
            await _emit_attempt(event)

        retry_reason = "transient_error"
        retry_error: BaseException | None = None
        try:
            # Opt-in global admission gate — excess attempts wait HERE
            # (client-side, visible) rather than queueing blind inside
            # the gateway. Backoff sleeps run outside the gate so a
            # waiting retry never holds a slot. The wait itself is
            # bounded by the wall deadline reserve, and the provider
            # timeout is recomputed after the slot is acquired so queue
            # time cannot leak past the loop budget.
            gate = _llm_gate()
            gate_acquired = False
            try:
                if gate is not None:
                    deadline_remaining, _deadline_reason = _nearest_deadline()
                    gate_wait_timeout = None
                    if deadline_remaining is not None:
                        gate_wait_timeout = (
                            deadline_remaining - _WALL_DEADLINE_FLOOR_S
                        )
                        if gate_wait_timeout <= 0:
                            _effective_timeout_or_deadline_exhausted(
                                attempt=attempt, reason="gate_wait",
                            )
                    if gate_wait_timeout is None:
                        await gate.acquire()
                    else:
                        try:
                            await asyncio.wait_for(
                                gate.acquire(), timeout=gate_wait_timeout,
                            )
                        except TimeoutError as exc:
                            raise LLMCallExhausted(
                                exc, _deadline_reason,
                            ) from exc
                    gate_acquired = True
                    effective_timeout = _effective_timeout_or_deadline_exhausted(
                        attempt=attempt, reason="post_gate",
                    )
                if attempt_delta is None:
                    response = await asyncio.wait_for(
                        llm_active.chat(messages_active, timeout=effective_timeout),
                        timeout=effective_timeout,
                    )
                else:
                    response = await _stream_llm_response(
                        llm_active, messages_active, effective_timeout, attempt_delta,
                        first_chunk_s=first_chunk_s,
                        reasoning_only_timeout_s=reasoning_only_timeout_s,
                        reasoning_only_max_tokens=reasoning_only_max_tokens,
                    )
                    empty_arg_tools = _stream_tool_calls_missing_required_arguments(
                        response, llm_active,
                    )
                    if empty_arg_tools:
                        # The stream has only been observed/assembled here: no
                        # assistant history or tool execution has happened yet,
                        # so replacing it with one non-streaming replay cannot
                        # duplicate a side effect.
                        streamed_response = response
                        stream_ended_at = time.monotonic()
                        logger.warning(
                            "Streamed tool call(s) %s had blank arguments despite "
                            "required schema fields (turn=%d, attempt=%d/%d); "
                            "replaying the same request non-streaming",
                            empty_arg_tools, turn, attempt + 1, max_retries,
                        )
                        recovered: LLMResponse | None = None
                        recovery_error: BaseException | None = None
                        try:
                            # Clamp to whatever is left of THIS attempt's own
                            # budget as well as the wall/logical deadline: the
                            # replay is a second physical request inside one
                            # attempt, so without the first term a turn could
                            # quietly cost 2x ``timeout`` whenever no deadline
                            # is stamped (direct loop use, SDK, tests).
                            recovery_timeout = min(
                                _effective_timeout_or_deadline_exhausted(
                                    attempt=attempt,
                                    reason="stream_empty_tool_arguments",
                                ),
                                max(
                                    effective_timeout
                                    - (stream_ended_at - attempt_started),
                                    0.0,
                                ),
                            )
                            if _stream_recovery_budget_too_small(
                                recovery_timeout, float(effective_timeout),
                            ):
                                raise TimeoutError(
                                    f"only {recovery_timeout:.0f}s of the "
                                    f"{effective_timeout:.0f}s attempt budget "
                                    f"left for the replay",
                                )
                            recovered = await asyncio.wait_for(
                                llm_active.chat(
                                    messages_active, timeout=recovery_timeout,
                                ),
                                timeout=recovery_timeout,
                            )
                        except Exception as exc:
                            # Recovery is opportunistic. Anything it raises —
                            # an exhausted deadline, a timeout, a provider 5xx
                            # — must not be worse than not having tried: keep
                            # the streamed response and let the loop apply its
                            # normal tool-validation feedback.
                            recovery_error = exc

                        if recovered is None:
                            logger.warning(
                                "Non-streaming replay failed (%s: %s); keeping "
                                "the streamed response with blank tool "
                                "arguments (turn=%d, attempt=%d/%d)",
                                type(recovery_error).__name__, recovery_error,
                                turn, attempt + 1, max_retries,
                            )
                            response.response_metadata = {
                                **(response.response_metadata or {}),
                                "stream_empty_args_fallback": False,
                                "stream_empty_args_tools": empty_arg_tools,
                                "stream_empty_args_recovery_error": type(
                                    recovery_error,
                                ).__name__,
                            }
                        else:
                            # Close the discarded stream as its own attempt.
                            # Every other discard path in this function does
                            # the same, and downstream depends on it twice:
                            # a protocol stream observer drains its sentence /
                            # ``<think>`` filters on a non-delivered outcome so
                            # the abandoned bytes cannot bleed into the replay,
                            # and attempt-finished is the billing record for a
                            # request whose payload never reaches the loop.
                            # That is also why the replay keeps its OWN usage
                            # untouched: merging the two would bill the stream
                            # a second time.
                            await _finish_attempt(
                                outcome=ATTEMPT_DISCARDED,
                                reason="stream_empty_tool_arguments",
                                recovery_action="replay_non_streaming",
                                response=streamed_response,
                                ended_at=stream_ended_at,
                            )
                            physical_attempt_index += 1
                            attempt_index = physical_attempt_index
                            attempt_started = stream_ended_at
                            attempt_first_delta = None
                            await _emit_attempt({
                                "phase": "started",
                                "attempt_index": attempt_index,
                                "max_tokens": active_cap,
                            })
                            response = recovered
                            response.response_metadata = {
                                **(response.response_metadata or {}),
                                "stream_empty_args_fallback": True,
                                "stream_empty_args_tools": empty_arg_tools,
                                "stream_finish_reason": (
                                    streamed_response.finish_reason or ""
                                ),
                            }
            finally:
                if gate is not None and gate_acquired:
                    gate.release()
            if _is_runaway_response(response):
                last_runaway_reason = "reasoning_runaway"
                if runaway_state is not None:
                    runaway_state["last_call_runaway_responses"] += 1
                    runaway_state["last_call_runaway_reasoning_chars"] += len(
                        getattr(response, "reasoning_content", "") or "",
                    )
                # Diagnostic only. ``consecutive_turns`` used to gate whether
                # this retry reduced the cap; the reduction is unconditional
                # now, so the counter survives purely so the log line (and a
                # post-mortem reading it) can tell a first-time runaway from a
                # model that has been running away turn after turn.
                prior_turn_runaway = bool(
                    runaway_state
                    and runaway_state.get("consecutive_turns", 0),
                )
                if (
                    runaway_retries < _RUNAWAY_MAX_RETRIES
                    and attempt < max_retries - 1
                    and _deadline_allows_retry_after(_RUNAWAY_BACKOFF_S)
                ):
                    runaway_retries += 1
                    # A first runaway is already enough evidence to stop
                    # spending the full output budget. The retry gets both a
                    # reduced cap and a transient instruction on a throwaway
                    # message copy; neither mutates durable history. A second
                    # runaway halves the cap again (derived from the already
                    # capped completion) down to the detection floor.
                    llm_active = _bind_reduced_max_tokens(
                        llm_active, response,
                    )
                    messages_active = [*messages, user_msg(_RUNAWAY_RECOVERY_GUIDANCE)]
                    reduced_cap = getattr(llm_active, "max_tokens", None)
                    await _finish_attempt(
                        outcome=ATTEMPT_DISCARDED,
                        reason="reasoning_runaway",
                        recovery_action="retry_reduced_cap",
                        response=response,
                    )
                    logger.warning(
                        "LLM reasoning runaway: capped completion with no "
                        "visible content (turn=%d, attempt=%d/%d, "
                        "runaway_retry=%d/%d, reduced_cap=%s, "
                        "prior_turn_runaway=%s); resampling",
                        turn, attempt + 1, max_retries,
                        runaway_retries, _RUNAWAY_MAX_RETRIES, reduced_cap,
                        prior_turn_runaway,
                    )
                    await asyncio.sleep(_RUNAWAY_BACKOFF_S)
                    continue
                if runaway_state is not None:
                    runaway_state["consecutive_turns"] = (
                        runaway_state.get("consecutive_turns", 0) + 1
                    )
                logger.error(
                    "LLM reasoning runaway persisted after %d resamples "
                    "(turn=%d); returning empty response for loop-level "
                    "nudge handling",
                    runaway_retries, turn,
                )
                if runaway_state is not None:
                    runaway_state["last_call_reason"] = "reasoning_runaway"
                # DELIVERED, not failed: this response is returned below, so
                # the loop appends it to history, bills it, and salvages the
                # turn with its no-tool nudge. Marking it ``failed`` would
                # make consumers drop bytes the loop actually used and would
                # flip the enclosing trace call to ``status="failed"`` even
                # though it produced a turn. ``reason`` carries the health.
                await _finish_attempt(
                    outcome=ATTEMPT_ACCEPTED_DEGRADED,
                    reason="reasoning_runaway",
                    recovery_action="return_to_loop",
                    response=response,
                )
                return response
            if runaway_state is not None:
                runaway_state["consecutive_turns"] = 0
                runaway_state["last_call_recovered"] = bool(runaway_retries)
                if runaway_retries:
                    runaway_state["last_call_reason"] = (
                        last_runaway_reason or "reasoning_runaway"
                    )
            await _finish_attempt(
                outcome=ATTEMPT_ACCEPTED,
                reason="",
                recovery_action="accepted",
                response=response,
            )
            return response
        except LLMCallExhausted as exc:
            await _finish_attempt(
                outcome=ATTEMPT_FAILED,
                reason=exc.reason,
                recovery_action="raise",
                error=exc.last_exc,
            )
            raise
        except LLMReasoningRunaway as exc:
            last_runaway_reason = "reasoning_runaway_early"
            partial_response = exc.partial_response
            if runaway_state is not None:
                runaway_state["last_call_runaway_responses"] += 1
                runaway_state["last_call_runaway_reasoning_chars"] += len(
                    getattr(partial_response, "reasoning_content", "") or "",
                )
            prior_turn_runaway = bool(
                runaway_state
                and runaway_state.get("consecutive_turns", 0)
            )
            if (
                runaway_retries < _RUNAWAY_MAX_RETRIES
                and attempt < max_retries - 1
                and _deadline_allows_retry_after(_RUNAWAY_BACKOFF_S)
            ):
                runaway_retries += 1
                llm_active = _bind_reduced_max_tokens(
                    llm_active,
                    active_cap=active_cap,
                )
                messages_active = [*messages, user_msg(_RUNAWAY_RECOVERY_GUIDANCE)]
                reduced_cap = getattr(llm_active, "max_tokens", None)
                await _finish_attempt(
                    outcome=ATTEMPT_DISCARDED,
                    reason="reasoning_runaway_early",
                    recovery_action="retry_reduced_cap",
                    response=partial_response,
                    error=exc,
                )
                logger.warning(
                    "LLM reasoning runaway stopped early: no visible/tool "
                    "progress (turn=%d, attempt=%d/%d, trigger=%s, "
                    "elapsed=%.1fs, estimated_tokens=%d, runaway_retry=%d/%d, "
                    "reduced_cap=%s, prior_turn_runaway=%s); resampling",
                    turn, attempt + 1, max_retries, exc.trigger,
                    exc.elapsed_s, exc.estimated_tokens,
                    runaway_retries, _RUNAWAY_MAX_RETRIES, reduced_cap,
                    prior_turn_runaway,
                )
                await asyncio.sleep(_RUNAWAY_BACKOFF_S)
                continue
            if runaway_state is not None:
                runaway_state["consecutive_turns"] = (
                    runaway_state.get("consecutive_turns", 0) + 1
                )
                runaway_state["last_call_reason"] = (
                    "reasoning_runaway_early"
                )
            logger.error(
                "LLM reasoning runaway stopped early but no resample slot "
                "remains (turn=%d, trigger=%s, elapsed=%.1fs, "
                "estimated_tokens=%d); returning partial response for "
                "loop-level nudge handling",
                turn, exc.trigger, exc.elapsed_s, exc.estimated_tokens,
            )
            await _finish_attempt(
                outcome=ATTEMPT_ACCEPTED_DEGRADED,
                reason="reasoning_runaway_early",
                recovery_action="return_to_loop",
                response=partial_response,
                error=exc,
            )
            return partial_response
        except LLMStreamStalled as exc:
            # Mid-stream silence (gateway queue black-hole / dropped
            # connection): the stream was already closed by the
            # watchdog; retry under the normal transient budget. Logged
            # distinctly from the total-timeout so traces show HOW the
            # attempt died, not just that it took too long.
            last_exc = exc
            retry_error = exc
            retry_reason = "stream_stalled"
            stream_stall_count += 1
            logger.warning(
                "LLM stream stalled: no chunks for %.0fs (turn=%d, "
                "attempt=%d/%d, chunks_seen=%d, elapsed=%.0fs, "
                "stall_count=%d); aborting stream and retrying",
                exc.stall_s, turn, attempt + 1, max_retries,
                exc.chunks_seen, exc.elapsed_s, stream_stall_count,
            )
            # Repeated mid-stream black-holes mean THIS endpoint is dead
            # for this call — a same-key retry just re-queues into the same
            # saturated gateway: one observed run burnt 56 stalls ×
            # ~180-330s = 207 minutes on a single endpoint this way.
            # When an outer chain is
            # active, stop burning the retry budget and surface
            # ``chain_advance``. ``agent_loop`` then either rotates the leg
            # (turn 1) or stops with ``llm_error`` for salvage (turn > 1) —
            # both stop the wall-burn. With no chain configured there's
            # nothing to advance to, so fall through to the normal transient
            # retry (wall-clamped).
            from frontier_agent.core.execution_context import (
                chain_fallback_active,
            )

            stall_max = _stream_stall_max_before_advance()
            if (
                stall_max > 0
                and stream_stall_count >= stall_max
                and chain_fallback_active()
            ):
                logger.error(
                    "LLM stream stalled %d× (turn=%d); surfacing for "
                    "chain advance instead of retrying the same "
                    "black-holed endpoint",
                    stream_stall_count, turn,
                )
                await _finish_attempt(
                    outcome=ATTEMPT_FAILED,
                    reason="stream_stalled",
                    recovery_action="chain_advance",
                    error=exc,
                )
                raise LLMCallExhausted(exc, "chain_advance") from exc
            backoff = _transient_backoff(attempt)
        except TimeoutError as exc:
            last_exc = exc
            retry_error = exc
            retry_reason = "timeout"
            deadline_remaining, deadline_reason = _nearest_deadline()
            if (
                deadline_remaining is not None
                and deadline_remaining <= 0
                and deadline_reason == "logical_call_deadline"
            ):
                await _finish_attempt(
                    outcome=ATTEMPT_FAILED,
                    reason=deadline_reason,
                    recovery_action="raise",
                    error=exc,
                )
                raise LLMCallExhausted(
                    exc, deadline_reason,
                ) from exc
            logger.warning(
                "LLM call timed out (turn=%d, attempt=%d/%d)",
                turn, attempt + 1, max_retries,
            )
            backoff = _transient_backoff(attempt)
        except Exception as exc:
            last_exc = exc
            retry_error = exc
            retry_reason = "transient_error"
            # Chain-aware shortcut: model_not_found / overload / credit
            # / safety_filter is deterministic on this (provider, input).
            # Skip the rest of the retry budget and surface so an outer
            # ``run_with_chain`` can advance the leg right now.
            if is_retriable_with_fallback(exc):
                # Overload (503) and empty completions frequently clear on a
                # same-key resample (temperature>0 re-rolls the sampler). When
                # NO outer chain is active to advance a leg
                # (``chain_fallback_active()`` is False),
                # short-circuiting these would trade a recoverable blip for a
                # turn-1 trial loss, so fall through to the transient-backoff
                # retry below (503 / no-status both land in the generic retry
                # path). Surface immediately when a chain IS active, or for the
                # genuinely deterministic failures (auth / model_unavailable /
                # credit / safety) where retrying the same key cannot help.
                from frontier_agent.core.execution_context import (
                    chain_fallback_active,
                )

                resample_may_recover = (
                    is_overloaded_error(exc) or is_empty_completion(exc)
                )
                if chain_fallback_active() or not resample_may_recover:
                    logger.error(
                        "LLM call hit chain-fallback signal (turn=%d, "
                        "attempt=%d/%d, %s); surfacing for layer advance: %s",
                        turn, attempt + 1, max_retries,
                        type(exc).__name__, exc,
                    )
                    await _finish_attempt(
                        outcome=ATTEMPT_FAILED,
                        reason="chain_advance",
                        recovery_action="chain_advance",
                        error=exc,
                    )
                    raise LLMCallExhausted(exc, "chain_advance") from exc
                logger.warning(
                    "Retriable-fallback signal but no active chain (turn=%d, "
                    "attempt=%d/%d, %s); same-key retry within budget: %s",
                    turn, attempt + 1, max_retries,
                    type(exc).__name__, exc,
                )
            status = _get_status_code(exc)
            if status and status in (400, 401, 403, 404):
                # Proxy-wrap escape hatch: OpenAI-compatible gateways
                # (new-api, etc.) sometimes package an upstream 5xx /
                # timeout as a 400 envelope (body carries
                # ``code=bad_response_status_code`` /
                # ``type=new_api_error``). The literal status is 400 but
                # the semantics are transient — sleeping and retrying
                # the same key fixes it. Vanilla 400 (bad JSON, schema
                # mismatch) still falls through to the non-transient
                # branch.
                if status == 400 and is_transient_network(exc):
                    backoff = _transient_backoff(attempt)
                    logger.warning(
                        "Proxy-wrapped transient %d (turn=%d, attempt=%d/%d): %s",
                        status, turn, attempt + 1, max_retries, exc,
                    )
                else:
                    logger.error(
                        "Non-transient LLM error %d (turn=%d): %s",
                        status, turn, exc,
                    )
                    await _finish_attempt(
                        outcome=ATTEMPT_FAILED,
                        reason="non_transient",
                        recovery_action="raise",
                        error=exc,
                    )
                    raise LLMCallExhausted(exc, "non_transient") from exc
            elif status == 429:
                retry_reason = "rate_limited"
                # 429 honours Retry-After (clamped at 300s ceiling so a
                # buggy upstream returning ``Retry-After: 86400`` cannot
                # silently stall the loop for a day); falls back to the
                # exponential rate-limit schedule when no header is set.
                # Workflows that opt into ``retry_wait_fixed`` still use
                # their fixed schedule — they're tuning for a known
                # worker-recovery cycle, not a true rate limit.
                if retry_wait_fixed is not None:
                    backoff = retry_wait_fixed
                else:
                    retry_after = _get_retry_after(exc)
                    backoff = (
                        min(retry_after, 300)
                        if retry_after
                        else _default_rate_limit_backoff(attempt)
                    )
                logger.warning(
                    "LLM rate-limited 429 (turn=%d, attempt=%d/%d, wait=%ds): %s",
                    turn, attempt + 1, max_retries, int(backoff), exc,
                )
            else:
                backoff = _transient_backoff(attempt)
                logger.warning(
                    "LLM call error (turn=%d, attempt=%d/%d): %s",
                    turn, attempt + 1, max_retries, exc,
                )

        if attempt < max_retries - 1:
            # Don't sleep past the nearest logical/run deadline: when the
            # backoff plus a useful attempt no longer fit, stop burning it
            # and surface now (salvage gets what's left).
            deadline_remaining, deadline_reason = _nearest_deadline()
            if (
                deadline_remaining is not None
                and backoff + _WALL_DEADLINE_FLOOR_S > deadline_remaining
            ):
                logger.warning(
                    "Abandoning LLM retries: backoff %ds would cross the "
                    "%s (%.0fs left, turn=%d, attempt=%d/%d)",
                    int(backoff), deadline_reason, deadline_remaining, turn,
                    attempt + 1, max_retries,
                )
                await _finish_attempt(
                    outcome=ATTEMPT_FAILED,
                    reason=deadline_reason,
                    recovery_action="abandon_retry",
                    error=retry_error,
                )
                if deadline_reason == "wall_deadline":
                    break
                deadline_exc = retry_error or TimeoutError(
                    f"{deadline_reason} reached before retry",
                )
                raise LLMCallExhausted(
                    deadline_exc, deadline_reason,
                ) from deadline_exc
            await _finish_attempt(
                outcome=ATTEMPT_DISCARDED,
                reason=retry_reason,
                recovery_action="retry_same_key",
                error=retry_error,
            )
            await asyncio.sleep(backoff)
        else:
            await _finish_attempt(
                outcome=ATTEMPT_FAILED,
                reason=retry_reason,
                recovery_action="raise_exhausted",
                error=retry_error,
            )

    logger.error("LLM call failed after %d retries (turn=%d)", max_retries, turn)
    # Should always have an exception captured here — every except clause
    # sets last_exc. Defensive RuntimeError covers a hypothetical
    # max_retries=0 invocation, which would skip the body entirely.
    if last_exc is None:
        last_exc = RuntimeError(
            f"call_llm exhausted with no captured exception "
            f"(max_retries={max_retries}, turn={turn})",
        )
    raise LLMCallExhausted(last_exc, "exhausted") from last_exc
