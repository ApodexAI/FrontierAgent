"""Domain-neutral, config-driven ReAct loop.

Workflow phases, terminal tools, and recovery policy are injected through
configuration and observers rather than implemented in this kernel.
"""
from __future__ import annotations

import asyncio
import contextlib
import inspect
import logging
import time
import uuid
from collections.abc import Awaitable, Callable, Coroutine
from typing import Any

from frontier_agent.core.llm import LLMClient
from frontier_agent.core.loop_types import (
    AgentLoopResult,
    CompactionEvent,
    LLMAttemptContext,
    LLMDeltaContext,
    LoopConfig,
    LoopPolicy,
    ToolResult,
    TurnContext,
    merge_interventions,
    notify_observers,
    notify_tool_call,
    notify_tool_result,
)
from frontier_agent.core.messages import (
    Message,
    is_assistant_msg,
    is_tool_msg,
    system_msg,
    tool_msg,
    user_msg,
)
from frontier_agent.core.runtime.loop.compact import (
    COMPACTION_SEQ_KEY,
    FORCE_COMPACTION_KEY,
    INPUT_ESTIMATE_KEY,
    DefaultCompactionPolicy,
    DefaultMessageCompactor,
    estimate_tokens,
)
from frontier_agent.core.runtime.loop.llm_client import (
    RUNAWAY_STATE_KEY,
    TRUNCATION_CONTINUATION_GUIDANCE,
    LLMCallExhausted,
    bind_session_id,
    bind_temperature,
    bind_tools,
    call_llm,
    estimate_message_tokens,
    estimate_text_tokens,
    extract_final_content,
    extract_leaked_reasoning,
    extract_usage,
    is_truncated_with_text,
)
from frontier_agent.core.runtime.loop.model_profile import (
    DefaultThinkingParser,
    HistoryPolicy,
    ModelProfile,
    NativeMessageNormalizer,
)
from frontier_agent.core.runtime.loop.tool_call_parser import (
    MultiFormatToolCallParser,
    ToolCallParser,
)
from frontier_agent.core.runtime.loop.tool_exec import (
    DefaultToolResultPostProcessor,
    execute_tools,
)
from frontier_agent.core.tool import Tool

logger = logging.getLogger(__name__)


# Rollback-attempts budget above ``cfg.max_turns``: when a rollback
# observer fires ``continue_to_next_turn=True`` we DON'T consume a turn
# from the ``max_turns`` budget, but we still cap total iterations at
# ``max_turns + EXTRA_ATTEMPTS_BUFFER`` to prevent runaway rollback
# loops (e.g. a flaky LLM that keeps emitting refusals/duplicates).
EXTRA_ATTEMPTS_BUFFER = 200


# Signature: (turn_index, messages_snapshot, metadata) -> awaitable None.
# Fires once per completed turn, after observer `on_turn_end` and any
# message compaction. Exceptions are caught by the loop — a failing
# checkpoint writer never kills a run.
TurnCompleteHook = Callable[
    [int, list[Message], dict[str, Any]], Awaitable[None],
]

# Signature: () -> awaitable bool. Returning ``True`` signals a graceful
# pause — the loop stops AFTER the current turn's checkpoint has been
# persisted. Exceptions are caught and treated as "no pause" so a
# broken pause-status reader can't brick a run.
PauseCheckHook = Callable[[], Awaitable[bool]]


async def _wait_for_tool_interrupt(
    observers: list[Any], ctx: TurnContext, tool_call: dict,
) -> bool:
    """Wait until any observer asks to interrupt a parked fan-in tool."""
    # ``observers`` is list[Any], so the hook has to be duck-typed off each one.
    # Annotating the getattr result states the expected shape: narrowing a bare
    # Any through callable() leaves a callable returning ``object``, which
    # create_task rejects.
    waiters: list[asyncio.Task[bool]] = []
    for observer in observers:
        fn: Callable[..., Coroutine[Any, Any, bool]] | None = getattr(
            observer, "wait_for_tool_interrupt", None,
        )
        if callable(fn):
            waiters.append(asyncio.create_task(fn(ctx, tool_call)))
    if not waiters:
        return False
    pending = set(waiters)
    try:
        while pending:
            done, pending = await asyncio.wait(
                pending, return_when=asyncio.FIRST_COMPLETED,
            )
            for task in done:
                try:
                    if bool(task.result()):
                        return True
                except asyncio.CancelledError:
                    raise
                except Exception:
                    logger.warning(
                        "Observer wait_for_tool_interrupt failed",
                        exc_info=True,
                    )
        return False
    finally:
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)


async def run_agent_loop(
    *,
    system_prompt: str,
    user_message: str,
    llm: LLMClient,
    tools: list[Tool],
    config: LoopConfig | None = None,
    observers: list[Any] | None = None,
    parser: ToolCallParser | None = None,
    model_profile: ModelProfile | None = None,
    history_policy: HistoryPolicy | None = None,
    initial_messages: list[Message] | None = None,
    on_turn_complete: TurnCompleteHook | None = None,
    pause_check: PauseCheckHook | None = None,
    scope_metadata: dict[str, Any] | None = None,
) -> AgentLoopResult:
    """Run a generic ReAct loop with observer and persistence hooks.

    Persistence precedes the pause probe, ensuring a paused run is resumable.
    Hook failures are isolated from the loop.
    """
    cfg = config or LoopConfig()
    obs = observers or []
    tc_parser = parser or MultiFormatToolCallParser()
    profile = model_profile or ModelProfile(model_id="default", provider="openai")
    policy = history_policy or HistoryPolicy()
    thinking_parser = DefaultThinkingParser()
    normalizer = NativeMessageNormalizer()

    tool_map: dict[str, Tool] = {t.name: t for t in tools}
    tool_names: set[str] = set(tool_map.keys())

    # Pin one conversation to one upstream worker when affinity is enabled.
    llm_session_id = cfg.llm_session_id or cfg.task_id
    llm_with_session = bind_session_id(llm, llm_session_id)
    llm_with_tools = bind_tools(llm_with_session, tools)

    # Empty user input resumes the supplied history without adding a turn.
    if initial_messages is not None:
        messages: list[Message] = list(initial_messages)
        if user_message:
            messages.append(user_msg(user_message))
    else:
        messages = [
            system_msg(system_prompt),
            user_msg(user_message),
        ]

    metadata: dict[str, Any] = {"role_id": cfg.role_id}

    from frontier_agent.core.execution_context import (
        ExecutionScope,
        reset_current_execution_scope,
        set_current_execution_scope,
    )
    scope_meta: dict[str, Any] = {"agent_id": cfg.role_id}
    if scope_metadata:
        scope_meta.update(scope_metadata)
    scope_meta.setdefault("llm_session_id", llm_session_id)
    scope = ExecutionScope(
        task_id=cfg.task_id,
        role_id=cfg.role_id,
        phase_id=_resolve_phase_id(cfg),
        metadata=scope_meta,
    )
    scope_token = set_current_execution_scope(scope)

    try:
        return await _run_loop_inner(
            cfg, obs, tc_parser, profile, policy, thinking_parser,
            normalizer, tool_map, tool_names, llm_with_session, llm_with_tools,
            messages, metadata, on_turn_complete, pause_check,
            scope=scope,
        )
    except asyncio.CancelledError:
        # The loop task was cancelled mid-flight (wall deadline, fan-out
        # timeout, caller gather teardown). ``on_loop_end`` never fires
        # on this path — it is the last statement of ``_run_loop_inner``
        # — so observers holding live resources (e.g. an observer's own
        # background snapshot-build task) would leak and later surface as
        # asyncio's "Task was destroyed but it is pending!". Give them
        # one bounded best-effort teardown pass, then let the
        # cancellation propagate unchanged.
        await _notify_loop_cancelled(obs)
        raise
    finally:
        reset_current_execution_scope(scope_token)


# Per-observer wall for the cancellation teardown pass. Deliberately short:
# the canceller is awaiting us, so this is cleanup-only — no finalize / LLM
# work belongs here (that's ``on_loop_end``'s job on the normal path).
_CANCEL_TEARDOWN_TIMEOUT_S = 10.0


async def _notify_loop_cancelled(observers: list[Any]) -> None:
    """Bounded best-effort ``on_loop_cancelled`` fan-out.

    Unlike ``notify_observers`` this is always awaited — fire-and-forget
    would recreate the very leak it exists to fix — and each observer
    gets its own short timeout so one stuck teardown can't pin the
    canceller. A repeat cancellation aborts the pass immediately (the
    caller re-raises CancelledError either way).
    """
    for observer in observers:
        fn = getattr(observer, "on_loop_cancelled", None)
        if fn is None:
            continue
        try:
            await asyncio.wait_for(fn(), timeout=_CANCEL_TEARDOWN_TIMEOUT_S)
        except asyncio.CancelledError:
            raise  # re-cancelled while tearing down — stop immediately
        except Exception as exc:
            logger.warning(
                "on_loop_cancelled failed for %s: %s",
                type(observer).__name__, exc,
            )


async def _run_loop_inner(
    cfg: LoopConfig,
    obs: list,
    tc_parser: Any,
    profile: Any,
    policy: Any,
    thinking_parser: Any,
    normalizer: Any,
    tool_map: dict[str, Tool],
    tool_names: set[str],
    llm_with_session: Any,
    llm_with_tools: Any,
    messages: list[Message],
    metadata: dict[str, Any],
    on_turn_complete: TurnCompleteHook | None = None,
    pause_check: PauseCheckHook | None = None,
    scope: Any = None,
) -> AgentLoopResult:
    """Inner loop extracted so run_agent_loop can wrap with ExecutionScope."""
    from frontier_agent.core.execution_context import chain_fallback_active

    await notify_observers(obs, "on_loop_start", cfg)

    stop_reason = ""
    total_tool_calls = 0
    no_tool_retries = 0
    truncation_continuations = 0

    last_input_tokens = 0
    last_output_tokens = 0

    stream_llm_tokens = bool(
        cfg.reasoning_only_timeout_s or cfg.reasoning_only_max_tokens
    ) or any(
        bool(getattr(observer, "wants_llm_delta", False))
        for observer in obs
    )
    if getattr(profile, "protocol", "chat_completions") in (
        "anthropic", "responses", "bedrock",
    ):
        stream_llm_tokens = False

    max_attempts = cfg.max_turns + EXTRA_ATTEMPTS_BUFFER
    turn = 0
    attempts = 0

    while turn < cfg.max_turns and attempts < max_attempts:
        turn += 1
        attempts += 1
        if scope is not None:
            scope.metadata["current_turn"] = turn

        llm_for_turn, messages_for_call, stop_reason = await _prepare_llm_request(
            cfg, obs, llm_with_session, llm_with_tools, messages, metadata, turn
        )
        if stop_reason:
            break

        (
            response, stop_reason, last_input_tokens, last_output_tokens,
            first_delta_at, llm_call_started, llm_call_finished,
            call_id, current_attempt_id, current_attempt_index
        ) = await _call_llm_with_callbacks(
            cfg, obs, profile, llm_for_turn, messages_for_call, metadata, turn,
            stream_llm_tokens, chain_fallback_active
        )
        if stop_reason:
            break

        (
            parsed_calls, ctx, stop_reason,
            continue_to_next_turn, skip_tool_execution,
            last_input_tokens, last_output_tokens,
        ) = await _process_llm_response(
            cfg, obs, tc_parser, profile, policy, thinking_parser, normalizer,
            tool_names, messages, metadata, turn, response, last_input_tokens,
            last_output_tokens, first_delta_at, llm_call_started, llm_call_finished,
            call_id, current_attempt_id, current_attempt_index
        )
        if stop_reason:
            break

        if continue_to_next_turn:
            turn -= 1
            continue

        # A reply the output cap cut off is the one case where we KNOW the model
        # was not finished, and it must never reach the ``no_tool`` branch below:
        # truncation and "the model chose to stop talking" are opposite signals
        # that happen to arrive with the same shape (no tool call), and under
        # ``no_tool_behavior="stop"`` sharing that exit ends the run on a
        # sentence cut mid-token. Checked before the branch, and on its own
        # budget, so a truncation never spends the nudge allowance.
        if not parsed_calls and is_truncated_with_text(response):
            truncation_continuations += 1
            if truncation_continuations <= cfg.truncation_max_continuations:
                logger.warning(
                    "turn=%d response truncated at the output cap with visible "
                    "text — continuing (%d/%d)",
                    turn, truncation_continuations, cfg.truncation_max_continuations,
                )
                # The partial text is already in history (``_process_llm_response``
                # appended it), so the work survives and the model is asked to
                # resume from it rather than restart.
                messages.append(user_msg(TRUNCATION_CONTINUATION_GUIDANCE))
                continue
            # A model that truncates every continuation gets its own stop reason
            # rather than ``no_tool``. The diagnosis was the invisible part of
            # this failure: the trajectory showed a short sentence and a
            # clean-looking ``no_tool``, which reads as a finished run.
            stop_reason = "response_truncated"
            logger.warning(
                "turn=%d response truncated after %d continuation(s) — stopping",
                turn, cfg.truncation_max_continuations,
            )
            break

        if not parsed_calls:
            no_tool_retries += 1
            if (
                cfg.loop_policy.no_tool_behavior != "nudge"
                or no_tool_retries >= cfg.no_tool_max_retries
            ):
                stop_reason = "no_tool"
                break
            messages.append(user_msg(_build_no_tool_nudge(cfg.loop_policy)))
            continue

        no_tool_retries = 0
        truncation_continuations = 0

        if not skip_tool_execution:
            stop_reason, tool_calls_executed = await _execute_tool_calls(
                cfg, obs, tool_map, messages, metadata, turn, total_tool_calls, ctx, parsed_calls
            )
            total_tool_calls += tool_calls_executed
            if stop_reason:
                break

        stop_reason = _handle_context_overflow(
            cfg, messages, turn, last_input_tokens, last_output_tokens
        )
        if stop_reason:
            break

        stop_reason = await _handle_turn_end(
            cfg, obs, messages, metadata, turn, ctx, on_turn_complete, pause_check
        )
        if stop_reason:
            break

    else:
        if attempts >= max_attempts and turn < cfg.max_turns:
            stop_reason = "max_attempts"
            logger.warning(
                "loop exhausted rollback budget at turn=%d (attempts=%d/%d)",
                turn, attempts, max_attempts,
            )
        else:
            stop_reason = "max_turns"

    return await _finalize_loop(
        obs, messages, metadata, turn, total_tool_calls, stop_reason
    )


async def _prepare_llm_request(
    cfg: LoopConfig, obs: list, llm_with_session: Any, llm_with_tools: Any,
    messages: list[Message], metadata: dict[str, Any], turn: int
) -> tuple[Any, list[Message], str]:
    temp_override = metadata.pop("_llm_temp_override", None)
    strip_tools = metadata.pop("_llm_strip_tools", False)
    llm_base = llm_with_session if strip_tools else llm_with_tools
    llm_for_turn = (
        bind_temperature(llm_base, temp_override)
        if temp_override is not None
        else llm_base
    )

    before_llm_ctx = TurnContext(
        turn=turn, max_turns=cfg.max_turns, task_id=cfg.task_id, role_id=cfg.role_id,
        ai_text="", thinking="", tool_calls=[], messages=messages, usage=None, metadata=metadata,
    )
    before_llm_interventions = await notify_observers(obs, "on_before_llm", before_llm_ctx)
    merged_before_llm = merge_interventions(before_llm_interventions)

    if merged_before_llm.inject_messages:
        for msg_text in merged_before_llm.inject_messages:
            messages.append(user_msg(msg_text))

    messages_for_call = messages
    if cfg.system_addendum_per_call and turn > cfg.system_addendum_min_turn:
        messages_for_call = [*messages, system_msg(cfg.system_addendum_per_call)]

    # Publish the estimate of THIS request, after observer injections and the
    # addendum. An observer comparing its own estimate against the provider's
    # reported ``prompt_tokens`` needs both sides measured on the same list;
    # sampling at turn end instead understates the ratio by whatever the
    # completion and tool results added.
    metadata[INPUT_ESTIMATE_KEY] = estimate_tokens(messages_for_call)

    return llm_for_turn, messages_for_call, merged_before_llm.stop_reason or ""


async def _call_llm_with_callbacks(
    cfg: LoopConfig, obs: list, profile: Any, llm_for_turn: Any, messages_for_call: list[Message],
    metadata: dict[str, Any], turn: int, stream_llm_tokens: bool, chain_fallback_active: Callable
) -> tuple[Any, str, int, int, float | None, float, float, str, str, int]:
    llm_call_started = time.perf_counter()
    first_delta_at: float | None = None
    call_id = f"llm_{uuid.uuid4().hex}"
    current_attempt_index = 1
    current_attempt_id = f"{call_id}_attempt_01"

    metadata["_llm_call_id"] = call_id
    metadata["_llm_attempt_id"] = current_attempt_id
    metadata["_llm_attempt_index"] = current_attempt_index
    metadata["_llm_attempt_outcome"] = ""
    metadata["_llm_attempt_count"] = 0

    async def _on_attempt(event: dict[str, Any]) -> None:
        nonlocal current_attempt_id, current_attempt_index
        current_attempt_index = int(event.get("attempt_index", 1) or 1)
        current_attempt_id = f"{call_id}_attempt_{current_attempt_index:02d}"
        phase = str(event.get("phase", "") or "")
        outcome = str(event.get("outcome", "") or "")
        if phase == "finished":
            metadata["_llm_call_id"] = call_id
            metadata["_llm_attempt_id"] = current_attempt_id
            metadata["_llm_attempt_index"] = current_attempt_index
            metadata["_llm_attempt_outcome"] = outcome
            metadata["_llm_attempt_count"] = max(
                int(metadata.get("_llm_attempt_count", 0) or 0), current_attempt_index
            )
        attempt_usage = event.get("usage")
        if isinstance(attempt_usage, dict):
            attempt_usage = dict(attempt_usage)
            if not attempt_usage.get("provider"):
                attempt_usage["provider"] = str(getattr(profile, "provider", "") or "")
            if not attempt_usage.get("model"):
                attempt_usage["model"] = str(getattr(profile, "model_id", "") or "")
        attempt_ctx = LLMAttemptContext(
            turn=turn, max_turns=cfg.max_turns, task_id=cfg.task_id, role_id=cfg.role_id,
            call_id=call_id, attempt_id=current_attempt_id, attempt_index=current_attempt_index,
            phase=phase, outcome=outcome, reason=str(event.get("reason", "") or ""),
            recovery_action=str(event.get("recovery_action", "") or ""),
            duration_ms=int(event.get("duration_ms", 0) or 0), ttft_ms=event.get("ttft_ms"),
            usage=attempt_usage, finish_reason=str(event.get("finish_reason", "") or ""),
            visible_chars=int(event.get("visible_chars", 0) or 0),
            reasoning_chars=int(event.get("reasoning_chars", 0) or 0),
            tool_calls_count=int(event.get("tool_calls_count", 0) or 0),
            max_tokens=event.get("max_tokens"), error_type=str(event.get("error_type", "") or ""),
            metadata=metadata,
        )
        await notify_observers(obs, "on_llm_attempt", attempt_ctx)

    async def _on_delta(
        delta: str, accumulated: str, delta_index: int, thinking_delta: str = "",
        *, tool_call_args_chunks: list[dict] | None = None,
    ) -> None:
        nonlocal first_delta_at
        if first_delta_at is None and (delta or thinking_delta or tool_call_args_chunks):
            first_delta_at = time.perf_counter()
        ctx = LLMDeltaContext(
            turn=turn, max_turns=cfg.max_turns, task_id=cfg.task_id, role_id=cfg.role_id,
            delta=delta, accumulated_text=accumulated, delta_index=delta_index,
            metadata=metadata, thinking_delta=thinking_delta,
            tool_call_args_chunks=tool_call_args_chunks or [],
            attempt_id=current_attempt_id, attempt_index=current_attempt_index, call_id=call_id,
        )
        await notify_observers(obs, "on_llm_delta", ctx)

    try:
        response = await call_llm(
            llm_for_turn, messages_for_call, cfg.llm_timeout, cfg.max_llm_retries, turn,
            on_delta=_on_delta if stream_llm_tokens else None,
            retry_wait_fixed=cfg.retry_wait_fixed,
            runaway_state=metadata.setdefault(RUNAWAY_STATE_KEY, {}),
            first_chunk_s=cfg.first_chunk_timeout,
            on_attempt=_on_attempt,
            reasoning_only_timeout_s=cfg.reasoning_only_timeout_s,
            reasoning_only_max_tokens=cfg.reasoning_only_max_tokens,
            logical_call_timeout_s=cfg.logical_call_timeout_s,
            max_completion_tokens_hint=(
                cfg.max_completion_tokens if (cfg.reasoning_only_timeout_s or cfg.reasoning_only_max_tokens) else None
            ),
        )
    except LLMCallExhausted as exhausted:
        if exhausted.reason == "wall_deadline":
            logger.warning(
                "agent_loop: wall deadline reached mid-turn %d; ending with wall_deadline for salvage: %s",
                turn, exhausted.last_exc,
            )
            return None, "wall_deadline", 0, 0, first_delta_at, llm_call_started, time.perf_counter(), call_id, current_attempt_id, current_attempt_index
        if turn == 1 and chain_fallback_active():
            logger.error(
                "agent_loop: surfacing call_llm failure on turn 1 (reason=%s) so chain wrapper can advance: %s",
                exhausted.reason, exhausted.last_exc,
            )
            raise exhausted.last_exc from exhausted
        logger.error(
            "agent_loop: call_llm exhausted after turn=%d (reason=%s); ending with llm_error to preserve partial content: %s",
            turn, exhausted.reason, exhausted.last_exc,
        )
        metadata["llm_error"] = str(exhausted.last_exc)
        metadata["llm_error_reason"] = exhausted.reason
        return None, "llm_error", 0, 0, first_delta_at, llm_call_started, time.perf_counter(), call_id, current_attempt_id, current_attempt_index

    llm_call_finished = time.perf_counter()
    if response is None:
        metadata.setdefault("llm_error", "LLM returned no response")
        return None, "llm_error", 0, 0, first_delta_at, llm_call_started, llm_call_finished, call_id, current_attempt_id, current_attempt_index

    return response, "", 0, 0, first_delta_at, llm_call_started, llm_call_finished, call_id, current_attempt_id, current_attempt_index


def _answer_dropped_tool_calls(
    messages: list[Message], history_msg: Message,
    parsed_calls: list[dict], tool_names: set[str],
) -> None:
    """Give every ``tool_call_id`` in the assistant turn a tool response.

    ``history_msg`` is built from the raw response, so it carries every native
    call the model emitted — including ones parsing then drops (an unknown
    companion name alongside a real action, or an over-cap call). The provider
    requires one ``tool`` message per id: an orphan is a hard HTTP 400 on Azure
    and others, which would turn a recoverable mistake into a dead run.

    The message doubles as the correction the model needs, so a dropped call is
    reported rather than silently vanishing and being reissued every turn.
    """
    recorded = history_msg.get("tool_calls") or []
    if not recorded:
        return
    answered = {
        message.get("tool_call_id")
        for message in messages
        if is_tool_msg(message)
    }
    answered.update(call.get("id") for call in parsed_calls)
    for call in recorded:
        call_id = call.get("id")
        if not call_id or call_id in answered:
            continue
        name = str((call.get("function") or {}).get("name") or call.get("name") or "")
        if name and name not in tool_names:
            detail = (
                f"unknown tool '{name}' is not available. It was not run; the "
                "other tool calls in this turn were. Use only the listed tools."
            )
        else:
            detail = "this tool call was not dispatched; re-issue it if still needed."
        messages.append(tool_msg(f"[tool call not executed] {detail}", call_id))
        answered.add(call_id)


def _pop_last_assistant_turn(messages: list[Message]) -> None:
    """Remove the assistant message a rollback observer rejected, in full.

    ``pop_last_message`` fires while the turn's tool calls are still
    unexecuted, so the tail is normally just the assistant message. It is
    NOT always: :func:`_answer_dropped_tool_calls` and the
    ``max_tool_calls_per_turn`` cap append ``tool`` messages *after* it to
    answer calls that will never run. Popping one message there would
    strip an answer and leave the assistant message holding an unanswered
    ``tool_call_id`` — which providers reject with a 400 on the next
    request. Drop the trailing tool answers first, then the assistant
    message itself.

    Only this turn's tail is in scope: real tool results are appended
    later, in ``_execute_tool_calls``.
    """
    while messages and is_tool_msg(messages[-1]):
        messages.pop()
    if messages and is_assistant_msg(messages[-1]):
        messages.pop()


async def _process_llm_response(
    cfg: LoopConfig, obs: list, tc_parser: Any, profile: Any, policy: Any, thinking_parser: Any, normalizer: Any,
    tool_names: set[str], messages: list[Message], metadata: dict[str, Any], turn: int,
    response: Any, last_input_tokens: int, last_output_tokens: int, first_delta_at: float | None,
    llm_call_started: float, llm_call_finished: float, call_id: str, current_attempt_id: str, current_attempt_index: int
) -> tuple[list[dict], TurnContext, str, bool, bool, int, int]:
    metadata["llm_duration_ms"] = int((llm_call_finished - llm_call_started) * 1000)
    metadata["llm_ttft_ms"] = int(((first_delta_at or llm_call_finished) - llm_call_started) * 1000)

    tr = thinking_parser.extract(response, profile)
    history_msg = normalizer.to_history(response, tr, policy, profile.thinking_format)
    messages.append(history_msg)

    if tr.thinking and profile.thinking_format == "tag":
        with contextlib.suppress(Exception):
            response.content = tr.visible_content

    parsed_calls = tc_parser.parse(response, tool_names)
    if not parsed_calls and tr.thinking and hasattr(tc_parser, "parse_text"):
        parsed_calls = tc_parser.parse_text(tr.thinking, tool_names)
        if parsed_calls:
            logger.warning("turn=%d recovered %d tool_call(s) leaked into <think>", turn, len(parsed_calls))

    cap = cfg.max_tool_calls_per_turn
    if cap and cap > 0 and len(parsed_calls) > cap:
        dropped_calls = parsed_calls[cap:]
        parsed_calls = parsed_calls[:cap]
        for dtc in dropped_calls:
            dtc_id = dtc.get("id")
            if not dtc_id:
                continue
            messages.append(tool_msg(
                f"[tool call skipped] exceeded the per-turn tool-call cap of {cap}; re-issue it in a later turn if still needed.",
                dtc_id,
            ))

    _answer_dropped_tool_calls(messages, history_msg, parsed_calls, tool_names)

    usage = extract_usage(response)
    if usage is None:
        model_id = str(getattr(profile, "model_id", "") or "")
        if model_id and model_id != "default":
            usage = {
                "provider": str(getattr(profile, "provider", "") or ""), "model": model_id,
                "prompt_tokens": 0, "completion_tokens": 0, "cached_tokens": 0, "cache_creation_tokens": 0,
                "reasoning_tokens": 0, "estimated": True,
            }
    if usage:
        last_input_tokens = int(usage.get("prompt_tokens", 0) or 0)
        last_output_tokens = int(usage.get("completion_tokens", 0) or 0)

    leaked_reasoning = extract_leaked_reasoning(response)
    rmd = getattr(response, "response_metadata", None) or {}
    metadata.pop("llm_fallback_used", None)
    metadata.pop("llm_model_actually_used", None)
    if "fallback_used" in rmd:
        metadata["llm_fallback_used"] = rmd["fallback_used"]
    if "model_actually_used" in rmd:
        metadata["llm_model_actually_used"] = rmd["model_actually_used"]
    metadata["finish_reason"] = getattr(response, "finish_reason", "") or ""

    post_content = getattr(response, "content", None)
    ai_text = post_content if isinstance(post_content, str) else tr.visible_content
    ctx = TurnContext(
        turn=turn, max_turns=cfg.max_turns, task_id=cfg.task_id, role_id=cfg.role_id,
        ai_text=ai_text, thinking=tr.thinking, tool_calls=parsed_calls, messages=messages,
        usage=usage, metadata=metadata, leaked_reasoning=leaked_reasoning,
        thinking_blocks=tr.raw_content_blocks or [],
    )

    llm_interventions = await notify_observers(obs, "on_llm_response", ctx)
    merged_llm = merge_interventions(llm_interventions)

    stop_reason = merged_llm.stop_reason or ""

    if merged_llm.pop_last_message and messages:
        _pop_last_assistant_turn(messages)
    if merged_llm.continue_to_next_turn:
        if merged_llm.inject_messages:
            for msg_text in merged_llm.inject_messages:
                messages.append(user_msg(msg_text))
        return (
            parsed_calls, ctx, stop_reason, True, False,
            last_input_tokens, last_output_tokens,
        )

    if merged_llm.inject_messages:
        for msg_text in merged_llm.inject_messages:
            messages.append(user_msg(msg_text))

    return (
        parsed_calls, ctx, stop_reason, False,
        merged_llm.skip_tool_execution, last_input_tokens, last_output_tokens,
    )


async def _execute_tool_calls(
    cfg: LoopConfig, obs: list, tool_map: dict[str, Tool], messages: list[Message], metadata: dict[str, Any],
    turn: int, total_tool_calls: int, ctx: TurnContext, parsed_calls: list[dict]
) -> tuple[str, int]:
    executable: list[tuple[int, dict]] = []
    synthetic: list[tuple[int, ToolResult]] = []
    for idx, tc in enumerate(parsed_calls):
        tcv = await notify_tool_call(obs, ctx, tc)
        if tcv.metadata_updates:
            metadata.update(tcv.metadata_updates)
        if tcv.rewrite_args is not None:
            tc = {**tc, "args": tcv.rewrite_args}
        if tcv.skip_with_result is not None:
            synthetic.append((idx, ToolResult(
                name=tc.get("name", ""), args=tc.get("args", {}) or {},
                result=tcv.skip_with_result, duration_ms=0,
                tool_call_id=tc.get("id") or f"call_{turn}_{idx}", is_error=False,
            )))
        else:
            executable.append((idx, tc))

    executed_results: list[ToolResult] = []
    if executable:
        has_tool_interrupt_waiter = any(
            callable(getattr(observer, "wait_for_tool_interrupt", None)) for observer in obs
        )
        executed_results = await execute_tools(
            [tc for _, tc in executable], tool_map, cfg.tool_timeout, turn, total_tool_calls,
            interrupt_waiter=(
                (lambda tool_call: _wait_for_tool_interrupt(obs, ctx, tool_call))
                if has_tool_interrupt_waiter else None
            ),
        )

    # Slots are pre-allocated so results reappear in the model's original call
    # order regardless of completion order.
    ordered: list[ToolResult | None] = [None] * len(parsed_calls)
    for (idx, _), tr in zip(executable, executed_results, strict=False):
        ordered[idx] = tr
    for idx, tr in synthetic:
        ordered[idx] = tr

    # ``executable`` and ``synthetic`` partition ``parsed_calls``, so every slot
    # is normally filled. The zip above still truncates if ``execute_tools``
    # returns fewer results than calls it was handed, and the placeholder is a
    # real None — previously typed away with a blanket ignore, which left the
    # attribute access below to raise. Drop unfilled slots and say so instead.
    results: list[ToolResult] = [tr for tr in ordered if tr is not None]
    if len(results) != len(ordered):
        logger.warning(
            "Tool execution returned %d result(s) for %d call(s); "
            "dropping the unfilled slot(s)", len(results), len(ordered),
        )

    processor = cfg.tool_result_post_processor or DefaultToolResultPostProcessor(cfg.tool_result_max_chars)
    can_recover = "recover_result" in tool_map
    for tr_result in results:
        tr_result = await notify_tool_result(obs, ctx, tr_result)
        # ``notify_tool_result`` ran FIRST, so the trajectory already holds
        # ``tr_result.result`` in full. The post-processor cuts only the string
        # that becomes the message — at a far smaller cap than the 150K upstream
        # (15_000 for sub-agents) — and persists nothing, so without a pointer
        # here the difference is simply lost to the model. Minted at this site
        # only: the two earlier cuts happen before the ``ToolResult`` exists, so
        # for those the trajectory holds the same preview the model already has.
        body = processor.process(tr_result)
        messages.append(tool_msg(
            _with_recovery_handle(body, tr_result, ctx.turn, enabled=can_recover),
            tr_result.tool_call_id,
        ))

    if any(result.interrupted for result in results):
        wait_interventions = await notify_observers(obs, "on_tool_wait_interrupted", ctx)
        merged_wait = merge_interventions(wait_interventions)
        if merged_wait.inject_messages:
            for msg_text in merged_wait.inject_messages:
                messages.append(user_msg(msg_text))
        if merged_wait.stop_reason:
            return merged_wait.stop_reason, len(parsed_calls)

    return "", len(parsed_calls)


def _handle_context_overflow(
    cfg: LoopConfig, messages: list[Message], turn: int, last_input_tokens: int, last_output_tokens: int
) -> str:
    if not cfg.context_overflow_guard or not messages:
        return ""

    trailing_tool_idx = len(messages)
    while trailing_tool_idx > 0 and is_tool_msg(messages[trailing_tool_idx - 1]):
        trailing_tool_idx -= 1
    buffer_factor = 1.5
    trailing_tool_tokens = 0
    for m in messages[trailing_tool_idx:]:
        trailing_tool_tokens += int(estimate_message_tokens(m) * buffer_factor)
    summary_tokens = int(estimate_text_tokens(cfg.summary_prompt) * buffer_factor)
    estimated_total = (
        last_input_tokens + last_output_tokens + trailing_tool_tokens +
        summary_tokens + cfg.max_completion_tokens + 1000
    )
    if estimated_total >= cfg.max_context_length:
        logger.warning(
            "Context overflow guard tripped at turn=%d "
            "(estimated=%d / limit=%d, last_input=%d, last_output=%d, "
            "trailing_tool=%d, summary=%d). "
            "Popping trailing ToolMessage(s) + last AIMessage and "
            "exiting loop with stopped_by='context_limit_reached'.",
            turn, estimated_total, cfg.max_context_length,
            last_input_tokens, last_output_tokens, trailing_tool_tokens, summary_tokens,
        )
        while messages and is_tool_msg(messages[-1]):
            messages.pop()
        if messages and is_assistant_msg(messages[-1]):
            messages.pop()
        return "context_limit_reached"
    return ""


def _with_recovery_handle(
    body: str, result: ToolResult, turn: int, *, enabled: bool,
) -> str:
    """Name the handle that fetches back what the post-processor cut.

    ``len(body) < len(result.result)`` is the exact condition under which
    recovery helps — it says the trajectory holds content the model cannot see —
    rather than an approximation of it. A processor that shortens a result some
    other way (URL stubbing, for instance) still satisfies it, and the handle is
    still correct there.

    The char count can drift by one case: ``notify_tool_result`` is last-mutation-
    wins, so an observer sitting AFTER the trajectory one that rewrites the result
    leaves the footer counting against a body the trajectory does not hold. The
    handle still resolves, and ``recover_result`` reports the real totals in its
    own header, so the agent sees the truth at the point it matters.

    Silent when the body already names a spill file. That is not a nicety: on a
    live agent-team run EVERY result site 3 shortened was a ``bash`` result
    carrying a gate-① spill pointer, the spill file held the full pre-cut output
    (42,770 chars behind an 8,000-char body), and the agent recovered by running
    ``cat`` on that path — 43 footers, zero tool calls. The footer only earns its
    place where nothing else covers the cut.

    Gated on ``recover_result`` being bound for THIS agent, not on a config flag:
    profiles carry their own tool lists (the stateful_react benchmark profile
    binds no reader at all), and a footer naming a tool the agent cannot call is
    worse than no footer — ``_spill_footer`` already carries a comment about that
    exact failure.

    Worded as prose naming a tool and its arguments, never as
    ``recover_result(turn=..., call_id="...")``. The callable form read as source
    to the model, which reproduced it inside a ```bash block instead of emitting a
    tool call; ``LeakedToolCallRetryObserver`` fired twice on that run.

    The note costs ~120 chars beyond the cap. The processors' own
    ``[... truncated N chars past M-char cap]`` marker is already appended after
    the cut, so a bounded overshoot is the existing behaviour rather than a
    regression this introduces; on a 15_000-char cap this roughly triples a
    45-char overshoot and stays under 1%.
    """
    if not enabled or not isinstance(body, str):
        return body
    original = result.result if isinstance(result.result, str) else ""
    if len(body) >= len(original):
        return body
    call_id = result.tool_call_id or ""
    if not call_id:
        # The handle is (turn, call_id); without an id it resolves to nothing,
        # and recover_result refuses empty ids rather than guessing.
        return body
    from plugins.tools._overflow import body_names_a_spill_file

    if body_names_a_spill_file(body):
        # Redundant, and measurably harmful as an alternative. Gate ① already
        # spilled the FULL pre-cut output and left a path in this body, so the
        # spill file is a strict superset of anything site 3 removed — verified on
        # a live run: 42,770 chars behind an 8,000-char body, with the elided
        # middle present in the file. Offering a second route to a subset of the
        # same bytes cost real turns: the agent quoted this footer, wrote "Let me
        # call recover_result", and then ran ``cat`` on the spill path anyway.
        return body
    return body + (
        f"\n\n[{len(original) - len(body):,} more chars were cut here. Use the "
        f"recover_result tool (a tool call, not a shell command) with "
        f"turn {turn} and call id {call_id}.]"
    )


async def _handle_turn_end(
    cfg: LoopConfig, obs: list, messages: list[Message], metadata: dict[str, Any], turn: int,
    ctx: TurnContext, on_turn_complete: TurnCompleteHook | None, pause_check: PauseCheckHook | None
) -> str:
    turn_interventions = await notify_observers(obs, "on_turn_end", ctx)
    merged_turn = merge_interventions(turn_interventions)

    if merged_turn.inject_messages:
        for msg_text in merged_turn.inject_messages:
            messages.append(user_msg(msg_text))

    if merged_turn.stop_reason:
        return merged_turn.stop_reason

    est_tokens = estimate_tokens(messages)
    compaction_policy = cfg.compaction_policy or DefaultCompactionPolicy(
        cfg.compact_after_turns, cfg.context_token_limit,
    )
    forced_compaction = bool(metadata.pop(FORCE_COMPACTION_KEY, False))
    if forced_compaction or compaction_policy.should_compact(turn, messages, est_tokens):
        compactor = cfg.compactor or DefaultMessageCompactor()
        result = compactor.compact(messages, cfg.keep_recent)
        messages[:] = await result if inspect.isawaitable(result) else result
        metadata[COMPACTION_SEQ_KEY] = int(
            metadata.get(COMPACTION_SEQ_KEY, 0) or 0,
        ) + 1
        # Compaction is the one history rewrite observers never saw: it edits
        # ``messages`` in place, so a trajectory built from the post-compaction
        # history shows the rollup with the replaced turns already gone. Report
        # it where the turn and sequence are known — the compactor knows neither.
        # Compactors that expose no event (the default one) simply aren't
        # reported, and ``notify_observers`` skips observers without the hook.
        compaction_event = getattr(compactor, "last_event", None)
        if isinstance(compaction_event, CompactionEvent):
            compaction_event.turn = turn
            compaction_event.seq = metadata[COMPACTION_SEQ_KEY]
            await notify_observers(obs, "on_compaction", compaction_event)

    if on_turn_complete is not None:
        try:
            await on_turn_complete(turn, messages, metadata)
        except Exception as exc:
            logger.warning("on_turn_complete hook failed at turn %d: %s", turn, exc)

    if pause_check is not None:
        try:
            if await pause_check():
                return "paused"
        except Exception as exc:
            logger.warning("pause_check failed at turn %d: %s", turn, exc)

    return ""


async def _finalize_loop(
    obs: list, messages: list[Message], metadata: dict[str, Any],
    turn: int, total_tool_calls: int, stop_reason: str
) -> AgentLoopResult:
    latched_answer = metadata.get("final_answer") if isinstance(metadata, dict) else None
    if isinstance(latched_answer, str) and latched_answer.strip():
        final_content = latched_answer
    else:
        final_content = extract_final_content(messages)
    result = AgentLoopResult(
        messages=messages,
        final_content=final_content,
        turns_used=turn,
        tool_calls_count=total_tool_calls,
        stopped_by=stop_reason,
        metadata=metadata,
    )

    await notify_observers(obs, "on_loop_end", result)
    return result

def _resolve_phase_id(cfg: LoopConfig) -> str:
    """Resolve the execution-scope phase ID for this loop run.

    Priority:
    1. Explicit LoopPolicy phase_id
    2. role_id (generic fallback)
    3. empty string
    """
    return cfg.loop_policy.phase_id or cfg.role_id or ""


def _build_no_tool_nudge(policy: LoopPolicy) -> str:
    """Render the no-tool recovery message from injected loop policy."""
    if policy.no_tool_nudge_message.strip():
        return policy.no_tool_nudge_message.strip()

    if policy.terminal_tool_names:
        terminals = ", ".join(f"`{name}`" for name in policy.terminal_tool_names)
        if len(policy.terminal_tool_names) == 1:
            finish_hint = f"If you are done, call {terminals}. "
        else:
            finish_hint = (
                "If you are done, call one of the terminal tools: "
                f"{terminals}. "
            )
    else:
        finish_hint = "If your workflow defines a terminal action, use it now. "

    return (
        "This loop requires a structured tool call to continue. "
        + finish_hint
        + "Otherwise, call an appropriate tool instead of replying in plain text."
    )
