from __future__ import annotations

import asyncio
import contextlib
import inspect
import logging
import time
from collections.abc import Awaitable, Callable
from typing import Any

from frontier_agent.core.errors import (
    LLMReasoningRunaway,
    LLMStreamStalled,
)
from frontier_agent.core.llm import LLMResponse
from frontier_agent.core.messages import Message, ToolCall

from ._runaway import _env_float, _env_int

logger = logging.getLogger(__name__)
# ── Stream-stall watchdog ─────────────────────────────────────────────
# A streaming request can be black-holed without a chunk, error, or connection
# close. Inter-chunk deadlines distinguish that state from slow decoding.
#
# The watchdog bounds the gap between consecutive stream chunks. Any
# chunk — visible text, reasoning delta, tool-call args — resets the
# timer, so reasoning-runaway streams (which decode at full speed) are
# NOT flagged. On a stall the stream is closed and the attempt retried
# under the normal transient-error budget, converting a 1200 s hang
# into a ``stall_timeout`` one.
#
# Default 180 s — generous against buffering proxies that hold chunks
# (a full-response buffer flush arrives as one late burst) and against
# providers that think silently without streaming reasoning. Tunable
# via FRONTIER_AGENT_LLM_STREAM_STALL_S; <= 0 disables.
_STREAM_STALL_DEFAULT_S = 180.0
_STREAM_STALL_ENV = "FRONTIER_AGENT_LLM_STREAM_STALL_S"

# First-chunk (TTFT) bound — a tighter leash on the FIRST chunk only.
# On a healthy gateway the first chunk (text or reasoning delta) lands
# within seconds (measured: TTFT under 3 s on every successful call),
# so a request with ZERO chunks after tens of seconds is almost
# certainly queued into a black hole — retrying immediately beats
# waiting out the generic 180 s stall bound. Inter-chunk gaps keep the
# looser stall bound (mid-generation pauses are legitimate).
# 0 / unset disables: the first chunk is then bounded by the stall
# timeout like any other gap. Deploys facing a large shared gateway
# set ~45.
_FIRST_CHUNK_ENV = "FRONTIER_AGENT_LLM_FIRST_CHUNK_S"

# After this many mid-stream stalls within a single ``call_llm``
# invocation, stop same-key retrying and surface ``chain_advance`` — a
# stalled stream is likely gateway black-holing, and same-key retries keep
# hitting the same dead backend.
#
# What "surface" buys depends on which turn stalled (``agent_loop`` decides):
# only a turn-1 exhaustion is re-raised to the outer ``run_with_chain`` to
# rotate to the next provider leg; past turn 1 the loop instead stops with
# ``llm_error`` to preserve partial content and hand off to salvage. Either
# way the win is the same — we stop burning the wall budget on a dead
# gateway instead of retrying it to exhaustion.
#
# Only fires when an outer chain is active: with no fallback leg
# configured (single-endpoint benchmark runs) there's nothing to advance
# to, so same-key retry within the wall budget stays the floor. Tunable
# via FRONTIER_AGENT_LLM_STREAM_STALL_MAX; default 2 (one retry, then
# advance). <= 0 disables the escape (retry to the budget as before).
_STREAM_STALL_MAX_ENV = "FRONTIER_AGENT_LLM_STREAM_STALL_MAX"
_STREAM_STALL_MAX_DEFAULT = 2


def _stream_stall_timeout_s() -> float:
    return _env_float(_STREAM_STALL_ENV, _STREAM_STALL_DEFAULT_S)


def _first_chunk_timeout_s() -> float:
    return _env_float(_FIRST_CHUNK_ENV, 0.0)


def _stream_stall_max_before_advance() -> int:
    return _env_int(_STREAM_STALL_MAX_ENV, _STREAM_STALL_MAX_DEFAULT)


class _ThinkTagSplitter:
    """Stateful split of an inline ``<think>...</think>`` text stream.

    Some providers (notably Qwen-style reasoning models behind
    OpenAI-compatible endpoints) inline their reasoning channel as
    literal ``<think>...</think>`` substrings inside the regular
    content stream rather than as a typed content block. To stream the
    two channels separately, we need a per-call state machine that
    survives across chunks (since a tag may straddle a chunk boundary).

    ``feed(text)`` returns ``(visible_text, thinking_text)`` extracted
    from this chunk. ``flush()`` drains buffered bytes at stream end
    (treating unmatched leftovers as visible if outside / as thinking
    if inside an unclosed ``<think>``). The splitter holds back at
    most ``len(CLOSE) - 1 == 7`` chars to disambiguate a partial tag
    from real content — flush() releases those.
    """

    OPEN = "<think>"
    CLOSE = "</think>"

    def __init__(self) -> None:
        self._inside = False
        self._buffer = ""

    @property
    def has_state(self) -> bool:
        """True when we're mid-tag or holding a partial-tag tail."""
        return self._inside or bool(self._buffer)

    @staticmethod
    def _suffix_overlap(buf: str, tag: str) -> int:
        """Return n s.t. ``buf[-n:] == tag[:n]`` (longest match).

        Used to decide how many trailing bytes to hold back as a
        possible partial-tag start. Returns 0 when no overlap — those
        bytes are safe to emit immediately.
        """
        max_n = min(len(tag) - 1, len(buf))
        for n in range(max_n, 0, -1):
            if buf[-n:] == tag[:n]:
                return n
        return 0

    def feed(self, text: str) -> tuple[str, str]:
        visible_parts: list[str] = []
        thinking_parts: list[str] = []
        buf = self._buffer + text
        self._buffer = ""

        while buf:
            if self._inside:
                idx = buf.find(self.CLOSE)
                if idx == -1:
                    hold = self._suffix_overlap(buf, self.CLOSE)
                    safe = len(buf) - hold
                    if safe:
                        thinking_parts.append(buf[:safe])
                    self._buffer = buf[safe:]
                    break
                thinking_parts.append(buf[:idx])
                buf = buf[idx + len(self.CLOSE):]
                self._inside = False
            else:
                idx = buf.find(self.OPEN)
                if idx == -1:
                    hold = self._suffix_overlap(buf, self.OPEN)
                    safe = len(buf) - hold
                    if safe:
                        visible_parts.append(buf[:safe])
                    self._buffer = buf[safe:]
                    break
                if idx:
                    visible_parts.append(buf[:idx])
                buf = buf[idx + len(self.OPEN):]
                self._inside = True

        return ("".join(visible_parts), "".join(thinking_parts))

    def flush(self) -> tuple[str, str]:
        remainder = self._buffer
        self._buffer = ""
        if not remainder:
            return ("", "")
        if self._inside:
            return ("", remainder)
        return (remainder, "")


async def _stream_llm_response(
    llm: Any,
    messages: list[Message],
    timeout: float,
    on_delta: Callable[..., Awaitable[None]],
    first_chunk_s: float | None = None,
    reasoning_only_timeout_s: float | None = None,
    reasoning_only_max_tokens: int | None = None,
) -> LLMResponse:
    """Stream ``StreamDelta``s through ``on_delta`` and fold them into an
    ``LLMResponse``.

    Native ``LLMClient.stream`` yields normalised ``StreamDelta``s, so the
    langchain ``AIMessageChunk`` merge / ``message_chunk_to_message`` dance
    is gone: visible content is concatenated, reasoning is accumulated, and
    tool-calls are stitched by ``index`` (id set-once, name/arguments
    appended). Note: OpenAI streaming surfaces usage / finish_reason / model
    only on the final chunk, which the client adapter does not forward into
    ``StreamDelta`` — so the assembled ``LLMResponse`` carries empty
    usage/finish_reason for streamed calls (the non-streaming path has them).

    Tool-call argument streaming: per-chunk ``tool_call_chunks`` are
    extracted from ``AIMessageChunk`` and forwarded to ``on_delta`` as
    the ``tool_call_args_chunks`` keyword arg so observers can decode
    a specific arg's value progressively (e.g. ``submit_report.content``
    markdown body → ``response.output_text.delta``). The callback is
    invoked when any of visible text / thinking / tool-call chunks is
    present in the chunk.

    Stall watchdog: the gap between consecutive chunks is bounded by the
    inter-chunk stall timeout (see :data:`_STREAM_STALL_ENV`). A stream
    that goes silent — gateway queue black-hole, dropped connection
    without FIN — raises :class:`LLMStreamStalled` after ``stall_s``
    instead of pinning the attempt for the full call ``timeout``. Any
    chunk (text / reasoning / tool args) resets the timer, so
    slow-but-alive generations are never flagged. The FIRST chunk gets
    an optionally tighter bound (:data:`_FIRST_CHUNK_ENV`) because a
    healthy gateway delivers TTFT in seconds — zero chunks after tens
    of seconds means black-holed, not thinking. The bound is one
    long-lived ``asyncio.timeout`` rescheduled per chunk — a single
    timer-handle mutation — rather than a per-chunk ``wait_for`` (which
    would allocate a future + timer on a loop that runs ~100k times for
    a long generation).

    Semantic reasoning watchdog: when enabled, the timer starts on the first
    reasoning delta and is never reset by more reasoning. Non-whitespace
    visible output or a non-empty tool-call delta permanently disarms it for
    that attempt. The token threshold is an approximate chars/4 liveness
    estimate only; provider terminal usage remains authoritative for billing.
    """
    accumulated = ""
    thinking_accum = ""
    delta_index = 0
    # Typed as ToolCall, not dict[str, Any]: the slots below are assembled
    # in the wire shape LLMResponse.tool_calls declares, and the literal
    # keeps ToolCall's fixed {id, type, function} key order.
    tool_call_acc: dict[int, ToolCall] = {}
    # Terminal metadata streamed late by the provider — kept so the assembled
    # LLMResponse carries usage/finish_reason/model (else streaming runs report
    # 0 usage and observers never see finish_reason="length").
    final_usage: dict[str, int] = {}
    final_finish_reason = ""
    final_model = ""
    # Vendor label stamped on the deltas by ``LLMFallbackChain.stream`` —
    # folded into the assembled response's ``response_metadata`` below so
    # streamed calls carry billing attribution like the non-streaming path.
    final_provider = ""
    think_splitter = _ThinkTagSplitter()
    accepts_tool_call_chunks = _accepts_tool_call_arg_chunks(on_delta)
    reasoning_timeout_s = max(float(reasoning_only_timeout_s or 0), 0.0)
    reasoning_token_limit = max(int(reasoning_only_max_tokens or 0), 0)
    reasoning_guard_enabled = bool(reasoning_timeout_s or reasoning_token_limit)
    reasoning_only_started: float | None = None
    reasoning_token_estimate = 0
    productive_output_seen = False
    stall_s = _stream_stall_timeout_s()
    # Per-call value (LoopConfig.first_chunk_timeout ← profile
    # ``agent.first_chunk_s``) wins over the process-wide env knob;
    # an explicit 0 disables even when the env is set.
    first_s = (
        first_chunk_s if first_chunk_s is not None else _first_chunk_timeout_s()
    )
    # The scope is armed with the (tighter) first-chunk bound when set,
    # then rescheduled to the inter-chunk stall cadence once chunks flow.
    initial_s = first_s if first_s > 0 else stall_s
    chunks_seen = 0
    stream_started = time.monotonic()
    loop = asyncio.get_running_loop()
    chunk_stream = llm.stream(messages, timeout=timeout)
    stall_scope: asyncio.Timeout | None = None

    def _assembled_response() -> LLMResponse:
        response_metadata = (
            {"provider_actually_used": final_provider} if final_provider else {}
        )
        visible_content = accumulated
        if visible_content:
            visible_content = (
                visible_content.lstrip() if visible_content.strip() else ""
            )
        # Drop slots that never received a function name. A slot is created
        # for ANY streamed tool-call delta carrying an index (see the
        # ``setdefault`` below), including a content-free one that merely opens
        # a tool-call block, and one whose stream was cut before the name
        # arrived — both observed against a production endpoint.
        #
        # Such a call is unexecutable, and keeping it is worse than dropping
        # it: the loop records it in DURABLE history as ``name=""``, and some
        # chat templates then fail to render that history at all ("can only
        # concatenate str (not \"NoneType\") to str", returned as HTTP 400).
        # Every later request in the session replays the same
        # history and is rejected the same way, so one malformed delta ends the
        # run — and it ends it looking like an ordinary empty submission, not
        # like the infrastructure fault it is.
        #
        # NAMED calls to tools that do not exist are deliberately kept: those
        # reach the executor and come back as "unknown tool 'x'", which the
        # model can read and act on.
        complete_tool_calls = [
            tool_call_acc[k]
            for k in sorted(tool_call_acc)
            if tool_call_acc[k]["function"]["name"]
        ]
        # Warn rather than drop silently — a provider emitting these
        # consistently is a real upstream defect, and this is the only place
        # that can still see it.
        dropped = len(tool_call_acc) - len(complete_tool_calls)
        if dropped:
            logger.warning(
                "dropped %d streamed tool_call(s) with no function name", dropped,
            )
        return LLMResponse(
            content=visible_content,
            tool_calls=complete_tool_calls,
            reasoning_content=thinking_accum,
            usage=final_usage,
            finish_reason=final_finish_reason,
            model=final_model,
            response_metadata=response_metadata,
        )

    async def _close_chunk_stream() -> None:
        with contextlib.suppress(Exception):
            await asyncio.wait_for(chunk_stream.aclose(), timeout=5.0)

    try:
        async with asyncio.timeout(timeout):
            async with asyncio.timeout(
                initial_s if initial_s > 0 else None,
            ) as stall_scope:
                async for delta in chunk_stream:
                    chunks_seen += 1
                    raw_visible = delta.content or ""
                    typed_thinking = delta.reasoning_content or ""
                    tc_chunks = delta.tool_call_deltas or []
                    # Capture terminal metadata as it arrives (usage on the
                    # late ``include_usage`` chunk, finish_reason on the last
                    # content chunk). Last non-empty wins.
                    if getattr(delta, "usage", None):
                        final_usage = delta.usage
                    if getattr(delta, "finish_reason", ""):
                        final_finish_reason = delta.finish_reason
                    if getattr(delta, "model", ""):
                        final_model = delta.model
                    if getattr(delta, "provider", ""):
                        final_provider = delta.provider
                    # Inline ``<think>...</think>`` tags (Qwen-style) are
                    # split out so ``delta`` carries answer-only text and
                    # ``thinking_delta`` collects both inline + typed
                    # reasoning. When neither the chunk nor the splitter
                    # has tag state, short-circuit to avoid scanning every
                    # clean chunk.
                    if raw_visible and (
                        "<think>" in raw_visible
                        or "</think>" in raw_visible
                        or think_splitter.has_state
                    ):
                        visible, inline_thinking = think_splitter.feed(
                            raw_visible,
                        )
                    else:
                        visible, inline_thinking = raw_visible, ""
                    thinking = (typed_thinking + inline_thinking) if (
                        typed_thinking or inline_thinking
                    ) else ""
                    # Stitch streamed tool-call deltas by index — id set
                    # once, name/arguments appended — into wire-shaped slots.
                    for tcd in tc_chunks:
                        idx = tcd.get("index") or 0
                        slot = tool_call_acc.setdefault(idx, {
                            "id": "", "type": "function",
                            "function": {"name": "", "arguments": ""},
                        })
                        if tcd.get("id"):
                            slot["id"] = tcd["id"]
                        if tcd.get("name"):
                            slot["function"]["name"] += tcd["name"]
                        if tcd.get("arguments"):
                            slot["function"]["arguments"] += tcd["arguments"]
                    if visible or thinking or tc_chunks:
                        if visible:
                            accumulated += visible
                        if thinking:
                            thinking_accum += thinking
                        tool_progress = any(
                            d.get("id") or d.get("name") or d.get("arguments")
                            for d in tc_chunks
                        )
                        if visible.strip() or tool_progress:
                            productive_output_seen = True
                        # Forward arg deltas in the {name, args, id, index}
                        # shape observers expect (``args`` = partial JSON
                        # fragment), mirroring the old chunk extractor.
                        arg_chunks = [
                            {"name": d.get("name"),
                             "args": d.get("arguments") or "",
                             "id": d.get("id"), "index": d.get("index")}
                            for d in tc_chunks
                        ]
                        if accepts_tool_call_chunks:
                            await on_delta(
                                visible, accumulated, delta_index, thinking,
                                tool_call_args_chunks=arg_chunks,
                            )
                        else:
                            await on_delta(
                                visible, accumulated, delta_index, thinking,
                            )
                        if (
                            reasoning_guard_enabled
                            and not productive_output_seen
                            and thinking
                        ):
                            now = loop.time()
                            if reasoning_only_started is None:
                                reasoning_only_started = now
                            # Liveness-only estimate. Keep it separate from
                            # provider usage: early cancellation commonly
                            # prevents the terminal billing chunk from arriving.
                            reasoning_token_estimate = (
                                len(thinking_accum) + 3
                            ) // 4
                            reasoning_elapsed = now - reasoning_only_started
                            time_exhausted = bool(
                                reasoning_timeout_s
                                and reasoning_elapsed >= reasoning_timeout_s
                            )
                            tokens_exhausted = bool(
                                reasoning_token_limit
                                and reasoning_token_estimate
                                >= reasoning_token_limit
                            )
                            if time_exhausted or tokens_exhausted:
                                trigger = "time" if time_exhausted else "tokens"
                                partial_response = _assembled_response()
                                await _close_chunk_stream()
                                raise LLMReasoningRunaway(
                                    elapsed_s=reasoning_elapsed,
                                    estimated_tokens=reasoning_token_estimate,
                                    trigger=trigger,
                                    partial_response=partial_response,
                                )
                        delta_index += 1
                    # One timer scope enforces both the resettable inter-chunk
                    # stall and the non-resettable semantic deadline.
                    # Reasoning chunks may move the stall edge forward, but
                    # min() keeps the first-reasoning deadline fixed.
                    # Productive output removes only the semantic candidate;
                    # ordinary stall handling remains.
                    watchdog_deadlines: list[float] = []
                    if stall_s > 0:
                        watchdog_deadlines.append(loop.time() + stall_s)
                    if (
                        reasoning_timeout_s
                        and reasoning_only_started is not None
                        and not productive_output_seen
                    ):
                        watchdog_deadlines.append(
                            reasoning_only_started + reasoning_timeout_s,
                        )
                    stall_scope.reschedule(
                        min(watchdog_deadlines)
                        if watchdog_deadlines
                        else None
                    )
    except TimeoutError as exc:
        if stall_scope is not None and stall_scope.expired():
            # OUR stall bound fired (the outer total-timeout raises with
            # its own scope expired and this one fresh; an external
            # cancellation re-raises CancelledError instead — neither is
            # misclassified). Close the generator now, while we're not
            # being cancelled, so the underlying HTTP stream is released
            # immediately.
            reasoning_elapsed = (
                loop.time() - reasoning_only_started
                if reasoning_only_started is not None
                else 0.0
            )
            if (
                reasoning_timeout_s
                and not productive_output_seen
                and reasoning_only_started is not None
                and reasoning_elapsed >= reasoning_timeout_s
            ):
                partial_response = _assembled_response()
                await _close_chunk_stream()
                raise LLMReasoningRunaway(
                    elapsed_s=reasoning_elapsed,
                    estimated_tokens=reasoning_token_estimate,
                    trigger="time",
                    partial_response=partial_response,
                ) from exc
            await _close_chunk_stream()
            raise LLMStreamStalled(
                initial_s if chunks_seen == 0 else stall_s, chunks_seen,
                time.monotonic() - stream_started,
            ) from exc
        raise

    # Drain any bytes the splitter held back at a partial-tag boundary.
    visible_flush, thinking_flush = think_splitter.flush()
    if visible_flush or thinking_flush:
        if visible_flush:
            accumulated += visible_flush
        if thinking_flush:
            thinking_accum += thinking_flush
        if accepts_tool_call_chunks:
            await on_delta(
                visible_flush, accumulated, delta_index, thinking_flush,
                tool_call_args_chunks=[],
            )
        else:
            await on_delta(visible_flush, accumulated, delta_index, thinking_flush)

    # Qwen chat templates delimit thinking from the visible/tool-call region
    # with ``</think>\n\n``. After SGLang's reasoning + tool parsers consume
    # both structured regions, those separators can be the only bytes left in
    # ``content`` (whitespace-only → drop entirely), or they lead the real
    # visible text (``\n\nAnswer…`` → lstrip the remnant). Either way they
    # carry no user-visible meaning; keeping the leading remnant doubles the
    # separator when ``thinking_in_history`` reconstructs the turn.
    return _assembled_response()


def _accepts_tool_call_arg_chunks(callback: Callable[..., Awaitable[None]]) -> bool:
    try:
        sig = inspect.signature(callback)
    except (TypeError, ValueError):
        return True
    for param in sig.parameters.values():
        if param.kind == inspect.Parameter.VAR_KEYWORD:
            return True
        if param.name == "tool_call_args_chunks":
            return True
    return False
