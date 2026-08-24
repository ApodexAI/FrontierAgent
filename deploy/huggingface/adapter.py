"""The Web adapter: FrontierAgent runtime → structured events.

This is the only place that knows how to start a FrontierAgent ``react`` run,
and the only place the Web UI talks to. It re-implements nothing: the agent
loop, tools, compaction, finalisation and salvage all stay in
``frontier_agent`` / ``workflows``. Two seams the runtime already exposes do
all the work:

``metadata['sdk_extra_observers']``
    The ``react_agent_node`` appends these to the observers it hands
    ``run_agent_loop``, so a plain :class:`BaseObserver` receives streaming
    deltas, tool calls and tool results — no terminal, no stdin, no ANSI.

``metadata['pause_check']``
    An async predicate the loop consults at each turn boundary. Setting the
    stop event makes the agent land cleanly (with whatever answer it has)
    instead of being killed mid-tool.

Usage::

    async for event in adapter.run(session=session, prompt="…"):
        ...
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import logging
import os
import secrets
import time
from collections import deque
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from deploy.huggingface.config import DemoConfig
from deploy.huggingface.containment import (
    PathContainmentObserver,
    SecretArgumentObserver,
)
from deploy.huggingface.errors import classify_error, classify_error_name
from deploy.huggingface.events import (
    ACTIVITY_FINISHED,
    ACTIVITY_STARTED,
    ARTIFACT_CREATED,
    ASSISTANT_DELTA,
    QUEUED,
    RUN_CANCELLED,
    RUN_COMPLETED,
    RUN_FAILED,
    RUN_STARTED,
    TASK_BOARD_UPDATED,
    WARNING,
    DemoEvent,
    event,
)
from deploy.huggingface.security import (
    Redactor,
    StreamRedactor,
    demo_safe_tool_policy,
    list_output_files,
)
from deploy.huggingface.sessions import DemoSession
from frontier_agent.components.task_board_types import BOARD_TOOLS
from frontier_agent.core.loop_types import (
    ATTEMPT_ACCEPTED_DEGRADED,
    ATTEMPT_DISCARDED,
    ATTEMPT_FAILED,
    DELIVERED_ATTEMPT_OUTCOMES,
    AgentLoopResult,
    BaseObserver,
    LLMAttemptContext,
    LLMDeltaContext,
    LoopConfig,
    ToolResult,
    TurnContext,
)

logger = logging.getLogger(__name__)

#: Grace added to the task wall before the adapter stops waiting altogether.
#: The scheduler and the in-loop wall-clock observer should both land first;
#: this is the backstop for a run wedged below them.
_TIMEOUT_GRACE_S = 20.0

#: How long a cancelled run may keep going before it is hard-cancelled. The
#: cooperative path only checks at a turn boundary, which a long tool call can
#: delay.
_CANCEL_GRACE_S = 20.0

#: Trusted state identifying the fallback ``outputs/answer.md`` written by the
#: adapter. It lives outside the agent-writable workspace and survives process
#: restarts, unlike in-memory ownership bookkeeping.
_GENERATED_ANSWER_MARKER = "generated-answer.sha256"


class RunRejected(RuntimeError):
    """The run was refused before it started (bad prompt, full queue)."""


# ── event transport ──────────────────────────────────────────────────────


class EventChannel:
    """Bounded, non-blocking hand-off from the runtime to the Web layer.

    ``push`` is synchronous and never awaits: the agent loop must not stall
    because a browser stopped reading. Under pressure the *oldest streaming
    delta* is dropped (text the user has already seen), never a lifecycle
    event — losing ``run_completed`` would hang the UI forever.
    """

    def __init__(self, maxlen: int = 4096) -> None:
        self._items: deque[DemoEvent] = deque()
        self._maxlen = max(int(maxlen), 16)
        self._wake = asyncio.Event()
        self._closed = False
        self.dropped = 0

    def push(self, item: DemoEvent) -> None:
        if self._closed:
            return
        if len(self._items) >= self._maxlen and not self._evict_delta():
            self.dropped += 1
            return
        self._items.append(item)
        self._wake.set()

    def _evict_delta(self) -> bool:
        for index, item in enumerate(self._items):
            if item.type == ASSISTANT_DELTA:
                del self._items[index]
                self.dropped += 1
                return True
        return False

    def close(self) -> None:
        self._closed = True
        self._wake.set()

    async def stream(self) -> AsyncIterator[DemoEvent]:
        """Yield events until the channel is closed and drained."""
        while True:
            while self._items:
                yield self._items.popleft()
            if self._closed:
                return
            self._wake.clear()
            if self._items or self._closed:
                continue
            await self._wake.wait()


# ── runtime → events ─────────────────────────────────────────────────────


class StructuredEventObserver(BaseObserver):
    """Turns agent-loop hooks into :class:`DemoEvent`s.

    ``critical = True`` is load-bearing, not an optimisation:
    ``loop_types.notify_observers`` awaits critical observers inline but fans
    non-critical ones out as background tasks, where hook order is not
    preserved. A UI fed out-of-order deltas renders scrambled text.
    """

    critical = True

    #: Opt into per-token streaming. ``run_agent_loop`` only asks the client to
    #: stream when some observer declares this (``agent_loop.py``:
    #: ``stream_llm_tokens``) — without it the loop makes a blocking call and
    #: ``on_llm_delta`` never fires, so the answer would appear all at once.
    wants_llm_delta = True

    def __init__(
        self,
        *,
        channel: EventChannel,
        session_id: str,
        run_id: str,
        redactor: Redactor,
        artifacts: ArtifactWatcher | None = None,
        arg_preview_chars: int = 400,
        result_preview_chars: int = 600,
    ) -> None:
        self._channel = channel
        self._session_id = session_id
        self._run_id = run_id
        self._redactor = redactor
        # Per-delta redaction is unsound on its own: SSE chunk boundaries are
        # arbitrary, so a secret split across two frames matches neither half.
        self._stream = StreamRedactor(redactor)
        self._artifacts = artifacts
        self._arg_chars = arg_preview_chars
        self._result_chars = result_preview_chars
        self.turns = 0
        self.tool_calls = 0
        self.stopped_by = ""
        self.answer_chars = 0
        #: The turn and logical-call the last attempt notification belonged to.
        #: A *second* logical call for a turn already seen means a rollback
        #: observer popped the first one — see :meth:`on_llm_attempt`.
        self._last_turn = 0
        self._last_call_id = ""
        #: Last provider failure seen, as ``(error_type, reason)``. The workflow
        #: absorbs an unreachable endpoint into a best-effort answer, so this is
        #: the only record left of *why* there was nothing to answer with.
        self.last_llm_error: tuple[str, str] | None = None

    # -- helpers ---------------------------------------------------------
    def _emit(self, type_: str, **data: Any) -> None:
        self._channel.push(event(
            type_, session_id=self._session_id, run_id=self._run_id, **data,
        ))

    def _preview(self, value: Any, limit: int) -> str:
        text = value if isinstance(value, str) else _to_text(value)
        text = self._redactor.redact(text)
        text = " ".join(text.split())
        return text if len(text) <= limit else text[: limit - 1] + "…"

    def _publish_new_artifacts(self) -> None:
        if self._artifacts is None:
            return
        for path in self._artifacts.poll():
            self._emit(
                ARTIFACT_CREATED,
                name=path.name,
                relpath=str(path.relative_path),
                size=path.size,
            )

    # -- loop hooks ------------------------------------------------------
    async def on_llm_delta(self, ctx: LLMDeltaContext) -> None:
        if ctx.delta:
            self.answer_chars += len(ctx.delta)
            self._emit_text(
                self._stream.feed(ctx.delta), ctx.turn, ctx.attempt_index,
            )
        return None

    def _emit_text(self, text: str, turn: int, attempt: int) -> None:
        """Emit already-redacted stream text, skipping empty hold-backs."""
        if text:
            self._emit(ASSISTANT_DELTA, text=text, turn=turn, attempt=attempt)

    async def on_llm_attempt(self, ctx: LLMAttemptContext) -> None:
        self._note_rolled_back_turn(ctx)
        if ctx.outcome == ATTEMPT_FAILED and ctx.error_type:
            self.last_llm_error = (ctx.error_type, ctx.reason)
        if ctx.outcome in DELIVERED_ATTEMPT_OUTCOMES:
            # The stream for this call is complete, so nothing more can arrive to
            # extend a partial secret: release whatever is still held back.
            self._emit_text(self._stream.flush(), ctx.turn, ctx.attempt_index)
        # A reply delivered *degraded* still reached the loop, so its bytes are
        # kept — but the guard that let it through is the only explanation for
        # a turn that then produces nothing readable. Without this the UI shows
        # an unexplained empty step.
        if ctx.outcome == ATTEMPT_ACCEPTED_DEGRADED:
            self._emit(
                WARNING,
                reason=ctx.reason or "llm_attempt_degraded",
                message=(
                    "the model's reply was accepted in a degraded state"
                    + (f" ({ctx.reason})" if ctx.reason else "")
                ),
                turn=ctx.turn,
                attempt=ctx.attempt_index,
            )
        # A discarded attempt means the deltas already streamed for it are not
        # part of the answer. Say so explicitly so the UI can reset its buffer
        # instead of showing two interleaved drafts.
        if ctx.outcome == ATTEMPT_DISCARDED:
            # Drop, do not flush: this attempt's text is being thrown
            # away, so releasing its tail would splice it onto the retry.
            self._stream.discard()
            self._emit(
                WARNING,
                reason="llm_attempt_discarded",
                message=(
                    "the model's response was discarded and retried"
                    + (f" ({ctx.reason})" if ctx.reason else "")
                ),
                discard_stream=True,
                turn=ctx.turn,
                attempt=ctx.attempt_index,
            )
        return None

    def _note_rolled_back_turn(self, ctx: LLMAttemptContext) -> None:
        """Detect a turn the loop re-ran after a rollback observer popped it.

        ``DuplicateQueryRollbackObserver`` (and any other observer returning
        ``pop_last_message``) rejects an assistant message in
        ``on_llm_response`` — *after* its deltas have already been streamed to
        the browser, and without the ``ATTEMPT_DISCARDED`` notification the
        retry path emits, because the attempt itself was delivered fine. The
        loop then decrements ``turn`` and issues a fresh logical call, so a new
        ``call_id`` under an unchanged ``turn`` is the signal: whatever the UI
        is showing for this turn is a draft the runtime has thrown away.

        Keyed on the attempt notification rather than the next delta because a
        re-run turn may emit no visible text at all (a tool-call-only turn),
        and the stale draft would then stay on screen unchallenged.
        """
        if not ctx.call_id:
            return
        rolled_back = (
            ctx.turn == self._last_turn and ctx.call_id != self._last_call_id
        )
        self._last_turn, self._last_call_id = ctx.turn, ctx.call_id
        if not rolled_back:
            return
        self._stream.discard()
        self._emit(
            WARNING,
            reason="turn_rolled_back",
            message=(
                "the model's last step was rolled back and re-planned "
                "(it repeated a request that had already run)"
            ),
            discard_stream=True,
            turn=ctx.turn,
            attempt=ctx.attempt_index,
        )

    async def on_tool_call(self, ctx: TurnContext, tool_call: dict) -> None:
        name, args = _tool_call_parts(tool_call)
        self.tool_calls += 1
        self._emit(
            ACTIVITY_STARTED,
            activity=name,
            turn=ctx.turn,
            call_id=str(tool_call.get("id") or ""),
            detail=self._preview(args, self._arg_chars),
        )
        return None

    async def on_tool_result(self, ctx: TurnContext, result: ToolResult) -> None:
        # ``is_error`` is only set when the tool *raised*. Tools in this repo
        # (and the containment gate) report refusals by returning an
        # "Error: …" string instead, and the UI must not paint those as
        # successes.
        failed = result.is_error or _looks_like_error(result.result)
        self._emit(
            ACTIVITY_FINISHED,
            activity=result.name,
            turn=ctx.turn,
            call_id=result.tool_call_id,
            ok=not failed,
            interrupted=result.interrupted,
            duration_ms=result.duration_ms,
            detail=self._preview(result.result, self._result_chars),
        )
        if failed:
            self._emit(
                WARNING,
                reason="tool_error",
                message=f"{result.name} failed",
                detail=self._preview(result.result, self._result_chars),
            )
        elif result.name in BOARD_TOOLS:
            # Project the runtime's authoritative board state. The UI should
            # never have to infer ids/statuses by parsing a tool's prose result.
            #
            # No masking here: ``SecretArgumentObserver`` scrubs every tool
            # call's arguments before the tool runs, so a description is already
            # clean by the time it reaches the board.
            from plugins.tools.task_board import snapshot_tasks

            self._emit(
                TASK_BOARD_UPDATED,
                tasks=snapshot_tasks(ctx.task_id),
            )
        self._publish_new_artifacts()
        return None  # leave the result unchanged for the next observer

    async def on_turn_end(self, ctx: TurnContext) -> None:
        self.turns = max(self.turns, ctx.turn)
        return None

    async def on_loop_end(self, result: AgentLoopResult) -> None:
        self.stopped_by = result.stopped_by or ""
        self.turns = max(self.turns, result.turns_used)
        # Backstop: a loop that ends without a delivered-attempt notification
        # must still not strand held-back text.
        self._emit_text(self._stream.flush(), self.turns, 1)
        self._publish_new_artifacts()

    async def on_loop_start(self, config: LoopConfig) -> None:
        return None


def _tool_call_parts(tool_call: Mapping[str, Any]) -> tuple[str, str]:
    """Extract ``(name, arguments)`` from either tool-call shape.

    Observers receive the loop's parsed form (``{"name", "args", "id"}``); the
    provider wire form (``{"function": {"name", "arguments"}}``) is accepted too
    so the activity feed is populated either way.
    """
    function = tool_call.get("function") or {}
    name = str(function.get("name") or tool_call.get("name") or "tool")
    for raw in (
        tool_call.get("args"),
        function.get("arguments"),
        tool_call.get("arguments"),
    ):
        if raw not in (None, "", {}):
            return name, _to_text(raw)
    return name, ""


#: How the repo's tools spell a refusal or failure in a *returned* string.
_ERROR_PREFIXES = ("error:", "error ", "[error", "traceback")


def _looks_like_error(result: str) -> bool:
    head = str(result or "").lstrip().lower()[:40]
    return head.startswith(_ERROR_PREFIXES) or "error]" in head


def _to_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return str(value)


# ── artifacts ────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Artifact:
    """A file the agent produced in the session's ``outputs/`` directory."""

    path: Path
    relative_path: Path
    size: int

    @property
    def name(self) -> str:
        return self.path.name


class ArtifactWatcher:
    """Reports files that appeared in ``outputs/`` since the last poll.

    ``prime_now=False`` defers the baseline: the staged mount point is seeded
    from the session *after* the watcher is constructed, and files carried over
    from an earlier prompt in the same session are not this run's output.
    """

    def __init__(
        self,
        outputs_dir: Path,
        *,
        prime_now: bool = True,
        redactor: Redactor | None = None,
    ) -> None:
        self._root = Path(outputs_dir)
        # Announce only what the visitor could actually download. Without this,
        # a file withheld for containing a secret would still be reported as
        # "produced" and then be absent from the download list.
        self._redactor = redactor
        self._seen: dict[Path, int] = {}
        if prime_now:
            self.prime()

    def _visible(self) -> list[Path]:
        return list_output_files(self._root, redactor=self._redactor)

    def prime(self) -> None:
        """Record the current contents as pre-existing (not this run's output)."""
        self._seen = {p: _size_of(p) for p in self._visible()}

    def poll(self) -> list[Artifact]:
        """New or grown files since the previous call."""
        fresh: list[Artifact] = []
        for path in self._visible():
            size = _size_of(path)
            if self._seen.get(path) == size:
                continue
            self._seen[path] = size
            with contextlib.suppress(ValueError):
                fresh.append(Artifact(
                    path=path,
                    relative_path=path.relative_to(self._root),
                    size=size,
                ))
        return fresh


def _size_of(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return -1


# ── the adapter ──────────────────────────────────────────────────────────


@dataclass
class RunHandle:
    """Bookkeeping for one in-flight run, so ``cancel`` can reach it."""

    run_id: str
    session_id: str
    stop: asyncio.Event = field(default_factory=asyncio.Event)
    task: asyncio.Task | None = None
    started_at: float = field(default_factory=time.time)


class FrontierAgentAdapter:
    """Runs the FrontierAgent ``react`` workflow and emits structured events."""

    def __init__(self, config: DemoConfig) -> None:
        self._config = config
        self._redactor = Redactor.for_secrets(config.secrets)
        # Serialised on purpose. The runtime keeps per-process global state —
        # the service registry (BenchmarkSession snapshots and restores it),
        # the bash policy / task sandbox context, the mount-dir env vars read
        # per run — so two concurrent runs in one process would not be
        # isolated from each other. Scale with replicas, not with this value.
        self._effective_concurrency = config.effective_concurrency
        self._slots = asyncio.Semaphore(self._effective_concurrency)
        self._waiting = 0
        self._runs: dict[str, RunHandle] = {}

    @property
    def config(self) -> DemoConfig:
        return self._config

    @property
    def redactor(self) -> Redactor:
        """The secret masker this adapter applies. Callers listing files for
        download should pass it to ``list_output_files`` so a file whose
        *contents* carry a secret is never offered."""
        return self._redactor

    @property
    def effective_concurrency(self) -> int:
        return self._effective_concurrency

    @property
    def active_run_ids(self) -> tuple[str, ...]:
        return tuple(self._runs)

    def busy_session_ids(self) -> set[str]:
        return {handle.session_id for handle in self._runs.values()}

    def cancel(self, run_id: str) -> bool:
        """Ask a run to stop at its next turn boundary. Returns True if known."""
        handle = self._runs.get(run_id)
        if handle is None:
            return False
        handle.stop.set()
        return True

    def cancel_session(self, session_id: str) -> int:
        """Cancel every run belonging to ``session_id``."""
        stopped = 0
        for handle in list(self._runs.values()):
            if handle.session_id == session_id:
                handle.stop.set()
                stopped += 1
        return stopped

    async def run(
        self,
        *,
        session: DemoSession,
        prompt: str,
        run_id: str | None = None,
    ) -> AsyncIterator[DemoEvent]:
        """Execute one prompt, yielding events until a terminal event.

        Exactly one of ``run_completed`` / ``run_failed`` / ``run_cancelled``
        is always the last event, including on rejection — the UI can rely on
        it to re-enable its controls.
        """
        rid = run_id or f"run-{secrets.token_hex(8)}"
        sid = session.session_id
        channel = EventChannel()

        text = str(prompt or "").strip()
        if not text:
            yield event(RUN_FAILED, session_id=sid, run_id=rid,
                        reason="empty_prompt", message="Enter a task first.")
            return
        if len(text) > self._config.max_prompt_chars:
            yield event(
                RUN_FAILED, session_id=sid, run_id=rid, reason="prompt_too_long",
                message=(
                    f"The prompt is {len(text)} characters; this demo accepts "
                    f"up to {self._config.max_prompt_chars}."
                ),
            )
            return
        if self._waiting >= self._config.queue_size:
            yield event(
                RUN_FAILED, session_id=sid, run_id=rid, reason="queue_full",
                message=(
                    f"The demo queue is full ({self._config.queue_size} waiting). "
                    "Please retry in a moment."
                ),
            )
            return

        handle = RunHandle(run_id=rid, session_id=sid)
        self._runs[rid] = handle
        # One try/finally covers the whole lifetime: a run abandoned while still
        # queued must leave no entry behind, or ``cancel`` would keep reporting
        # success for it and the queue accounting would drift upwards forever.
        acquired = False
        self._waiting += 1
        try:
            if self._slots.locked():
                yield event(
                    QUEUED, session_id=sid, run_id=rid,
                    position=self._waiting,
                    message="Waiting for the demo runner to free up…",
                )
            try:
                await self._slots.acquire()
                acquired = True
            finally:
                self._waiting -= 1

            session.ensure_dirs()
            # Remove only an unchanged adapter-generated fallback, and do it
            # before the agent can author a new file at the same path.
            _retire_generated_answer(session)
            artifacts = ArtifactWatcher(
                session.outputs, redactor=self._redactor,
            )
            observer = StructuredEventObserver(
                channel=channel, session_id=sid, run_id=rid,
                redactor=self._redactor, artifacts=artifacts,
            )
            yield event(
                RUN_STARTED, session_id=sid, run_id=rid,
                workflow=self._config.workflow,
                model=self._config.model_id,
                max_turns=self._config.max_turns,
                timeout_s=int(self._config.task_timeout_s),
            )

            handle.task = asyncio.create_task(
                self._execute(session, text, handle, observer),
                name=f"frontier-demo-{rid}",
            )
            waiter = asyncio.create_task(
                self._await_result(session, handle, channel, observer),
                name=f"frontier-demo-wait-{rid}",
            )
            try:
                async for item in channel.stream():
                    yield item
            except GeneratorExit:
                # The browser went away mid-run: stop the agent rather than
                # leaving it burning tokens for nobody.
                handle.stop.set()
                raise
            finally:
                channel.close()
                for pending in (handle.task, waiter):
                    if pending is not None and not pending.done():
                        pending.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await asyncio.gather(
                        *(p for p in (handle.task, waiter) if p is not None),
                        return_exceptions=True,
                    )
        finally:
            if acquired:
                self._slots.release()
            self._runs.pop(rid, None)

    # -- internals -------------------------------------------------------

    async def _settle(self, handle: RunHandle) -> dict[str, Any]:
        """Await the run under two deadlines: the task wall and the stop grace.

        Cooperative cancellation only takes effect at a turn boundary, which a
        hung LLM or tool call postpones indefinitely. Without a bound on that,
        pressing Stop would keep the single runner — and therefore every queued
        visitor — waiting for the full task timeout.

        Raises ``asyncio.TimeoutError`` for either deadline; the caller
        distinguishes them by whether the stop event was set.
        """
        task = handle.task
        assert task is not None
        stopper = asyncio.create_task(
            handle.stop.wait(), name=f"frontier-demo-stop-{handle.run_id}",
        )
        try:
            done, _pending = await asyncio.wait(
                {asyncio.shield(task), stopper},
                timeout=self._config.task_timeout_s + _TIMEOUT_GRACE_S,
                return_when=asyncio.FIRST_COMPLETED,
            )
            if not done:
                raise TimeoutError  # the task wall elapsed
            if task.done():
                return task.result()
            # Stop was requested while the run is still going: allow the
            # cooperative path a bounded window to land with a partial answer.
            return await asyncio.wait_for(
                asyncio.shield(task), timeout=_CANCEL_GRACE_S,
            )
        finally:
            stopper.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await stopper

    async def _await_result(
        self,
        session: DemoSession,
        handle: RunHandle,
        channel: EventChannel,
        observer: StructuredEventObserver,
    ) -> None:
        """Await the run task and push exactly one terminal event."""
        sid, rid = handle.session_id, handle.run_id
        assert handle.task is not None
        try:
            state = await self._settle(handle)
        except TimeoutError:
            # Either the task wall elapsed, or Stop was requested and the
            # cooperative path did not land inside its grace. Which one it was
            # decides what the visitor is told.
            stopped = handle.stop.is_set()
            handle.stop.set()
            handle.task.cancel()
            with contextlib.suppress(BaseException):
                await handle.task
            if stopped:
                channel.push(event(
                    RUN_CANCELLED, session_id=sid, run_id=rid,
                    reason="cancel_grace_expired",
                    message=(
                        "Stopped. The agent was mid-call and did not finish "
                        f"within {int(_CANCEL_GRACE_S)}s, so the run was ended."
                    ),
                    turns=observer.turns, tool_calls=observer.tool_calls,
                ))
            else:
                channel.push(event(
                    RUN_FAILED, session_id=sid, run_id=rid, reason="timeout",
                    message=(
                        "The run exceeded this demo's time budget "
                        f"({int(self._config.task_timeout_s)}s) and was stopped."
                    ),
                    turns=observer.turns, tool_calls=observer.tool_calls,
                ))
        except asyncio.CancelledError:
            channel.push(event(
                RUN_CANCELLED, session_id=sid, run_id=rid,
                reason="cancelled", turns=observer.turns,
                tool_calls=observer.tool_calls,
            ))
        except Exception as exc:
            logger.warning("demo run %s failed", rid, exc_info=True)
            reason, message = classify_error(exc)
            channel.push(event(
                RUN_FAILED, session_id=sid, run_id=rid,
                reason=reason, message=self._redactor.redact(message),
                turns=observer.turns, tool_calls=observer.tool_calls,
            ))
        else:
            final = self._redactor.redact(_final_answer(state))
            stopped_by = str(state.get("stopped_by") or observer.stopped_by)
            created = (
                _ensure_answer_artifact(
                    session, final, redactor=self._redactor,
                )
                if stopped_by != "llm_error"
                else None
            )
            if created is not None:
                channel.push(event(
                    ARTIFACT_CREATED,
                    session_id=sid,
                    run_id=rid,
                    name=created.name,
                    relpath=str(created.relative_to(session.outputs)),
                    size=_size_of(created),
                    generated=True,
                ))
            artifacts = [
                str(path.relative_to(session.outputs))
                for path in list_output_files(
                    session.outputs, redactor=self._redactor,
                )
            ]
            common: dict[str, Any] = {
                "stopped_by": stopped_by,
                "answer_status": str(state.get("answer_status") or ""),
                "turns": observer.turns,
                "tool_calls": observer.tool_calls,
                "dropped_events": channel.dropped,
                "artifacts": artifacts,
            }
            # The react workflow deliberately fails *open*: when the endpoint is
            # unreachable it still returns a placeholder "best available result"
            # so a benchmark row is never empty. A demo must not present that as
            # an answer — the endpoint, not the model, is what needs fixing.
            if stopped_by == "llm_error":
                error_type, detail = observer.last_llm_error or ("", "")
                reason, message = classify_error_name(error_type, detail)
                channel.push(event(
                    RUN_FAILED, session_id=sid, run_id=rid,
                    reason=reason, message=self._redactor.redact(message),
                    partial_answer=final, **common,
                ))
            else:
                terminal = RUN_CANCELLED if handle.stop.is_set() else RUN_COMPLETED
                channel.push(event(
                    terminal, session_id=sid, run_id=rid, answer=final, **common,
                ))
        finally:
            channel.close()

    async def _execute(
        self,
        session: DemoSession,
        prompt: str,
        handle: RunHandle,
        observer: StructuredEventObserver,
    ) -> dict[str, Any]:
        """Bootstrap the runtime and run one ``react`` task."""
        from benchmarks.public.core.kernel_adapter import BenchmarkSession
        from frontier_agent.core.runtime import registry
        from frontier_agent.core.runtime.resources.manager import ResourceManager

        config = self._config
        metadata: dict[str, Any] = {
            # Which workflow profile YAML to load, and the demo's bounds on top.
            "profile": config.workflow_profile,
            "profile_overrides": config.profile_overrides(),
            # Structured events instead of a terminal UI.
            "sdk_extra_observers": [
                observer,
                PathContainmentObserver(
                    workspace=session.workspace,
                    read_roots=(session.inputs,),
                    session_label=session.short_id,
                ),
                SecretArgumentObserver(
                    self._redactor, session_label=session.short_id,
                ),
            ],
            # Cooperative cancellation at the next turn boundary.
            "pause_check": _stop_predicate(handle.stop),
            # Session affinity upstream + per-session artifact/trace locations.
            "session_id": session.session_id,
            "turn_index": 1,
            "_trial_dir": str(session.state),
            # The runtime's authorised write root. Deliberately the workspace
            # rather than the session root: ``outputs/`` lives inside the
            # workspace, so this still covers every legitimate write while
            # leaving ``inputs/`` (read-only) and ``state/`` (traces) out of
            # reach — see ``plugins/tools/_path_auth``.
            "coding_workspace_root": str(session.workspace),
            "run_id": handle.run_id,
            "run_type": "hf_space_demo",
        }

        with _scoped_env(self._runtime_env(session)):
            async with BenchmarkSession() as runtime:
                # Fail closed: install the demo toolset before anything runs.
                registry.get(ResourceManager).set_global_tool_policy(
                    demo_safe_tool_policy(
                        config.allowed_tools, public_mode=config.public_mode,
                    ),
                )
                run = runtime.run(
                    prompt,
                    meta=metadata,
                    pipeline_id=config.pipeline_id,
                    extra_input={"current_query": prompt},
                )
                return await asyncio.wait_for(
                    run, timeout=config.task_timeout_s,
                )

    def _runtime_env(self, session: DemoSession) -> dict[str, str]:
        """Per-run environment the runtime reads while the task executes.

        These are process-wide by construction (``resolve_mount_dirs`` and the
        wall-time budget read ``os.environ`` inside the node), which is the
        second reason runs are serialised.
        """
        config = self._config
        return {
            # ``resolve_mount_dirs`` reads these, and in native mode the react
            # prompt names them verbatim — so the agent is told about *this*
            # session's directories and nobody else's.
            "FRONTIER_AGENT_WORKSPACE_DIR": str(session.workspace),
            "FRONTIER_AGENT_OUTPUTS_DIR": str(session.outputs),
            "FRONTIER_AGENT_INPUTS_DIR": str(session.inputs),
            # Authorises the file tools for this session's tree and nothing
            # else (``plugins/tools/_path_auth``).
            "CODING_WORKSPACE_ROOT": str(session.workspace),
            # Scheduler-level hard ceiling for the whole graph execution.
            "FRONTIER_AGENT_TASK_WALL_TIME_S": str(config.hard_wall_time_s),
            # Defence in depth: even though ``bash`` is not in the demo
            # toolset, keep the command allowlist enforced.
            "BASH_ALLOWLIST_MODE": "enforce",
            # No model-authored command can run (bash/run_python_code are
            # denied), so the absent unprivileged tool account is not a gap.
            "FRONTIER_AGENT_TOOL_USER": "off",
            "OPENAI_MAX_TOKENS": str(config.max_output_tokens),
        }


def _stop_predicate(stop: asyncio.Event) -> Callable[[], Awaitable[bool]]:
    async def _check() -> bool:
        return stop.is_set()

    return _check


@contextlib.contextmanager
def _scoped_env(values: Mapping[str, str]) -> Iterator[None]:
    """Apply ``values`` to ``os.environ``, restoring the previous state after."""
    previous: dict[str, str | None] = {k: os.environ.get(k) for k in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for key, old in previous.items():
            if old is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old


def _final_answer(state: Mapping[str, Any] | None) -> str:
    if not state:
        return ""
    for key in ("final_answer", "final_content"):
        value = str(state.get(key) or "").strip()
        if value:
            return value
    return ""


def _ensure_answer_artifact(
    session: DemoSession,
    answer: str,
    *,
    redactor: Redactor,
) -> Path | None:
    """Keep the download pane useful when a text-only run made no file.

    Agent-authored deliverables remain untouched. Only when the outputs tree is
    genuinely empty do we persist the already-redacted final response as a
    small Markdown fallback.

    The prior run's fallback is retired before execution begins. Ownership is
    recorded in trusted session state so it survives process restarts and can
    never be confused with an agent-authored file that happens to use the same
    name.
    """
    if not answer.strip() or list_output_files(session.outputs, redactor=redactor):
        return None
    session.outputs.mkdir(parents=True, exist_ok=True)
    path = session.outputs / "answer.md"
    content = answer.rstrip() + "\n"
    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
    marker = session.state / _GENERATED_ANSWER_MARKER
    marker.write_text(digest + "\n", encoding="ascii")
    path.write_text(content, encoding="utf-8")
    return path


def _retire_generated_answer(session: DemoSession) -> None:
    """Remove the preceding run's unchanged fallback before this run starts.

    A digest, rather than the filename alone, proves ownership. If anything
    replaced or edited ``answer.md`` after the fallback was created, preserve
    it as a genuine session deliverable and consume the stale marker only.
    """
    marker = session.state / _GENERATED_ANSWER_MARKER
    try:
        expected = marker.read_text(encoding="ascii").strip()
    except FileNotFoundError:
        return
    except OSError:
        logger.warning("could not read generated-answer marker", exc_info=True)
        return

    path = session.outputs / "answer.md"
    try:
        if path.is_file():
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            if actual == expected:
                path.unlink()
        marker.unlink(missing_ok=True)
    except OSError:
        # Keep the marker so a later run can retry cleanup. More importantly,
        # never make a run fail merely because optional fallback cleanup failed.
        logger.warning("could not retire generated answer", exc_info=True)


__all__ = [
    "Artifact",
    "ArtifactWatcher",
    "EventChannel",
    "FrontierAgentAdapter",
    "RunHandle",
    "RunRejected",
    "StructuredEventObserver",
]
