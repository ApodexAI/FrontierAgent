"""Observers for the native stateful ReAct agent."""

from __future__ import annotations

import asyncio
import datetime as _dt
import logging
import re
import time
from typing import Any

from rich.console import Console

from frontier_agent.components.finalization import (
    has_malformed_tool_protocol,
    remaining_phase_budget_s,
)
from frontier_agent.components.observers.console import (
    RichConsoleObserver as RichConsoleObserver,
)
from frontier_agent.core.llm import LLMClient
from frontier_agent.core.loop_types import (
    AgentLoopResult,
    BaseObserver,
)
from frontier_agent.core.messages import Message, assistant_msg_with_reasoning, user_msg
from frontier_agent.infra.nonblocking_stream import nonblocking_stderr
from workflows._shared.sdk_shim import (
    ReporterDeltaEmitter,
    record_reporter_usage,
)
from workflows.stateful_react_agent._runtime import (
    _build_recovery_messages,
    _forced_final_stop_reasons,
    _minimal_best_effort_answer,
    _strip_leaked_tool_calls,
    force_final_answer,
    scrub_leaked_tool_calls,
    unwrap_fenced_images,
)
from workflows.stateful_react_agent.prompts import get_report_prompt

logger = logging.getLogger(__name__)
# NonBlockingStream implements only the part of the file protocol Rich uses
# (write / flush / isatty); see its docstring.
_console = Console(
    file=nonblocking_stderr(),  # pyright: ignore[reportArgumentType]
    width=200,
    force_terminal=True,
)

_CONTENT_PREVIEW = 2000
_ARGS_PREVIEW = 400
_RESULT_PREVIEW = 400




class FinalAnswerSalvageObserver(BaseObserver):
    """Synthesise a reporter-disabled final answer before streaming/terminal.

    On bounded/abnormal stops (``max_turns`` / ``context_limit_reached`` / infra
    errors) the agent produced no clean no-tool final turn, so the answer is
    synthesised by :func:`force_final_answer`. Historically that ran in the node
    tail — AFTER the loop's terminal ``final`` and the reporter stream — so the
    wire carried the pre-salvage draft while the node returned a different,
    synthesised answer.

    Reporter-enabled runs use :class:`ReportSynthesisObserver` directly and do
    not mount this observer. In reporter-disabled runs, running it here
    (``critical``, ordered before :class:`ReporterStreamObserver` and the
    serve chain's protocol stream observer) sets ``result.final_content`` and
    ``result.metadata["final_answer"]`` before either reads them, so the raw
    stream, terminal ``final``, and node return agree. No-op on normal
    ``no_tool`` completion and on cancel/pause.
    """

    critical: bool = True

    def __init__(
        self,
        *,
        llm: LLMClient,
        timeout: float,
        task_description: str,
        thinking_format: str,
        salvage_infra_errors: bool,
        final_prompt: str | None = None,
        language: str = "",
        phase_deadline_monotonic: float | None = None,
    ) -> None:
        self._llm = llm
        self._timeout = timeout
        self._task_description = task_description
        self._thinking_format = thinking_format
        self._salvage_infra_errors = salvage_infra_errors
        self._final_prompt = final_prompt
        self._language = language
        self._phase_deadline_monotonic = phase_deadline_monotonic

    async def on_loop_end(self, result: AgentLoopResult) -> None:
        # D5b: cancel/pause leaves no terminal — don't synthesise a salvage
        # answer for a run that was stopped.
        if result.stopped_by == "paused":
            return
        # force_final_answer self-gates: it returns early unless stopped_by is a
        # forced reason and no final_answer exists yet. It never raises (falls
        # back to an explicit best-effort status), so no guard is needed here.
        await force_final_answer(
            result,
            self._llm,
            # Never ask for more time than the external ceiling still allows —
            # being cancelled mid-rescue would leave the run with no answer at
            # all, which is exactly what the rescue exists to prevent.
            remaining_phase_budget_s(
                self._timeout, self._phase_deadline_monotonic,
            ),
            task_description=self._task_description,
            thinking_format=self._thinking_format,
            salvage_infra_errors=self._salvage_infra_errors,
            final_prompt=self._final_prompt,
            language=self._language,
        )


class ReporterStreamObserver(BaseObserver):
    """Re-stream the final answer as reporter ``llm_delta`` frames, pre-terminal.

    The stateful agent's final answer IS the report, but the frontend renders
    the report from ``response.swarm.llm_delta`` frames with
    ``agent_id="reporter"``. The agent's own turns stream under
    ``agent_id="stateful_react"`` and ``top_terminal_tool=None`` for this
    pipeline, so nothing reporter-attributed otherwise reaches the wire.

    This observer emits the reporter run envelope + the resolved answer as
    ``output_text`` deltas at ``on_loop_end``. It is ``critical`` and MUST be
    ordered BEFORE the serve chain's protocol stream observer in the observer
    list (``notify_observers`` awaits critical hooks inline in list order): the
    protocol observer emits the terminal ``final`` in ITS ``on_loop_end``, so
    the reporter stream lands immediately before the terminal.

    Cancellation/pause emits nothing — a hard cancel fires ``on_loop_cancelled``
    (not ``on_loop_end``), and the ``paused`` stop is skipped here, matching the
    D5b "no terminal on cancel" convention.

    On salvage stops (``max_turns`` / ``context_limit_reached`` / infra errors)
    the resolved answer comes from :class:`FinalAnswerSalvageObserver`, which
    MUST be ordered before this one so ``result`` already holds the synthesised
    answer here — otherwise the stream would carry the pre-salvage draft.
    """

    critical: bool = True

    def __init__(self, emitter: object) -> None:
        self._stream = ReporterDeltaEmitter(emitter)

    async def on_loop_end(self, result: AgentLoopResult) -> None:
        # D5b: cancellation / pause leaves no terminal — emit nothing.
        if result.stopped_by == "paused":
            return
        text = _strip_leaked_tool_calls(str(
            result.metadata.get("final_answer") or result.final_content or ""
        ))
        if not text:
            return
        self._stream.start()
        self._stream.stream_output(text)
        self._stream.finish(final_content=text)


_THINK_OPEN_RE = re.compile(r"<\s*think\s*>", re.IGNORECASE)
_THINK_CLOSE_RE = re.compile(r"<\s*/\s*think\s*>", re.IGNORECASE)
# Leaked qwen text-mode markup, split into openers + their closers so a partially
# streamed block can be held back instead of shipped raw (the final-text
# equivalents live in ``_runtime._LEAKED_*``).
_LEAK_OPEN_RE = re.compile(r"<\s*(tool_call|tool_response|function)\b", re.IGNORECASE)
_LEAK_CLOSE_RE = {
    "tool_call": re.compile(r"<\s*/\s*tool_call\s*>", re.IGNORECASE),
    "tool_response": re.compile(r"<\s*/\s*tool_response\s*>", re.IGNORECASE),
    "function": re.compile(r"<\s*/\s*function\s*>", re.IGNORECASE),
}
# A trailing fragment that could still grow into any of these must be withheld.
_PARTIAL_TAGS = ("<think>", "</think>", "<tool_call", "<tool_response", "<function")
_PARTIAL_CLOSE_TAGS = ("</think>",)


def _partial_tail_len(text: str, literals: tuple[str, ...]) -> int:
    """Length of the trailing suffix that could still become one of ``literals``."""
    limit = min(len(text), max(len(literal) for literal in literals) - 1)
    for size in range(limit, 0, -1):
        tail = text[-size:].lower()
        if any(literal.startswith(tail) for literal in literals):
            return size
    return 0


def _drop_think_prefix(text: str) -> str:
    """Everything after the LAST ``</think>`` (mirrors mtv2's ``_strip_thinking``).

    Belt-and-braces for the persisted answer: a closing tag that survives the
    streaming filter (duplicated tags, a tag nested inside held markup) still
    must not reach ``final_answer`` / the trace.
    """
    if "</think>" in text.lower():
        return re.split(_THINK_CLOSE_RE, text)[-1]
    return text


class _ReportStreamFilter:
    """Incrementally sanitise a streamed report body.

    The delta stream and the persisted answer must agree (a consumer may
    assemble the report from ``output_text`` deltas — the documented contract in
    ``reporter_stream.py``), so this applies BOTH of the final text's removals
    per chunk:

    * ``<think>…</think>`` blocks — including the SGLang / Qwen quirk where the
      OPENING tag is missing but the closing tag is always emitted. For an
      OPENER-LESS close the **last** ``</think>`` wins, matching ``_strip_thinking``
      in mtv2 (``rsplit("</think>", 1)[-1]``): a close arriving after visible text
      was already streamed retroactively drops that text from the persisted report
      (a live stream cannot be retracted). A PAIRED ``<think>…</think>`` after
      visible text is a block embedded in the report — it is excised and the
      surrounding text kept.
    * leaked qwen text-mode tool-call markup
      (:func:`~workflows.stateful_react_agent._runtime.scrub_leaked_tool_calls`).

    Tags and whole markup blocks may straddle provider chunks, so any suffix
    that could still become a delimiter — and any unclosed markup block — is
    withheld until the following ``feed``. Visible leading/trailing whitespace is
    deferred so the result matches a final ``.strip()`` without buffering the
    body.

    ``assume_think_prefix`` (tag-mode thinking with ``enable_thinking``, i.e. the
    Apodex profiles' live configuration) starts the filter INSIDE an implicit
    think block: with an absent opener that is the only leak-free reading of the
    stream. Two escapes keep it from swallowing a report: endpoints that do
    separate reasoning (SGLang ``--reasoning-parser``) release the hold on the
    first ``reasoning_content`` delta via :meth:`note_reasoning_channel`, and a
    stream that produced neither a tag nor a reasoning delta is flushed verbatim
    by :meth:`finish` (that call loses live streaming, never the report).
    """

    def __init__(self, *, assume_think_prefix: bool = False) -> None:
        # ``_assumed_hold``: inside an *implicit* (opener-less) think block —
        # everything is withheld rather than trimmed, so the hold can be
        # released later with the report body intact.
        self._assumed_hold = bool(assume_think_prefix)
        self._in_think = bool(assume_think_prefix)
        self._pending = ""
        self._visible = ""
        self._after_think = False
        self._visible_started = False
        self._pending_whitespace = ""
        self._saw_tag = False
        # ``_paired_open``: the think block currently open was entered via a real
        # ``<think>`` opener (not the implicit hold / an opener-less close), so
        # its close is a block EMBEDDED in the report — the surrounding visible
        # text must survive rather than be retracted by last-close-wins.
        self._paired_open = False

    @property
    def visible_text(self) -> str:
        """Sanitised report text so far — authoritative for the persisted answer."""
        return self._visible

    def _emit(self, text: str) -> str:
        text = scrub_leaked_tool_calls(text)
        if not text:
            return ""
        if self._after_think:
            text = text.lstrip()
            self._after_think = False
        if not self._visible_started:
            text = text.lstrip()
        if not text:
            return ""

        trailing = len(text) - len(text.rstrip())
        if trailing:
            core = text[:-trailing]
            whitespace = text[-trailing:]
        else:
            core = text
            whitespace = ""
        if not core:
            self._pending_whitespace += whitespace
            return ""

        out = self._pending_whitespace + core
        self._pending_whitespace = whitespace
        self._visible_started = True
        self._visible += out
        return out

    def _close_think(self, end: int) -> None:
        """Leave a think block whose ``</think>`` ends at ``end`` in ``_pending``."""
        self._pending = self._pending[end:]
        self._in_think = False
        self._assumed_hold = False
        self._after_think = True
        self._saw_tag = True
        paired = self._paired_open
        self._paired_open = False
        if self._visible and not paired:
            # Last close wins for an OPENER-LESS close (missing-opener SGLang/Qwen
            # quirk, duplicated tags): what looked like the report was reasoning
            # after all. The wire already has it; drop it from the persisted text
            # so surfaces B and C stay correct. A PAIRED <think>…</think> after
            # visible text is instead a block embedded in the report — excise it
            # and keep the surrounding text (``_after_think`` trims its seam).
            logger.warning(
                "reporter: late </think> after %d streamed chars — "
                "dropping them from the persisted report", len(self._visible),
            )
            self._visible = ""
            self._visible_started = False
            self._pending_whitespace = ""

    def _safe_end(self) -> int:
        """End of the portion of ``_pending`` that can be emitted now."""
        text = self._pending
        for match in _LEAK_OPEN_RE.finditer(text):
            closer = _LEAK_CLOSE_RE[match.group(1).lower()]
            if closer.search(text, match.end()) is None:
                return match.start()
        return len(text) - _partial_tail_len(text, _PARTIAL_TAGS)

    def feed(self, text: str) -> str:
        """Consume one raw content delta and return its safe visible portion."""
        if not text:
            return ""
        self._pending += text
        emitted: list[str] = []

        while self._pending:
            if self._in_think:
                close = _THINK_CLOSE_RE.search(self._pending)
                if close is None:
                    if not self._assumed_hold:
                        # Confirmed reasoning: discard all but a partial close tag.
                        keep = _partial_tail_len(self._pending, _PARTIAL_CLOSE_TAGS)
                        self._pending = self._pending[len(self._pending) - keep:] if keep else ""
                    break
                self._close_think(close.end())
                continue

            open_m = _THINK_OPEN_RE.search(self._pending)
            close_m = _THINK_CLOSE_RE.search(self._pending)
            if close_m is not None and (open_m is None or close_m.start() < open_m.start()):
                # Stray close with no opener → everything before it is reasoning.
                self._close_think(close_m.end())
                continue
            if open_m is not None:
                visible = self._emit(self._pending[:open_m.start()])
                if visible:
                    emitted.append(visible)
                self._pending = self._pending[open_m.end():]
                self._in_think = True
                self._assumed_hold = False
                self._saw_tag = True
                self._paired_open = True
                continue

            end = self._safe_end()
            safe, self._pending = self._pending[:end], self._pending[end:]
            visible = self._emit(safe)
            if visible:
                emitted.append(visible)
            break

        return "".join(emitted)

    def note_reasoning_channel(self) -> str:
        """Release an ``assume_think_prefix`` hold: the endpoint separates reasoning.

        A native ``reasoning_content`` delta proves thinking is NOT inlined in
        ``content``, so the withheld text is report body. Returns whatever
        becomes emittable (usually ``""`` — reasoning deltas precede content).
        """
        if not self._assumed_hold:
            return ""
        self._assumed_hold = False
        self._in_think = False
        pending, self._pending = self._pending, ""
        return self.feed(pending)

    def finish(self) -> str:
        """Flush the settled tail; discard thinking and trailing whitespace."""
        tail = ""
        if self._in_think:
            if self._assumed_hold and not self._saw_tag:
                # Neither tag ever arrived and the endpoint never used the
                # reasoning channel: the implicit-think hold was unnecessary, so
                # flush verbatim rather than swallow the report.
                self._in_think = False
                self._assumed_hold = False
                tail = self._emit(self._pending)
        else:
            tail = self._emit(self._pending)
        self._pending = ""
        self._pending_whitespace = ""
        return tail


class ReportSynthesisObserver(BaseObserver):
    """Lightweight single-LLM reporter (opt-in via ``agent.reporter: true``).

    On a natural loop end this runs ONE tool-free, streaming LLM call over the
    whole conversation to synthesise a structured, cited report — the harness
    analogue of a *standard-mode* reporter: append a summarize prompt to the
    agent message history and stream one summary call. The LLM call itself is
    streamed (for the ``reasoning`` channel + the incremental think-tag filter),
    but the report BODY is buffered until the call finishes, cleaned (think-tag
    strip + :func:`~workflows.stateful_react_agent._runtime.unwrap_fenced_images`,
    which needs the whole document to match an image against its citation), and
    only then re-chunked onto serve stdout as ``response.swarm.llm_delta`` frames
    (``agent_id="reporter"``, ``channel="output_text"``) via
    :meth:`~workflows._shared.sdk_shim.ReporterDeltaEmitter.stream_output`.
    This trades live token-by-token streaming for a guarantee that ``output_text``
    and ``final.answer`` are byte-identical (a fenced-then-unwrapped image can
    never render differently on the wire than in the persisted report). This
    observer REPLACES :class:`ReporterStreamObserver` (which merely re-emits the
    raw last turn) — wire one XOR the other, never both.

    Ordering: ``critical`` and placed after
    :class:`FinalAnswerSalvageObserver` (so bounded exits already have a real
    clean-context baseline) and before the serve chain's protocol stream
    observer, so the reporter stream lands before the terminal ``final``.

    Fail-open: any error keeps the pre-reporter salvage/last-turn answer, or
    installs an explicit deterministic best-effort baseline when none exists.
    The reporter is never a hard dependency. Cancellation/pause emits nothing
    (D5b: no terminal on a stopped run).
    """

    critical: bool = True

    def __init__(
        self,
        *,
        llm: LLMClient,
        timeout: float | None,
        task_description: str,
        emitter: object = None,
        usage_aggregator: object = None,
        language: str = "English",
        thinking_format: str = "tag",
        inline_thinking: bool = False,
        extra_observers: object = None,
        context_max_tokens: int = 220_000,
        phase_timeout: float | None = None,
        phase_deadline_monotonic: float | None = None,
    ) -> None:
        self._llm = llm
        self._timeout = timeout
        self._task_description = task_description
        self._language = language or "English"
        self._thinking_format = thinking_format
        # Tag-mode thinking that the endpoint may NOT split onto its own channel
        # (``enable_thinking`` + ``thinking_format: tag``): the ``<think>`` opener
        # can be missing while ``</think>`` is always emitted, so the stream
        # filter must hold the leading body back until the close proves where
        # reasoning ended. See :class:`_ReportStreamFilter`.
        self._inline_thinking = bool(inline_thinking)
        self._emitter = emitter
        self._usage_aggregator = usage_aggregator
        # A worker-trace observer lives among the serve chain's extra
        # observers; used to append an ``agent_type="reporter"`` timing row
        # so the trace's llm_call_timings stays consistent with
        # message_history + usage.
        self._extra_observers = extra_observers
        self._context_max_tokens = max(1_024, int(context_max_tokens))
        self._phase_timeout = phase_timeout
        # Absolute instant the whole task must finish by, when an external
        # ceiling is known. ``phase_timeout`` is the *planned* budget; on a
        # short wall the research loop's reserve does not survive intact, so
        # the phase clamps itself to whatever is really left at start.
        self._phase_deadline_monotonic = phase_deadline_monotonic

    def _effective_phase_timeout(self) -> float | None:
        """Planned phase budget, clamped to the time the hard wall still allows."""
        if self._phase_deadline_monotonic is None:
            return self._phase_timeout
        if self._phase_timeout is None:
            return max(self._phase_deadline_monotonic - time.monotonic(), 1.0)
        return remaining_phase_budget_s(
            self._phase_timeout, self._phase_deadline_monotonic,
        )

    async def on_loop_end(self, result: AgentLoopResult) -> None:
        # D5b: cancel/pause leaves no terminal — synthesise nothing.
        if result.stopped_by == "paused":
            return

        stream = ReporterDeltaEmitter(self._emitter)

        # Baseline answer to keep if the reporter fails / returns empty.
        metadata_answer = result.metadata.get("final_answer")
        baseline = _strip_leaked_tool_calls(str(
            metadata_answer or result.final_content or "",
        ))
        if baseline and not result.metadata.get("final_answer_source"):
            source = (
                "existing_partial"
                if not metadata_answer
                and result.stopped_by in _forced_final_stop_reasons(True)
                else "agent"
            )
            result.metadata["final_answer"] = baseline
            result.metadata["final_answer_source"] = source
        if not baseline:
            baseline = _minimal_best_effort_answer(
                self._task_description,
                result.stopped_by,
                language=self._language,
            )
            result.metadata["final_answer"] = baseline
            result.metadata["final_answer_source"] = "deterministic_fallback"
            result.final_content = baseline

        report_prompt = get_report_prompt(self._task_description, self._language)
        history = list(result.messages)
        if has_malformed_tool_protocol(history):
            # A malformed/orphan tool call can make every reporter fallback leg
            # fail with the same provider-side 400. Recover only in that case;
            # healthy runs preserve their full structured conversation.
            messages = _build_recovery_messages(
                history,
                task_description=self._task_description,
                final_prompt=report_prompt,
                context_max_tokens=self._context_max_tokens,
            )
        else:
            messages = history
            if (
                messages
                and isinstance(messages[-1], dict)
                and messages[-1].get("role") == "user"
            ):
                messages.pop()
            messages.append(user_msg(report_prompt))

        stream.start()
        cancelled = False
        # Stream through an incremental filter: reasoning models may inline
        # ``<think>…</think>`` in ``delta.content`` (opener sometimes absent),
        # with tags split across chunks, and may leak ``<tool_call>`` markup.
        # The filter's per-chunk output is accumulated into ``visible_text``
        # (below) but deliberately NOT sent to ``output_text`` live: the report
        # body is only put on the wire once, fully cleaned, after the call
        # finishes (see the class docstring) — that is what keeps the delta
        # stream and ``final.answer`` byte-identical. Native
        # ``reasoning_content`` still streams live on its own channel; it never
        # reaches the persisted report so it has no consistency requirement.
        think_filter = _ReportStreamFilter(assume_think_prefix=self._inline_thinking)
        raw_parts: list[str] = []
        # Capture terminal stream metadata (usage on the late include_usage
        # chunk, model/provider stamps) — last non-empty wins, matching the
        # loop's own streaming extraction (llm_client.py).
        rep_usage: dict = {}
        rep_model = rep_provider = ""
        started_ts = _dt.datetime.now(_dt.UTC)
        status = "success"
        error_message = ""
        try:
            async with asyncio.timeout(self._effective_phase_timeout()):
                async for delta in self._llm.stream(
                    messages, timeout=self._timeout,
                ):
                    chunk = getattr(delta, "content", None)
                    if chunk:
                        raw_parts.append(chunk)
                        think_filter.feed(chunk)
                    rc = getattr(delta, "reasoning_content", "")
                    if rc:
                        # A native reasoning delta proves thinking is NOT
                        # inlined in ``content`` → release any implicit-think
                        # hold (the report text it frees up is picked up from
                        # ``visible_text`` below; nothing goes out live here).
                        think_filter.note_reasoning_channel()
                        stream.reasoning(rc)
                    if getattr(delta, "usage", None):
                        rep_usage = delta.usage
                    if getattr(delta, "model", ""):
                        rep_model = delta.model
                    if getattr(delta, "provider", ""):
                        rep_provider = delta.provider
        except asyncio.CancelledError:
            cancelled = True
            raise
        except Exception as exc:
            status = "error"
            error_message = f"{type(exc).__name__}: {exc}"
            logger.warning(
                "ReportSynthesisObserver: synthesis failed (%s) — "
                "falling back to baseline answer", error_message,
            )

        if cancelled:
            return

        think_filter.finish()

        # The LLM call happened whenever ANY terminal metadata arrived (usage /
        # model / a content chunk), even if it then errored or returned empty —
        # record it on the aggregator so billing never under-counts a real call.
        raw_response = "".join(raw_parts)
        # Persisted answer = exactly what the filter let through, plus a final
        # ``</think>``-tail drop + strip as belt-and-braces (``final_answer`` /
        # the trace must never carry leaked reasoning) and the citation-gated
        # image unwrap. This is also what gets chunked onto the wire below, so
        # the delta stream and ``final_answer`` are the same text by construction.
        report = unwrap_fenced_images(
            _strip_leaked_tool_calls(_drop_think_prefix(think_filter.visible_text)),
        )
        call_happened = bool(rep_usage or rep_model or raw_parts)
        if call_happened:
            record_reporter_usage(
                self._usage_aggregator,
                usage=rep_usage, provider=rep_provider, model=rep_model,
            )

        if status == "success" and report:
            # Append a compact report-prompt + response pair to the user-visible
            # history. The exact protocol-clean reporter request is captured in
            # the reporter timing row below.
            if (
                result.messages
                and isinstance(result.messages[-1], dict)
                and result.messages[-1].get("role") == "user"
            ):
                result.messages.pop()
            result.messages.append(user_msg(report_prompt))
            result.messages.append(
                assistant_msg_with_reasoning(
                    report, "", thinking_format=self._thinking_format,
                ),
            )
            result.metadata["final_answer"] = report
            result.metadata["final_answer_source"] = "reporter_llm"
            result.metadata["final_answer_rescued"] = False
            result.metadata["final_answer_rescue_mode"] = ""
            result.final_content = report
            # Put the fully-cleaned report on the wire in one shot — deltas are
            # literal slices of ``report``, so they and ``final.answer`` agree
            # by construction (see the class docstring).
            stream.stream_output(report)
        else:
            # Failed OR empty synthesis: nothing was streamed live (the body is
            # always buffered now), so fall back to the baseline answer as the
            # thing put on the wire.
            if status == "success":
                status = "error"
                error_message = "reporter returned empty content"
            logger.warning(
                "ReportSynthesisObserver: %s — keeping baseline answer",
                error_message,
            )
            if baseline:
                stream.stream_output(baseline)

        # Worker-trace reporter timing row (agent_type="reporter") — keeps
        # llm_call_timings consistent with the appended message + usage. Runs
        # BEFORE the worker-trace observer's ``on_loop_end`` (this observer
        # precedes the serve chain), so the row is present when the trace
        # serialises.
        self._append_reporter_timing(
            started_ts=started_ts, status=status, error=error_message,
            model=rep_model, provider=rep_provider, usage=rep_usage,
            input_messages=messages, raw_response=raw_response,
        )

        stream.finish(
            final_content=str(
                result.metadata.get("final_answer") or result.final_content or "",
            ),
            status=status,
            stop_reason="final_answer" if status == "success" else "error",
            error_message=error_message,
        )

    def _append_reporter_timing(
        self,
        *,
        started_ts: _dt.datetime,
        status: str,
        error: str,
        model: str,
        provider: str,
        usage: dict[str, Any],
        input_messages: list[Message],
        raw_response: Any,
    ) -> None:
        """Best-effort: append an ``agent_type="reporter"`` row to the worker trace."""
        obs = self._extra_observers
        if not obs:
            return
        try:
            from workflows._shared.sdk_shim import find_trace_observer
            trace_obs = find_trace_observer(obs)
        except Exception:
            trace_obs = None
        if trace_obs is None:
            return
        end_ts = _dt.datetime.now(_dt.UTC)
        try:
            trace_obs.append_reporter_llm_call_timing(
                purpose="Reporter | Synthesize Report",
                model=model,
                provider=provider,
                status="success" if status == "success" else "failed",
                error=error,
                start_time=started_ts.isoformat(), end_time=end_ts.isoformat(),
                duration_ms=int((end_ts - started_ts).total_seconds() * 1000),
                ttft_ms=None, usage=dict(usage) if usage else None,
                input_messages=[
                    dict(message)
                    for message in input_messages
                    if isinstance(message, dict)
                ],
                raw_response=raw_response,
            )
        except Exception as exc:
            logger.debug("ReportSynthesisObserver: timing append failed: %s", exc)
