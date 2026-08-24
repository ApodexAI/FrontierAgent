"""The TUI render sink + approver.

:class:`TuiSink` implements the same method surface the ``TerminalObserver`` and
``TerminalSession`` already call on the line-mode ``Renderer`` (``note``,
``tool_call``, ``diff_preview``, ``tool_result``, ``todos``, ``final``,
``content_delta`` …). Instead of writing stdout it posts :class:`Render`
messages, so the engine and observers need no changes — only ``session.r`` is
swapped for this object.

Rich formatting (glyphs, arg summaries) is reused from the line-mode
``Renderer`` so the two surfaces look the same.
"""

from __future__ import annotations

import logging
from collections import deque
from collections.abc import Callable
from copy import deepcopy
from typing import Any

from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text

from apodex.observers import Decision
from apodex.render import (
    _MAX_CHANGE_ROWS,
    _MAX_RESULT_CHARS,
    Renderer,
    diff_to_text,
    preserve_line_breaks,
    tool_label,
)
from apodex.tui.logo import render_logo
from apodex.tui.messages import Render
from apodex.tui.screens import ApprovalScreen
from apodex.tui.state import PresentationPhase
from apodex.tui.themes import GLYPHS, rich_style
from apodex.tui.widgets import ActivityState

logger = logging.getLogger(__name__)

# Terminal phases a *new* task may start from directly. ``TaskRunnerMixin``
# re-enters ``run_task`` inside the same session worker to run queued steering
# input, so the next task can begin before the finished one has settled back to
# IDLE. ``ERROR`` is absent on purpose: it resumes the same task instead
# (``resume_after_error``).
_RESTARTABLE_PHASES = frozenset({
    PresentationPhase.DONE,
    PresentationPhase.INCOMPLETE,
})


class TuiSink:
    """Drop-in for ``Renderer`` that drives Textual widgets."""

    def __init__(self, app: Any) -> None:
        self.app = app
        self._verbose = True
        # Whether assistant content streamed this task — if so, ``final`` shows a
        # compact footer rather than re-rendering the whole (already-shown) answer.
        self._content_shown = False
        self._stream_chunks: list[tuple[str, str]] = []
        self._stream_render_pending = False
        self._fallback_activity_ids: deque[str] = deque()
        self._activity_sequence = 0
        self._task_board_calls: dict[str, tuple[str, dict]] = {}
        self._process_status = "complete"
        # ``history`` can reconstruct coordinator calls, but Agent Team worker
        # rows/events only exist in collect_reports snapshots.  Keep a small,
        # JSON-safe semantic mirror for session checkpoints; widget instances
        # themselves are deliberately never serialized.
        self._saved_subagents: dict[str, dict[str, object]] = {}

    def snapshot_state(self) -> dict[str, object]:
        """Return the durable part of the TUI presentation state."""
        return {
            "version": 1,
            "subagents": deepcopy(list(self._saved_subagents.values())),
        }

    def restore_state(self, state: object) -> None:
        """Load a checkpoint mirror before replaying it into fresh widgets."""
        self._saved_subagents.clear()
        if not isinstance(state, dict):
            return
        snapshots = state.get("subagents")
        if not isinstance(snapshots, list):
            return
        for index, raw in enumerate(snapshots):
            if not isinstance(raw, dict):
                continue
            item = deepcopy(raw)
            if str(item.get("status") or "") in {"queued", "submitted", "running"}:
                # A restored process has no live worker behind it.  Preserve
                # where it stopped without reviving a spinner that can never
                # receive another heartbeat.
                item["status"] = "aborted"
                item["active"] = False
            session_id = str(item.get("session_id") or item.get("name") or index)
            self._saved_subagents[session_id] = item

    def clear_saved_activity(self) -> None:
        """Forget presentation-only activity for /clear and new sessions."""
        self._saved_subagents.clear()

    def saved_subagents(self) -> list[dict[str, object]]:
        """Return worker snapshots in their original stable display order."""
        return deepcopy(list(self._saved_subagents.values()))

    def _style(self, role: str, *, bold: bool = False) -> str:
        return rich_style(self.app._ui_theme, role, bold=bold)

    @property
    def _theme(self) -> str:
        return self.app._ui_theme

    # ── plumbing ──────────────────────────────────────────────────────────
    def _post(self, fn: Callable[[Any], Any]) -> bool:
        try:
            self.app.post_message(Render(fn))
            return True
        except Exception:
            logger.exception("failed to post a TUI render message")
            return False

    def _block(
        self, renderable: object, *, classes: str = "", process: bool = False,
        step: bool = True,
    ) -> None:
        self._post(
            lambda app: app.transcript.add(
                renderable, classes=classes, process=process, step=step,
            )
        )

    def _queue_stream(self, kind: str, text: str) -> None:
        """Coalesce token callbacks into ordered UI-message batches."""
        self._stream_chunks.append((kind, text))
        if self._stream_render_pending:
            return
        self._stream_render_pending = True
        if not self._post(self._flush_stream):
            self._stream_render_pending = False

    async def _flush_stream(self, app: Any) -> None:
        chunks, self._stream_chunks = self._stream_chunks, []
        self._stream_render_pending = False
        if not chunks:
            return
        # Preserve thinking/content boundaries while joining adjacent tokens.
        batches: list[tuple[str, str]] = []
        for kind, text in chunks:
            if batches and batches[-1][0] == kind:
                previous_kind, previous_text = batches[-1]
                batches[-1] = (previous_kind, previous_text + text)
            else:
                batches.append((kind, text))
        for kind, text in batches:
            await app.transcript.stream(kind, text)

    def begin_task(self) -> None:
        """Reset per-task presentation state before a new agent worker starts."""
        self._content_shown = False
        self._fallback_activity_ids.clear()
        self._process_status = "complete"
        self.app.presentation.begin_task()
        async def _f(app: Any) -> None:
            app.show_task_workspace()
            await app.transcript.begin_process()

        self._post(_f)

    def finish_task(self) -> None:
        """Freeze the final task state before the app returns to idle."""
        self.sync_queued_steers()
        async def _f(app: Any) -> None:
            app.finish_active_activities(ActivityState.SKIPPED)
            await app.transcript.end_process(self._process_status)

        self._post(_f)
        self.app.presentation.finish()

    def interrupted(self) -> None:
        self._process_status = "interrupted"
        self._post(lambda app: app.finish_active_activities(ActivityState.INTERRUPTED))
        self.app.presentation.interrupt()

    def reset_presentation(self) -> None:
        self.app.presentation.reset()

    def tick(self) -> None:
        self.app.presentation.settle()
        self.sync_queued_steers()

    def sync_queued_steers(self) -> None:
        inbox = getattr(self.app.session, "_inbox", None)
        queue = getattr(inbox, "queue", ()) if inbox is not None else ()
        self.app.presentation.set_queued_steers(len(queue))

    async def finish_stream(self) -> None:
        """Drain queued deltas and finalize the live block before returning control."""
        if self._stream_chunks:
            await self._flush_stream(self.app)
        await self.app.transcript.end_stream()

    # ── state hooks (sync; read by the status bar) ────────────────────────
    def set_usage(self, usage: Any, window: int) -> None:
        self.app.set_usage(usage, window)

    def set_verbose(self, on: bool) -> None:
        self._verbose = bool(on)

    def working_on(self) -> None:
        if self.app.presentation.phase in _RESTARTABLE_PHASES:
            # A queued steer may start a follow-up from inside the same session
            # worker, after the prior result has already been rendered. Every
            # phase here is terminal, so ``transition`` below would be a no-op
            # and the status bar would carry the finished run's label (and its
            # frozen timer) through the whole follow-up task.
            self.app.presentation.begin_task()
        elif self.app.presentation.phase == PresentationPhase.ERROR:
            self.app.presentation.resume_after_error()
        else:
            self.app.presentation.transition(PresentationPhase.THINKING)
        self.sync_queued_steers()

    def working_off(self) -> None:
        self.sync_queued_steers()

    def subagent_status(
        self, snapshots: list[dict[str, object]], *, done: bool = False,
        timeout_s: int = 0,
    ) -> None:
        """Update the stable team overview and the current wait card."""
        # A session may run several Agent Team tasks.  Each heartbeat only
        # describes the current task, while the Activity pane intentionally
        # retains earlier workers, so merge by stable session id rather than
        # replacing the checkpoint with the newest task alone.
        for index, raw in enumerate(snapshots):
            if not isinstance(raw, dict):
                continue
            item = deepcopy(raw)
            session_id = str(item.get("session_id") or item.get("name") or index)
            self._saved_subagents[session_id] = item

        async def _f(app: Any) -> None:
            app.activity.update_subagents(snapshots, done=done)
            await app.transcript.show_subagent_status(
                snapshots, done=done, timeout_s=timeout_s,
            )

        self._post(_f)

    def banner(self, **_: Any) -> None:
        # The status bar carries model/mode/cwd live; no separate banner block.
        pass

    def logo(self) -> None:
        """Mount the pixel-art logo at the top of an empty transcript.

        Sized from the transcript's own content width rather than the terminal's:
        the sidebar takes 45 columns when it is open, and a logo laid out against
        the full width would wrap into rubble in the pane it actually lands in
        — ``text-wrap: nowrap`` does not save it, because Textual measures the
        wrapped height before the clip applies and reserves all of it.

        That width is zero until layout has run, and on_mount is before layout,
        so the fit here is only a first guess; ``TranscriptView.on_resize`` does
        the real one once the pane has a size. Not a step, so it never counts
        toward a run's step tally.
        """
        async def _f(app: Any) -> None:
            width = app.transcript.content_size.width or app.size.width
            await app.transcript.add(
                render_logo(app._ui_theme, max(0, width - 2)),
                classes="startup-logo", step=False,
            )

        self._post(_f)

    def rule(self) -> None:
        pass

    # ── notes / plain text ────────────────────────────────────────────────
    def echo_user(self, text: str) -> None:
        self._block(Text(f"{GLYPHS['user']} {text}",
                         style=self._style("user", bold=True)),
                    classes="user-message turn-start")

    def note(self, msg: str) -> None:
        self._block(Text(msg, style=self._style("muted")))

    def queued(self, text: str) -> None:
        self.sync_queued_steers()
        self._block(Text(
            f"{GLYPHS['queued']} queued — will steer at the next step: {text}",
            style=self._style("accent"),
        ))

    def error(self, msg: str) -> None:
        self._process_status = "failed"
        self._post(lambda app: app.finish_active_activities(ActivityState.FAILED))
        self.app.presentation.finish(PresentationPhase.ERROR)
        # Classed so ``/filter errors`` surfaces run-level failures too, not
        # only the tool results that happen to carry a non-zero exit.
        self._block(Text(f"error: {msg}", style=self._style("err")),
                    classes="transcript-error")

    def llm_failure(self, msg: str, *, configuration_error: bool = False) -> None:
        """Show an unmistakable run-level LLM failure, not a final report."""
        self._process_status = "failed"
        self._content_shown = False
        self._post(lambda app: app.finish_active_activities(ActivityState.FAILED))
        self.app.presentation.finish(PresentationPhase.ERROR)
        title = "LLM configuration error" if configuration_error else "LLM call failed"
        status = self._process_status

        async def _f(app: Any) -> None:
            await app.transcript.end_stream()
            await app.transcript.end_process(status)
            await app.transcript.add(Panel(
                Text(msg, style=self._style("err")),
                title=f"{GLYPHS['fail']} {title}",
                border_style=self._style("err"),
                title_align="left",
            ), classes="transcript-error")

        self._post(_f)

    def incomplete(
        self,
        text: str,
        *,
        turns: int = 0,
        tool_calls: int = 0,
        stopped_by: str = "",
    ) -> None:
        """Render partial work without saving or labelling it as a report."""
        meta = f"turns={turns} · tools={tool_calls}"
        if stopped_by:
            meta += f" · {stopped_by}"
        self._process_status = "incomplete"
        shown = self._content_shown
        self._content_shown = False
        self._post(lambda app: app.finish_active_activities(ActivityState.SKIPPED))
        self.app.presentation.finish(PresentationPhase.INCOMPLETE)
        status = self._process_status

        async def _f(app: Any) -> None:
            await app.transcript.end_stream()
            await app.transcript.end_process(status)
            # No ``save_final_report``, and no ``show_completed_deliverables``:
            # partial work is not promoted to a deliverable, and revealing the
            # outputs tab would present the run as delivered. The other two
            # calls still belong here. ``final_text`` only ever resets on a
            # whole-transcript replacement, so leaving it alone lets Ctrl-Y copy
            # the PREVIOUS task's report and announce it as this one's; and a run
            # can stop short long after its publisher wrote real files to
            # /outputs, which the pane must list even while it stays in back.
            app.transcript.set_final_text(text)
            app.refresh_deliverables()
            if shown:
                await app.transcript.add(
                    Text(
                        f"{GLYPHS['stopped']} Run incomplete · {meta} · "
                        "partial output was not saved as a final report",
                        style=self._style("warn"),
                    ),
                    classes="report-footer transcript-incomplete",
                )
                return
            await app.transcript.add(Panel(
                Markdown(preserve_line_breaks(text) or "_(no partial output)_"),
                title=f"{GLYPHS['stopped']} Incomplete output",
                subtitle=meta,
                subtitle_align="right",
                border_style=self._style("warn"),
            ), classes="transcript-incomplete")

        self._post(_f)

    # ── streaming ─────────────────────────────────────────────────────────
    def content_delta(self, s: str) -> None:
        if not s:
            return
        self.app.presentation.transition(PresentationPhase.RESPONDING)
        self._content_shown = True
        self._queue_stream("content", s)

    def thinking_delta(self, s: str) -> None:
        if not s:
            return
        self.app.presentation.transition(PresentationPhase.THINKING)
        if not self._verbose:
            return
        self._queue_stream("thinking", s)

    def end_turn_text(self) -> None:
        """Close the narration block at the end of one assistant message.

        Workflows whose tools present themselves in the timeline mount nothing
        into the transcript between messages, so without an explicit boundary a
        whole task's prose grew into a single never-finalized live block: no
        Markdown, and no separation beyond the blank lines the model emitted.
        """
        self._post(lambda app: app.transcript.end_stream())

    def turn_text_fallback(self, ai_text: str, thinking: str) -> None:
        if thinking.strip() and self._verbose:
            self.app.presentation.transition(PresentationPhase.THINKING)
            self._post(lambda app: app.transcript.add_thinking(thinking.strip()))
        if ai_text.strip():
            self.app.presentation.transition(PresentationPhase.RESPONDING)
            self._content_shown = True
            self._block(
                Markdown(preserve_line_breaks(ai_text.strip())),
                classes="assistant-content",
            )

    # ── tool calls / diffs / results ──────────────────────────────────────
    def _start_activity_id(self, call_id: str) -> str:
        if call_id:
            return call_id
        self._activity_sequence += 1
        generated = f"tui-call-{self._activity_sequence}"
        self._fallback_activity_ids.append(generated)
        return generated

    def _result_activity_id(self, call_id: str) -> str:
        if call_id:
            return call_id
        if self._fallback_activity_ids:
            return self._fallback_activity_ids.popleft()
        self._activity_sequence += 1
        return f"tui-result-{self._activity_sequence}"

    def activity_call(self, name: str, args: dict, *, call_id: str = "") -> None:
        activity_id = self._start_activity_id(call_id)
        detail = Renderer._summarize_args(name, args)
        self._post(lambda app: app.start_activity(activity_id, name, detail))

    def tool_call(
        self, name: str, args: dict, risk_reason: str = "", danger: bool = False,
        *, call_id: str = "",
    ) -> None:
        label = tool_label(name)
        detail = Renderer._summarize_args(name, args)
        summary = f"{name} {detail}".strip()
        activity_id = self._start_activity_id(call_id)
        if name in {"add_task", "update_task"}:
            self._task_board_calls[activity_id] = (name, args)
        self.app.presentation.transition(PresentationPhase.RUNNING_TOOL, tool=summary)
        line = Text()
        line.append(label, style=self._style("tool", bold=True))
        line.append(f" {detail}")
        if risk_reason:
            mark = f"{GLYPHS['danger']} " if danger else ""
            line.append(
                f"  ({mark}{risk_reason})",
                style=self._style("err", bold=True) if danger else self._style("muted"),
            )
        if name == "bash":
            cmd = str(args.get("command", "")).strip()
            if cmd and cmd != detail:
                shown = cmd if len(cmd) <= 500 else cmd[:500] + " …"
                line.append(f"\n    $ {shown}", style=self._style("muted"))

        def _f(app: Any) -> Any:
            app.start_activity(activity_id, name, detail)
            return app.transcript.add(line, classes="tool-call", process=True)

        self._post(_f)

    def diff_preview(self, diff_text: str, *, stats: tuple[int, int] | None = None) -> None:
        title = "proposed change"
        if stats is not None:
            title += f"  +{stats[0]} -{stats[1]}"
        self._block(
            Panel(diff_to_text(diff_text, theme=self._theme), title=title,
                  border_style=self._style("border"), title_align="left"),
            process=True,
            # Detail attached to the edit call above it.
            step=False,
        )

    def activity_result(
        self, name: str, *, call_id: str = "", is_error: bool,
        ms: int = 0, outcome: str = "",
    ) -> None:
        activity_id = self._result_activity_id(call_id)
        state = ActivityState.SKIPPED if outcome == "skipped" else None
        self._post(lambda app: app.finish_activity(
            activity_id, name, is_error=is_error, ms=ms, state=state,
        ))

    def tool_result(
        self, name: str, result: str, *, is_error: bool, ms: int = 0,
        call_id: str = "",
    ) -> None:
        activity_id = self._result_activity_id(call_id)
        task_board_op = self._task_board_calls.pop(activity_id, None)
        body = result if isinstance(result, str) else str(result)
        truncated = ""
        if len(body) > _MAX_RESULT_CHARS:
            truncated = (f"\n… [+{len(body) - _MAX_RESULT_CHARS} chars hidden — "
                         "full output in the trace (/log)]")
            body = body[:_MAX_RESULT_CHARS]
        summary = self._result_summary(body)
        # The words match the call row above it; the ✓/✗ replaces that row's
        # family marker rather than sitting beside it.
        shown = tool_label(name, marker=False)
        header = (f"{GLYPHS['fail']} {shown} failed" if is_error else f"{GLYPHS['ok']} {shown}")
        if summary:
            header += f" · {summary}"
        if ms:
            header += f" · {ms} ms"
        def _f(app: Any) -> Any:
            app.finish_activity(activity_id, name, is_error=is_error, ms=ms)
            if task_board_op and not is_error:
                operation, args = task_board_op
                app.apply_task_board_operation(operation, args)
                # The sidebar Plan is the authoritative current board. The
                # tool-call row already preserves the mutation in the process
                # history, so repeating a success note in the transcript adds
                # visual noise without adding information.
                return None
            result_text = Text(
                body + truncated,
                # State belongs in the compact heading. Keeping a long error
                # body muted avoids turning an expanded traceback into a wall
                # of red while preserving the failed title as a strong cue.
                style=self._style("muted"),
            )
            result_class = "tool-result tool-result-error" if is_error else "tool-result"
            return app.transcript.add_collapsible(
                header,
                result_text,
                classes=result_class,
                collapsed=not is_error,
                process=True,
                # The outcome of the call above it, not a step of its own.
                step=False,
            )

        self._post(_f)
        self.app.presentation.transition(PresentationPhase.THINKING)
        self.sync_queued_steers()

    @staticmethod
    def _result_summary(result: str) -> str:
        """Keep a useful outcome visible while the full output is folded."""
        lines = [" ".join(line.split()) for line in result.splitlines() if line.strip()]
        if not lines:
            return "no output"
        first = lines[0]
        if len(first) > 64:
            first = first[:61] + "…"
        if len(lines) > 1:
            first += f" · {len(lines)} lines"
        return first

    # ── todos / changes / plan / final ────────────────────────────────────
    def todos(self, items: list) -> None:
        self._post(lambda app: app.show_todos(items))

    def changes(self, stats: list) -> None:
        if not stats:
            return
        shown = stats[:_MAX_CHANGE_ROWS]
        body = Text()
        for path, add, dele in shown:
            body.append(f"{path}  ", style=self._style("text", bold=True))
            body.append(f"+{add}", style=self._style("ok"))
            body.append(" ")
            body.append(f"-{dele}\n", style=self._style("err"))
        if len(stats) > len(shown):
            body.append(
                f"… and {len(stats) - len(shown)} more\n", style=self._style("muted")
            )
        title = f"Changed files ({len(stats)})  ·  /revert to undo"
        self._block(Panel(body, title=title, border_style=self._style("todo"), title_align="left"))

    def plan_review(self, plan: str) -> None:
        text = plan.strip() or "_(no plan text provided)_"
        self._block(Panel(
            Markdown(preserve_line_breaks(text)),
            title=f"{GLYPHS['proposal']} Proposed plan  ·  approve to unlock edits",
            border_style=self._style("todo"), title_align="left",
        ))

    def final(self, text: str, *, turns: int = 0, tool_calls: int = 0, stopped_by: str = "") -> None:
        meta = f"turns={turns} · tools={tool_calls}" + (f" · {stopped_by}" if stopped_by else "")
        usage = self.app.usage
        if usage is not None and getattr(usage, "total", 0):
            meta += f" · {usage.summary()}"
        shown = self._content_shown
        self._content_shown = False
        self.app.presentation.finish()
        # Read now, not inside the coroutine: an answer arriving after a failed
        # or interrupted step must not relabel that history as a clean run.
        status = self._process_status
        if shown:
            async def _f(app: Any) -> None:
                await app.transcript.end_stream()
                await app.transcript.end_process(status)
                app.save_final_report(text)
                app.show_completed_workspace()
                # Unconditionally, so /copy works even if the prose was not
                # streamed into a block this method could promote.
                app.transcript.set_final_text(text)
                await app.transcript.promote_last_content()
                await app.transcript.add(
                    Text(f"{GLYPHS['ok']} Report complete · {meta}",
                         style=self._style("muted")),
                    classes="report-footer",
                )

            self._post(_f)
            return
        async def _f(app: Any) -> None:
            await app.transcript.end_process(status)
            app.save_final_report(text)
            app.show_completed_workspace()
            app.transcript.set_final_text(text)
            await app.transcript.add(Panel(
                Markdown(preserve_line_breaks(text) or "_(no answer)_"),
                title=f"{GLYPHS['ok']} Final report", subtitle=meta,
                subtitle_align="right", border_style=self._style("ok"),
            ), classes="final-report")

        self._post(_f)


class TuiApprover:
    """Modal-backed approval gate; same ``confirm`` contract as ``Approver``."""

    def __init__(self, app: Any, auto_approve: bool = False, auto_for_me: bool = False) -> None:
        self.app = app
        self.auto_approve = auto_approve
        self.auto_for_me = auto_for_me
        self.interactive = True
        self.inbox: Any = None  # set by the session each run; unused in the TUI

    async def confirm(
        self, name: str, target: str, reason: str, *, dangerous: str = "",
        preview: str = "", preview_kind: str = "",
    ) -> Decision:
        if self.auto_approve:
            return Decision(True)
        tool = self.app.presentation.current_tool or name
        self.app.presentation.transition(PresentationPhase.AWAITING_APPROVAL, tool=tool)
        try:
            outcome = await self.app.push_screen_wait(
                ApprovalScreen(
                    name, reason, dangerous, target=target,
                    preview=preview, preview_kind=preview_kind,
                )
            )
        finally:
            self.app.presentation.transition(PresentationPhase.RUNNING_TOOL, tool=tool)
        if outcome is None:  # screen dismissed without a value → fail safe
            return Decision(False)
        if outcome.all_session:  # 'allow all' flips auto-approve for the session
            self.auto_approve = True
        if getattr(outcome, "auto_for_me", False):  # 'auto for me' flips docker/trusted env mode
            self.auto_for_me = True
        return outcome.decision


__all__ = ["TuiApprover", "TuiSink"]
