"""Custom widgets for the TUI: the scrolling transcript, status bar, todo pane.

These hold no agent logic — they are pure view. The transcript embeds Rich
renderables (``Panel`` / ``Text`` / ``Markdown``) so the look matches line mode's
``Renderer`` without re-implementing the formatting.
"""

from __future__ import annotations

import json
import re
import time
from collections import deque
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, ClassVar

from rich.console import RenderableType
from rich.markdown import Markdown
from rich.text import Text
from textual.binding import Binding, BindingType
from textual.containers import Vertical, VerticalScroll
from textual.widget import Widget
from textual.widgets import Collapsible, OptionList, Static
from textual.widgets.option_list import Option

from apodex.render import diff_to_text, preserve_line_breaks, tool_label
from apodex.tui.logo import render_logo
from apodex.tui.state import PresentationPhase, TuiPresentationState, format_elapsed
from apodex.tui.themes import (
    GLYPHS,
    active_theme,
    agent_identity_role,
    agent_kind,
    agent_kind_glyph,
    rich_style,
)
from frontier_agent.core.messages import Message, text_of
from frontier_agent.core.runtime.loop.llm_client import extract_final_content
from plugins.tools._coerce import coerce_json_list, coerce_json_object

_MAX_TRANSCRIPT_BLOCKS = 300
_MAX_RENDERED_MARKDOWN_CHARS = 200_000
_LIVE_RENDER_MIN_CHARS = 128
_LIVE_RENDER_INTERVAL = 0.05
_MAX_ACTIVITY_RECORDS = 100
# Thinking stays visible while it is short, then gets out of the reading path.
# Both limits matter: prose reaches the character limit first, while lists and
# shell-like notes can be visually long without containing many characters.
_THINKING_COLLAPSE_CHARS = 900
_THINKING_COLLAPSE_LINES = 10
# Sourced from the shared vocabulary rather than spelled out: this was a colour
# emoji, which paints itself and so ignored every theme (see ``themes`` module
# docstring) *and* occupied two cells in a column of one-cell markers.
_THINKING_ICON = GLYPHS["thinking"]
# Used only before first layout, when ``content_size`` is still zero.
_ACTIVITY_FALLBACK_WIDTH = 36
# Braille-free quarter-circle spinner: the same four shapes the status bar's
# ``running`` glyph belongs to, so a spinning row reads as the same "busy" as
# everything else in the UI.
_SPINNER_FRAMES = ("◐", "◓", "◑", "◒")


def _renderable_text(renderable: object) -> str:
    """Flatten a Rich renderable to searchable text.

    Only the shapes the transcript actually mounts are handled: ``Text`` and
    ``Markdown`` carry their source directly, and ``Panel`` wraps one of those.
    Rendering through a console for the general case would cost a full layout
    pass per block on every keystroke of a search.
    """
    plain = getattr(renderable, "plain", None)
    if isinstance(plain, str):
        return plain
    markup = getattr(renderable, "markup", None)
    if isinstance(markup, str):
        return markup
    inner = getattr(renderable, "renderable", None)
    if inner is not None and inner is not renderable:
        return _renderable_text(inner)
    return str(renderable)


_BLANK_RUN = re.compile(r"\n[ \t]*(?:\n[ \t]*)+")


def _tighten(text: str) -> str:
    """Normalise the vertical rhythm of streamed prose.

    Model narration carries its own spacing — a trailing blank line per message,
    two or three between paragraphs — and the live block renders it verbatim.
    Concatenated across a task that never mounts anything between messages, that
    spacing became screens of empty transcript. One blank line separates
    paragraphs; the edges carry none, so a block starts and ends on prose.
    """
    return _BLANK_RUN.sub("\n\n", text).strip("\n")


def _bounded_markdown(text: str) -> Markdown:
    """Keep pathological model output from monopolising the UI renderer.

    The complete answer remains in the session checkpoint and trace.  This is
    only a front-end safety valve; ordinary answers never approach the limit.
    """
    text = preserve_line_breaks(text)
    if len(text) <= _MAX_RENDERED_MARKDOWN_CHARS:
        return Markdown(text)
    hidden = len(text) - _MAX_RENDERED_MARKDOWN_CHARS
    visible = text[:_MAX_RENDERED_MARKDOWN_CHARS]
    return Markdown(
        f"{visible}\n\n---\n\n"
        f"_… {hidden:,} additional characters hidden in the TUI; "
        "the complete answer is preserved in the session trace._"
    )


class TailScroll(VerticalScroll):
    """A scroll container that keeps its newest content in view.

    Textual's ``anchor()`` does the real work once a pane actually scrolls,
    including releasing itself when the user scrolls away. What it cannot handle
    is a pane holding *less* content than its own height: anchoring resolves
    before the layout that would report the new content height, and with a stale
    ``virtual_size`` it settles on a negative offset. Nothing clamps that back —
    with nothing scrollable, no later event recomputes it — so the content ends
    up floating at the bottom of an empty pane, which is what a fresh session's
    welcome note and a one-line plan both did.

    So the decision is deferred one refresh, by which point ``max_scroll_y`` is
    real, and made explicitly: pin to the top when everything fits, otherwise
    hand back to ``anchor()``. Deciding synchronously reads the stale size and
    gets it wrong in one direction or the other every time.
    """

    # ``anchor()`` cannot safely represent this intent while the contents are
    # shorter than the viewport (see ``_pin_tail``), so it is retained
    # separately until the pane actually becomes scrollable. Declared on the
    # class, not in ``on_mount``: the first layout pass fires ``watch_scroll_y``
    # and ``watch_virtual_size``, and ``follow_tail`` is public, so both can run
    # before the widget's ``Mount`` event has been processed.
    _wants_tail: bool = True
    #: ``(scroll_y, max_scroll_y)`` captured when an anchor release was seen,
    #: pending the refresh that tells us where the scroll actually landed.
    _release_from: tuple[int, int] | None = None

    def on_mount(self) -> None:
        self._wants_tail = True
        self._release_from = None
        self.follow_tail()

    def follow_tail(self) -> None:
        """Put the newest content in view; call after mounting content.

        Skipped while the user is reading further up, so this cannot yank them
        back down. That is a stored *intent* rather than a position check: by
        the time content has grown, ``max_scroll_y`` is already the new end, so
        "am I at the bottom?" reads false for a pane that was following.
        """
        if self._wants_tail:
            self.call_after_refresh(self._pin_tail)

    def _pin_tail(self) -> None:
        """Deferred by one refresh so ``max_scroll_y`` is the *new* size.

        Deciding synchronously is the trap: at mount, and immediately after
        content is added, the layout has not run yet, so the size still reads as
        it was. Textual's own anchoring hits this and parks short content at a
        negative offset that nothing clamps back — nothing is scrollable, so no
        later event re-runs the calculation.
        """
        if not self._wants_tail:
            return
        if self.max_scroll_y <= 0:
            # Anchoring a short container makes Textual's compositor derive a
            # negative ``content height - viewport height`` offset. Keep the
            # *intent* above, but only enable Textual's anchor once there is a
            # real scroll range.
            self.anchor(False)
            self.scroll_to(y=0, animate=False, immediate=True, release_anchor=False)
        else:
            # Hand back to Textual once there is something to scroll: it also
            # releases the anchor when the user scrolls away, which we want.
            self.anchor()

    def resume_tail(self) -> None:
        """Re-arm following; for callers that replace the contents wholesale."""
        self._wants_tail = True
        self._release_from = None
        self.follow_tail()

    def release_anchor(self) -> None:
        """Note a scroll that *might* be the user leaving the newest content.

        Textual funnels keyboard, scrollbar-drag and ``scroll_to`` movement
        through this hook — but it calls it *before* the scroll lands, and it
        calls it for movement that cannot move: page-down, or a click on the
        scrollbar track, while the view is already pinned at the end. Deciding
        here would end tail-following for the rest of the session, because a
        scroll that does not move fires no ``watch_scroll_y`` to put the intent
        back. So record where the scroll started and settle once it has been
        applied. Programmatic tail pinning uses ``release_anchor=False`` and
        never reaches this at all.
        """
        super().release_anchor()
        if not self._wants_tail:
            return
        self._release_from = (self.scroll_offset.y, self.max_scroll_y)
        self.call_after_refresh(self._settle_release)

    def _settle_release(self) -> None:
        """Stop following only if the scroll really ended up short of the end."""
        release, self._release_from = self._release_from, None
        if release is None or not self._wants_tail or self.max_scroll_y <= 0:
            return
        start_y, start_max = release
        if self.scroll_offset.y == start_y and start_y >= start_max:
            return  # nothing moved: the view is still sitting on the tail
        if self.scroll_offset.y >= self.max_scroll_y:
            return  # it landed on the end, or new content pulled it back there
        self._wants_tail = False

    def _check_anchor(self) -> None:
        """Keep Textual's own anchor subordinate to the tail intent.

        Textual re-engages the anchor whenever ``scroll_y`` lands on the end —
        including when a shrink *clamped* it there under a reader who had
        scrolled away. Only ``_wants_tail`` can tell those apart, so it gates
        the re-engagement; the compositor would otherwise pin the view to the
        bottom behind ``follow_tail``'s back.
        """
        if not self._wants_tail:
            return
        super()._check_anchor()

    def watch_scroll_y(self, old_value: float, new_value: float) -> None:
        """Treat keyboard/programmatic arrival at the bottom like a thumb drag.

        Only movement *towards* the end counts. Content shrinking under a reader
        who scrolled up — a ``Collapsible`` folding, ``_trim_blocks`` pruning —
        clamps ``scroll_y`` *down* onto the new end, and that is the layout
        moving the view, not a request to resume following.

        The intent is updated *before* delegating, because the base watcher is
        what calls ``_check_anchor``, and that now reads it.
        """
        if (
            self.max_scroll_y > 0
            and new_value > old_value
            and new_value >= self.max_scroll_y
        ):
            self._wants_tail = True
        super().watch_scroll_y(old_value, new_value)

    def watch_virtual_size(self, old_size: object, new_size: object) -> None:
        """Re-pin after content reflow, not only after mounting a new widget.

        Streaming updates grow one existing ``Static``; Markdown finalization
        and Collapsible state changes also resize existing widgets. None of
        those operations mount a block, so mount-only following leaves the
        scrollbar thumb above the real end and the active turn off-screen.
        """
        if old_size != new_size and self._wants_tail:
            self.follow_tail()


class TranscriptView(TailScroll):
    """A vertical, scrollable log of message blocks.

    Discrete blocks (tool calls, results, notes, the final answer) are mounted
    as ``Static`` widgets wrapping a Rich renderable. Streaming text (assistant
    content / thinking) accumulates into a single *live* block that is appended
    to per delta and, for assistant content, re-rendered as Markdown once the
    turn ends — mirroring line mode's "stream live, render once" behaviour.
    """

    def __init__(self, *, max_blocks: int = _MAX_TRANSCRIPT_BLOCKS) -> None:
        super().__init__(id="transcript")
        self.max_blocks = max(1, max_blocks)
        self._live: Static | None = None
        self._live_container: Widget | None = None
        self._live_kind: str | None = None  # "content" | "thinking" | None
        self._buf = ""
        self._last_live_render_at = 0.0
        self._last_live_render_size = 0
        self._thinking_auto_collapsed = False
        self._process: Collapsible | None = None
        self._process_body: Vertical | None = None
        self._process_started_at = 0.0
        self._process_steps = 0
        self._process_failed = False
        self._review_block: Widget | None = None
        self._logo_width = -1
        self._subagent_status: Static | None = None
        self.final_text = ""
        self.filter_mode = "all"
        self.filter_query = ""
        self.filter_matches = 0
        self._collapse_snapshot: list[tuple[Collapsible, bool]] = []
        self._pruned_blocks = 0
        self._prune_notice: Static | None = None

    def _style(self, role: str, *, bold: bool = False) -> str:
        return rich_style(active_theme(self), role, bold=bold)

    async def add(
        self, renderable: RenderableType, *, classes: str = "", process: bool = False,
        step: bool = True,
    ) -> None:
        """Finalize any live stream, then append a discrete block."""
        await self.end_stream()
        await self._mount_block(
            renderable, classes=classes, process=process, step=step,
        )

    async def begin_process(self) -> None:
        """Start one progressively-disclosed execution group for a task."""
        await self.end_stream()
        # A filter describes the transcript the user was reviewing. Leaving it
        # armed would mount the new run's blocks visible among hidden ones — a
        # half-filtered view that matches neither the header nor any filter.
        self.clear_filter()
        if self._process is not None:
            await self.end_process("interrupted")
        self._process_body = Vertical(classes="process-body")
        self._process = Collapsible(
            self._process_body,
            title=f"{GLYPHS['pending']} Working · starting",
            collapsed=False,
            collapsed_symbol="›",
            expanded_symbol="⌄",
            classes="block process-group",
        )
        self._process_started_at = time.monotonic()
        self._process_steps = 0
        self._process_failed = False
        await self._mount_widget(self._process)

    async def end_process(self, status: str = "complete") -> None:
        """Freeze and usually collapse the current execution group."""
        await self.end_stream()
        if self._process is None:
            return
        elapsed = max(0, int(time.monotonic() - self._process_started_at))
        if self._process_failed or status in {"failed", "error"}:
            glyph, label, clean = GLYPHS["fail"], "issues found", False
        elif status == "incomplete":
            glyph, label, clean = GLYPHS["stopped"], "incomplete", False
        elif status == "interrupted":
            glyph, label, clean = GLYPHS["cancelled"], "interrupted", False
        else:
            glyph, label, clean = GLYPHS["ok"], status, True
        self._process.title = self._process_title(
            glyph, label, elapsed=elapsed,
        )
        # Only a clean run leaves the reading path. A failed or interrupted one
        # is exactly the history the user needs, so it stays open.
        self._process.collapsed = clean
        self._process = None
        self._process_body = None

    def _process_title(self, glyph: str, label: str, *, elapsed: int) -> str:
        steps = self._process_steps
        return (
            f"{glyph} Working · {steps} step{'' if steps == 1 else 's'} · "
            f"{label} · {format_elapsed(elapsed)}"
        )

    def _touch_process(self, *, failed: bool = False, step: bool = True) -> None:
        """Record one mounted row against the open group.

        ``step`` counts what the user would call a step — a thought, a tool
        invocation. A result or a diff is the *outcome* of the call above it,
        so counting those too reported three steps for a single tool call.
        """
        if self._process is None:
            return
        self._process_steps += int(step)
        self._process_failed = self._process_failed or failed
        self.refresh_process()

    def refresh_process(self) -> None:
        """Refresh elapsed time even during one long-running tool call."""
        if self._process is None:
            return
        elapsed = max(0, int(time.monotonic() - self._process_started_at))
        self._process.title = self._process_title(
            GLYPHS["running"], "running", elapsed=elapsed,
        )

    async def show_subagent_status(
        self, snapshots: list[dict[str, object]], *, done: bool = False,
        timeout_s: int = 0,
    ) -> None:
        """Show one live, in-place status card during collect_reports.

        The card is the *live* view — one line per worker, replaced in place
        rather than appended, so a ten-minute fan-in does not bury the
        transcript under a thousand progress blocks. It disappears on ``done``;
        the sidebar keeps the durable per-worker timeline.
        """
        if done:
            if self._subagent_status is not None and self._subagent_status.is_attached:
                await self._subagent_status.remove()
            self._subagent_status = None
            return
        if not snapshots:
            return

        spinner = _SPINNER_FRAMES[
            int(time.monotonic() * 2) % len(_SPINNER_FRAMES)
        ]
        views = [
            SubAgentView.from_snapshot(item, index)
            for index, item in enumerate(snapshots)
        ]
        running = sum(view.state is ActivityState.RUNNING for view in views)
        ready = sum(view.status == "ready" for view in views)
        # Name column padded to a common width so the status column lines up
        # and the card scans as a table instead of ragged prose. Bounded so one
        # verbose name cannot push everything else off a narrow pane.
        name_width = min(28, max((len(view.name) for view in views), default=0))
        text = Text()
        text.append(
            f"{spinner} Sub-agents · {running}/{len(views)} running",
            style=self._style("tool", bold=True),
        )
        if ready:
            text.append(f" · {ready} ready", style=self._style("ok", bold=True))
        if timeout_s:
            text.append(f" · waiting up to {format_elapsed(timeout_s)}", style=self._style("subtle"))
        for view in views:
            state_glyph, role = ActivityPane._DISPLAY[view.state]
            if view.state is ActivityState.RUNNING:
                state_glyph = spinner
            text.append("\n  ")
            text.append(f"{state_glyph} ", style=self._style(role, bold=True))
            text.append(
                f"{view.kind_glyph} ", style=self._style(view.identity, bold=True),
            )
            text.append(
                view.name.ljust(name_width),
                style=self._style(view.identity, bold=True),
            )
            text.append(f"  {view.display_label}", style=self._style(role))
            if view.elapsed_s:
                text.append(
                    f" · {format_elapsed(int(view.elapsed_s))}",
                    style=self._style("subtle"),
                )
            if view.queue_note:
                text.append(f" · {view.queue_note}", style=self._style("subtle"))

        if self._subagent_status is None or not self._subagent_status.is_attached:
            self._subagent_status = Static(text, classes="block subagent-status")
            await self._mount_widget(
                self._subagent_status, process=True, step=False,
            )
        else:
            self._subagent_status.update(text)
        self.follow_tail()

    async def add_collapsible(
        self,
        title: str,
        renderable: RenderableType,
        *,
        classes: str,
        collapsed: bool = True,
        process: bool = False,
        step: bool = True,
    ) -> Collapsible:
        """Append a keyboard- and mouse-expandable transcript section."""
        await self.end_stream()
        # Modifier classes (for example ``tool-result-error``) belong on the
        # container; the body keeps one stable selector for tests and styling.
        names = classes.split()
        body = Static(renderable, classes=f"{names[0]}-body")
        block = Collapsible(
            body,
            title=title,
            collapsed=collapsed,
            collapsed_symbol="›",
            expanded_symbol="⌄",
            classes=f"block {classes}",
        )
        await self._mount_widget(
            block, process=process, failed="tool-result-error" in names, step=step,
        )
        return block

    async def add_thinking(self, thinking: str) -> Collapsible:
        """Render non-streamed reasoning with the same policy as live reasoning."""
        visible = thinking[:_MAX_RENDERED_MARKDOWN_CHARS]
        if len(thinking) > _MAX_RENDERED_MARKDOWN_CHARS:
            visible += "\n… additional thinking hidden in the TUI"
        collapsed = self._thinking_is_long(thinking)
        return await self.add_collapsible(
            self._thinking_title(thinking, complete=True),
            Text(visible, style=self._style("subtle")),
            classes="thinking-block",
            collapsed=collapsed,
            process=True,
        )

    async def _mount_block(
        self, renderable: RenderableType, *, classes: str = "", process: bool = False,
        step: bool = True,
    ) -> None:
        block_classes = "block" + (f" {classes}" if classes else "")
        await self._mount_widget(
            Static(renderable, classes=block_classes), process=process, step=step,
        )

    async def _mount_widget(
        self, widget: Widget, *, process: bool = False, failed: bool = False,
        step: bool = True,
    ) -> None:
        if process and self._process_body is not None:
            await self._process_body.mount(widget)
            self._touch_process(failed=failed, step=step)
        else:
            await self.mount(widget)
        await self._trim_blocks()
        self.follow_tail()

    def _process_steps_of(self, group: Widget) -> list[Widget]:
        body = next(iter(group.query(".process-body")), None)
        if body is None:
            return []
        return [step for step in body.children if step.has_class("block")]

    def _block_weight(self, block: Widget) -> int:
        """Count a group as its own row plus every step folded inside it."""
        if not block.has_class("process-group"):
            return 1
        return 1 + len(self._process_steps_of(block))

    async def _trim_blocks(self) -> None:
        """Bound mounted widgets so multi-hour sessions stay responsive.

        Steps live *inside* a process group, so counting only direct children
        would let one long task mount thousands of widgets behind a single row.
        The budget is therefore charged against every mounted block, and a task
        that outgrows it on its own has its earliest steps pruned in place.
        """
        top = [child for child in self.children if child.has_class("block")]
        overflow = sum(self._block_weight(block) for block in top) - self.max_blocks
        if overflow <= 0:
            return
        pruned = 0
        stale: list[Widget] = []
        for block in top:
            # The open group is still being written to; its steps are trimmed
            # below rather than dropping the run the user is watching.
            if pruned >= overflow or block is self._process:
                break
            stale.append(block)
            pruned += self._block_weight(block)
        if stale:
            await self.remove_children(stale)
        if pruned < overflow and self._process_body is not None:
            steps = [
                step for step in self._process_body.children
                if step.has_class("block") and step is not self._live_container
            ]
            drop = steps[:overflow - pruned]
            if drop:
                await self._process_body.remove_children(drop)
                pruned += len(drop)
        if not pruned:
            return
        self._pruned_blocks += pruned
        await self._show_prune_notice()

    async def _show_prune_notice(self) -> None:
        message = Text(
            f"{GLYPHS['pruned']} {self._pruned_blocks:,} older transcript blocks "
            "hidden to keep the TUI responsive (full history remains in the session)",
            style=self._style("subtle"),
        )
        if self._prune_notice is None:
            self._prune_notice = Static(message, classes="transcript-pruned")
            first = next(iter(self.children), None)
            if first is None:
                await self.mount(self._prune_notice)
            else:
                await self.mount(self._prune_notice, before=first)
        else:
            self._prune_notice.update(message)

    async def stream(self, kind: str, text: str) -> None:
        """Append streamed ``text`` to the live block of the given ``kind``."""
        if not text:
            return
        if kind == "content" and self._live_kind != "content" and not text.strip():
            # A reasoning model routes a tool-only turn through the thinking
            # channel and leaves a stray newline on the visible one. Opening a
            # block for that spends a separator rule on an empty message.
            return
        if kind != self._live_kind:
            await self.end_stream()
            self._live_kind = kind
            self._buf = ""
            self._last_live_render_at = 0.0
            self._last_live_render_size = 0
            if kind == "thinking":
                self._live = Static("", classes="thinking-block-body")
                self._live_container = Collapsible(
                    self._live,
                    title=f"{_THINKING_ICON} Thinking…",
                    collapsed=False,
                    collapsed_symbol="›",
                    expanded_symbol="⌄",
                    classes="block thinking-block",
                )
            else:
                self._live = Static("", classes="assistant-content-body")
                self._live_container = self._live
                self._live.add_class("block", "assistant-content")
                # Tool rows for this turn may live inside the (collapsed)
                # process group, so two messages usually sit flush against each
                # other. A rule marks the boundary the blank lines used to.
                if self._last_block_is_content():
                    self._live.add_class("assistant-continued")
            if kind == "thinking" and self._process_body is not None:
                await self._process_body.mount(self._live_container)
                self._touch_process()
            else:
                await self.mount(self._live_container)
            await self._trim_blocks()
            self.follow_tail()
        self._buf += text
        now = time.monotonic()
        visible_size = min(len(self._buf), _MAX_RENDERED_MARKDOWN_CHARS)
        enough_text = visible_size - self._last_live_render_size >= _LIVE_RENDER_MIN_CHARS
        enough_time = now - self._last_live_render_at >= _LIVE_RENDER_INTERVAL
        if self._live is not None and (enough_text or enough_time):
            self._live.update(self._render_live())
            self._last_live_render_at = now
            self._last_live_render_size = visible_size
        if (
            kind == "thinking"
            and not self._thinking_auto_collapsed
            and self._thinking_is_long(self._buf)
            and isinstance(self._live_container, Collapsible)
        ):
            self._live_container.collapsed = True
            self._thinking_auto_collapsed = True

    def _last_block_is_content(self) -> bool:
        """Whether the newest top-level block is assistant prose."""
        previous = next(
            (child for child in reversed(list(self.children)) if child.has_class("block")),
            None,
        )
        return previous is not None and previous.has_class("assistant-content")

    def _render_live(self) -> Text:
        visible = self._buf[:_MAX_RENDERED_MARKDOWN_CHARS]
        if len(self._buf) > _MAX_RENDERED_MARKDOWN_CHARS:
            visible += "\n… additional streaming output hidden in the TUI"
        if self._live_kind == "thinking":
            # Reasoning is supporting material, one tier below compact labels
            # and tool output. An explicit palette colour (rather than Rich's
            # terminal-dependent ``dim``) keeps it quiet but still measurable.
            return Text(visible, style=self._style("subtle"))
        # Prose is tightened while it streams too: this is the render the user
        # actually watches, and Markdown finalization comes much later.
        return Text(_tighten(visible), style=self._style("text"))

    @staticmethod
    def _thinking_is_long(text: str) -> bool:
        return (
            len(text) > _THINKING_COLLAPSE_CHARS
            or text.count("\n") + 1 > _THINKING_COLLAPSE_LINES
        )

    @staticmethod
    def _thinking_title(text: str, *, complete: bool) -> str:
        if not complete:
            return f"{_THINKING_ICON} Thinking…"
        summary = TranscriptView._thinking_summary(text)
        lines = text.count("\n") + 1
        suffix = f" · {lines} lines" if lines > 2 else ""
        if TranscriptView._thinking_is_long(text):
            suffix += " · expand to review"
        return f"{_THINKING_ICON} {summary}{suffix}"

    @staticmethod
    def _thinking_summary(text: str) -> str:
        """Derive a stable, scannable title without another model call."""
        line = next((part.strip() for part in text.splitlines() if part.strip()), "Reasoning")
        line = re.sub(r"^[#>*\-\d.\s]+", "", line)
        line = re.sub(
            r"^(?:let me|i(?:'ll| will| need to)|first,?)\s+",
            "",
            line,
            flags=re.IGNORECASE,
        )
        line = re.sub(r"\s+", " ", line).rstrip(".:;，。；")
        return (line[:57] + "…") if len(line) > 58 else (line or "Reasoning")

    async def end_stream(self) -> None:
        """Close the live block; re-render assistant content as Markdown."""
        if self._live is not None and self._live_kind == "content":
            if self._buf.strip():
                self._live.update(_bounded_markdown(_tighten(self._buf)))
            elif self._live.is_attached:
                # Whitespace that arrived *after* real prose opened the block is
                # harmless, but a block holding nothing else must not survive as
                # a rule and a blank row — nor be promoted as the final report.
                await self._live.remove()
        elif (
            self._live is not None
            and self._live_kind == "thinking"
            and isinstance(self._live_container, Collapsible)
        ):
            self._live.update(self._render_live())
            self._live_container.title = self._thinking_title(self._buf, complete=True)
        self._live = None
        self._live_container = None
        self._live_kind = None
        self._buf = ""
        self._last_live_render_at = 0.0
        self._last_live_render_size = 0
        self._thinking_auto_collapsed = False

    async def promote_last_content(self) -> None:
        """Give the most recent assistant prose the final-report visual tier."""
        blocks = list(self.query(".assistant-content"))
        if blocks:
            report = blocks[-1]
            report.add_class("final-report")
            # The heading mounted below draws the boundary now; keeping the
            # continuation rule too would stack two rules on one seam.
            report.remove_class("assistant-continued")
            heading = Static(
                Text(f"{GLYPHS['ok']} Final report", style=self._style("ok", bold=True)),
                classes="block report-heading",
            )
            await self.mount(heading, before=report)

    def set_final_text(self, text: str) -> None:
        self.final_text = text

    def clear_filter(self) -> bool:
        """Restore every block, and the disclosure state the filter overrode."""
        if self.filter_mode == "all":
            return False
        for block in self.query(".block"):
            block.display = True
        for block, collapsed in self._collapse_snapshot:
            if block.is_attached:
                block.collapsed = collapsed
        self._collapse_snapshot = []
        self.filter_mode = "all"
        self.filter_query = ""
        self.filter_matches = 0
        return True

    def apply_filter(self, mode: str, query: str = "") -> int:
        """Filter transcript blocks in place; return the number of matches."""
        mode = mode.lower()
        query = query.casefold().strip()
        if mode == "all":
            self.clear_filter()
            return len(list(self.query(".block")))
        if self.filter_mode == "all":
            # Snapshot on the way in only: successive filters must restore the
            # folding the user chose, not whatever an earlier filter forced.
            self._collapse_snapshot = [
                (block, block.collapsed) for block in self.query(Collapsible)
            ]
        matches = 0
        for block in self.query(".block"):
            # The group is a structural parent, not a searchable result. Its
            # visibility is derived from the child results below.
            if block.has_class("process-group"):
                continue
            classes = set(block.classes)
            if mode == "thinking":
                visible = "thinking-block" in classes
            elif mode == "tools":
                visible = bool(classes & {"tool-call", "tool-result"})
            elif mode == "errors":
                visible = bool(classes & {"tool-result-error", "transcript-error"})
            elif mode == "report":
                visible = bool(classes & {"final-report", "report-heading", "report-footer"})
            elif mode == "search":
                visible = query in self._plain_text(block).casefold()
            else:
                raise ValueError(f"unknown transcript filter: {mode}")
            block.display = visible
            matches += int(visible)
            if visible and isinstance(block, Collapsible) and mode == "search":
                block.collapsed = False
        for process in self.query(".process-group"):
            children_visible = any(child.display for child in process.query(".block"))
            process.display = children_visible
            if mode != "report" and children_visible and isinstance(process, Collapsible):
                process.collapsed = False
        self.filter_mode = mode
        self.filter_query = query
        self.filter_matches = matches
        return matches

    @staticmethod
    def _plain_text(widget: Widget) -> str:
        """Searchable text for one block, including the block itself.

        ``query`` walks descendants only, so a bare ``Static`` block — a tool
        call line, a note, an echoed prompt — would contribute nothing and
        could never be found. A collapsible's title counts too: while folded it
        is the only part of the block on screen.
        """
        parts: list[str] = []
        if isinstance(widget, Collapsible):
            parts.append(str(widget.title))
        statics = [widget] if isinstance(widget, Static) else []
        statics.extend(widget.query(Static))
        parts.extend(_renderable_text(item.render()) for item in statics)
        return "\n".join(parts)

    def _visible_blocks(self) -> list[Widget]:
        return [block for block in self.query(".block") if block.display]

    def _select_review_block(self, block: Widget) -> None:
        if self._review_block is not None and self._review_block.is_attached:
            self._review_block.remove_class("review-active")
        self._review_block = block
        block.add_class("review-active")
        block.scroll_visible(animate=False)

    def review_move(self, delta: int) -> None:
        """Move the review cursor, entering at the newest block.

        The cursor tracks the widget rather than its position: new blocks
        arriving, a filter, or pruning would all silently shift an index out
        from under the user mid-review.
        """
        blocks = self._visible_blocks()
        if not blocks:
            return
        if self._review_block in blocks:
            index = blocks.index(self._review_block) + delta
            index = max(0, min(len(blocks) - 1, index))
        else:
            index = len(blocks) - 1
        self._select_review_block(blocks[index])

    def toggle_review_block(self) -> None:
        blocks = self._visible_blocks()
        if not blocks:
            return
        block = self._review_block if self._review_block in blocks else blocks[-1]
        self._select_review_block(block)
        if isinstance(block, Collapsible):
            block.collapsed = not block.collapsed

    def on_resize(self) -> None:
        """Re-fit the logo whenever the pane changes width.

        The logo is laid out to an exact column count, and that count is not
        knowable until Textual has run the layout — not at mount, and not from
        the app's own Resize, which arrives before the sidebar has taken its 45
        columns back off this pane. This is the first moment the number is real,
        which is why the re-fit lives here and not at the call sites that change
        the width.
        """
        self.refresh_logo(active_theme(self))

    def refresh_logo(self, theme: str, *, force: bool = False) -> None:
        """Repaint the startup logo for ``theme``, fitted to the current width.

        Its colours are baked into the ``Text`` at mount time, like every other
        Rich renderable in the transcript, so a CSS reload alone would leave the
        one thing on screen whose whole job is to wear the palette still wearing
        the previous one.

        Repainting can resize the block and so post another Resize; an unchanged
        width is therefore skipped unless ``force``, which is what a theme switch
        passes — new colours at the same fit.
        """
        width = max(0, self.content_size.width - 2)
        if width == self._logo_width and not force:
            return
        self._logo_width = width
        for block in self.query(".startup-logo").results(Static):
            block.update(render_logo(theme, width))

    async def clear_all(self) -> None:
        await self.end_stream()
        await self.remove_children()
        # A wholesale replacement is a new transcript: whatever the user had
        # scrolled away from is gone, so following starts armed again.
        self.resume_tail()
        self._pruned_blocks = 0
        self._prune_notice = None
        self._process = None
        self._process_body = None
        self._review_block = None
        self._subagent_status = None
        self.final_text = ""
        self.filter_mode = "all"
        self.filter_query = ""
        self.filter_matches = 0
        self._collapse_snapshot = []

    async def replay_history(self, history: list[Message]) -> int:
        """Replace the transcript with the complete saved visible history.

        Tool protocol messages are regrouped into the same progressively
        disclosed ``Working`` block used by a live run.  Older replay code
        mounted every call/result at top level, which made /resume flatten the
        coordinator timeline and visually lose the main-screen activity.
        """
        await self.clear_all()
        recent: list[tuple[str, object]] = []
        eligible = 0
        tool_names: dict[str, str] = {}
        for message in history:
            role = message.get("role")
            if role in ("user", "assistant"):
                body = (
                    extract_final_content([message])
                    if role == "assistant"
                    else text_of(message.get("content")).strip()
                )
                if body:
                    eligible += 1
                    visible_role = (
                        "assistant-step"
                        if role == "assistant" and message.get("tool_calls")
                        else role
                    )
                    recent.append((visible_role, body))
            if role == "assistant":
                for call in message.get("tool_calls") or []:
                    function = call.get("function") or {}
                    name = str(function.get("name") or call.get("name") or "tool")
                    raw_args = function.get("arguments", call.get("args", {}))
                    try:
                        args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                    except (TypeError, ValueError):
                        args = raw_args
                    call_id = str(call.get("id") or "")
                    if call_id:
                        tool_names[call_id] = name
                    eligible += 1
                    recent.append(("tool-call", (name, args)))
            if role == "tool":
                call_id = str(message.get("tool_call_id") or "")
                name = str(message.get("name") or tool_names.get(call_id) or "tool")
                eligible += 1
                recent.append(("tool-result", (
                    name,
                    text_of(message.get("content")),
                    bool(message.get("is_error", False)),
                )))

        # Live runs remain bounded, but explicit resume is a history-review
        # operation: do not silently discard old turns or tool calls.
        self.max_blocks = max(self.max_blocks, eligible)
        process_open = False

        async def begin_process_if_needed() -> None:
            nonlocal process_open
            if process_open:
                return
            await self.begin_process()
            process_open = True

        async def end_process_if_needed() -> None:
            nonlocal process_open
            if not process_open:
                return
            await self.end_process("complete")
            process_open = False

        for role, payload in recent:
            if role == "user":
                await end_process_if_needed()
                await self._mount_block(
                    Text(f"{GLYPHS['user']} {payload}", style=self._style("user", bold=True)),
                    classes="history-user turn-start",
                )
            elif role in {"assistant", "assistant-step"}:
                # In the native loop a plain assistant message is the final
                # answer, so the coordinator work group closes before it.
                if role == "assistant":
                    await end_process_if_needed()
                await self._mount_block(
                    _bounded_markdown(str(payload)), classes="history-assistant",
                )
            elif role == "tool-call":
                await begin_process_if_needed()
                name, args = payload  # type: ignore[misc]
                detail = json.dumps(args, ensure_ascii=False) if isinstance(args, (dict, list)) else str(args)
                await self.add_collapsible(
                    tool_label(name),
                    Text(detail, style=self._style("muted")),
                    classes="tool-call history-tool-call",
                    process=True,
                )
            else:
                await begin_process_if_needed()
                name, body, is_error = payload  # type: ignore[misc]
                glyph = GLYPHS["fail"] if is_error else GLYPHS["ok"]
                await self.add_collapsible(
                    f"{glyph} {tool_label(name, marker=False)}",
                    Text(str(body), style=self._style("muted")),
                    classes=("tool-result tool-result-error" if is_error else "tool-result"),
                    process=True,
                    step=False,
                )
        await end_process_if_needed()
        return len(recent)


class ActivityState(StrEnum):
    RUNNING = "running"
    # Work accepted but not started — a sub-agent sitting behind the task its
    # session is already running. Distinct from RUNNING so the row neither
    # animates nor claims elapsed time it has not spent.
    QUEUED = "queued"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"
    INTERRUPTED = "interrupted"


@dataclass
class ActivityRecord:
    call_id: str
    name: str
    summary: str
    state: ActivityState
    started_at: float
    finished_at: float | None = None
    duration_ms: int = 0
    # Sub-agent rows only: the inferred specialty (drives the marker glyph)
    # and the semantic colour role that identifies this worker. Empty on
    # ordinary tool rows, which are identified by their tool name alone.
    kind: str = ""
    identity: str = ""
    parent_agent: str = ""
    event_kind: str = ""


# Bus status → (activity state, row label). The bus vocabulary is wider than
# the pane's five states, and mapping it in one place is what keeps the
# transcript card and the sidebar from disagreeing about what a worker is
# doing. Anything unlisted is deliberately *not* treated as success: an
# unrecognised state is unknown, and a green ✓ would assert otherwise.
# States whose duration is still accruing, so the row's elapsed time is
# recomputed from the clock rather than read from a frozen ``duration_ms``.
_LIVE_STATES = frozenset({ActivityState.RUNNING, ActivityState.QUEUED})

_SUBAGENT_STATES: dict[str, tuple[ActivityState, str]] = {
    "submitted": (ActivityState.RUNNING, "working"),
    "running": (ActivityState.RUNNING, "working"),
    "queued": (ActivityState.QUEUED, "queued"),
    "ready": (ActivityState.SUCCESS, "report ready"),
    "completed": (ActivityState.SUCCESS, "done"),
    "idle": (ActivityState.SUCCESS, "idle"),
    "failed": (ActivityState.FAILED, "failed"),
    "aborted": (ActivityState.INTERRUPTED, "aborted"),
    "unassigned": (ActivityState.SKIPPED, "unassigned"),
}


def _as_float(value: object, default: float = 0.0) -> float:
    """Coerce a snapshot field that crossed a ``dict[str, object]`` boundary."""
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _as_int(value: object, default: int = 0) -> int:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class SubAgentView:
    """One ``describe_sessions_for_task`` snapshot, resolved for display.

    Both sub-agent surfaces — the live transcript card and the persistent
    sidebar rows — build from this, so a worker's marker, colour and wording
    cannot drift between them.
    """

    session_id: str
    name: str
    kind: str
    kind_glyph: str
    identity: str
    status: str
    state: ActivityState
    label: str
    active: bool
    elapsed_s: float
    queued: int
    completed: int

    @classmethod
    def from_snapshot(cls, item: dict[str, object], index: int = 0) -> SubAgentView:
        name = str(item.get("name") or "sub-agent")
        session_id = str(item.get("session_id") or name)
        status = str(item.get("status") or "unknown")
        state, label = _SUBAGENT_STATES.get(
            status, (ActivityState.SKIPPED, status),
        )
        kind = agent_kind(name)
        return cls(
            session_id=session_id,
            name=name,
            kind=kind,
            kind_glyph=agent_kind_glyph(kind),
            identity=agent_identity_role(index),
            status=status,
            state=state,
            label=label,
            active=bool(item.get("active", state is ActivityState.RUNNING)),
            elapsed_s=max(0.0, _as_float(item.get("elapsed_s"))),
            queued=max(0, _as_int(item.get("queued"))),
            completed=max(0, _as_int(item.get("completed"))),
        )

    @property
    def display_label(self) -> str:
        """``label``, absorbing the backlog count when it *is* the state."""
        if self.status == "queued" and self.queued:
            return f"queued ×{self.queued}"
        return self.label

    @property
    def queue_note(self) -> str:
        """``"N queued"``, or empty when ``display_label`` already says it."""
        if not self.queued or self.status == "queued":
            return ""
        return f"{self.queued} queued"

    @property
    def details(self) -> str:
        """The label plus only the counts it does not already imply.

        Each segment has to earn its width: on a 36-column sidebar a repeated
        count ("report ready · 1 ready") costs the tail of the row and reads
        as two separate things.
        """
        parts = [self.display_label]
        if self.queue_note:
            parts.append(self.queue_note)
        if self.completed and self.status != "ready":
            parts.append(f"{self.completed} ready")
        elif self.completed > 1:
            parts.append(f"+{self.completed - 1} more")
        return " · ".join(parts)


class ActivityPane(OptionList):
    """Bounded, selectable tool timeline whose rows update in place."""

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("space", "preview", "Details", priority=True),
        Binding("escape", "close", "Return to prompt", priority=True),
    ]

    # State → (glyph, role). The glyph alone carries the state: each is a
    # distinct shape, so the row does not depend on colour, and dropping the
    # spelled-out word ("success", "interrupted") returns ~9 of the sidebar's 36
    # columns to the command text — which is the part worth scanning.
    _DISPLAY: ClassVar[Any] = {
        ActivityState.RUNNING: (GLYPHS["running"], "tool"),
        ActivityState.QUEUED: (GLYPHS["queued"], "subtle"),
        ActivityState.SUCCESS: (GLYPHS["ok"], "ok"),
        ActivityState.FAILED: (GLYPHS["fail"], "err"),
        ActivityState.SKIPPED: (GLYPHS["skipped"], "subtle"),
        ActivityState.INTERRUPTED: (GLYPHS["stopped"], "warn"),
    }

    def __init__(
        self, *, max_records: int = _MAX_ACTIVITY_RECORDS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        super().__init__(id="activity")
        self.max_records = max(1, max_records)
        self.clock = clock
        self.records: list[ActivityRecord] = []
        self.agent_records: dict[str, ActivityRecord] = {}
        self.agent_events: dict[str, dict[str, ActivityRecord]] = {}
        self.expanded_agents: set[str] = set()
        self._rendered_records: list[ActivityRecord | None] = []
        self._pending: dict[str, deque[ActivityRecord]] = {}
        self._sequence = 0

    def start(self, call_id: str, name: str, summary: str) -> str:
        record = self._create_record(call_id, name, summary)
        self._trim()
        self._render_records()
        return record.call_id

    def _create_record(
        self, call_id: str, name: str, summary: str,
    ) -> ActivityRecord:
        self._sequence += 1
        public_id = call_id.strip() or f"activity-{self._sequence}"
        record = ActivityRecord(
            call_id=public_id,
            name=name or "tool",
            summary=" ".join(summary.split()),
            state=ActivityState.RUNNING,
            started_at=self.clock(),
        )
        self.records.append(record)
        self._pending.setdefault(public_id, deque()).append(record)
        return record

    def finish(
        self, call_id: str, name: str, *, is_error: bool, ms: int = 0,
        state: ActivityState | None = None,
    ) -> None:
        pending = self._pending.get(call_id)
        record = pending.popleft() if pending else None
        if pending is not None and not pending:
            self._pending.pop(call_id, None)
        if record is None:
            # Preserve an unmatched result for diagnosis; never guess by name.
            record = self._create_record(
                call_id, name, "result without matching start",
            )
            pending = self._pending[record.call_id]
            pending.popleft()
            if not pending:
                self._pending.pop(record.call_id, None)
        elif record not in self.records:
            while len(self.records) >= self.max_records:
                self._evict_one()
            self.records.append(record)

        finished_at = self.clock()
        record.finished_at = finished_at
        record.duration_ms = max(0, ms) or max(0, int((finished_at - record.started_at) * 1000))
        record.state = state or (ActivityState.FAILED if is_error else ActivityState.SUCCESS)
        self._trim()
        self._render_records()

    def finish_active(self, state: ActivityState) -> None:
        if state in _LIVE_STATES:
            raise ValueError("active activity rows need a terminal state")
        now = self.clock()
        visible_records = {id(record) for record in self.records}
        for record in self.records:
            if record.state not in _LIVE_STATES:
                continue
            self._settle_record(record, state, now)
        for pending in self._pending.values():
            for record in pending:
                if id(record) in visible_records:
                    continue
                self._settle_record(record, state, now)
        # Sub-agent rows outlive a single tool call, but they must not outlive
        # the *turn*: nothing publishes further snapshots once the loop stops,
        # so a worker left RUNNING here would spin forever and keep the pane
        # repainting on every tick with an ever-growing clock.
        for record in self.agent_records.values():
            if record.state in _LIVE_STATES:
                self._settle_record(record, state, now)
        self._pending.clear()
        self._trim()
        self._render_records()

    def refresh_running(self) -> None:
        if any(
            record.state in _LIVE_STATES
            for record in [*self.records, *self.agent_records.values()]
        ):
            self._render_records()

    def update_subagents(
        self, snapshots: list[dict[str, object]], *, done: bool = False,
    ) -> None:
        """Upsert one stable Activity row per Agent Team worker."""
        now = self.clock()
        seen: set[str] = set()
        for index, item in enumerate(snapshots):
            view = SubAgentView.from_snapshot(item, index)
            seen.add(view.session_id)
            record = self.agent_records.get(view.session_id)
            if record is None:
                self.agent_records[view.session_id] = ActivityRecord(
                    call_id=f"subagent:{view.session_id}",
                    name=view.name,
                    summary=view.details,
                    state=view.state,
                    started_at=now - view.elapsed_s,
                    kind=view.kind,
                    identity=view.identity,
                )
                self._settle_agent_duration(
                    self.agent_records[view.session_id], view, now,
                )
                record = self.agent_records[view.session_id]
            else:
                was_live = record.state in _LIVE_STATES
                record.name = view.name
                record.summary = view.details
                record.state = view.state
                record.kind = view.kind
                # Identity is assigned once. Positions only shift if the bus stops
                # listing a session, and a row changing colour mid-run would read
                # as a different worker.
                record.identity = record.identity or view.identity
                if view.active:
                    anchor = now - view.elapsed_s
                    # Re-anchor outright when the worker was idle and has picked up
                    # a new task; while it stays live, only move the start earlier,
                    # so a snapshot that momentarily under-reports elapsed time
                    # cannot make the row's clock jump backwards.
                    if not was_live or record.started_at > anchor:
                        record.started_at = anchor
                self._settle_agent_duration(record, view, now)
            self._update_agent_events(view.session_id, item.get("events"), now)
            event_count = len(self.agent_events.get(view.session_id, {}))
            if event_count:
                record.summary = f"{view.details} · {event_count} events"

        # A missing worker during an active snapshot was removed from the bus;
        # retain its row but stop animating it. On the final callback, snapshots
        # are still supplied, so workers that remain active stay visibly active.
        if snapshots and not done:
            for session_id, record in self.agent_records.items():
                if session_id not in seen and record.state == ActivityState.RUNNING:
                    self._settle_record(record, ActivityState.INTERRUPTED, now)
        self._render_records()

    def _update_agent_events(
        self, session_id: str, raw_events: object, now: float,
    ) -> None:
        if not isinstance(raw_events, list):
            return
        bucket = self.agent_events.setdefault(session_id, {})
        for raw in raw_events:
            if not isinstance(raw, dict):
                continue
            event_id = str(raw.get("id") or f"event-{len(bucket) + 1}")
            kind = str(raw.get("kind") or "activity")
            title = str(raw.get("title") or kind)
            detail = str(raw.get("detail") or "").strip() or "(no detail)"
            turn = max(0, _as_int(raw.get("turn")))
            if kind == "thinking":
                name = f"{GLYPHS['thinking']} thinking"
            elif kind == "message":
                name = "assistant"
            elif kind == "tool_call":
                name = f"{GLYPHS['tool']} {title}"
            elif kind == "tool_error":
                name = f"{GLYPHS['fail']} {title}"
            else:
                name = f"{GLYPHS['ok']} {title}"
            summary = detail
            if turn:
                summary = f"turn {turn} · {summary}"
            state = (
                ActivityState.FAILED
                if bool(raw.get("is_error")) or kind == "tool_error"
                else ActivityState.SUCCESS
            )
            record = bucket.get(event_id)
            if record is None:
                bucket[event_id] = ActivityRecord(
                    call_id=f"subevent:{session_id}:{event_id}",
                    name=name,
                    summary=summary,
                    state=state,
                    started_at=_as_float(raw.get("at"), now),
                    finished_at=now,
                    parent_agent=session_id,
                    event_kind=kind,
                )
            else:
                record.name = name
                record.summary = summary
                record.state = state

    @staticmethod
    def _settle_agent_duration(
        record: ActivityRecord, view: SubAgentView, now: float,
    ) -> None:
        """Keep a live row ticking; freeze a finished one exactly once.

        ``_duration`` recomputes ``now - started_at`` for live rows, so a
        finished worker must have ``finished_at``/``duration_ms`` written and
        then left alone. Recomputing on every snapshot is what made completed
        sub-agents appear to keep counting for as long as the fan-in blocked.
        """
        if record.state in _LIVE_STATES:
            # A worker can go live again on its next assigned task; clearing
            # the stamp is what lets the row restart rather than stay frozen.
            record.finished_at = None
            record.duration_ms = 0
            return
        if record.finished_at is not None:
            return
        record.finished_at = now
        # The bus reports the authoritative wall-clock duration of the task
        # that just ended; the pane's own ``now - started_at`` is only a
        # fallback for a snapshot that omits it (it also counts the time the
        # row spent visible before the first snapshot arrived).
        elapsed = view.elapsed_s or max(0.0, now - record.started_at)
        record.duration_ms = max(0, int(elapsed * 1000))

    @staticmethod
    def _settle_record(
        record: ActivityRecord, state: ActivityState, now: float,
    ) -> None:
        """Move a row to a terminal ``state`` and stop its clock."""
        record.state = state
        record.finished_at = now
        record.duration_ms = max(0, int((now - record.started_at) * 1000))

    def rerender(self) -> None:
        """Repaint every row — the styles are baked into the written ``Text``,
        so a theme change needs the rows rebuilt, not just a CSS reload."""
        self._render_records()

    def on_resize(self) -> None:
        """Re-fit and re-follow whenever the pane changes size.

        Rows are ellipsized to the pane width, so a terminal resize needs to
        rebuild the option prompts after Textual has calculated the new layout.
        """
        self._render_records()

    def clear_records(self) -> None:
        self.records.clear()
        self.agent_records.clear()
        self.agent_events.clear()
        self.expanded_agents.clear()
        self._rendered_records.clear()
        self._pending.clear()
        self._render_records()

    def latest(self, call_id: str) -> ActivityRecord | None:
        return next((record for record in reversed(self.records) if record.call_id == call_id), None)

    def _trim(self) -> None:
        while len(self.records) > self.max_records:
            self._evict_one()

    def _evict_one(self) -> None:
        index = next(
            (i for i, record in enumerate(self.records)
             if record.state not in _LIVE_STATES),
            0,
        )
        self.records.pop(index)

    @staticmethod
    def _duration(record: ActivityRecord, now: float) -> str:
        ms = (
            max(0, int((now - record.started_at) * 1000))
            if record.state in _LIVE_STATES
            else record.duration_ms
        )
        return f"{ms} ms" if ms < 1000 else f"{ms / 1000:.1f}s"

    def _render_records(self) -> None:
        # Follow the newest call until the user enters this list. Once focused,
        # preserve their review cursor across running-duration refreshes.
        selected = self.selected_record if self.has_focus else None
        now = self.clock()
        self.clear_options()
        self._rendered_records = []
        if not self.records and not self.agent_records:
            # A blank pane read as a rendering failure rather than an idle
            # timeline — the plan board already says "no plan yet".
            self.add_option(Option(
                Text("no tool calls yet", style=self._style("subtle")),
                disabled=True,
            ))
            self._rendered_records.append(None)
            return
        width = self.content_size.width or _ACTIVITY_FALLBACK_WIDTH
        if self.agent_records:
            self.add_option(Option(
                self._group_heading("SUB-AGENTS", self.agent_records.values()),
                disabled=True,
            ))
            self._rendered_records.append(None)
            for session_id, record in self.agent_records.items():
                self.add_option(Option(
                    self._render_agent_row(session_id, record, now, width),
                    id=f"agent-{len(self._rendered_records)}",
                ))
                self._rendered_records.append(record)
                if session_id in self.expanded_agents:
                    for event_record in self.agent_events.get(session_id, {}).values():
                        self.add_option(Option(
                            self._render_event_row(event_record, width),
                            id=f"agent-event-{len(self._rendered_records)}",
                        ))
                        self._rendered_records.append(event_record)
        if self.records and self.agent_records:
            self.add_option(Option(
                self._group_heading("COORDINATOR", self.records),
                disabled=True,
            ))
            self._rendered_records.append(None)
        for record in self.records:
            self.add_option(Option(
                self._render_row(record, now, width),
                id=f"activity-{len(self._rendered_records)}",
            ))
            self._rendered_records.append(record)
        selectable = [
            index for index, record in enumerate(self._rendered_records)
            if record is not None
        ]
        if not selectable:
            return
        # ``index`` matches by value and ``ActivityRecord`` is a plain
        # dataclass, so identity is what must be compared — two rows can
        # legitimately hold equal fields.
        restored = next(
            (
                index for index in selectable
                if self._rendered_records[index] is selected
            ),
            None,
        )
        self.highlighted = selectable[-1] if restored is None else restored

    def _group_heading(
        self, label: str, records: Iterable[ActivityRecord],
    ) -> Text:
        """A section divider that also carries the group's live counts.

        The pane scrolls, so a bare label leaves the user counting rows to
        answer "how many are still working?" — which is the whole question
        during a fan-in.
        """
        records = list(records)
        live = sum(record.state in _LIVE_STATES for record in records)
        failed = sum(
            record.state in (ActivityState.FAILED, ActivityState.INTERRUPTED)
            for record in records
        )
        heading = Text(label, style=self._style("subtle", bold=True))
        heading.append(f"  {len(records)}", style=self._style("subtle"))
        if live:
            heading.append(f" · {live} live", style=self._style("tool"))
        if failed:
            heading.append(f" · {failed} failed", style=self._style("err"))
        return heading

    @property
    def selected_record(self) -> ActivityRecord | None:
        index = self.highlighted
        if index is None or not 0 <= index < len(self._rendered_records):
            return None
        return self._rendered_records[index]

    def action_preview(self) -> None:
        record = self.selected_record
        if record is None:
            return
        if record.call_id.startswith("subagent:"):
            session_id = record.call_id.removeprefix("subagent:")
            if session_id in self.expanded_agents:
                self.expanded_agents.remove(session_id)
            else:
                self.expanded_agents.add(session_id)
            self._render_records()
            return
        self.app.open_activity_preview(record)  # type: ignore[attr-defined]

    def action_close(self) -> None:
        self.app.focus_prompt()  # type: ignore[attr-defined]

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        event.stop()
        self.action_preview()

    def _render_row(self, record: ActivityRecord, now: float, width: int) -> Text:
        """One timeline row, ellipsized to the pane.

        Duration precedes the summary so it survives truncation: it is
        fixed-width and always worth seeing, while the command's tail is what a
        narrow sidebar can afford to drop. Ellipsizing here rather than letting
        the row overflow is also what removes the horizontal scrollbar that used
        to sit across the bottom of the sidebar.

        A sub-agent row carries two markers, not one: the state glyph (shared
        with tool rows) and a specialty glyph, with the worker's name in its
        own identity colour. Which worker a line belongs to is the thing being
        scanned for when several run at once, and the state glyph alone cannot
        answer it — five workers all "running" look identical.
        """
        glyph, style = self._DISPLAY[record.state]
        line = Text(no_wrap=True, overflow="ellipsis")
        line.append(f"{glyph} ", style=self._style(style, bold=True))
        if record.kind:
            line.append(
                f"{agent_kind_glyph(record.kind)} ",
                style=self._style(record.identity or "text", bold=True),
            )
        line.append(
            record.name,
            style=self._style(record.identity or "text", bold=True),
        )
        line.append(f" {self._duration(record, now)}", style=self._style("subtle"))
        if record.summary:
            line.append(f"  {record.summary}", style=self._style("muted"))
        line.truncate(max(1, width), overflow="ellipsis")
        return line

    def _render_agent_row(
        self, session_id: str, record: ActivityRecord, now: float, width: int,
    ) -> Text:
        disclosure = "⌄" if session_id in self.expanded_agents else "›"
        line = Text(f"{disclosure} ", style=self._style("subtle", bold=True))
        line.append_text(self._render_row(record, now, max(1, width - 2)))
        line.truncate(max(1, width), overflow="ellipsis")
        return line

    def _render_event_row(self, record: ActivityRecord, width: int) -> Text:
        role = "err" if record.state is ActivityState.FAILED else "muted"
        line = Text("    ", style=self._style("subtle"))
        line.append(record.name, style=self._style(role, bold=True))
        if record.summary:
            one_line = " ".join(record.summary.split())
            line.append(f"  {one_line}", style=self._style("subtle"))
        line.truncate(max(1, width), overflow="ellipsis")
        return line

    def summary(self) -> str:
        """Compact counts for the sidebar heading.

        The pane scrolls now, so the heading has to carry what is off-screen.
        """
        if not self.records:
            return ""
        running = sum(1 for r in self.records if r.state == ActivityState.RUNNING)
        failed = sum(1 for r in self.records if r.state == ActivityState.FAILED)
        # Glyph-prefixed rather than spelled out ("1 running · 2 failed" ran past
        # the pane edge and got clipped), reusing the same marks as the rows.
        parts = [str(len(self.records))]
        if running:
            # Keep a cell between a Unicode state mark and its count. Some
            # terminal fonts render the quarter-circle slightly wider than
            # wcwidth reports, so ``◐2`` visually overprints the digit.
            parts.append(f"{GLYPHS['running']} {running}")
        if failed:
            parts.append(f"{GLYPHS['fail']} {failed}")
        return " ".join(parts)

    def _style(self, role: str, *, bold: bool = False) -> str:
        return rich_style(active_theme(self), role, bold=bold)


class StatusBar(Static):
    """One-line responsive footer for the current UI presentation state."""

    _PHASES: ClassVar[Any] = {
        PresentationPhase.IDLE: (GLYPHS["pending"], "idle", "muted"),
        PresentationPhase.THINKING: (GLYPHS["running"], "thinking", "tool"),
        PresentationPhase.RESPONDING: (GLYPHS["responding"], "responding", "accent"),
        PresentationPhase.RUNNING_TOOL: (GLYPHS["tool"], "tool", "tool"),
        PresentationPhase.AWAITING_APPROVAL: (GLYPHS["approval"], "approval", "err"),
        PresentationPhase.DONE: (GLYPHS["ok"], "done", "ok"),
        PresentationPhase.INCOMPLETE: (GLYPHS["stopped"], "incomplete", "warn"),
        PresentationPhase.INTERRUPTED: (GLYPHS["stopped"], "interrupted", "warn"),
        PresentationPhase.ERROR: (GLYPHS["error"], "error", "err"),
    }

    def __init__(self) -> None:
        super().__init__("", id="status")

    def show(
        self, *, presentation: TuiPresentationState, mode: str,
        ctx: str, tools: int, width: int,
        changes: tuple[int, int, int] = (0, 0, 0),
    ) -> None:
        icon, label, phase_style = self._PHASES[presentation.phase]
        elapsed = format_elapsed(presentation.elapsed_seconds())
        tool = " ".join(presentation.current_tool.split())
        tool_name = tool.split(maxsplit=1)[0] if tool else ""
        compact_ctx = ctx.removesuffix(" left")

        parts: list[tuple[str, str]] = [(f"{icon} {label}", phase_style)]
        if elapsed:
            parts.append((elapsed, ""))
        if tool_name:
            parts.append((tool_name, "tool"))
        parts.append((mode, ""))

        # Gated like every segment below it: ungated, these three sit ahead of
        # the whole line and the ellipsis eats the ``queued N`` indicator that
        # tells the user their steer is still waiting.
        files, added, removed = changes
        if files and width >= 120:
            parts.append((f"{files} file{'s' if files != 1 else ''}", ""))
            parts.append((f"+{added}", "ok"))
            parts.append((f"-{removed}", "err"))
        elif files and width >= 100:
            parts.append((f"{files}f +{added} -{removed}", ""))

        if width >= 60 and compact_ctx:
            parts.append((f"ctx {compact_ctx}", ""))

        if width >= 120:
            parts.extend([(f"tools {tools}", ""), (f"queued {presentation.queued_steers}", "")])
        elif width >= 100:
            parts.extend([(f"t {tools}", ""), (f"q {presentation.queued_steers}", "")])
        elif width >= 60:
            parts.append((f"t {tools}", ""))
            if presentation.queued_steers:
                parts.append((f"queued {presentation.queued_steers}", "accent"))
        elif presentation.queued_steers:
            parts.append((f"q {presentation.queued_steers}", "accent"))

        line = Text(no_wrap=True, overflow="ellipsis")
        for index, (value, style) in enumerate(parts):
            if index:
                line.append(" · ", style=self._style("muted"))
            line.append(value, style=self._style(style, bold=style not in {"", "muted"}))
        # The widget has one cell of horizontal padding on each side.
        line.truncate(max(1, width - 2), overflow="ellipsis")
        self.update(line)

    def _style(self, role: str, *, bold: bool = False) -> str:
        if not role:
            return ""
        return rich_style(active_theme(self), role, bold=bold)


class TodoPane(Static):
    """Persistent sidebar board for both local todos and Agent Team tasks."""

    def __init__(self) -> None:
        # Plain text at construction: ``self.app`` (and therefore the active
        # theme) is not reachable until mount, and ``_update_view`` repaints
        # this the first time a plan or a theme change arrives.
        super().__init__("no plan yet", id="todos")
        self._items: list[dict[str, str]] = []

    def clear(self) -> None:
        """Clear the task board back to 'no plan yet' when a new query starts."""
        self._items = []
        self._update_view()

    def show_todos(self, items: list, *, force: bool = False) -> None:
        if not items and not force and self._items:
            # Preserve existing task board items after delivery/completion;
            # do not wipe them out to 'no plan yet' unless explicit or empty.
            return
        self._items = [
            {
                "id": str(getattr(it, "id", "")),
                "content": str(getattr(it, "content", it)),
                "status": str(getattr(it, "status", "pending")),
            }
            for it in items
        ]
        self._update_view()

    def apply_task_board_operation(self, name: str, args: dict) -> None:
        """Project Agent Team's add_task/update_task tools into this pane.

        Coerces arguments exactly as the task_board tools do, so a
        double-encoded item can't leave the pane disagreeing with the board.
        """
        if name == "add_task":
            for raw in coerce_json_list(args.get("tasks", [])) or []:
                task = coerce_json_object(raw)
                if task is None:
                    continue
                content = str(task.get("description", "")).strip()
                if not content or any(item["content"] == content for item in self._items):
                    continue
                self._items.append({"id": f"t{len(self._items) + 1}", "content": content, "status": "open"})
        elif name == "update_task":
            by_id = {item["id"]: item for item in self._items}
            for raw in coerce_json_list(args.get("updates", [])) or []:
                update = coerce_json_object(raw)
                if update is None:
                    continue
                item = by_id.get(str(update.get("id", "")))
                resolution = str(update.get("resolution", "")).strip()
                if item is not None and resolution in {"open", "in_progress", "resolved", "cancelled"}:
                    item["status"] = resolution
        self._update_view()

    def refresh_theme(self) -> None:
        self._update_view()

    def summary(self) -> str:
        """``done/total`` for the sidebar heading — the pane scrolls, so the
        heading has to carry the progress that scrolled out of view."""
        if not self._items:
            return ""
        done = sum(
            1 for item in self._items
            if item["status"] in {"completed", "resolved", "cancelled"}
        )
        return f"{done}/{len(self._items)}"

    def _update_view(self) -> None:
        if not self._items:
            self.update(Text("no plan yet", style=self._style("muted")))
            return
        body = Text()
        glyphs = {
            "completed": GLYPHS["ok"], "resolved": GLYPHS["ok"],
            "in_progress": GLYPHS["in_progress"], "cancelled": GLYPHS["cancelled"],
        }
        for item in self._items:
            status = item["status"]
            role = "ok" if status in {"completed", "resolved"} else (
                "tool" if status == "in_progress" else (
                    "subtle" if status == "cancelled" else "muted"
                )
            )
            label = f"{item['id']} " if item["id"] else ""
            body.append(
                f"{glyphs.get(status, GLYPHS['pending'])} {label}{item['content']}\n",
                style=self._style(role),
            )
        self.update(body)

    def _style(self, role: str) -> str:
        return rich_style(active_theme(self), role)


class DiffScroll(VerticalScroll):
    """Scroll host for :class:`DiffPane`.

    A ``Static`` clips whatever overflows its box — a plain widget's virtual
    size never exceeds its container, so ``overflow-y: auto`` on the pane
    itself buys nothing and everything past the first screenful of a diff is
    simply unreachable. Only a scroll view carries a content-sized virtual
    size, which is also where the arrow / page / home / end bindings live.
    """

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("escape", "close", "Close", priority=True)
    ]

    def action_close(self) -> None:
        self.app.close_diff()  # type: ignore[attr-defined]


class DiffPane(Static):
    """Session-scoped unified diff for the sidebar; scrolled by DiffScroll."""

    def __init__(self) -> None:
        super().__init__("No session file changes", id="workspace-diff")
        self._diff_text = ""
        self._stats: tuple[tuple[str, int, int], ...] = ()
        self._showing_error = False

    def show_diff(self, diff_text: str, stats: list[tuple[str, int, int]]) -> None:
        """Replace the view only when its semantic content changed."""
        frozen = tuple(stats)
        # ``_diff_text``/``_stats`` describe the last *good* reading, not what
        # is on screen: ``show_error`` paints a banner over them and leaves
        # them alone. Skipping the repaint on an unchanged reading would then
        # strand that banner — and since ``report()`` is memoised per
        # fingerprint, "unchanged" is the normal state of a polled session, so
        # one transient failure would leave it up until the tree moves again.
        if (
            not self._showing_error
            and diff_text == self._diff_text
            and frozen == self._stats
        ):
            return
        self._showing_error = False
        self._diff_text = diff_text
        self._stats = frozen
        if not diff_text:
            self.update(Text("No session file changes", style=self._style("muted")))
            return
        files = len(stats)
        added = sum(item[1] for item in stats)
        removed = sum(item[2] for item in stats)
        body = Text()
        body.append(f"{files} file{'s' if files != 1 else ''} changed  ", style=self._style("text", bold=True))
        body.append(f"+{added}", style=self._style("ok", bold=True))
        body.append("  ")
        body.append(f"-{removed}\n\n", style=self._style("err", bold=True))
        body.append_text(diff_to_text(diff_text, max_lines=2_000, theme=active_theme(self)))
        self.update(body)

    def show_error(self, detail: str) -> None:
        """Say the diff could not be read, keeping the last body below it.

        Rendering a failed read as "No session file changes" would tell a user
        whose session rewrote half the tree that nothing happened.
        """
        body = Text()
        body.append(
            f"⚠ could not read session changes — showing the last successful "
            f"reading\n{detail}\n\n",
            style=self._style("err", bold=True),
        )
        if self._diff_text:
            body.append_text(
                diff_to_text(self._diff_text, max_lines=2_000, theme=active_theme(self))
            )
        # ``_diff_text``/``_stats`` are left untouched: they still describe the
        # last good reading, which is what the body below the banner shows.
        # ``_showing_error`` is what forces the next good read to repaint over
        # the banner even when that reading is identical.
        self._showing_error = True
        self.update(body)

    def refresh_theme(self) -> None:
        diff_text, stats = self._diff_text, list(self._stats)
        self._diff_text = ""
        self._stats = ()
        self._showing_error = False
        self.show_diff(diff_text, stats)

    def _style(self, role: str, *, bold: bool = False) -> str:
        return rich_style(active_theme(self), role, bold=bold)


class DeliverablesPane(OptionList):
    """Keyboard-selectable files produced by the current session."""

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("space", "preview", "Preview", priority=True),
        Binding("escape", "close", "Close", priority=True),
    ]

    def __init__(self) -> None:
        super().__init__(id="deliverables")
        self._files: list[Path] = []

    def show_files(self, root: Path, files: list[Path]) -> None:
        """Replace the list while preserving the highlighted file if possible."""
        selected = self.selected_path
        self.clear_options()
        self._files = list(files)
        if not files:
            self.add_option(Option("No deliverables yet", disabled=True))
            return
        for index, path in enumerate(files):
            relative = path.relative_to(root)
            size = path.stat().st_size
            self.add_option(Option(f"{relative}  ({size:,} B)", id=f"file-{index}"))
        if selected in self._files:
            self.highlighted = self._files.index(selected)
        else:
            self.highlighted = 0

    @property
    def selected_path(self) -> Path | None:
        index = self.highlighted
        if index is None or not 0 <= index < len(self._files):
            return None
        return self._files[index]

    def action_preview(self) -> None:
        path = self.selected_path
        if path is not None:
            self.app.open_deliverable_preview(path)  # type: ignore[attr-defined]

    def action_close(self) -> None:
        self.app.close_deliverables()  # type: ignore[attr-defined]

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        event.stop()
        self.action_preview()


__all__ = [
    "ActivityPane", "ActivityRecord", "ActivityState", "DeliverablesPane",
    "DiffPane", "DiffScroll", "StatusBar", "TailScroll", "TodoPane",
    "TranscriptView",
]
