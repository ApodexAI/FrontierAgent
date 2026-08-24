"""``FrontierAgentApp`` — the full-screen Textual front end for FrontierAgent.

The app is the foreground; each agent run is a Textual *worker* on the same
asyncio loop, so the observers (running inside that worker) drive widgets and
open the approval modal directly. The engine is untouched: we only swap
``session.r`` for a :class:`TuiSink` and ``session.approver`` for a
:class:`TuiApprover`, and set ``session.tui_mode`` so the stdin steer-reader is
not attached (the input box feeds steering instead).
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import json
import logging
import os
import shlex
import time
from collections.abc import Callable
from os.path import commonprefix
from pathlib import Path
from typing import Any, ClassVar

from textual import events, on, work
from textual.app import App, ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Horizontal, Vertical
from textual.widgets import Button, Collapsible, Input, Static, Tab, Tabs
from textual.widgets._collapsible import CollapsibleTitle

from apodex.tui.ime import widen_escape_sequence_limit
from apodex.tui.messages import Render
from apodex.tui.screens import (
    BINARY_PREVIEWABLE_SUFFIXES,
    PREVIEWABLE_SUFFIXES,
    ActivityDetailScreen,
    ApprovalScreen,
    CommandScreen,
    ContextScreen,
    FilePreviewScreen,
    HelpScreen,
    ModelScreen,
    ResumeScreen,
    SettingsScreen,
    ThemeScreen,
    WorkflowScreen,
)
from apodex.tui.sink import TuiApprover, TuiSink
from apodex.tui.state import TuiPresentationState
from apodex.tui.themes import THEME_PICKER_NAMES, TUI_THEME_NAMES, register_themes
from apodex.tui.widgets import (
    ActivityPane,
    ActivityRecord,
    ActivityState,
    DeliverablesPane,
    DiffPane,
    DiffScroll,
    StatusBar,
    TailScroll,
    TodoPane,
    TranscriptView,
)
from frontier_agent.core.messages import Message

logger = logging.getLogger(__name__)


_COMMANDS = (
    ("/help", "shortcuts and interaction help"),
    ("/mode", "switch coding, research, React, or Agent Team workflow"),
    ("/model", "select a model"),
    ("/settings", "open theme and workflow settings"),
    ("/cwd", "show or change working directory"),
    ("/clear", "clear conversation context and plan"),
    ("/new", "save this session and start a fresh one"),
    ("/fork", "branch the current context into a new session"),
    ("/sessions", "list recent saved sessions"),
    ("/rename", "give the current session a readable name"),
    ("/plan", "toggle plan mode"),
    ("/revert", "undo session file changes"),
    ("/compact", "summarize earlier turns"),
    ("/cost", "show token and context usage"),
    ("/context", "visualize current context usage"),
    ("/config", "show safe local runtime configuration"),
    ("/init", "write or improve AGENTS.md"),
    ("/resume", "continue a saved session"),
    ("/log", "show local trace path"),
    ("/auto", "toggle auto-approval"),
    ("/bypass", "bypass all permission checks"),
    ("/autome", "auto-approve for me (docker / trusted env)"),
    ("/verbose", "toggle thinking details"),
    ("/filter", "show all, thinking, tools, errors, or report"),
    ("/find", "search the visible transcript"),
    ("/report", "jump to the final report"),
    ("/copy", "copy the final report"),
    ("/attach", "attach a workspace file or directory"),
    ("/attachments", "list files attached to this session"),
    ("/detach", "remove a session attachment"),
    ("/paste", "attach files or an image from the macOS clipboard"),
    ("/theme", "switch visual theme"),
    ("/workflow", "select React or Agent Team workflow"),
    ("/exit", "leave apodex"),
)
_TRANSCRIPT_VIEW_IDLE = "View: all · /filter thinking|tools|errors|report · /find <text>"
_COMMAND_NAMES = frozenset(command for command, _ in _COMMANDS)
_COMMANDS_WITH_ARGUMENTS = frozenset({
    "/mode", "/model", "/cwd", "/theme", "/workflow", "/filter", "/find",
    "/attach", "/detach",
    "/rename",
})
_WORKSPACE_SEARCH_LIMIT = 20_000
_PASTE_DEDUP_WINDOW_SECONDS = 0.35
_PASTE_INLINE_CHAR_LIMIT = 1_000
# How long an ``@`` file index stays usable before the next mention refreshes
# it in the background. Long enough that a burst of mentions walks the tree
# once, short enough that files the agent just wrote show up.
_WORKSPACE_INDEX_TTL_SECONDS = 20.0
_WORKSPACE_SEARCH_SKIP_DIRS = frozenset({
    ".apodex", ".git", ".hg", ".mypy_cache", ".pytest_cache", ".ruff_cache",
    ".svn", ".tox", ".venv", "__pycache__", "build", "dist", "node_modules",
    "target", "venv",
})


def _is_slash_command(value: str) -> bool:
    """Whether ``value`` is a slash command rather than text that starts with "/".

    ``@`` completion is deliberately off inside commands: ``/attach`` takes a
    raw path, so offering a mention there would corrupt the argument. But an
    ordinary message can also begin with "/" — ``/outputs/report.md summarise
    @notes`` — and that one must keep file completion, so the leading "/" alone
    is not enough; the first word has to name a real command.
    """
    if not value.startswith("/"):
        return False
    head, separator, _rest = value.partition(" ")
    return not separator or head in _COMMAND_NAMES


def _shared_completion(candidates: list[str], fragment: str) -> str:
    """Longest prefix every candidate shares — never shorter than ``fragment``.

    ``@`` matching is case-insensitive, so the shared prefix has to be measured
    on the folded spellings. Measuring it on the raw ones returns "" for a set
    like ``["README.md", "requirements.txt"]``, which replaced a typed ``@re``
    with a bare ``@`` — deleting the user's input instead of completing it.
    """
    if not candidates:
        return fragment
    shared = commonprefix([name.casefold() for name in candidates])
    if len(shared) <= len(fragment):
        return fragment
    # Case folding can change length (``ß`` → ``ss``), so the offset is only
    # trustworthy once it round-trips against every candidate.
    completed = candidates[0][:len(shared)]
    folded = completed.casefold()
    if not all(name.casefold().startswith(folded) for name in candidates):
        return fragment
    return completed


def _walk_workspace_files(root: Path, limit: int) -> tuple[str, ...]:
    """Index files under ``root`` for ``@`` completion. Runs off the UI thread.

    ``os.scandir`` rather than ``os.walk`` + ``Path.is_symlink``: the dirent
    already carries the entry type, so skipping links costs no extra ``lstat``
    per file. That is the difference between a large checkout indexing in
    milliseconds and the indexer spending a syscall on every file twice.
    """
    # ``rstrip`` rather than ``len(str(root)) + 1``: the filesystem root prints
    # as "/", which already carries its separator, so the naive +1 ate the
    # first character of every indexed path under ``--cwd /``.
    base = str(root).rstrip(os.sep)
    prefix_length = len(base) + 1
    found: list[str] = []
    stack: list[str] = [str(root)]
    while stack and len(found) < limit:
        try:
            entries = list(os.scandir(stack.pop()))
        except OSError:
            continue
        for entry in entries:
            if len(found) >= limit:
                break
            try:
                if entry.is_symlink():
                    continue
                if entry.is_dir(follow_symlinks=False):
                    if entry.name not in _WORKSPACE_SEARCH_SKIP_DIRS:
                        stack.append(entry.path)
                    continue
                if not entry.is_file(follow_symlinks=False):
                    continue
            except OSError:
                continue
            found.append(entry.path[prefix_length:].replace(os.sep, "/"))
    return tuple(found)


class PromptInput(Input):
    """Prompt that routes paste through attachment and multiline handling."""

    def _on_paste(self, event: events.Paste) -> None:
        if event.text:
            self.app.paste_clipboard_text(event.text)  # type: ignore[attr-defined]
        else:
            # Some macOS terminals represent Cmd-V on a non-text clipboard as
            # an empty bracketed paste. They cannot carry the image bytes, but
            # the host bridge can now read NSPasteboard explicitly. Terminals
            # that swallow image Cmd-V entirely still require Ctrl-V or /paste.
            self.app.action_paste_clipboard()  # type: ignore[attr-defined]
        event.stop()


class FrontierAgentApp(App):
    """A Claude-Code-style TUI driving the FrontierAgent agent loop."""

    # Chrome is themed entirely through the tokens ``themes.register_themes``
    # pins from each palette: ``$text`` / ``$text-muted`` / ``$text-disabled``
    # are the three measured contrast tiers and ``$border`` the fitted line
    # colour. The title and status bars deliberately sit on ``$panel`` with
    # coloured *text* rather than a saturated fill — a full-width block of
    # ``$accent`` fought every light palette and forced Textual to guess a
    # readable foreground with ``color: auto``.
    CSS = """
    /* Screen carries the tier so unstyled text anywhere inherits the fitted
       foreground; without it Textual falls back to pure white/black, which is
       brighter than any palette intends. */
    Screen { background: $background; color: $text; }
    #topbar { height: 1; background: $panel; }
    #title {
        width: 1fr;
        height: 1; background: $panel; color: $primary; text-style: bold;
        padding: 0 1; text-wrap: nowrap; overflow-x: hidden;
    }
    /* The menu was the one control on screen with no affordance: muted text on
       the bar's own background read as a label, not a button. It now sits on a
       raised surface in the accent colour, and inverts on hover/focus. */
    #menu-button {
        width: 16; min-width: 16; height: 1; min-height: 1; padding: 0 1;
        margin-left: 1; border: none;
        background: $surface; color: $primary; text-style: bold;
    }
    #menu-button:focus, #menu-button:hover {
        background: $primary; color: $background; text-style: bold;
    }
    #main { height: 1fr; }
    #transcript-column { width: 3fr; height: 1fr; }
    #transcript-view {
        height: 1; padding: 0 2; color: $text-muted; background: $background;
        text-wrap: nowrap; overflow-x: hidden;
    }
    #transcript { width: 1fr; height: 1fr; padding: 0 1; background: $background; color: $text; }
    /* The logo is laid out to the pane's exact width; wrapping it would fold
       the peaks back under themselves, so clip instead. */
    .startup-logo {
        height: auto; padding: 1 0 0 1; margin-bottom: 1;
        text-wrap: nowrap; overflow-x: hidden;
    }
    #sidebar {
        width: 44; border-left: solid $border; padding: 0 1;
        background: $background; color: $text;
    }
    /* One cell, not Textual's default two. At the sidebar's width a 2-cell
       gutter is a visible slab down the pane, and the transcript's was the
       widest single element on screen. */
    #transcript, #activity, #todos-box, #deliverables, #sidebar-diff {
        scrollbar-size-vertical: 1;
    }
    #sidebar-tabs {
        height: 3; background: $background;
    }
    #sidebar-tabs Tab { color: $text-muted; background: $background; }
    #sidebar-tabs Tab.-active { color: $primary; text-style: bold; }
    .sidebar-panel { height: 1fr; background: $background; }
    #todos-box {
        height: 1fr;
        background: $background; color: $text;
    }
    #activity {
        height: 1fr; background: $background; color: $text;
        border: none; padding: 0; overflow-x: hidden;
    }
    #activity:focus { outline: none; }
    #activity > .option-list--option-highlighted {
        color: $text; background: $surface;
    }
    #deliverables-box { height: 1fr; }
    #deliverables-location {
        height: auto; max-height: 2; color: $text-disabled; padding: 0 1;
        text-wrap: nowrap; overflow-x: hidden;
    }
    #deliverables {
        height: 1fr; color: $text-muted;
        background: $background; border: none; padding: 0;
    }
    #deliverables:focus { outline: none; }
    #deliverables > .option-list--option-highlighted {
        color: $text; background: $surface;
    }
    #sidebar-diff {
        height: 1fr; padding: 0 1; background: $background;
        overflow-y: auto; overflow-x: auto;
    }
    #sidebar-diff:focus { outline: none; }
    /* The diff is one tall renderable; the scroll host above owns the
       overflow, so the pane itself must size to its content. */
    #workspace-diff { height: auto; width: auto; color: $text-muted; }
    .block { margin-bottom: 0; min-height: 0; height: auto; }
    .process-group {
        height: auto; margin: 0; padding: 0;
        border: round $border; background: $background;
    }
    .process-group > CollapsibleTitle {
        padding: 0 1; color: $text; background: $panel; text-style: bold;
    }
    .process-group > CollapsibleTitle:hover,
    .process-group > CollapsibleTitle:focus { color: $primary; background: $surface; }
    .process-group > Contents { padding: 0; }
    .process-body { min-height: 0; height: auto; padding: 0 1; }
    /* Transcript hierarchy: reasoning recedes, tools form compact operational
       rows, and answer prose keeps the strongest reading contrast. */
    .thinking-block {
        height: auto; margin-bottom: 0; padding: 0;
        background: $background; border: none;
    }
    .thinking-block CollapsibleTitle {
        padding: 0; color: $text-muted; background: $background;
        text-style: bold;
    }
    .thinking-block CollapsibleTitle:hover,
    .thinking-block CollapsibleTitle:focus {
        color: $text; background: $surface;
    }
    .thinking-block Contents { padding: 0 0 0 2; }
    .thinking-block-body {
        height: auto; padding: 0 1; color: $text-disabled;
        border-left: solid $border;
    }
    .tool-call {
        padding: 0 1; color: $text-muted; background: $surface;
        border-left: solid $primary;
    }
    .tool-result {
        height: auto; margin: 0; padding: 0;
        background: $surface; border: none;
    }
    .tool-result CollapsibleTitle {
        width: 1fr; padding: 0 1; color: $success;
        background: $surface; text-style: bold;
    }
    .tool-result-error CollapsibleTitle { color: $error; }
    .tool-result CollapsibleTitle:hover,
    .tool-result CollapsibleTitle:focus { background: $panel; }
    .tool-result Contents { padding: 0 1 1 2; }
    .tool-result-body { height: auto; color: $text-muted; }
    /* The live fan-in card. It replaces itself in place rather than
       appending, so it is given a little breathing room and a tinted bed to
       read as one standing panel instead of another transcript block. Its
       rail is ``outer`` rather than ``thick``; see ``.review-active``. */
    .subagent-status {
        height: auto; padding: 0 1; margin: 1 0 0 0;
        color: $text-muted; background: $surface;
        border-left: outer $primary;
    }
    .assistant-content { color: $text; padding: 0 1; }
    /* One assistant message per block. Consecutive messages are adjacent — the
       tool rows between them are folded into the process group — so a one-row
       dashed rule marks the seam that the model's own blank lines used to.
       Alpha-blended toward the bed: this seam recurs every few lines, so it
       reads below the rules that frame a user prompt or the final report. */
    .assistant-continued { border-top: dashed $border 60%; }
    .report-heading {
        color: $success; text-style: bold; padding: 1 1 0 1; margin-bottom: 0;
        border-top: solid $border;
    }
    .final-report {
        color: $text; padding: 0 2; max-width: 110;
        border-left: solid $primary;
    }
    .report-footer { color: $text-muted; padding-left: 1; }
    /* A user prompt is also the semantic start of a turn. The dashed rule
       separates conversations without wrapping every message in a heavy card;
       the accent rail makes the prompt itself easy to find while scanning.
       ``outer`` rather than ``thick``; see ``.review-active``. */
    .turn-start {
        height: auto; margin-top: 1; padding: 0 1;
        color: $primary; background: $surface;
        border-top: dashed $border; border-left: outer $primary;
    }
    .user-message, .history-user { color: $primary; }
    .transcript-error { color: $error; }
    /* An outline, not a border: the review cursor is drawn over the block and
       must not reflow the transcript under the user as it moves. It therefore
       overwrites whatever rail the block already has, so no rail above may use
       ``thick $primary`` — the cursor would be invisible on exactly the blocks
       (prompts, the fan-in card) a reader is most likely to jump to. */
    .review-active { outline-left: thick $primary; }
    .transcript-pruned { height: auto; }
    #status {
        height: 1; background: $panel; color: $text-muted; padding: 0 1;
        text-wrap: nowrap; overflow-x: hidden;
    }
    #command-hint {
        display: none; height: auto; max-height: 2; color: $text-muted; padding: 0 1;
    }
    #prompt { height: 3; border: tall $border; background: $surface; color: $text; }
    #prompt:focus { border: tall $primary; }
    #attachments-bar {
        display: none; height: auto; max-height: 3; padding: 0 2;
        color: $text-muted; background: $surface;
    }
    """

    # Override the default ctrl+c-quits: while an agent runs, ctrl+c cancels the
    # run (state is checkpointed per turn); idle, it quits.
    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("ctrl+c", "interrupt", "Interrupt / Quit", priority=True),
        Binding("f1", "show_help", "Help"),
        Binding("f2", "settings", "Menu"),
        Binding("ctrl+p", "command_palette", "Commands"),
        Binding("ctrl+b", "toggle_sidebar", "Sidebar"),
        Binding("ctrl+o", "toggle_deliverables", "Deliverables"),
        Binding("ctrl+tab", "next_sidebar_tab", "Next sidebar tab", show=False),
        Binding(
            "ctrl+shift+tab", "previous_sidebar_tab", "Previous sidebar tab",
            show=False,
        ),
        Binding("alt+j", "review_next", "Next block", show=False),
        Binding("alt+k", "review_previous", "Previous block", show=False),
        Binding("alt+enter", "review_toggle", "Expand / collapse", show=False),
        Binding("ctrl+g", "jump_report", "Final report", show=False),
        Binding("ctrl+y", "copy_report", "Copy report", show=False),
        Binding("ctrl+v", "paste_clipboard", "Paste clipboard attachment", priority=True),
        Binding("up", "history_previous", "Previous input", show=False),
        Binding("down", "history_next", "Next input", show=False),
    ]

    def __init__(
        self, session: Any, *, resumed: bool = False, initial_task: str | None = None,
        theme: str = "catppuccin",
    ) -> None:
        super().__init__()
        # Before any input is read: Textual's escape-sequence collector gives up
        # at 32 characters, which is shorter than the sequence a terminal sends
        # for a five-character input-method commit. See ``apodex.tui.ime``.
        widen_escape_sequence_limit()
        self.session = session
        self.presentation = TuiPresentationState()
        self.sink = TuiSink(self)
        self.busy = False
        self.usage: Any = None
        self._window = 0
        self._tools = 0
        self._resumed = resumed
        self._initial_task = initial_task
        register_themes(self)
        self._ui_theme = theme if theme in TUI_THEME_NAMES else "catppuccin"
        self._input_history: list[str] = []
        self._history_index: int | None = None
        self._history_draft = ""
        self._sidebar_user_visible = True
        self._terminal_width = 120
        self._deliverables_root = self._resolve_deliverables_root()
        self._sidebar_tab = "plan"
        self._sidebar_titles: dict[str, str] = {}
        self._workspace_diff_stats: list[tuple[str, int, int]] = []
        self._workspace_diff_text = ""
        self._workspace_diff_pending = False
        # Last diff-read failure, if any. Kept so the pane can say the view is
        # stale instead of impersonating a clean tree.
        self._workspace_diff_error = ""
        self._completed_workspace_pending = False
        self._completed_workspace_needs_followup = False
        self._transcript_view_line = _TRANSCRIPT_VIEW_IDLE
        self._workspace_file_cache_root: Path | None = None
        self._workspace_file_cache: tuple[str, ...] = ()
        self._workspace_file_cache_ready = False
        self._workspace_file_cache_at = 0.0
        self._workspace_index_pending = False
        self._workspace_mention_active = False
        self._last_terminal_paste: tuple[str, float] | None = None
        self._last_inserted_paste: tuple[str, float] | None = None
        self._pasted_text_blocks: dict[str, str] = {}
        self._pasted_text_markers: dict[str, str] = {}
        self._pasted_text_sequence = 0

    def _resolve_deliverables_root(self) -> Path:
        """Return the stable output mount, falling back to a local TUI folder."""
        configured = os.environ.get("FRONTIER_AGENT_OUTPUTS_DIR", "").strip()
        return Path(configured) if configured else Path(self.session.cwd) / ".apodex" / "outputs"

    def _deliverables_location_lines(self) -> list[str]:
        """Show deliverables plus the separate run-private work location."""
        host = os.environ.get("APODEX_HOST_OUTPUTS_DIR", "").strip()
        if host:
            lines = [f"Host: {host}", f"Agent: {self._deliverables_root}"]
        else:
            lines = [str(self._deliverables_root)]
        work = os.environ.get("APODEX_HOST_WORKSPACE_DIR", "").strip()
        if work:
            lines.append(f"Work: {work} (intermediate)")
        return lines

    def _title_text(self) -> str:
        name = str(getattr(self.session, "session_name", "") or "").strip()
        label = f"  ·  {name}" if name else ""
        return f"FrontierAgent  ·  {self.session.mode}{label}  ·  {self.session.cwd}"

    # ── layout ────────────────────────────────────────────────────────────
    def compose(self) -> ComposeResult:
        with Horizontal(id="topbar"):
            yield Static(
                self._title_text(), id="title",
            )
            yield Button("☰ Menu · F2", id="menu-button", compact=True, flat=True)
        with Horizontal(id="main"):
            with Vertical(id="transcript-column"):
                yield Static(_TRANSCRIPT_VIEW_IDLE, id="transcript-view")
                yield TranscriptView()
            with Vertical(id="sidebar"):
                yield Tabs(
                    Tab("Plan", id="sidebar-tab-plan"),
                    Tab("Activity", id="sidebar-tab-activity"),
                    Tab("Files", id="sidebar-tab-deliverables"),
                    Tab("Diff", id="sidebar-tab-diff"),
                    active="sidebar-tab-plan", id="sidebar-tabs",
                )
                with (
                    Vertical(id="sidebar-plan", classes="sidebar-panel"),
                    TailScroll(id="todos-box"),
                ):
                    yield TodoPane()
                with Vertical(id="sidebar-activity", classes="sidebar-panel"):
                    yield ActivityPane()
                with Vertical(
                    id="deliverables-box", classes="sidebar-panel",
                ):
                    yield Static("", id="deliverables-location")
                    yield DeliverablesPane()
                with DiffScroll(id="sidebar-diff", classes="sidebar-panel"):
                    yield DiffPane()
        yield StatusBar()
        yield Static("", id="command-hint")
        yield Static("", id="attachments-bar")
        yield PromptInput(
            placeholder="Type a task · Ctrl-V paste attachment · /help for commands",
            id="prompt",
        )

    async def on_mount(self) -> None:
        # Cache widget refs the sink/helpers use.
        self.transcript = self.query_one(TranscriptView)
        self.activity = self.query_one(ActivityPane)
        self.todos_pane = self.query_one(TodoPane)
        self.status = self.query_one(StatusBar)
        self.deliverables = self.query_one(DeliverablesPane)
        self.diff_pane = self.query_one(DiffPane)
        self.diff_scroll = self.query_one(DiffScroll)
        self.diff_tab = self.query_one("#sidebar-tab-diff", Tab)
        self.deliverables_location = self.query_one("#deliverables-location", Static)
        self.deliverables_box = self.query_one("#deliverables-box", Vertical)
        self.attachments_bar = self.query_one("#attachments-bar", Static)
        self.query_one("#sidebar-activity").display = False
        self.deliverables_box.display = False
        self.query_one("#sidebar-diff").display = False
        self.diff_tab.display = False
        # TailScroll anchors itself on mount; see widgets.TailScroll.
        self.todos_box = self.query_one("#todos-box", TailScroll)
        self._terminal_width = self.size.width
        self._apply_theme(self._ui_theme)

        # Redirect the session at the TUI: widget sink + modal approver + the
        # tui_mode flag (no stdin steer-reader). Carry over auto-approve / model.
        self.session.r = self.sink
        self.session.tui_mode = True
        self.sink.restore_state(getattr(self.session, "tui_state", {}))
        self.session.approver = TuiApprover(
            self, auto_approve=self.session.approver.auto_approve,
        )
        self.set_usage(self.session.usage, self.session.cfg.context_window)

        self.query_one("#prompt", Input).focus()
        self.set_interval(0.4, self._refresh_status)
        self.set_interval(1.0, self._request_workspace_diff)
        self._request_workspace_diff()
        self._refresh_chrome()
        self._refresh_attachments()
        # Warm the ``@`` file index now, in the background, so the first mention
        # has candidates to offer instead of waiting on a cold walk.
        self._workspace_files()

        # Before the replay, so a resumed session's history reads as history
        # under the logo rather than above it.
        self.sink.logo()
        if self._resumed:
            await self._replay_session_history()
        welcome: Callable[[], str] | None = getattr(
            self.session, "demo_welcome", None,
        )
        self.sink.note(
            welcome() if callable(welcome) else (
                "Ready for a local BYOK demo. Type a task below. "
                "/config settings · F1 help · Ctrl-P commands · Ctrl-C interrupt."
            )
        )
        if getattr(self.session.plan_state, "active", False):
            self.sink.note("▤ plan mode ON — edits locked until you approve a plan (/plan to toggle)")
        if self._resumed:
            self.sink.note(
                f"resumed session {self.session.session_id} "
                f"({len(self.session.history)} prior messages)"
            )
        if self._initial_task:
            self._remember_input(self._initial_task)
            self.sink.echo_user(self._initial_task)
            self._run_agent(self._initial_task)

    def on_resize(self, event: events.Resize) -> None:
        """Keep the transcript usable in narrow local and SSH terminals."""
        try:
            self._terminal_width = event.size.width
            self._sync_sidebar()
            if hasattr(self, "status"):
                # The activity pane re-fits and re-follows itself on its own
                # Resize, which Textual delivers after layout.
                self._refresh_status()
        except Exception:
            logger.exception("failed to update responsive TUI layout")

    def _sync_sidebar(self) -> None:
        self.query_one("#sidebar").display = (
            self._sidebar_user_visible and self._terminal_width >= 100
        )

    # ── sink helpers (called on the message pump) ─────────────────────────
    def set_usage(self, usage: Any, window: int) -> None:
        self.usage = usage
        self._window = window

    def start_activity(self, call_id: str, name: str, summary: str) -> None:
        self.activity.start(call_id, name, summary)
        self._tools += 1

    def finish_activity(
        self, call_id: str, name: str, *, is_error: bool, ms: int = 0,
        state: ActivityState | None = None,
    ) -> None:
        self.activity.finish(call_id, name, is_error=is_error, ms=ms, state=state)

    def finish_active_activities(self, state: ActivityState) -> None:
        self.activity.finish_active(state)

    def clear_activity(self, *, reset_checkpoint: bool = True) -> None:
        self.activity.clear_records()
        if reset_checkpoint:
            self.sink.clear_saved_activity()
        self._tools = 0

    def show_todos(self, items: list, *, force: bool = False) -> None:
        self.todos_pane.show_todos(items, force=force)
        # The board is a Static inside the scroll box, so its content changes
        # through ``update()`` — there is no mount to hook, and the box has to be
        # told to re-follow the newest task itself.
        self.todos_box.follow_tail()

    def clear_todos(self) -> None:
        self.todos_pane.clear()
        self.todos_box.follow_tail()

    def apply_task_board_operation(self, name: str, args: dict) -> None:
        self.todos_pane.apply_task_board_operation(name, args)
        self.todos_box.follow_tail()

    async def on_render(self, message: Render) -> None:
        """Apply one queued UI mutation, in order (see messages.Render)."""
        try:
            result = message.fn(self)
            if inspect.isawaitable(result):
                await result
        except Exception:
            logger.exception("TUI render mutation failed")

    def _refresh_sidebar_titles(self) -> None:
        """Keep tab labels' off-screen counts current.

        Driven from the status tick rather than wired into every mutation, so no
        code path can forget it. Cached because ``Static.update`` refreshes
        unconditionally and this runs 2.5×/s.
        """
        for widget_id, label, summary in (
            ("#sidebar-tab-plan", "Plan", self.todos_pane.summary()),
            ("#sidebar-tab-activity", "Activity", self.activity.summary()),
            ("#sidebar-tab-diff", "Diff", str(len(self._workspace_diff_stats)) if self._workspace_diff_stats else ""),
        ):
            heading = f"{label}  {summary}" if summary else label
            if self._sidebar_titles.get(widget_id) != heading:
                self._sidebar_titles[widget_id] = heading
                with contextlib.suppress(Exception):
                    self.query_one(widget_id, Tab).label = heading

    def _refresh_status(self) -> None:
        self.sink.tick()
        self.transcript.refresh_process()
        self._refresh_transcript_view()
        self.activity.refresh_running()
        self._refresh_sidebar_titles()
        ctx = ""
        if self.usage is not None and self._window:
            status = getattr(self.usage, "context_status", None)
            if callable(status):
                ctx = str(status(self._window))
            else:
                pct = self.usage.context_pct_left(self._window)
                if pct is not None:
                    ctx = f"{pct}% left"
        self.status.show(
            presentation=self.presentation, mode=self.session.mode,
            ctx=ctx, tools=self._tools,
            width=self._terminal_width,
            changes=(
                len(self._workspace_diff_stats),
                sum(item[1] for item in self._workspace_diff_stats),
                sum(item[2] for item in self._workspace_diff_stats),
            ),
        )

    def _request_workspace_diff(self) -> None:
        """Refresh journal-backed diff data without blocking the UI thread."""
        if self._workspace_diff_pending:
            return
        journal = getattr(self.session, "journal", None)
        if journal is None:
            self._store_workspace_diff([], "")
            return
        self._workspace_diff_pending = True
        self._read_workspace_diff(journal)

    @work(exclusive=True, group="workspace-diff", thread=True)
    def _read_workspace_diff(self, journal: Any) -> None:
        stats: list[tuple[str, int, int]] = []
        diff_text = ""
        failure = ""
        try:
            # One pass: the header counts and the hunks they describe have to
            # come from the same read, or a file written between two passes
            # makes them disagree.
            stats, diff_text = journal.report()
        except Exception as exc:
            logger.exception("failed to build workspace diff")
            failure = f"{type(exc).__name__}: {exc}"
        finally:
            try:
                self.call_from_thread(
                    self._store_workspace_diff, stats, diff_text, journal,
                    failure,
                )
            except Exception:
                self._workspace_diff_pending = False

    def _store_workspace_diff(
        self, stats: list[tuple[str, int, int]], diff_text: str,
        journal: Any | None = None, failure: str = "",
    ) -> None:
        self._workspace_diff_pending = False
        if failure:
            # An empty result from a failed read is indistinguishable from a
            # clean tree, which would hide the tab, drop the status counters
            # and tell the user nothing changed on disk. Keep the last good
            # snapshot and say the reading is stale instead.
            self._workspace_diff_error = failure
            self.diff_pane.show_error(failure)
            self._refresh_sidebar_titles()
            if self._completed_workspace_pending:
                # Resolve the reveal off the last good snapshot rather than
                # leaving the sidebar waiting on a read that will not arrive.
                self._completed_workspace_pending = False
                self._completed_workspace_needs_followup = False
                self._show_sidebar_tab(
                    "diff" if self._workspace_diff_stats else "deliverables",
                    focus=self.query_one("#sidebar").display,
                )
            return
        self._workspace_diff_error = ""
        if journal is not None and journal is not getattr(self.session, "journal", None):
            # A /new, /cwd or resumed session replaced the journal while this
            # worker was reading. Never flash the previous session's changes.
            self._request_workspace_diff()
            return
        self._workspace_diff_stats = list(stats)
        self._workspace_diff_text = diff_text
        self.diff_pane.show_diff(diff_text, stats)
        # ``App.query_one`` searches the active screen. A settings/approval
        # modal may be active when the periodic diff poll completes, so keep
        # using the main-screen widget captured during mount.
        self.diff_tab.display = bool(stats)
        self._refresh_sidebar_titles()
        if self._completed_workspace_pending:
            if self._completed_workspace_needs_followup:
                # Completion may have arrived while an older periodic read was
                # in flight. Confirm once more so the last tool's writes decide
                # which tab is revealed.
                self._completed_workspace_needs_followup = False
                self._request_workspace_diff()
                return
            self._completed_workspace_pending = False
            target = "diff" if stats else "deliverables"
            sidebar_visible = self.query_one("#sidebar").display
            self._show_sidebar_tab(target, focus=sidebar_visible)
        elif not stats and self._sidebar_tab == "diff":
            self._show_sidebar_tab("deliverables", focus=False)

    def _refresh_chrome(self) -> None:
        """Synchronize the title, sidebar and status with mutable session state."""
        self.query_one("#title", Static).update(
            self._title_text()
        )
        self.set_usage(self.session.usage, self.session.cfg.context_window)
        try:
            from apodex.todo import get_todos
            self.show_todos(get_todos())
        except Exception:
            logger.exception("failed to restore TUI plan state")
        self._refresh_status()
        self._refresh_attachments()

    def _refresh_attachments(self) -> None:
        """Render compact attachment chips without exposing original host paths."""
        bar = getattr(self, "attachments_bar", None)
        manager = getattr(self.session, "attachments", None)
        if bar is None or manager is None:
            return
        try:
            items = manager.list()
        except Exception:
            logger.exception("failed to list TUI attachments")
            items = []
        if not items:
            bar.display = False
            bar.update("")
            return
        from apodex.attachments import format_size
        labels = [f"[{item.relative_path} · {format_size(item.size)}]" for item in items]
        bar.update("Attachments  " + "  ".join(labels))
        bar.display = True

    def _refresh_transcript_view(self) -> None:
        """Mirror the transcript's own filter state into the view bar.

        The transcript owns the state — it drops a filter by itself when a new
        run starts — so the bar is derived from it on every tick rather than
        written once by the command that set it.
        """
        mode = self.transcript.filter_mode
        if mode == "all":
            line = _TRANSCRIPT_VIEW_IDLE
        elif mode == "search":
            line = (f"Search: {self.transcript.filter_query} · "
                    f"{self.transcript.filter_matches} matches · /filter all to reset")
        else:
            line = (f"View: {mode} · {self.transcript.filter_matches} blocks · "
                    "/filter all to reset")
        if line != self._transcript_view_line:
            self._transcript_view_line = line
            self.query_one("#transcript-view", Static).update(line)

    # ``Toggled`` is the base of ``Collapsed``/``Expanded``, and the naming
    # convention only dispatches on the concrete class — the decorator matches
    # the whole message MRO, so one handler covers both directions.
    @on(Collapsible.Toggled)
    def _on_transcript_toggled(self, event: Collapsible.Toggled) -> None:
        """Keep the prompt focused when a transcript section is clicked open.

        ``CollapsibleTitle`` is focusable, so clicking one silently moved
        keyboard focus out of the input and the next thing typed went nowhere.
        Focus is only pulled back when a title actually holds it, so a
        background auto-collapse cannot yank it from elsewhere.
        """
        if isinstance(self.focused, CollapsibleTitle):
            self._focus_prompt()

    def _focus_prompt(self) -> None:
        """Return keyboard control after workers and modal screens finish."""
        if not isinstance(self.screen, ApprovalScreen):
            self.query_one("#prompt", Input).focus()

    def focus_prompt(self) -> None:
        """Public focus hook used by selectable sidebar panes."""
        self._focus_prompt()

    # ── input routing ─────────────────────────────────────────────────────
    def on_input_submitted(self, event: Input.Submitted) -> None:
        text = self._expand_pasted_text(event.value).strip()
        event.input.value = ""
        self._pasted_text_blocks.clear()
        self._pasted_text_markers.clear()
        if not text:
            return
        self._remember_input(text)
        if self.busy:
            # While the agent works: typing steers it; slash commands wait.
            if text.startswith("/"):
                self.notify("busy — Ctrl-C to interrupt first", severity="warning")
            elif self.session._inbox is not None:
                # Enqueue through the inbox so a coordinator parked in
                # collect_reports is woken immediately; the observer injects
                # the line as the next main-agent user message.
                self.session._inbox.enqueue(text)
                self._refresh_status()
            else:
                self.notify("can't steer right now", severity="warning")
            return
        if text.startswith("/"):
            self._handle_slash(text)
        else:
            self.sink.echo_user(text)
            self._run_agent(text)

    def action_paste_clipboard(self) -> None:
        """Read Finder files or image data from the macOS Pasteboard."""
        self._paste_clipboard()

    def paste_clipboard_text(self, text: str) -> None:
        """Route bracketed terminal text through host-path detection."""
        now = time.monotonic()
        previous = self._last_terminal_paste
        if (
            previous is not None
            and previous[0] == text
            and now - previous[1] <= _PASTE_DEDUP_WINDOW_SECONDS
        ):
            logger.debug("ignored duplicate terminal paste event")
            return
        self._last_terminal_paste = (text, now)
        self._paste_clipboard(text)

    def _insert_pasted_text(self, prompt: Input, text: str) -> None:
        """Insert pasted text without losing newlines or flooding the prompt.

        ``Input`` is deliberately one line high.  Multiline and very large
        pastes therefore appear as a compact marker, while the exact original
        text is retained and expanded when the user submits the prompt.
        """
        normalized = text.replace("\r\n", "\n").replace("\r", "\n")
        if not normalized:
            return
        now = time.monotonic()
        previous = self._last_inserted_paste
        if (
            previous is not None
            and previous[0] == normalized
            and now - previous[1] <= _PASTE_DEDUP_WINDOW_SECONDS
        ):
            logger.debug("ignored duplicate clipboard text insertion")
            return
        shown = normalized
        if "\n" in normalized or len(normalized) > _PASTE_INLINE_CHAR_LIMIT:
            shown = self._fold_pasted_text(normalized)
        prompt.insert_text_at_cursor(shown)
        self._last_inserted_paste = (normalized, now)

    def _fold_pasted_text(self, text: str) -> str:
        """Store ``text`` and return its compact, editable prompt marker.

        Identical text always folds to the same marker.  Scrolling through
        prompt history re-folds the same entry on every keypress, and minting a
        fresh marker each time would retain another full copy of it until the
        next submit.
        """
        existing = self._pasted_text_markers.get(text)
        if existing is not None:
            return existing
        self._pasted_text_sequence += 1
        lines = text.count("\n") + 1
        marker = (
            f"[Pasted text #{self._pasted_text_sequence} · "
            f"{lines:,} lines · {len(text):,} chars]"
        )
        self._pasted_text_blocks[marker] = text
        self._pasted_text_markers[text] = marker
        return marker

    def _expand_pasted_text(self, value: str) -> str:
        """Replace intact paste markers with their exact multiline payloads."""
        expanded = value
        for marker, text in self._pasted_text_blocks.items():
            expanded = expanded.replace(marker, text)
        return expanded

    @work(exclusive=True, group="clipboard")
    async def _paste_clipboard(self, pasted_text: str | None = None) -> None:
        from apodex.clipboard import ClipboardError, paste_from_clipboard

        manager = getattr(self.session, "attachments", None)
        if manager is None:
            self.sink.error("clipboard attachments are unavailable in this session")
            return
        prompt = self.query_one("#prompt", Input)
        try:
            result = await asyncio.to_thread(
                paste_from_clipboard, manager, pasted_text=pasted_text,
            )
        except ClipboardError as exc:
            if pasted_text is not None:
                self._insert_pasted_text(prompt, pasted_text)
                self.notify(
                    f"attachment detection unavailable; pasted as text ({exc})",
                    severity="warning",
                )
            else:
                self.sink.error(str(exc))
            self._focus_prompt()
            return
        if result.kind == "attachments":
            self._refresh_attachments()
            names = ", ".join(result.attachments)
            self.sink.note(f"pasted attachment{'s' if len(result.attachments) != 1 else ''}: {names}")
            if self.busy and result.attachments:
                by_name = {
                    item.relative_path: item.agent_path for item in manager.list()
                }
                paths = [by_name[name] for name in result.attachments if name in by_name]
                prompt.insert_text_at_cursor(
                    "[Attached files: " + ", ".join(paths) + "] "
                )
        elif result.kind == "text":
            self._insert_pasted_text(prompt, result.text)
        else:
            self.notify(result.message or "clipboard is empty or unsupported", severity="warning")
        self._focus_prompt()

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id != "prompt":
            return
        value = event.value.strip()
        hint = self.query_one("#command-hint", Static)

        if _is_slash_command(value):
            if " " not in value:
                matches = [command for command, _ in _COMMANDS if command.startswith(value)]
                hint.update("Tab complete · " + "  ".join(matches[:6]))
                hint.display = bool(matches)
            else:
                hint.display = False
            self._workspace_mention_active = False
            return

        self._refresh_mention_hint(event.value)

    @staticmethod
    def _mention_token(value: str) -> str | None:
        """Return the ``@`` fragment the prompt's last word holds, else ``None``."""
        token = value[value.rfind(" ") + 1:]
        if not token.startswith("@") or '"' in token or "'" in token:
            return None
        return token[1:]

    def _refresh_mention_hint(self, value: str | None = None) -> None:
        """Render the ``@`` candidate hint for the prompt's current token."""
        prompt = self.query_one("#prompt", Input)
        hint = self.query_one("#command-hint", Static)
        fragment = self._mention_token(prompt.value if value is None else value)
        if fragment is None:
            self._workspace_mention_active = False
            hint.display = False
            return
        self._workspace_mention_active = True
        candidates = self._mention_candidates(fragment)
        if candidates:
            references = [
                f'@"{name}"' if " " in name else "@" + name
                for name in candidates[:4]
            ]
            hint.update("Files · Tab complete · " + "  ".join(references))
        elif not self._workspace_index_ready():
            # Say so rather than claiming the project is empty: the index lands
            # a moment later and this hint is rewritten from _store_workspace_index.
            hint.update(f"Indexing files under {self.session.cwd}…")
        elif fragment:
            hint.update(f"No files match @{fragment} under {self.session.cwd}")
        else:
            hint.update(f"No files found under {self.session.cwd}")
        hint.display = True

    def _workspace_root(self) -> Path | None:
        """The resolved ``--cwd`` to index, or ``None`` when it is unreadable."""
        try:
            return Path(self.session.cwd).expanduser().resolve(strict=True)
        except (OSError, RuntimeError):
            return None

    def _workspace_index_ready(self) -> bool:
        """Whether the cache already describes the current ``--cwd``."""
        root = self._workspace_root()
        if root is None:
            # Nothing to index, so "no files found" is the accurate wording.
            return True
        return self._workspace_file_cache_ready and self._workspace_file_cache_root == root

    def _workspace_files(self) -> tuple[str, ...]:
        """Return the current index of files under ``--cwd``, refreshing it lazily.

        The walk used to run inline here, and this is reached from
        ``on_input_changed`` — a synchronous Textual handler. On a large
        checkout that froze the whole TUI (keystrokes, streaming output and the
        status bar alike) for the length of the walk, once per ``@`` mention,
        because the cache was invalidated whenever a new mention started. The
        walk now runs in a worker thread and callers read whatever snapshot is
        current; a stale one is refreshed in the background rather than inline.
        """
        root = self._workspace_root()
        if root is None:
            return ()
        fresh = (
            self._workspace_file_cache_ready
            and self._workspace_file_cache_root == root
            and time.monotonic() - self._workspace_file_cache_at
            <= _WORKSPACE_INDEX_TTL_SECONDS
        )
        if not fresh and not self._workspace_index_pending:
            self._workspace_index_pending = True
            self._index_workspace_files(root)
        if self._workspace_file_cache_root != root:
            # A ``/cwd`` switch retires the previous project's paths outright —
            # completing them would reference files this session cannot reach.
            return ()
        return self._workspace_file_cache

    @work(exclusive=True, group="workspace-index", thread=True)
    def _index_workspace_files(self, root: Path) -> None:
        """Walk the project tree off the UI thread; see :meth:`_workspace_files`."""
        files: tuple[str, ...] = ()
        try:
            files = _walk_workspace_files(root, _WORKSPACE_SEARCH_LIMIT)
        except OSError:
            logger.exception("failed to index workspace files for @ completion")
        finally:
            # Publish even after a failed walk: leaving ``_workspace_index_pending``
            # set would wedge every later refresh behind one bad traversal.
            try:
                self.call_from_thread(self._store_workspace_index, root, files)
            except Exception:  # app already shutting down
                self._workspace_index_pending = False

    def _store_workspace_index(self, root: Path, files: tuple[str, ...]) -> None:
        """Adopt a finished index on the UI thread and repaint a live hint."""
        self._workspace_index_pending = False
        self._workspace_file_cache_root = root
        self._workspace_file_cache = files
        self._workspace_file_cache_ready = True
        self._workspace_file_cache_at = time.monotonic()
        if self._workspace_mention_active:
            self._refresh_mention_hint()

    @staticmethod
    def _mention_match_rank(name: str, fragment: str) -> int | None:
        """Rank basename/path prefix and substring matches for ``@`` search."""
        if not fragment:
            return 0
        folded_name = name.casefold()
        folded_fragment = fragment.casefold()
        basename = Path(folded_name).name
        if basename.startswith(folded_fragment):
            return 0
        if any(part.startswith(folded_fragment) for part in folded_name.split("/")):
            return 1
        if folded_fragment in basename:
            return 2
        if folded_fragment in folded_name:
            return 3
        return None

    def _mention_candidates(self, fragment: str) -> list[str]:
        """Search session attachments and workspace files for an ``@`` token."""
        names: list[tuple[str, int]] = [
            (name, 0) for name in self._attached_file_names()
        ]
        names.extend((name, 1) for name in self._workspace_files())

        ranked: list[tuple[int, int, int, int, str, str]] = []
        seen: set[str] = set()
        for name, source_rank in names:
            folded = name.casefold()
            if folded in seen:
                continue
            seen.add(folded)
            match_rank = self._mention_match_rank(name, fragment)
            if match_rank is None:
                continue
            ranked.append((
                match_rank,
                source_rank,
                name.count("/"),
                len(name),
                folded,
                name,
            ))
        ranked.sort()
        return [item[-1] for item in ranked[:100]]

    def _attached_file_names(self) -> list[str]:
        """List attachment names without letting a staging error break input."""
        manager = getattr(self.session, "attachments", None)
        if manager is not None:
            try:
                return [item.relative_path for item in manager.list()]
            except Exception:
                logger.exception("failed to list attachments for @ completion")
        return []

    def on_key(self, event: events.Key) -> None:
        """Complete slash commands and attached/workspace ``@`` references."""
        prompt = self.query_one("#prompt", Input)
        if event.key != "tab" or not prompt.has_focus:
            return
        raw_value = prompt.value
        value = raw_value.strip()
        if _is_slash_command(value):
            if " " in value:
                return
            matches = [command for command, _ in _COMMANDS if command.startswith(value)]
            if not matches:
                return
            completion = commonprefix(matches)
            if len(matches) == 1:
                completion = matches[0] + (" " if matches[0] in _COMMANDS_WITH_ARGUMENTS else "")
            prompt.value = completion
            prompt.cursor_position = len(completion)
            event.prevent_default()
            event.stop()
            return

        token_start = raw_value.rfind(" ") + 1
        token = raw_value[token_start:]
        if not token.startswith("@") or '"' in token or "'" in token:
            return
        fragment = token[1:]
        candidates = self._mention_candidates(fragment)
        if not candidates:
            return
        prefix_candidates = [
            name for name in candidates
            if name.casefold().startswith(fragment.casefold())
        ]
        attachment_prefix_candidates = [
            name for name in self._attached_file_names()
            if name.casefold().startswith(fragment.casefold())
        ]
        unique_completion = (
            len(attachment_prefix_candidates) == 1
            or len(candidates) == 1
            or len(prefix_candidates) == 1
        )
        if len(attachment_prefix_candidates) == 1:
            completed = attachment_prefix_candidates[0]
        elif len(prefix_candidates) == 1:
            completed = prefix_candidates[0]
        elif len(candidates) == 1:
            completed = candidates[0]
        else:
            completed = _shared_completion(prefix_candidates, fragment)
        reference = f'@"{completed}"' if " " in completed else "@" + completed
        if unique_completion:
            reference += " "
        prompt.value = raw_value[:token_start] + reference
        prompt.cursor_position = len(prompt.value)
        event.prevent_default()
        event.stop()

    def _remember_input(self, text: str) -> None:
        if not self._input_history or self._input_history[-1] != text:
            self._input_history.append(text)
            del self._input_history[:-100]
        self._history_index = None
        self._history_draft = ""

    def _move_history(self, delta: int) -> None:
        prompt = self.query_one("#prompt", Input)
        if not prompt.has_focus or not self._input_history:
            return
        if self._history_index is None:
            if delta > 0:
                return
            self._history_draft = prompt.value
            self._history_index = len(self._input_history) - 1
        else:
            self._history_index += delta
            if self._history_index >= len(self._input_history):
                self._history_index = None
                prompt.value = self._history_draft
                prompt.cursor_position = len(prompt.value)
                return
            self._history_index = max(0, self._history_index)
        history_value = self._input_history[self._history_index]
        if "\n" in history_value or len(history_value) > _PASTE_INLINE_CHAR_LIMIT:
            prompt.value = self._fold_pasted_text(history_value)
        else:
            prompt.value = history_value
        prompt.cursor_position = len(prompt.value)

    # ── workers ───────────────────────────────────────────────────────────
    @work(exclusive=True, group="agent")
    async def _run_agent(self, task: str) -> None:
        self.sink.begin_task()
        self.busy = True
        self.clear_todos()
        self._refresh_status()
        try:
            await self.session.run_task(task)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # CancelledError is BaseException → not caught here
            self.sink.error(f"agent loop failed: {exc}")
        finally:
            try:
                await self.sink.finish_stream()
            except Exception:
                logger.exception("failed to finalize TUI stream")
            self.sink.finish_task()
            self.busy = False
            self._refresh_chrome()
            self._refresh_deliverables()
            self._focus_prompt()

    @work
    async def _handle_slash(self, line: str) -> None:
        cmd = line.split()[0].lower()
        if cmd in ("/exit", "/quit"):
            self.exit()
            return
        if cmd == "/help":
            self.push_screen(HelpScreen())
            return
        if cmd in ("/settings", "/menu"):
            await self._settings_modal().wait()
            return
        if cmd == "/resume":
            await self._resume_modal()
            return
        if cmd in {"/new", "/fork"}:
            starter: Callable[..., tuple[str, str]] | None = getattr(
                self.session, "start_new_session", None,
            )
            if callable(starter):
                try:
                    previous, current = starter(fork=cmd == "/fork")
                    if cmd == "/new":
                        await self.transcript.clear_all()
                    self.clear_activity()
                    self.clear_todos()
                    self.sink.reset_presentation()
                    self._deliverables_root = self._resolve_deliverables_root()
                    self._refresh_chrome()
                    action = "started new session" if cmd == "/new" else "forked context into"
                    self.sink.note(f"saved {previous} · {action} {current}")
                except Exception as exc:
                    self.sink.error(f"could not create session: {exc}")
                finally:
                    self._focus_prompt()
                return
        if cmd == "/model" and len(line.split()) == 1:
            # Bare /model → arrow-key picker. "/model <name|n>" falls through to
            # the session's text handler (same as line mode).
            await self._model_modal()
            return
        if cmd == "/theme" and len(line.split()) == 1:
            await self._theme_modal()
            return
        if cmd == "/theme":
            self._handle_theme(line)
            return
        if cmd == "/workflow" and len(line.split()) == 1:
            await self._workflow_modal()
            return
        if cmd == "/workflow":
            # Workflows are represented by their native profile modes.
            line = "/mode" + line.removeprefix("/workflow")
        if cmd == "/filter":
            parts = line.split(maxsplit=1)
            mode = parts[1].strip().lower() if len(parts) == 2 else "all"
            if mode not in {"all", "thinking", "tools", "errors", "report"}:
                self.notify("usage: /filter all|thinking|tools|errors|report", severity="warning")
            else:
                self.transcript.apply_filter(mode)
                self._refresh_transcript_view()
            self._focus_prompt()
            return
        if cmd == "/find":
            query = line.partition(" ")[2].strip()
            if not query:
                self.notify("usage: /find <text>", severity="warning")
            else:
                count = self.transcript.apply_filter("search", query)
                self._refresh_transcript_view()
                if not count:
                    self.notify(f"no transcript matches for {query!r}", severity="warning")
            self._focus_prompt()
            return
        if cmd == "/report":
            self.action_jump_report()
            self._focus_prompt()
            return
        if cmd == "/context":
            self.push_screen(ContextScreen(
                self.session.usage,
                window=int(getattr(self.session.cfg, "context_window", 0) or 0),
                output_reserve=int(getattr(self.session.cfg, "max_tokens", 0) or 0),
                model=str(getattr(self.session.cfg, "model", "") or ""),
            ))
            return
        if cmd == "/copy":
            self.action_copy_report()
            self._focus_prompt()
            return
        if cmd in {"/attach", "/attachments", "/detach"}:
            self._handle_attachment_command(line)
            self._focus_prompt()
            return
        if cmd == "/paste":
            self.action_paste_clipboard()
            self._focus_prompt()
            return
        if cmd == "/init":
            from apodex.session import _INIT_PROMPT
            self._run_agent(_INIT_PROMPT)
            return
        if cmd == "/clear":
            await self.transcript.clear_all()
            self.clear_activity()
            self.clear_todos()
            # /clear returns the pane to its start-of-session state, logo included.
            self.sink.logo()
        try:
            await self.session._slash(line)
        except Exception as exc:
            self.sink.error(f"command failed: {exc}")
        finally:
            self._refresh_chrome()
            self._focus_prompt()

    def _handle_attachment_command(self, line: str) -> None:
        from apodex.attachments import AttachmentError, format_size

        manager = getattr(self.session, "attachments", None)
        if manager is None:
            self.sink.error("attachments are unavailable in this session")
            return
        try:
            parts = shlex.split(line)
        except ValueError as exc:
            self.sink.error(f"invalid attachment path: {exc}")
            return
        cmd, args = parts[0].lower(), parts[1:]
        try:
            if cmd == "/attach":
                if not args:
                    self.sink.error("usage: /attach <path> [path ...]")
                    return
                added = manager.attach_many(args)
                if not added:
                    self.sink.note("no files were attached")
                else:
                    self.sink.note(
                        "attached: " + ", ".join(item.relative_path for item in added)
                    )
            elif cmd == "/detach":
                if len(args) != 1:
                    self.sink.error("usage: /detach <attachment>")
                    return
                removed = manager.detach(args[0])
                self.sink.note(f"detached {args[0]} ({removed} file{'s' if removed != 1 else ''})")
            else:
                items = manager.list()
                if not items:
                    self.sink.note("no files attached")
                else:
                    self.sink.note("attached files:\n" + "\n".join(
                        f"  {item.relative_path} · {format_size(item.size)} · {item.agent_path}"
                        for item in items
                    ))
        except AttachmentError as exc:
            self.sink.error(str(exc))
        finally:
            self._refresh_attachments()

    async def _model_modal(self) -> None:
        models = list(getattr(self.session, "models", []) or [])
        current = self.session.cfg.model
        chosen = await self.push_screen_wait(ModelScreen(models, current))
        if chosen and chosen != current:
            # Reuse the session's switch logic (rebuild client + refresh env).
            await self.session._slash(f"/model {chosen}")
        self._refresh_chrome()
        self._focus_prompt()

    @work(exclusive=True, group="settings")
    async def _settings_modal(self) -> None:
        from apodex.session import list_saved_sessions, load_session_state

        rules = getattr(self.session, "rules", None)
        recent_sessions = tuple(
            (
                item["session_id"],
                f"{item['name'] + ' · ' if item.get('name') else ''}"
                f"{item['modified_at']} · {item['mode']} · "
                f"{item['message_count']} msgs · {item['cwd']}",
            )
            for item in list_saved_sessions()[:10]
            if item["session_id"] != self.session.session_id
        )
        chosen = await self.push_screen_wait(SettingsScreen(
            THEME_PICKER_NAMES,
            self._ui_theme,
            self.session.mode,
            plan_mode=bool(getattr(self.session.plan_state, "active", False)),
            verbose=bool(getattr(self.session, "verbose", self.sink._verbose)),
            auto_approve=bool(self.session.approver.auto_approve),
            auto_for_me=bool(getattr(self.session.approver, "auto_for_me", False)),
            permission_allow=tuple(sorted(getattr(rules, "allow", ()))),
            permission_deny=tuple(sorted(getattr(rules, "deny", ()))),
            sessions=recent_sessions,
            current_session=self.session.session_id,
        ))
        if chosen is not None:
            changes: list[str] = []
            if chosen.theme != self._ui_theme:
                self._apply_theme(chosen.theme)
                changes.append(f"theme → {chosen.theme}")
            if chosen.verbose != bool(getattr(self.session, "verbose", True)):
                self.session.verbose = chosen.verbose
                self.sink.set_verbose(chosen.verbose)
                changes.append(f"full thinking → {chosen.verbose}")
            if chosen.auto_approve != self.session.approver.auto_approve:
                self.session.approver.auto_approve = chosen.auto_approve
                changes.append(f"bypass permissions → {chosen.auto_approve}")
            if chosen.auto_for_me != getattr(self.session.approver, "auto_for_me", False):
                self.session.approver.auto_for_me = chosen.auto_for_me
                changes.append(f"auto for me → {chosen.auto_for_me}")

            # Persist user settings to ~/.config/apodex/settings.json
            user_settings = getattr(self.session, "user_settings", None)
            if user_settings is not None:
                user_settings.theme = chosen.theme
                user_settings.workflow = chosen.workflow
                user_settings.plan_mode = chosen.plan_mode
                user_settings.verbose = chosen.verbose
                user_settings.auto_approve = chosen.auto_approve
                user_settings.auto_for_me = chosen.auto_for_me
                user_settings.save()
            if chosen.resume_session_id:
                state = load_session_state(chosen.resume_session_id)
                if state is None:
                    self.sink.error("could not load that session")
                else:
                    await self._apply_resume_state(chosen.resume_session_id, state)
            else:
                if chosen.workflow != self.session.mode:
                    await self.session._slash(f"/mode {chosen.workflow}")
                    changes.append(f"workflow → {chosen.workflow}")
                if chosen.plan_mode != bool(self.session.plan_state.active):
                    self.session.plan_state.active = chosen.plan_mode
                    changes.append(f"plan mode → {chosen.plan_mode}")
            if changes:
                self.sink.note("settings applied · " + " · ".join(changes))
        self._refresh_chrome()
        self._focus_prompt()

    async def _theme_modal(self) -> None:
        chosen = await self.push_screen_wait(ThemeScreen(THEME_PICKER_NAMES, self._ui_theme))
        if chosen and chosen != self._ui_theme:
            self._apply_theme(chosen)
            self.sink.note(f"theme → {chosen}")
        self._refresh_chrome()
        self._focus_prompt()

    async def _workflow_modal(self) -> None:
        chosen = await self.push_screen_wait(WorkflowScreen(self.session.mode))
        if chosen and chosen != self.session.mode:
            await self.session._slash(f"/mode {chosen}")
        self._refresh_chrome()
        self._focus_prompt()

    async def _resume_modal(self) -> None:
        from apodex.session import load_session_state
        sid = await self.push_screen_wait(ResumeScreen())
        if not sid:
            return
        state = load_session_state(sid)
        if state is None:
            self.sink.error("could not load that session")
            self._focus_prompt()
            return
        await self._apply_resume_state(sid, state)

    async def _apply_resume_state(self, sid: str, state: dict) -> bool:
        """Apply and render a checkpoint selected by the resume modal."""
        try:
            switch = getattr(self.session, "switch_session", None)
            if switch is not None:
                switch(state, fallback_id=sid)
            else:  # lightweight session doubles and third-party integrations
                self.session.session_id = state.get("session_id", sid)
                self.session.mode = state.get("mode", self.session.mode)
                self.session.restore(state)
        except Exception as exc:
            self.sink.error(f"could not resume session: {exc}")
            self._focus_prompt()
            return False
        restored_tui = (
            state.get("tui")
            if "tui" in state
            else getattr(self.session, "tui_state", {})
        )
        self.sink.restore_state(restored_tui)
        await self._replay_session_history()
        self.sink.reset_presentation()
        self._refresh_chrome()
        self._refresh_deliverables()
        self._focus_prompt()
        self.sink.note(
            f"resumed {self.session.session_id} ({len(self.session.history)} prior messages)"
        )
        return True

    async def _replay_session_history(self) -> None:
        """Restore transcript and activity rows from the same full history."""
        provider: Callable[[], list[Message]] | None = getattr(
            self.session, "replay_history", None,
        )
        history = provider() if callable(provider) else list(self.session.history)
        restored_tools = sum(
            len(message.get("tool_calls") or [])
            + int(message.get("role") == "tool" and not message.get("tool_call_id"))
            for message in history
        )
        self.activity.max_records = max(self.activity.max_records, restored_tools)
        await self.transcript.replay_history(history)
        # Keep the just-loaded semantic checkpoint while replacing widget
        # rows; /clear and /new still discard it through clear_activity().
        self.clear_activity(reset_checkpoint=False)
        names: dict[str, str] = {}
        sequence = 0
        for message in history:
            if message.get("role") == "assistant":
                for call in message.get("tool_calls") or []:
                    sequence += 1
                    function = call.get("function") or {}
                    name = str(function.get("name") or call.get("name") or "tool")
                    call_id = str(call.get("id") or f"resumed-call-{sequence}")
                    raw_args = function.get("arguments", call.get("args", {}))
                    try:
                        args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                    except (TypeError, ValueError):
                        args = raw_args
                    summary = json.dumps(args, ensure_ascii=False) if isinstance(args, (dict, list)) else str(args)
                    names[call_id] = name
                    self.start_activity(call_id, name, summary)
            elif message.get("role") == "tool":
                sequence += 1
                call_id = str(message.get("tool_call_id") or f"resumed-result-{sequence}")
                name = str(message.get("name") or names.get(call_id) or "tool")
                self.finish_activity(
                    call_id, name,
                    is_error=bool(message.get("is_error", False)),
                    ms=int(message.get("duration_ms") or 0),
                )
        saved_subagents = self.sink.saved_subagents()
        if saved_subagents:
            self.activity.update_subagents(saved_subagents, done=True)
        # A checkpoint can be written after the last completed model turn but
        # before an interrupted worker publishes another heartbeat.  Never
        # revive such rows as permanently-running activity on resume.
        self.activity.finish_active(ActivityState.INTERRUPTED)

    def _apply_theme(self, name: str) -> None:
        """Switch both halves of the UI to ``name``.

        ``dark`` and ``light`` are our own registered palettes now, not
        Textual's built-ins: those had no Rich half at all, so the transcript
        silently fell back to Catppuccin and painted dark-theme colours onto a
        light background.
        """
        self._ui_theme = name
        self.theme = name
        # Widgets that keep their own Rich renderables must be repainted — the
        # styles are baked into the ``Text`` objects, so a CSS reload alone
        # leaves them on the previous palette.
        if hasattr(self, "todos_pane"):
            self.todos_pane.refresh_theme()
            self.activity.rerender()
            self.diff_pane.refresh_theme()
            self.transcript.refresh_logo(name, force=True)
            self._refresh_status()

    def _handle_theme(self, line: str) -> None:
        parts = line.split(maxsplit=1)
        name = parts[1].strip().lower() if len(parts) == 2 else (
            "light" if self._ui_theme == "dark" else "dark"
        )
        if name == "mono":
            self.sink.note("mono theme uses the color-free line UI; restart with --theme mono")
            return
        if name not in TUI_THEME_NAMES:
            self.sink.error("usage: /theme " + "|".join(TUI_THEME_NAMES))
            return
        self._apply_theme(name)
        self.sink.note(f"theme → {name}")

    # ── global interaction actions ──────────────────────────────────────
    def action_show_help(self) -> None:
        if not isinstance(self.screen, HelpScreen):
            self.push_screen(HelpScreen())

    def action_settings(self) -> None:
        if self.busy:
            self.notify("busy — Ctrl-C to interrupt first", severity="warning")
            return
        self._settings_modal()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "menu-button":
            self.action_settings()

    def action_command_palette(self) -> None:
        if self.busy:
            self.notify("busy — Ctrl-C to interrupt first", severity="warning")
            return
        self.push_screen(CommandScreen(_COMMANDS), self._insert_command)

    def _insert_command(self, command: str | None) -> None:
        if not command:
            self._focus_prompt()
            return
        value = command + (" " if command in _COMMANDS_WITH_ARGUMENTS else "")
        prompt = self.query_one("#prompt", Input)
        prompt.value = value
        prompt.cursor_position = len(value)
        prompt.focus()

    def action_toggle_sidebar(self) -> None:
        self._sidebar_user_visible = not self._sidebar_user_visible
        self._sync_sidebar()
        state = "shown" if self._sidebar_user_visible else "hidden"
        if self._sidebar_user_visible and self._terminal_width < 100:
            state = "enabled (hidden until terminal is at least 100 columns)"
        self.notify(f"sidebar {state}")

    def action_toggle_deliverables(self) -> None:
        target = "plan" if self._sidebar_tab == "deliverables" else "deliverables"
        self._show_sidebar_tab(target, focus=True)

    def close_deliverables(self) -> None:
        """Return from the file tab to the default Plan workspace."""
        self._show_sidebar_tab("plan", focus=True)

    def close_diff(self) -> None:
        self._show_sidebar_tab("plan", focus=True)

    def _show_sidebar_tab(self, name: str, *, focus: bool) -> None:
        """Show exactly one workspace tab and optionally enter its item list."""
        if name not in {"plan", "activity", "deliverables", "diff"}:
            return
        if name == "diff" and not self._workspace_diff_stats:
            return
        self._sidebar_tab = name
        for section in ("plan", "activity", "deliverables", "diff"):
            selector = "#deliverables-box" if section == "deliverables" else f"#sidebar-{section}"
            self.query_one(selector).display = section == name
        tabs = self.query_one("#sidebar-tabs", Tabs)
        tab_id = f"sidebar-tab-{name}"
        if tabs.active != tab_id:
            tabs.active = tab_id
        if name == "deliverables":
            self._refresh_deliverables()
        if not focus:
            return
        if name == "activity":
            self.activity.focus()
        elif name == "deliverables":
            self.deliverables.focus()
        elif name == "diff":
            self._request_workspace_diff()
            self.diff_scroll.focus()
        else:
            self._focus_prompt()

    def show_task_workspace(self) -> None:
        """Reset the sidebar to the live task board when work begins."""
        self._completed_workspace_pending = False
        self._completed_workspace_needs_followup = False
        self._show_sidebar_tab("plan", focus=False)

    def show_completed_workspace(self) -> None:
        """Reveal Diff after a changed run, otherwise reveal its output files."""
        self._completed_workspace_pending = True
        self._completed_workspace_needs_followup = self._workspace_diff_pending
        if not self._workspace_diff_pending:
            self._request_workspace_diff()

    def show_completed_deliverables(self) -> None:
        """Compatibility alias for callers predating the conditional Diff tab."""
        self.show_completed_workspace()

    def refresh_deliverables(self) -> None:
        """Reload the outputs pane in place, leaving the visible tab alone.

        A run that stops short must not be presented as delivered, so it never
        reveals the tab — but its publisher may already have written real files,
        which the pane still has to list once the user opens it.
        """
        self._refresh_deliverables()

    @on(Tabs.TabActivated, "#sidebar-tabs")
    def _on_sidebar_tab_activated(self, event: Tabs.TabActivated) -> None:
        event.stop()
        name = (event.tab.id or "sidebar-tab-plan").removeprefix("sidebar-tab-")
        self._show_sidebar_tab(
            name, focus=self.query_one("#sidebar").display,
        )

    def action_next_sidebar_tab(self) -> None:
        tabs = self._available_sidebar_tabs()
        self._show_sidebar_tab(tabs[(self._sidebar_tab_index() + 1) % len(tabs)], focus=True)

    def action_previous_sidebar_tab(self) -> None:
        tabs = self._available_sidebar_tabs()
        self._show_sidebar_tab(tabs[(self._sidebar_tab_index() - 1) % len(tabs)], focus=True)

    def _sidebar_tab_index(self) -> int:
        """Position of the visible tab, tolerating one that just disappeared.

        Diff leaves the list the moment its stats go empty, and there is a
        window inside ``_store_workspace_diff`` where that has happened but the
        active tab has not yet been moved off it. ``tuple.index`` would raise
        straight out of the key handler; stepping from -1 lands on "plan".
        """
        tabs = self._available_sidebar_tabs()
        return tabs.index(self._sidebar_tab) if self._sidebar_tab in tabs else -1

    def _available_sidebar_tabs(self) -> tuple[str, ...]:
        tabs = ("plan", "activity", "deliverables")
        return tabs + (("diff",) if self._workspace_diff_stats else ())

    def open_activity_preview(self, record: ActivityRecord) -> None:
        duration = self.activity._duration(record, self.activity.clock())
        self.push_screen(ActivityDetailScreen(
            name=record.name,
            call_id=record.call_id,
            state=record.state.value,
            duration=duration,
            summary=record.summary,
        ))

    def open_deliverable_preview(self, path: Path) -> None:
        """Open a selected text deliverable after re-validating its boundary."""
        root = self._deliverables_root.resolve()
        candidate = path.resolve()
        try:
            relative = candidate.relative_to(root)
        except ValueError:
            self.notify("deliverable is outside the output directory", severity="error")
            return
        if candidate.suffix.lower() not in PREVIEWABLE_SUFFIXES:
            self.notify(f"preview is not available for {candidate.suffix or 'this file type'}")
            return
        try:
            if not candidate.is_file():
                self.notify("preview is only available for files", severity="warning")
                return
            if candidate.suffix.lower() not in BINARY_PREVIEWABLE_SUFFIXES:
                with candidate.open("rb") as source:
                    if b"\x00" in source.read(8192):
                        self.notify("preview is only available for text files", severity="warning")
                        return
        except OSError as exc:
            self.notify(f"could not read deliverable: {exc}", severity="error")
            return
        self.push_screen(FilePreviewScreen(candidate, label=str(relative)))

    def action_review_next(self) -> None:
        self.transcript.review_move(1)

    def action_review_previous(self) -> None:
        self.transcript.review_move(-1)

    def action_review_toggle(self) -> None:
        self.transcript.toggle_review_block()

    def action_jump_report(self) -> None:
        reports = list(self.transcript.query(".final-report"))
        if not reports:
            self.notify("no final report yet", severity="warning")
            return
        # Jumping out of a filtered view drops the filter, rather than forcing
        # one block visible inside a view that claims to hide it.
        if self.transcript.clear_filter():
            self._refresh_transcript_view()
        reports[-1].scroll_visible(animate=False)

    def action_copy_report(self) -> None:
        text = self.transcript.final_text.strip()
        if not text:
            self.notify("no final report yet", severity="warning")
            return
        self.copy_to_clipboard(text)
        self.notify("final report copied")

    def _refresh_deliverables(self) -> None:
        """Reload after a workflow finishes, without creating an empty output dir."""
        expected = self._resolve_deliverables_root()
        if expected != self._deliverables_root:
            self._deliverables_root = expected
        root = self._deliverables_root
        self.deliverables_location.update("\n".join(self._deliverables_location_lines()))
        if not root.is_dir():
            self.deliverables.show_files(root, [])
            return
        files = sorted(path for path in root.rglob("*") if path.is_file())[:100]
        self.deliverables.show_files(root, files)

    def save_final_report(self, text: str) -> Path | None:
        """Persist a text-only answer when no explicit deliverable exists.

        The workflow's ``/outputs`` manifest is authoritative. Turning the
        coordinator's chat summary into ``final-report.md`` after publishers
        have already populated that directory both violates the manifest and
        makes internal status prose look like a user-requested artifact.
        """
        root = self._resolve_deliverables_root()
        target = root / "final-report.md"
        temporary = root / ".final-report.md.tmp"
        try:
            if root.is_dir() and any(
                path.is_file()
                and path.name not in {target.name, temporary.name}
                and "scratch" not in path.relative_to(root).parts[:1]
                for path in root.rglob("*")
            ):
                return None
            root.mkdir(parents=True, exist_ok=True)
            body = text.strip() or "_(no answer)_"
            temporary.write_text(body + "\n", encoding="utf-8")
            temporary.replace(target)
        except OSError as exc:
            logger.exception("failed to persist final report")
            self.notify(f"could not save final report: {exc}", severity="warning")
            return None
        self._deliverables_root = root
        if self._sidebar_tab == "deliverables":
            self._refresh_deliverables()
        return target

    def action_history_previous(self) -> None:
        self._move_history(-1)

    def action_history_next(self) -> None:
        self._move_history(1)

    # ── interrupt ─────────────────────────────────────────────────────────
    def action_interrupt(self) -> None:
        if self.busy:
            self.sink.interrupted()
            if isinstance(self.screen, ApprovalScreen):
                self.pop_screen()
            self.workers.cancel_group(self, "agent")
            self.sink.note("■ interrupted (state saved — resume continues from the last turn)")
        else:
            self.exit()


__all__ = ["FrontierAgentApp"]
