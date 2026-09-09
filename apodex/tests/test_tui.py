"""Textual TUI smoke tests — drive ``FrontierAgentApp`` with a fake session (no LLM).

These exercise the parts the TUI adds on top of the (separately tested) engine:
the transcript stream, the input → worker routing, busy-time steering, and the
approval modal's key handling. The agent loop itself is covered by test_smoke.
"""

from __future__ import annotations

import asyncio
import re
import threading
from pathlib import Path
from unittest import mock

import pytest
from rich.cells import cell_len
from rich.panel import Panel
from rich.style import Style
from rich.text import Text
from textual import events
from textual.widgets import Collapsible, Input, OptionList, Static, Tab
from textual.widgets._collapsible import CollapsibleTitle

import apodex.tui.app as app_module
from apodex.observers import Decision
from apodex.steer import SteerInbox
from apodex.tui.app import FrontierAgentApp
from apodex.tui.screens import (
    ActivityDetailScreen,
    ApprovalOutcome,
    ApprovalScreen,
    CommandScreen,
    ContextScreen,
    FilePreviewScreen,
    HelpScreen,
    ModelScreen,
    SettingsScreen,
    ThemeScreen,
    WorkflowScreen,
)
from apodex.tui.state import PresentationPhase
from apodex.tui.themes import (
    GLYPHS,
    agent_kind,
    agent_kind_glyph,
    palette,
    rich_style,
)
from apodex.tui.widgets import (
    _MAX_RENDERED_MARKDOWN_CHARS,
    ActivityPane,
    ActivityState,
    SubAgentView,
    TailScroll,
    _bounded_markdown,
)
from frontier_agent.core.messages import assistant_msg, system_msg, user_msg

pytestmark = pytest.mark.asyncio


class _Cfg:
    model = "fake-model"
    context_window = 100_000


class _Usage:
    total = 0

    def context_pct_left(self, window: int) -> int:
        return 50

    def summary(self) -> str:
        return "0 tokens"


class _Plan:
    active = False


class _Approver:
    auto_approve = False


class _FakeSession:
    """Just enough of ``TerminalSession`` for the TUI to drive."""

    def __init__(self) -> None:
        self.mode = "coding"
        self.cwd = "/tmp"
        self.cfg = _Cfg()
        self.models = ["fake-model", "other-model"]
        self.usage = _Usage()
        self.plan_state = _Plan()
        self.history: list = []
        self.session_id = "test-session"
        self.session_name = ""
        self.approver = _Approver()
        self.verbose = True
        self.rules = type("Rules", (), {"allow": set(), "deny": set()})()
        self.tui_mode = False
        self.r = None
        self._inbox = None
        self.tasks: list[str] = []
        self.slash_commands: list[str] = []
        self.hold = asyncio.Event()
        self.hold.set()  # released by default; cleared to keep a run "busy"

    async def run_task(self, task: str) -> None:
        self.tasks.append(task)
        self._inbox = SteerInbox(self.r)
        self.r.content_delta("hello ")
        await self.hold.wait()

    async def _slash(self, line: str) -> bool:
        self.slash_commands.append(line)
        return False

    def restore(self, state: dict) -> None:
        self.history = list(state.get("history") or [])

    def switch_session(self, state: dict, *, fallback_id: str = "") -> None:
        self.session_id = state.get("session_id", fallback_id)
        self.session_name = state.get("name", "")
        self.mode = state.get("mode", self.mode)
        self.cwd = state.get("cwd", self.cwd)
        self.cfg.model = state.get("model", self.cfg.model)
        self.plan_state.active = bool(state.get("plan_active", False))
        self.restore(state)


async def _wait_until(cond, timeout: float = 2.0) -> None:
    """Poll ``cond`` (called with no args) until true or timeout."""
    for _ in range(int(timeout / 0.02)):
        if cond():
            return
        await asyncio.sleep(0.02)
    raise AssertionError("condition not met within timeout")


async def test_app_boots_streams_and_routes_task() -> None:
    session = _FakeSession()
    app = FrontierAgentApp(session)
    async with app.run_test() as pilot:
        # The TUI took over the session.
        assert session.tui_mode is True
        assert session.r is app.sink

        # Keep the run "busy" so we can observe streaming + steering mid-run.
        session.hold.clear()

        # Type a task → it starts an agent worker and reaches run_task.
        prompt = app.query_one("#prompt", Input)
        prompt.value = "do something"
        await pilot.press("enter")
        await _wait_until(lambda: session.tasks == ["do something"])
        await _wait_until(lambda: app.busy is True)

        # The streamed delta landed in the transcript's live block.
        await _wait_until(lambda: "hello" in app.transcript._buf)

        # A line typed while busy steers (queued) instead of starting a task.
        prompt.value = "also do this"
        await pilot.press("enter")
        await _wait_until(lambda: "also do this" in session._inbox.queue)
        assert session.tasks == ["do something"]  # no second task started
        assert app.presentation.queued_steers == 1
        assert "queued 1" in app.status.render().plain

        session._inbox.queue.clear()
        app._refresh_status()
        assert app.presentation.queued_steers == 0

        # Release the held run so the worker can finish cleanly.
        session.hold.set()
        await _wait_until(lambda: app.busy is False)


async def test_preflight_warnings_render_in_the_transcript_after_mount() -> None:
    """stderr is gone once Textual owns the screen, so the TUI shows them."""
    app = FrontierAgentApp(
        _FakeSession(),
        startup_warnings=["SERPER_API_KEY is not set; web_search will error."],
    )
    async with app.run_test() as pilot:
        await pilot.pause()
        assert app.transcript.apply_filter("search", "SERPER_API_KEY") == 1


async def test_attachment_commands_update_bar_and_remove_copy(
    monkeypatch, tmp_path,
) -> None:
    from apodex.attachments import AttachmentManager

    source = tmp_path / "source" / "claim.pdf"
    source.parent.mkdir()
    source.write_bytes(b"claim")
    staging = tmp_path / "inputs"
    monkeypatch.setenv("APODEX_INPUT_STAGING_DIR", str(staging))
    monkeypatch.setenv("FRONTIER_AGENT_INPUTS_DIR", "/inputs")
    session = _FakeSession()
    session.attachments = AttachmentManager(str(tmp_path), session.session_id)
    app = FrontierAgentApp(session)
    process_cwd = tmp_path / "unrelated-process-cwd"
    process_cwd.mkdir()
    monkeypatch.chdir(process_cwd)

    async with app.run_test() as pilot:
        prompt = app.query_one("#prompt", Input)
        prompt.value = "/attach source/claim.pdf"
        await pilot.press("enter")
        await _wait_until(lambda: len(session.attachments.list()) == 1)
        assert app.attachments_bar.display is True
        assert "claim.pdf" in str(app.attachments_bar.render())

        prompt.value = "review @cla"
        await pilot.press("tab")
        assert prompt.value == "review @claim.pdf "

        prompt.value = "/detach claim.pdf"
        await pilot.press("enter")
        await _wait_until(lambda: not session.attachments.list())
        assert app.attachments_bar.display is False


async def test_at_sign_shows_attachment_candidates_before_tab_completion(
    monkeypatch, tmp_path,
) -> None:
    from apodex.attachments import AttachmentManager

    sources = tmp_path / "source"
    sources.mkdir()
    observability = sources / "observability-requirements.md"
    incidents = sources / "support-incidents.md"
    observability.write_text("requirements")
    incidents.write_text("incidents")
    monkeypatch.setenv("APODEX_INPUT_STAGING_DIR", str(tmp_path / "inputs"))
    session = _FakeSession()
    session.attachments = AttachmentManager(str(tmp_path), session.session_id)
    session.attachments.attach_many([str(observability), str(incidents)])
    app = FrontierAgentApp(session)

    async with app.run_test() as pilot:
        prompt = app.query_one("#prompt", Input)
        hint = app.query_one("#command-hint", Static)

        prompt.value = "review @"
        await pilot.pause()
        assert hint.display is True
        assert "@observability-requirements.md" in hint.render().plain
        assert "@support-incidents.md" in hint.render().plain

        # With no shared prefix, Tab keeps focus in the prompt while the hint
        # exposes the choices instead of appearing to do nothing.
        await pilot.press("tab")
        assert prompt.value == "review @"
        assert prompt.has_focus

        prompt.value = "review @OBS"
        await pilot.pause()
        await pilot.press("tab")
        assert prompt.value == "review @observability-requirements.md "


async def test_at_sign_explains_when_no_files_are_attached() -> None:
    session = _FakeSession()
    session.cwd = "/path/that/does/not/exist"
    session.attachments = type("Attachments", (), {"list": lambda _self: []})()
    app = FrontierAgentApp(session)

    async with app.run_test() as pilot:
        prompt = app.query_one("#prompt", Input)
        prompt.value = "review @"
        await pilot.pause()

        hint = app.query_one("#command-hint", Static)
        assert hint.display is True
        assert "No files found under" in hint.render().plain


async def test_at_sign_searches_files_under_session_cwd(
    monkeypatch, tmp_path,
) -> None:
    workspace = tmp_path / "workspace"
    source = workspace / "src" / "api"
    ignored = workspace / "node_modules" / "dependency"
    source.mkdir(parents=True)
    ignored.mkdir(parents=True)
    (source / "client.py").write_text("client")
    (workspace / "README.md").write_text("readme")
    (ignored / "client.js").write_text("ignored")

    session = _FakeSession()
    session.cwd = str(workspace)
    session.attachments = type("Attachments", (), {"list": lambda _self: []})()
    app = FrontierAgentApp(session)

    async with app.run_test() as pilot:
        prompt = app.query_one("#prompt", Input)
        hint = app.query_one("#command-hint", Static)

        prompt.value = "inspect @"
        # The tree is indexed in a worker thread, so the first hint may land a
        # beat later; ``_store_workspace_index`` repaints it when it does.
        await _wait_until(lambda: "@README.md" in hint.render().plain)
        assert hint.display is True
        assert "node_modules" not in hint.render().plain

        prompt.value = "inspect @CLIENT"
        await pilot.pause()
        assert "@src/api/client.py" in hint.render().plain
        assert "node_modules" not in hint.render().plain
        await pilot.press("tab")
        assert prompt.value == "inspect @src/api/client.py "


async def test_at_sign_indexing_never_blocks_the_ui_thread(tmp_path) -> None:
    """The tree walk must not run inside the synchronous input handler.

    It used to: ``on_input_changed`` walked the whole ``--cwd`` inline, and
    invalidated its cache at the start of every mention, so a large checkout
    froze keystrokes, streaming output and the status bar once per ``@``.
    """
    (tmp_path / "ledger.md").write_text("ledger")
    session = _FakeSession()
    session.cwd = str(tmp_path)
    session.attachments = type("Attachments", (), {"list": lambda _self: []})()
    app = FrontierAgentApp(session)

    walk_threads: list[str] = []
    real_walk = app_module._walk_workspace_files

    def recording_walk(root, limit):
        walk_threads.append(threading.current_thread().name)
        return real_walk(root, limit)

    async with app.run_test() as pilot:
        # Let the mount-time warm index finish first, so the only walk the
        # recorder sees is the one this test triggers.
        await app.workers.wait_for_complete()
        with mock.patch.object(app_module, "_walk_workspace_files", recording_walk):
            app._workspace_file_cache_at = 0.0  # force a refresh
            app.query_one("#prompt", Input).value = "read @led"
            await pilot.pause()
            await _wait_until(lambda: bool(walk_threads))
            await app.workers.wait_for_complete()

    assert len(walk_threads) == 1, f"expected one refresh, got {walk_threads}"
    assert threading.main_thread().name not in walk_threads
    assert app._workspace_file_cache == ("ledger.md",)


def test_workspace_index_keeps_whole_names_when_cwd_is_the_filesystem_root(
    tmp_path,
) -> None:
    """``str(Path("/"))`` already ends in a separator.

    Adding one anyway trimmed the first character of every indexed path, so
    ``--cwd /`` offered ``tc/hosts`` for ``/etc/hosts``.
    """
    (tmp_path / "etc").mkdir()
    (tmp_path / "etc" / "hosts").write_text("127.0.0.1 localhost")

    class _Entry:
        def __init__(self, path: str, directory: bool) -> None:
            self.path = "/" + path
            self.name = path.rsplit("/", 1)[-1]
            self._directory = directory

        def is_symlink(self) -> bool:
            return False

        def is_dir(self, follow_symlinks: bool = True) -> bool:
            return self._directory

        def is_file(self, follow_symlinks: bool = True) -> bool:
            return not self._directory

    tree = {
        "/": [_Entry("etc", True), _Entry("kernel", False)],
        "/etc": [_Entry("etc/hosts", False)],
    }
    with mock.patch.object(app_module.os, "scandir", lambda path: tree.get(path, [])):
        found = app_module._walk_workspace_files(Path("/"), 100)

    assert sorted(found) == ["etc/hosts", "kernel"]


def test_folding_the_same_text_twice_reuses_one_marker(tmp_path) -> None:
    """Scrolling history re-folds the same entry on every keypress.

    Minting a fresh marker each time retained another full copy of the text
    until the next submit, so a large paste grew memory per arrow press.
    """
    session = _FakeSession()
    session.cwd = str(tmp_path)
    app = FrontierAgentApp(session)
    payload = "line one\nline two\nline three"

    first = app._fold_pasted_text(payload)
    second = app._fold_pasted_text(payload)

    assert first == second
    assert app._pasted_text_blocks == {first: payload}
    assert app._expand_pasted_text(f"see {first}") == f"see {payload}"


async def test_tab_on_case_insensitive_matches_keeps_the_typed_fragment(
    tmp_path,
) -> None:
    """Tab must never delete input it cannot extend.

    ``@`` search folds case, so ``commonprefix`` over the raw spellings of
    ``README.md`` and ``requirements.txt`` is empty — using it directly replaced
    the typed ``@re`` with a bare ``@``.
    """
    (tmp_path / "README.md").write_text("readme")
    (tmp_path / "requirements.txt").write_text("requirements")
    session = _FakeSession()
    session.cwd = str(tmp_path)
    session.attachments = type("Attachments", (), {"list": lambda _self: []})()
    app = FrontierAgentApp(session)

    async with app.run_test() as pilot:
        prompt = app.query_one("#prompt", Input)
        hint = app.query_one("#command-hint", Static)
        prompt.value = "review @re"
        await _wait_until(lambda: "@README.md" in hint.render().plain)

        await pilot.press("tab")
        assert prompt.value == "review @re"

        # A shared prefix that IS longer still completes, case-folded.
        prompt.value = "review @REQ"
        await pilot.pause()
        await pilot.press("tab")
        assert prompt.value == "review @requirements.txt "


async def test_at_sign_completion_is_disabled_inside_slash_commands(
    tmp_path,
) -> None:
    (tmp_path / "example.md").write_text("example")
    session = _FakeSession()
    session.cwd = str(tmp_path)
    session.attachments = type("Attachments", (), {"list": lambda _self: []})()
    app = FrontierAgentApp(session)

    async with app.run_test() as pilot:
        prompt = app.query_one("#prompt", Input)
        prompt.value = "/attach @"
        await pilot.pause()

        hint = app.query_one("#command-hint", Static)
        assert hint.display is False
        await pilot.press("tab")
        assert prompt.value == "/attach @"


async def test_at_sign_completion_survives_a_message_starting_with_a_path(
    tmp_path,
) -> None:
    """A leading "/" is not a command unless the first word actually is one."""
    (tmp_path / "example.md").write_text("example")
    session = _FakeSession()
    session.cwd = str(tmp_path)
    session.attachments = type("Attachments", (), {"list": lambda _self: []})()
    app = FrontierAgentApp(session)

    async with app.run_test() as pilot:
        prompt = app.query_one("#prompt", Input)
        hint = app.query_one("#command-hint", Static)
        prompt.value = "/outputs/report.md compare with @exa"
        await _wait_until(lambda: "@example.md" in hint.render().plain)

        await pilot.press("tab")
        assert prompt.value == "/outputs/report.md compare with @example.md "


async def test_ctrl_v_pastes_clipboard_image_as_attachment(
    monkeypatch, tmp_path,
) -> None:
    from apodex.attachments import AttachmentManager
    from apodex.clipboard import ClipboardPaste

    monkeypatch.setenv("APODEX_INPUT_STAGING_DIR", str(tmp_path / "inputs"))
    monkeypatch.setenv("FRONTIER_AGENT_INPUTS_DIR", "/inputs")
    session = _FakeSession()
    session.attachments = AttachmentManager(str(tmp_path), session.session_id)

    def fake_paste(manager, *, pasted_text=None):
        source = tmp_path / "clipboard.png"
        source.write_bytes(b"png")
        added = manager.attach(str(source))
        return ClipboardPaste("attachments", tuple(item.relative_path for item in added))

    monkeypatch.setattr("apodex.clipboard.paste_from_clipboard", fake_paste)
    app = FrontierAgentApp(session)
    async with app.run_test() as pilot:
        await pilot.press("ctrl+v")
        await _wait_until(lambda: len(session.attachments.list()) == 1)
        assert "clipboard.png" in str(app.attachments_bar.render())


async def test_multiline_terminal_paste_is_preserved_and_submitted_once(
    monkeypatch,
) -> None:
    from apodex.clipboard import ClipboardPaste

    monkeypatch.setattr(
        "apodex.clipboard.paste_from_clipboard",
        lambda _manager, *, pasted_text=None: ClipboardPaste(
            "text", text=pasted_text or "",
        ),
    )
    session = _FakeSession()
    session.attachments = type("Attachments", (), {"list": lambda _self: []})()
    app = FrontierAgentApp(session)
    async with app.run_test() as pilot:
        prompt = app.query_one("#prompt", Input)
        pasted = "first line\nsecond line\nthird line"
        prompt.value = "Analyze this: "
        prompt.cursor_position = len(prompt.value)
        prompt._on_paste(events.Paste(pasted))
        await _wait_until(
            lambda: "Pasted text #1" in prompt.value
        )
        assert "3 lines" in prompt.value
        assert "second line" not in prompt.value
        prompt.insert_text_at_cursor(" End.")

        await pilot.press("enter")
        expected = f"Analyze this: {pasted} End."
        await _wait_until(lambda: session.tasks == [expected])
        assert app._input_history[-1] == expected


async def test_duplicate_terminal_paste_event_inserts_text_once(
    monkeypatch,
) -> None:
    from apodex.clipboard import ClipboardPaste

    calls = 0

    def fake_paste(_manager, *, pasted_text=None):
        nonlocal calls
        calls += 1
        return ClipboardPaste("text", text=pasted_text or "")

    monkeypatch.setattr("apodex.clipboard.paste_from_clipboard", fake_paste)
    session = _FakeSession()
    session.attachments = type("Attachments", (), {"list": lambda _self: []})()
    app = FrontierAgentApp(session)
    async with app.run_test():
        prompt = app.query_one("#prompt", Input)
        prompt._on_paste(events.Paste("same text"))
        prompt._on_paste(events.Paste("same text"))

        await _wait_until(lambda: prompt.value == "same text")
        assert calls == 1


async def test_duplicate_clipboard_routes_commit_text_once() -> None:
    session = _FakeSession()
    app = FrontierAgentApp(session)
    async with app.run_test() as pilot:
        prompt = app.query_one("#prompt", Input)

        app._insert_pasted_text(prompt, "same text")
        app._insert_pasted_text(prompt, "same text")
        app._insert_pasted_text(prompt, "different text")

        assert prompt.value == "same textdifferent text"
        await pilot.pause()


async def test_empty_terminal_paste_reads_image_clipboard(
    monkeypatch, tmp_path,
) -> None:
    from apodex.attachments import AttachmentManager
    from apodex.clipboard import ClipboardPaste

    monkeypatch.setenv("APODEX_INPUT_STAGING_DIR", str(tmp_path / "inputs"))
    monkeypatch.setenv("FRONTIER_AGENT_INPUTS_DIR", "/inputs")
    session = _FakeSession()
    session.attachments = AttachmentManager(str(tmp_path), session.session_id)

    def fake_paste(manager, *, pasted_text=None):
        assert pasted_text is None
        source = tmp_path / "cmd-v-image.png"
        source.write_bytes(b"png")
        added = manager.attach(str(source))
        return ClipboardPaste("attachments", tuple(item.relative_path for item in added))

    monkeypatch.setattr("apodex.clipboard.paste_from_clipboard", fake_paste)
    app = FrontierAgentApp(session)
    async with app.run_test():
        app.query_one("#prompt", Input)._on_paste(events.Paste(""))
        await _wait_until(lambda: len(session.attachments.list()) == 1)
        assert "cmd-v-image.png" in str(app.attachments_bar.render())


async def test_presentation_phase_sequence_reaches_approval_and_done() -> None:
    session = _FakeSession()
    app = FrontierAgentApp(session)
    async with app.run_test() as pilot:
        phases: list[PresentationPhase] = []

        def capture() -> None:
            app._refresh_status()
            phases.append(app.presentation.phase)

        app.sink.begin_task()
        capture()
        assert "thinking" in app.status.render().plain

        app.sink.thinking_delta("considering")
        capture()
        app.sink.content_delta("answering")
        capture()
        app.sink.tool_call("bash", {"command": "pwd"})
        capture()

        async def approve(_screen):
            capture()
            return ApprovalOutcome(Decision(True))

        app.push_screen_wait = approve
        decision = await session.approver.confirm("bash", "pwd", "run command")
        assert decision.approved is True
        capture()

        app.sink.tool_result("bash", "/tmp", is_error=False, ms=4)
        capture()
        app.sink.final("done", turns=1, tool_calls=1)
        capture()
        await pilot.pause()

        assert phases == [
            PresentationPhase.THINKING,
            PresentationPhase.THINKING,
            PresentationPhase.RESPONDING,
            PresentationPhase.RUNNING_TOOL,
            PresentationPhase.AWAITING_APPROVAL,
            PresentationPhase.RUNNING_TOOL,
            PresentationPhase.THINKING,
            PresentationPhase.DONE,
        ]
        assert "done" in app.status.render().plain


async def test_activity_rows_use_call_id_when_same_tool_finishes_out_of_order() -> None:
    app = FrontierAgentApp(_FakeSession())
    async with app.run_test() as pilot:
        app.sink.tool_call("bash", {"command": "first"}, call_id="call-a")
        app.sink.tool_call("bash", {"command": "second"}, call_id="call-b")
        app.sink.tool_result(
            "bash", "second failed", is_error=True, ms=20, call_id="call-b",
        )
        app.sink.tool_result(
            "bash", "first done", is_error=False, ms=10, call_id="call-a",
        )
        await pilot.pause()

        assert app.activity.latest("call-a").state is ActivityState.SUCCESS
        assert app.activity.latest("call-b").state is ActivityState.FAILED
        assert app.activity.latest("call-a").duration_ms == 10
        assert app.activity.latest("call-b").duration_ms == 20


async def test_agent_team_activity_groups_workers_and_coordinator() -> None:
    app = FrontierAgentApp(_FakeSession())
    async with app.run_test() as pilot:
        app.start_activity("collect", "collect_reports", "timeout=1800")
        app.activity.update_subagents([
            {
                "session_id": "root::researcher",
                "name": "researcher",
                "status": "running",
                "elapsed_s": 12,
                "queued": 0,
                "completed": 0,
            },
            {
                "session_id": "root::builder",
                "name": "builder",
                "status": "ready",
                "elapsed_s": 8,
                "queued": 1,
                "completed": 1,
            },
        ])
        await pilot.pause()

        researcher = app.activity.agent_records["root::researcher"]
        builder = app.activity.agent_records["root::builder"]
        assert researcher.state is ActivityState.RUNNING
        assert builder.state is ActivityState.SUCCESS
        assert "report ready" in builder.summary
        assert "1 queued" in builder.summary
        rendered = [record for record in app.activity._rendered_records if record]
        assert rendered[:2] == [researcher, builder]
        assert rendered[-1].call_id == "collect"


async def test_sub_agent_row_toggles_expandable_thinking_and_tool_events() -> None:
    app = FrontierAgentApp(_FakeSession())
    async with app.run_test() as pilot:
        app.activity.update_subagents([{
            "session_id": "root::researcher",
            "name": "researcher",
            "status": "running",
            "elapsed_s": 12,
            "queued": 0,
            "completed": 0,
            "events": [
                {
                    "id": "thought-1", "kind": "thinking",
                    "title": "thinking", "detail": "Inspect ClinVar first",
                    "turn": 1, "is_error": False, "at": 1.0,
                },
                {
                    "id": "tool-1", "kind": "tool_call",
                    "title": "web_search", "detail": "ClinVar API",
                    "turn": 1, "is_error": False, "at": 2.0,
                },
            ],
        }])
        await pilot.pause()

        assert len([r for r in app.activity._rendered_records if r]) == 1
        app.activity.highlighted = 1  # heading is row 0, worker is row 1
        app.activity.action_preview()
        await pilot.pause()

        visible = [r for r in app.activity._rendered_records if r]
        assert len(visible) == 3
        assert visible[1].event_kind == "thinking"
        assert visible[2].event_kind == "tool_call"
        assert "root::researcher" in app.activity.expanded_agents

        app.activity.highlighted = 1
        app.activity.action_preview()
        assert "root::researcher" not in app.activity.expanded_agents


def _snapshot(name: str, status: str, **extra: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "session_id": f"root::{name}",
        "name": name,
        "role_id": "agent_team_sub",
        "status": status,
        "active": status in {"queued", "submitted", "running"},
        "elapsed_s": 0.0,
        "queued": 0,
        "completed": 0,
    }
    payload.update(extra)
    return payload


def test_finished_subagent_duration_freezes_instead_of_ticking() -> None:
    """A worker that reported must stop counting.

    ``describe_sessions_for_task`` publishes the *frozen* duration for a
    finished session, so re-deriving it from the pane's own clock on every
    one-second heartbeat made completed workers appear to run forever.
    """
    now = [1000.0]
    pane = ActivityPane(clock=lambda: now[0])

    pane.update_subagents([_snapshot("scout", "running", elapsed_s=5.0)])
    now[0] = 1010.0
    pane.update_subagents([_snapshot("scout", "ready", elapsed_s=10.0)])
    record = pane.agent_records["root::scout"]
    frozen = record.duration_ms

    for later in (1040.0, 1100.0, 1300.0):
        now[0] = later
        pane.update_subagents([_snapshot("scout", "ready", elapsed_s=10.0)])
        assert record.duration_ms == frozen
        assert pane._duration(record, later) == "10.0s"


def test_finished_subagent_resumes_its_clock_on_the_next_task() -> None:
    """Sessions are reusable, so freezing must not be permanent."""
    now = [1000.0]
    pane = ActivityPane(clock=lambda: now[0])

    pane.update_subagents([_snapshot("scout", "ready", elapsed_s=4.0)])
    now[0] = 1020.0
    pane.update_subagents([_snapshot("scout", "running", elapsed_s=2.0)])
    record = pane.agent_records["root::scout"]

    assert record.state is ActivityState.RUNNING
    assert record.finished_at is None
    now[0] = 1023.0
    assert pane._duration(record, now[0]) == "5.0s"


def test_interrupting_a_turn_settles_sub_agent_rows() -> None:
    """Nothing publishes snapshots after the loop stops.

    A worker left RUNNING would keep the spinner glyph, keep growing its
    elapsed time from the pane's clock, and keep ``refresh_running`` repainting
    the sidebar on every tick for the rest of the session.
    """
    now = [1000.0]
    pane = ActivityPane(clock=lambda: now[0])
    pane.start("collect", "collect_reports", "timeout=1800")
    pane.update_subagents([_snapshot("scout", "running", elapsed_s=5.0)])

    now[0] = 1002.0
    pane.finish_active(ActivityState.INTERRUPTED)
    record = pane.agent_records["root::scout"]

    assert record.state is ActivityState.INTERRUPTED
    now[0] = 2000.0
    assert pane._duration(record, now[0]) == "7.0s"
    pane._render_records = lambda: pytest.fail("idle pane must not repaint")
    pane.refresh_running()


def test_queued_sub_agent_is_not_reported_as_running() -> None:
    pane = ActivityPane(clock=lambda: 0.0)
    pane.update_subagents([_snapshot("scout", "queued", queued=2)])
    record = pane.agent_records["root::scout"]

    assert record.state is ActivityState.QUEUED
    # The backlog count is the state, not an extra fact about it.
    assert record.summary == "queued ×2"
    assert ActivityPane._DISPLAY[ActivityState.QUEUED][0] == GLYPHS["queued"]


def test_unknown_sub_agent_status_is_not_painted_as_success() -> None:
    """A green ✓ asserts the worker delivered; an unknown state asserts
    nothing, so it must not borrow the success colour."""
    view = SubAgentView.from_snapshot(_snapshot("scout", "wedged"))

    assert view.state is ActivityState.SKIPPED
    assert view.label == "wedged"


def test_ready_sub_agent_does_not_double_count_its_report() -> None:
    ready = SubAgentView.from_snapshot(
        _snapshot("scout", "ready", completed=1, queued=1),
    )
    assert ready.details == "report ready · 1 queued"

    two = SubAgentView.from_snapshot(_snapshot("scout", "ready", completed=2))
    assert two.details == "report ready · +1 more"

    idle = SubAgentView.from_snapshot(_snapshot("scout", "idle", completed=3))
    assert idle.details == "idle · 3 ready"


def test_sub_agent_specialty_is_inferred_from_the_coordinator_name() -> None:
    """All Agent Team workers share one ``role_id``; the name the coordinator
    invented is the only signal for what a worker actually does."""
    assert agent_kind("market_research_2") == "research"
    assert agent_kind("code_review_1") == "verify"
    assert agent_kind("draft_writer") == "write"
    assert agent_kind("revenue_data") == "data"
    assert agent_kind("zzz") == "generic"


def test_sub_agent_rows_are_distinguishable_by_marker_and_colour() -> None:
    pane = ActivityPane(clock=lambda: 0.0)
    pane.update_subagents([
        _snapshot("solar_research", "running"),
        _snapshot("solar_writeup", "running"),
    ])
    research = pane.agent_records["root::solar_research"]
    writeup = pane.agent_records["root::solar_writeup"]

    # Same state, same topic — the state glyph cannot tell them apart.
    assert research.state is writeup.state
    assert research.kind != writeup.kind
    assert research.identity != writeup.identity

    rendered = pane._render_row(research, 0.0, 60).plain
    assert agent_kind_glyph("research") in rendered
    # Colour is carried in the row's spans, not its text.
    colours = {
        str(span.style) for span in pane._render_row(research, 0.0, 60).spans
    }
    other = {
        str(span.style) for span in pane._render_row(writeup, 0.0, 60).spans
    }
    assert colours != other


def test_sub_agent_identity_colour_is_stable_across_re_renders() -> None:
    pane = ActivityPane(clock=lambda: 0.0)
    pane.update_subagents([_snapshot("scout", "running")])
    first = pane.agent_records["root::scout"].identity
    pane.update_subagents([_snapshot("scout", "ready")])

    assert pane.agent_records["root::scout"].identity == first


def test_activity_group_headings_carry_live_counts() -> None:
    pane = ActivityPane(clock=lambda: 0.0)
    pane.update_subagents([
        _snapshot("a_research", "running"),
        _snapshot("b_writeup", "ready"),
        _snapshot("c_audit", "failed"),
    ])
    heading = pane._group_heading("SUB-AGENTS", pane.agent_records.values()).plain

    assert heading.startswith("SUB-AGENTS")
    assert "3" in heading
    assert "1 live" in heading
    assert "1 failed" in heading


async def test_activity_reused_call_id_completes_pending_records_in_order() -> None:
    app = FrontierAgentApp(_FakeSession())
    async with app.run_test():
        app.start_activity("reused", "bash", "first")
        app.start_activity("reused", "bash", "second")
        app.finish_activity("reused", "bash", is_error=False, ms=1)

        first, second = app.activity.records[-2:]
        assert first.state is ActivityState.SUCCESS
        assert second.state is ActivityState.RUNNING

        app.finish_activity("reused", "bash", is_error=True, ms=2)
        assert second.state is ActivityState.FAILED


async def test_activity_timeline_is_bounded_but_tool_count_is_session_total() -> None:
    app = FrontierAgentApp(_FakeSession())
    async with app.run_test() as pilot:
        for index in range(150):
            call_id = f"call-{index}"
            app.start_activity(call_id, "bash", f"command {index}")
            app.finish_activity(call_id, "bash", is_error=False, ms=index)
        await pilot.pause()

        assert len(app.activity.records) == 100
        assert app.activity.records[0].call_id == "call-50"
        assert app.activity.records[-1].call_id == "call-149"
        assert app._tools == 150


async def test_activity_timeline_stays_bounded_with_150_running_calls() -> None:
    app = FrontierAgentApp(_FakeSession())
    async with app.run_test():
        for index in range(150):
            app.start_activity(f"call-{index}", "bash", f"command {index}")

        assert len(app.activity.records) == 100
        assert app.activity.records[0].call_id == "call-50"
        assert all(
            record.state is ActivityState.RUNNING
            for record in app.activity.records
        )

        app.finish_activity("call-0", "bash", is_error=False, ms=5)
        assert len(app.activity.records) == 100
        assert app.activity.latest("call-0").state is ActivityState.SUCCESS


async def test_activity_interrupt_closes_running_rows() -> None:
    app = FrontierAgentApp(_FakeSession())
    async with app.run_test() as pilot:
        app.sink.tool_call("bash", {"command": "sleep 10"}, call_id="slow")
        app.sink.interrupted()
        await pilot.pause()

        assert app.activity.latest("slow").state is ActivityState.INTERRUPTED


async def test_skipped_activity_closes_without_tool_result_panel() -> None:
    app = FrontierAgentApp(_FakeSession())
    async with app.run_test() as pilot:
        await app.transcript.clear_all()
        app.sink.tool_call("bash", {"command": "blocked"}, call_id="skip-me")
        app.sink.activity_result(
            "bash", call_id="skip-me", is_error=False, outcome="skipped",
        )
        await pilot.pause()

        assert app.activity.latest("skip-me").state is ActivityState.SKIPPED
        panels = [
            renderable
            for block in app.transcript.query(".block")
            if isinstance(
                (renderable := getattr(block.render(), "_renderable", None)), Panel,
            )
        ]
        assert panels == []


async def test_tool_results_use_progressive_disclosure_and_keep_reviewable_output() -> None:
    app = FrontierAgentApp(_FakeSession())
    long_success = "summary line\n" + "success-tail\n" * 100
    long_error = "failure detail\n" + "error-tail\n" * 100
    async with app.run_test() as pilot:
        await app.transcript.clear_all()
        app.sink.tool_call("bash", {"command": "success"}, call_id="ok")
        app.sink.tool_result(
            "bash", long_success, is_error=False, call_id="ok",
        )
        app.sink.tool_call("bash", {"command": "failure"}, call_id="bad")
        app.sink.tool_result(
            "bash", long_error, is_error=True, call_id="bad",
        )
        await pilot.pause()

        results = list(app.transcript.query(".tool-result"))
        assert len(results) == 2
        success, error = results
        assert isinstance(success, Collapsible) and success.collapsed is True
        assert isinstance(error, Collapsible) and error.collapsed is False
        # The same human label the call row used, not the raw tool name.
        assert "Bash failed" in str(error.title)
        assert "failure detail" in str(error.title)
        assert "101 lines" in str(error.title)

        # Collapsing changes presentation, not retention: review can expand the
        # actual output rather than leaving the user with only a /log pointer.
        success_body = success.query_one(".tool-result-body", Static).render().plain
        error_body = error.query_one(".tool-result-body", Static).render().plain
        assert "summary line" in success_body and "success-tail" in success_body
        assert "failure detail" in error_body and "error-tail" in error_body

        success.collapsed = False
        await pilot.pause()
        assert success.collapsed is False


async def test_long_thinking_auto_collapses_and_can_be_reopened() -> None:
    app = FrontierAgentApp(_FakeSession())
    async with app.run_test() as pilot:
        await app.transcript.clear_all()
        app.sink.thinking_delta("short thought")
        await pilot.pause()

        thinking = app.transcript.query_one(".thinking-block", Collapsible)
        assert thinking.collapsed is False
        assert f"{GLYPHS['thinking']} Thinking" in str(thinking.title)

        app.sink.thinking_delta("\n" + "a long line\n" * 12)
        await pilot.pause()
        assert thinking.collapsed is True

        await app.sink.finish_stream()
        assert "short thought" in str(thinking.title)
        assert "expand to review" in str(thinking.title)

        thinking.collapsed = False
        await pilot.pause()
        assert thinking.collapsed is False
        assert "short thought" in thinking.query_one(
            ".thinking-block-body", Static,
        ).render().plain


async def test_task_process_groups_steps_and_collapses_before_final_report() -> None:
    app = FrontierAgentApp(_FakeSession())
    async with app.run_test() as pilot:
        await app.transcript.clear_all()
        app.sink.begin_task()
        app.sink.thinking_delta("Inspect the inputs")
        app.sink.tool_call("bash", {"command": "pwd"}, call_id="grouped")
        app.sink.tool_result(
            "bash", "/tmp", is_error=False, ms=8, call_id="grouped",
        )
        app.sink.content_delta("The result is ready.")
        app.sink.final("The result is ready.", turns=1, tool_calls=1)
        await pilot.pause()

        process = app.transcript.query_one(".process-group", Collapsible)
        assert process.collapsed is True
        # One thought and one tool call. The result is that call's outcome, not
        # a third step the user took.
        assert "2 steps" in str(process.title)
        assert "complete" in str(process.title)
        assert len(list(process.query(".thinking-block"))) == 1
        assert len(list(process.query(".tool-result"))) == 1
        assert app.transcript.query_one(".final-report").display is True
        assert app.transcript.final_text == "The result is ready."


async def test_transcript_filters_search_and_restore_nested_process() -> None:
    app = FrontierAgentApp(_FakeSession())
    async with app.run_test() as pilot:
        await app.transcript.clear_all()
        app.sink.begin_task()
        app.sink.thinking_delta("Inspect alpha configuration")
        app.sink.tool_call("bash", {"command": "echo beta"}, call_id="filter")
        app.sink.tool_result(
            "bash", "beta result", is_error=False, call_id="filter",
        )
        app.sink.final("Gamma report")
        await pilot.pause()

        assert app.transcript.apply_filter("thinking") == 1
        process = app.transcript.query_one(".process-group", Collapsible)
        assert process.display is True and process.collapsed is False
        assert app.transcript.query_one(".thinking-block").display is True
        assert app.transcript.query_one(".tool-call").display is False

        assert app.transcript.apply_filter("search", "beta result") == 1
        result = app.transcript.query_one(".tool-result", Collapsible)
        assert result.display is True and result.collapsed is False

        app.transcript.apply_filter("report")
        assert process.display is False
        assert app.transcript.query_one(".final-report").display is True

        app.transcript.apply_filter("all")
        assert all(block.display for block in app.transcript.query(".block"))


async def test_process_steps_count_against_the_transcript_block_budget() -> None:
    """Steps nested in a group are widgets too: one long run cannot outgrow the cap."""
    app = FrontierAgentApp(_FakeSession())
    async with app.run_test() as pilot:
        await app.transcript.clear_all()
        app.transcript.max_blocks = 6
        app.sink.begin_task()
        for i in range(30):
            app.sink.tool_call("bash", {"command": f"echo {i}"}, call_id=f"c{i}")
            app.sink.tool_result("bash", f"out {i}", is_error=False, call_id=f"c{i}")
        await pilot.pause()

        assert len(list(app.transcript.query(".block"))) <= 6
        assert app.transcript._pruned_blocks > 0
        # The newest steps are the ones kept.
        assert "out 29" in app.transcript._plain_text(
            list(app.transcript.query(".tool-result"))[-1],
        )


async def test_find_matches_plain_blocks_and_collapsed_titles() -> None:
    """Bare Statics and folded titles are searchable, not just expanded bodies."""
    app = FrontierAgentApp(_FakeSession())
    async with app.run_test() as pilot:
        await app.transcript.clear_all()
        app.sink.begin_task()
        app.sink.tool_call("bash", {"command": "echo unicorn"}, call_id="f1")
        app.sink.tool_result("bash", "zebra output", is_error=False, call_id="f1")
        app.sink.echo_user("unicorn question")
        await pilot.pause()

        # Present only in a bare Static tool-call line and the echoed prompt.
        assert app.transcript.apply_filter("search", "unicorn") == 2
        # Present only in the folded result's summary title.
        assert app.transcript.apply_filter("search", "zebra") == 1


async def test_filter_resets_when_a_new_run_starts() -> None:
    """A stale filter would mount new blocks visible among hidden ones."""
    app = FrontierAgentApp(_FakeSession())
    async with app.run_test() as pilot:
        await app.transcript.clear_all()
        app.sink.begin_task()
        app.sink.thinking_delta("alpha reasoning")
        app.sink.tool_call("bash", {"command": "pwd"}, call_id="r1")
        app.sink.final("first answer")
        await pilot.pause()
        done = app.transcript.query_one(".process-group", Collapsible)
        assert done.collapsed is True

        assert app.transcript.apply_filter("thinking") == 1
        assert app.transcript.filter_mode == "thinking"
        assert done.collapsed is False

        app.sink.begin_task()
        await pilot.pause()
        assert app.transcript.filter_mode == "all"
        assert all(block.display for block in app.transcript.query(".block"))
        # The filter forced the finished run open; resetting hands it back.
        assert done.collapsed is True
        app._refresh_transcript_view()
        assert "View: all" in app.query_one("#transcript-view", Static).render().plain


async def test_clicking_a_collapsible_returns_focus_to_the_prompt() -> None:
    """Collapsible titles are focusable, so a click would swallow the next keystroke."""
    app = FrontierAgentApp(_FakeSession())
    async with app.run_test() as pilot:
        await app.transcript.clear_all()
        app.sink.begin_task()
        app.sink.tool_call("bash", {"command": "pwd"}, call_id="k1")
        app.sink.tool_result("bash", "ok", is_error=False, call_id="k1")
        await pilot.pause()

        await pilot.click(app.transcript.query(CollapsibleTitle).first())
        await pilot.pause()
        assert app.focused is app.query_one("#prompt", Input)
        await pilot.press("h", "i")
        assert app.query_one("#prompt", Input).value == "hi"


async def test_interrupted_and_failed_runs_keep_their_status() -> None:
    """An answer arriving later must not relabel a broken run as a clean one."""
    app = FrontierAgentApp(_FakeSession())
    async with app.run_test() as pilot:
        await app.transcript.clear_all()
        app.sink.begin_task()
        app.sink.tool_call("bash", {"command": "sleep"}, call_id="s1")
        await pilot.pause()
        process = app.transcript.query_one(".process-group", Collapsible)
        app.sink.interrupted()
        app.sink.finish_task()
        await pilot.pause()
        assert "interrupted" in str(process.title)
        assert GLYPHS["ok"] not in str(process.title)
        assert process.collapsed is False

        app.sink.begin_task()
        app.sink.tool_call("bash", {"command": "boom"}, call_id="s2")
        await pilot.pause()
        failed = list(app.transcript.query(".process-group"))[-1]
        app.sink.error("boom")
        app.sink.final("recovered answer")
        app.sink.finish_task()
        await pilot.pause()
        assert "issues found" in str(failed.title)
        assert failed.collapsed is False


async def test_errors_filter_includes_run_level_failures() -> None:
    """A run that died outside a tool is exactly what /filter errors is for."""
    app = FrontierAgentApp(_FakeSession())
    async with app.run_test() as pilot:
        await app.transcript.clear_all()
        app.sink.begin_task()
        app.sink.tool_call("bash", {"command": "ok"}, call_id="e1")
        app.sink.tool_result("bash", "fine", is_error=False, call_id="e1")
        app.sink.error("the provider hung up")
        await pilot.pause()

        assert app.transcript.apply_filter("errors") == 1
        failure = app.transcript.query_one(".transcript-error")
        assert failure.display is True
        assert "the provider hung up" in app.transcript._plain_text(failure)


async def test_review_cursor_is_visible_and_survives_new_blocks() -> None:
    """The cursor tracks a widget: later blocks must not shift it silently."""
    app = FrontierAgentApp(_FakeSession())
    async with app.run_test() as pilot:
        await app.transcript.clear_all()
        app.sink.begin_task()
        app.sink.tool_call("bash", {"command": "first"}, call_id="v1")
        app.sink.tool_result("bash", "one", is_error=False, call_id="v1")
        await pilot.pause()

        app.action_review_previous()  # Enters at the newest block.
        selected = app.transcript.query_one(".review-active")
        assert selected.has_class("tool-result")
        app.action_review_previous()
        moved = app.transcript.query_one(".review-active")
        assert moved.has_class("tool-call")
        assert len(list(app.transcript.query(".review-active"))) == 1

        app.sink.tool_call("bash", {"command": "second"}, call_id="v2")
        await pilot.pause()
        assert app.transcript.query_one(".review-active") is moved

        app.action_review_toggle()
        assert app.transcript.query_one(".review-active") is moved


async def test_copy_report_works_without_streamed_prose() -> None:
    """/copy reads the answer the model returned, not whatever got promoted."""
    app = FrontierAgentApp(_FakeSession())
    async with app.run_test() as pilot:
        await app.transcript.clear_all()
        app.sink.begin_task()
        app.sink.content_delta("streamed prose")
        app.sink.final("the authoritative answer", turns=1)
        await pilot.pause()

        assert app.transcript.final_text == "the authoritative answer"
        assert app.transcript.query_one(".final-report") is not None


async def test_elapsed_freezes_then_terminal_status_returns_to_idle() -> None:
    app = FrontierAgentApp(_FakeSession())
    now = [100.0]
    app.presentation.clock = lambda: now[0]
    async with app.run_test() as pilot:
        app.sink.begin_task()
        now[0] += 1.2
        app._refresh_status()
        assert "1s" in app.status.render().plain

        app.sink.final("done")
        app._refresh_status()
        frozen = app.presentation.elapsed_seconds()
        assert frozen == 1

        now[0] += 1.0
        app._refresh_status()
        assert app.presentation.elapsed_seconds() == frozen
        assert app.presentation.phase is PresentationPhase.DONE

        now[0] += 1.1
        app._refresh_status()
        await pilot.pause()
        assert app.presentation.phase is PresentationPhase.IDLE
        assert app.presentation.elapsed_seconds() is None


async def test_recoverable_error_and_hidden_thinking_resume_task_phase() -> None:
    app = FrontierAgentApp(_FakeSession())
    async with app.run_test():
        app.sink.begin_task()
        started_at = app.presentation.task_started_at
        app.sink.error("temporary tool failure")
        app._refresh_status()
        assert app.presentation.phase is PresentationPhase.ERROR
        assert "error" in app.status.render().plain

        app.sink.working_on()
        assert app.presentation.phase is PresentationPhase.THINKING
        assert app.presentation.task_started_at == started_at

        app.sink.content_delta("answer")
        assert app.presentation.phase is PresentationPhase.RESPONDING
        app.sink.set_verbose(False)
        app.sink.thinking_delta("hidden reasoning")
        assert app.presentation.phase is PresentationPhase.THINKING


async def test_welcome_identifies_local_byok_demo_without_login() -> None:
    app = FrontierAgentApp(_FakeSession())
    async with app.run_test() as pilot:
        await pilot.pause()
        transcript = "\n".join(
            block.render().plain for block in app.transcript.query(".block")
        )
        assert "local BYOK demo" in transcript
        assert "/config" in transcript
        assert "login" not in transcript.lower()


async def test_prompt_history_restores_draft_after_navigation() -> None:
    app = FrontierAgentApp(_FakeSession())
    async with app.run_test() as pilot:
        app._remember_input("first task")
        app._remember_input("second task")
        prompt = app.query_one("#prompt", Input)
        prompt.value = "unfinished draft"

        await pilot.press("up")
        assert prompt.value == "second task"
        await pilot.press("up")
        assert prompt.value == "first task"
        await pilot.press("down")
        assert prompt.value == "second task"
        await pilot.press("down")
        assert prompt.value == "unfinished draft"


async def test_prompt_history_replays_multiline_input_without_flattening() -> None:
    session = _FakeSession()
    app = FrontierAgentApp(session)
    pasted = "line one\nline two\nline three"
    async with app.run_test() as pilot:
        app._remember_input(pasted)
        prompt = app.query_one("#prompt", Input)

        await pilot.press("up")
        assert "Pasted text #1" in prompt.value

        await pilot.press("enter")
        await _wait_until(lambda: session.tasks == [pasted])


async def test_slash_command_hint_and_tab_completion() -> None:
    app = FrontierAgentApp(_FakeSession())
    async with app.run_test() as pilot:
        prompt = app.query_one("#prompt", Input)
        prompt.value = "/the"
        await pilot.pause()
        hint = app.query_one("#command-hint", Static)
        assert hint.display is True
        assert "/theme" in hint.render().plain

        await pilot.press("tab")
        assert prompt.value == "/theme "
        assert prompt.cursor_position == len(prompt.value)

        prompt.value = "/conf"
        await pilot.pause()
        assert "/config" in hint.render().plain
        await pilot.press("tab")
        assert prompt.value == "/config"


async def test_command_palette_inserts_selected_command() -> None:
    app = FrontierAgentApp(_FakeSession())
    async with app.run_test() as pilot:
        await pilot.press("ctrl+p")
        await _wait_until(lambda: isinstance(app.screen, CommandScreen))
        command_list = app.screen.query_one("#cmd-list", OptionList)
        command_list.highlighted = 2  # /model
        await pilot.press("enter")
        await _wait_until(lambda: not isinstance(app.screen, CommandScreen))
        prompt = app.query_one("#prompt", Input)
        assert prompt.value == "/model "
        assert prompt.has_focus

        await pilot.press("ctrl+p")
        await _wait_until(lambda: isinstance(app.screen, CommandScreen))
        command_list = app.screen.query_one("#cmd-list", OptionList)
        command_ids = {
            command_list.get_option_at_index(index).id
            for index in range(command_list.option_count)
        }
        assert "/config" in command_ids
        await pilot.press("escape")


async def test_f1_help_opens_and_closes() -> None:
    app = FrontierAgentApp(_FakeSession())
    async with app.run_test() as pilot:
        await pilot.press("f1")
        await _wait_until(lambda: isinstance(app.screen, HelpScreen))
        help_text = app.screen.query_one("#help-body Static", Static).render().plain
        assert "/config shows safe local settings" in help_text
        await pilot.press("f1")
        await _wait_until(lambda: not isinstance(app.screen, HelpScreen))
        assert app.query_one("#prompt", Input).has_focus


async def test_streaming_deltas_batch_and_flush_without_losing_text() -> None:
    """High-frequency deltas are coalesced and lose no text."""
    app = FrontierAgentApp(_FakeSession())
    async with app.run_test() as pilot:
        for fragment in ("**hello", " world", "**\n", "```python\n", "x = 1\n", "```\n"):
            app.sink.content_delta(fragment)
        await _wait_until(lambda: "x = 1" in app.transcript._buf)
        await app.transcript.end_stream()
        await pilot.pause()

        assert app.transcript.query_one(Static) is not None
        assert app.sink._stream_chunks == []
        assert app.sink._stream_render_pending is False


async def test_long_session_prunes_old_transcript_widgets() -> None:
    """Mounted widgets stay bounded even though the session keeps full history."""
    app = FrontierAgentApp(_FakeSession())
    async with app.run_test() as pilot:
        await app.transcript.clear_all()
        app.transcript.max_blocks = 12
        for index in range(50):
            await app.transcript.add(Text(f"block {index}"))
        await pilot.pause()

        assert len(app.transcript.query(".block")) == 12
        assert app.transcript._pruned_blocks == 38
        assert len(app.transcript.query(".transcript-pruned")) == 1


async def test_large_history_replays_complete_window() -> None:
    session = _FakeSession()
    session.history = [user_msg(f"message {index}") for index in range(80)]
    app = FrontierAgentApp(session)
    async with app.run_test() as pilot:
        app.transcript.max_blocks = 10
        shown = await app.transcript.replay_history(session.history)
        await pilot.pause()

        blocks = list(app.transcript.query(".history-user"))
        assert shown == len(blocks) == 80
        assert blocks[0].render().plain == "› message 0"
        assert blocks[-1].render().plain == "› message 79"
        assert app.transcript._pruned_blocks == 0


async def test_narration_turns_become_separate_blocks_one_rule_apart() -> None:
    """Prose is split per assistant message, not run together with blank lines."""
    app = FrontierAgentApp(_FakeSession())
    async with app.run_test(size=(84, 30)) as pilot:
        await app.transcript.clear_all()
        for segment in ("first message.\n\n\n\n", "\n\nsecond message.\n\n\n",
                        "\n\n\nthird message.\n"):
            app.sink.content_delta(segment)
            await app.sink.finish_stream()
            app.sink.end_turn_text()
            await pilot.pause()

        blocks = list(app.transcript.query(".assistant-content"))
        assert len(blocks) == 3
        assert not blocks[0].has_class("assistant-continued")
        assert all(block.has_class("assistant-continued") for block in blocks[1:])
        # One prose row each, plus the single rule row a continuation adds — not
        # the four-to-six blank rows the model's own spacing used to occupy.
        assert [block.region.height for block in blocks] == [1, 2, 2]


async def test_tool_only_turns_leave_no_empty_narration_blocks() -> None:
    """A reasoning workflow's tool-only turns must not each cost a rule.

    Replays the shape of a real ``stateful_react`` run: seven turns whose
    visible channel carried nothing but a newline, then the report.
    """
    app = FrontierAgentApp(_FakeSession())
    async with app.run_test(size=(84, 30)) as pilot:
        await app.transcript.clear_all()
        for _ in range(7):
            app.sink.content_delta("\n")
            await app.sink.finish_stream()
            app.sink.end_turn_text()
            await pilot.pause()
        app.sink.content_delta("## Report\n\nthe answer.\n")
        await app.sink.finish_stream()
        await pilot.pause()

        blocks = list(app.transcript.query(".assistant-content"))
        assert len(blocks) == 1
        assert not blocks[0].has_class("assistant-continued")


async def test_whitespace_only_narration_block_is_removed_on_close() -> None:
    """Whitespace can also open the block one delta before any prose arrives."""
    app = FrontierAgentApp(_FakeSession())
    async with app.run_test() as pilot:
        await app.transcript.clear_all()
        await app.transcript.stream("content", "prose")
        app.transcript._buf = "   \n\n"  # provider revised the turn to nothing
        await app.transcript.end_stream()
        await pilot.pause()

        assert not list(app.transcript.query(".assistant-content"))


async def test_live_narration_collapses_the_models_blank_line_runs() -> None:
    """The pre-Markdown live block is tightened too — that is what streams."""
    app = FrontierAgentApp(_FakeSession())
    async with app.run_test() as pilot:
        await app.transcript.clear_all()
        app.sink.content_delta("one.\n\n\n\n\ntwo.\n\n\n")
        await _wait_until(lambda: "two." in app.transcript._buf)
        await pilot.pause()

        live = app.transcript.query_one(".assistant-content", Static)
        assert live.render().plain == "one.\n\ntwo."


async def test_promoted_report_drops_the_continuation_rule() -> None:
    """The final-report heading owns that seam; two rules would stack on it."""
    app = FrontierAgentApp(_FakeSession())
    async with app.run_test() as pilot:
        await app.transcript.clear_all()
        for segment in ("draft note.", "final answer."):
            app.sink.content_delta(segment)
            await app.sink.finish_stream()
            app.sink.end_turn_text()
            await pilot.pause()
        await app.transcript.promote_last_content()
        await pilot.pause()

        report = list(app.transcript.query(".final-report"))[-1]
        assert not report.has_class("assistant-continued")
        assert len(app.transcript.query(".report-heading")) == 1


async def test_many_stream_fragments_coalesce_into_one_bounded_block() -> None:
    """A tens-of-thousands-of-token-like burst does not create token widgets."""
    app = FrontierAgentApp(_FakeSession())
    async with app.run_test() as pilot:
        await app.transcript.clear_all()
        fragments = ["0123456789" for _ in range(10_000)]
        for fragment in fragments:
            app.sink.content_delta(fragment)
        await app.sink.finish_stream()
        await pilot.pause()

        assert len(app.transcript.query(".block")) == 1
        assert app.sink._stream_chunks == []
        assert app.transcript._live is None


async def test_pathological_markdown_is_bounded_for_frontend_rendering() -> None:
    text = "x" * (_MAX_RENDERED_MARKDOWN_CHARS + 123)
    rendered = _bounded_markdown(text)
    assert len(rendered.markup) < len(text) + 200
    assert "123 additional characters hidden" in rendered.markup


async def test_tui_theme_and_responsive_sidebar() -> None:
    app = FrontierAgentApp(_FakeSession(), theme="light")
    async with app.run_test(size=(120, 30)) as pilot:
        # ``dark`` / ``light`` are our own registered palettes, not Textual's
        # built-ins — those had no Rich half, so the transcript fell back to
        # Catppuccin and painted dark colours onto a light background.
        assert app.theme == "light"
        assert app.query_one("#sidebar").display is True

        await pilot.resize_terminal(80, 24)
        assert app.query_one("#sidebar").display is False

        app.query_one("#prompt", Input).value = "/theme dark"
        await pilot.press("enter")
        await _wait_until(lambda: app.theme == "dark")


async def test_light_theme_paints_rich_content_from_the_light_palette() -> None:
    """Regression: selecting ``light`` used to leave every Rich renderable on
    the Catppuccin *dark* palette, because ``light`` had no palette of its own."""
    from apodex.tui.themes import palette

    app = FrontierAgentApp(_FakeSession(), theme="light")
    async with app.run_test() as pilot:
        await pilot.pause()
        light = palette("light")
        for role, token in (("ok", light.success), ("err", light.error),
                            ("muted", light.muted), ("text", light.foreground)):
            assert app.sink._style(role) == token, role
        assert app.sink._style("ok") != palette("catppuccin").success


async def test_agent_team_task_board_is_kept_in_the_sidebar() -> None:
    app = FrontierAgentApp(_FakeSession())
    async with app.run_test(size=(120, 30)) as pilot:
        app.sink.begin_task()
        app.sink.tool_call("add_task", {
            "tasks": [
                {"description": "find primary sources"},
                {"description": "verify licenses"},
            ],
        }, call_id="board-add")
        app.sink.tool_result("add_task", "Added ['t1', 't2'].", is_error=False,
                             call_id="board-add")
        await pilot.pause()

        board = app.todos_pane.render().plain
        assert "t1 find primary sources" in board
        assert "t2 verify licenses" in board
        process = app.transcript.query_one(".process-group", Collapsible)
        assert len(list(process.query(".tool-call"))) == 1
        assert len(list(process.query(".tool-result"))) == 0

        app.sink.tool_call("update_task", {
            "updates": [
                {"id": "t1", "resolution": "in_progress"},
                {"id": "t2", "resolution": "cancelled"},
            ],
        }, call_id="board-update")
        app.sink.tool_result("update_task", "Updated ['t1', 't2'].", is_error=False,
                             call_id="board-update")
        await pilot.pause()
        board = app.todos_pane.render().plain
        assert "▶ t1 find primary sources" in board
        assert "⊘ t2 verify licenses" in board
        assert len(list(process.query(".tool-call"))) == 2
        assert len(list(process.query(".tool-result"))) == 0


async def test_task_board_applies_updates_the_model_double_encoded() -> None:
    """Models serialise list elements as JSON strings on some calls but not
    others; a strict isinstance(dict) check dropped exactly those, leaving the
    sidebar showing a task as unfinished after the board had resolved it."""
    app = FrontierAgentApp(_FakeSession())
    async with app.run_test(size=(120, 30)) as pilot:
        app.sink.begin_task()
        app.sink.tool_call("add_task", {
            "tasks": [
                {"description": "explain money creation"},
                '{"description": "provide a learning path"}',
            ],
        }, call_id="board-add")
        app.sink.tool_result("add_task", "Added ['t1', 't2'].", is_error=False,
                             call_id="board-add")
        await pilot.pause()
        assert "t2 provide a learning path" in app.todos_pane.render().plain

        app.sink.tool_call("update_task", {
            "updates": ['{"id":"t1","resolution":"in_progress"}'],
        }, call_id="board-progress")
        app.sink.tool_result("update_task", "Updated ['t1'].", is_error=False,
                             call_id="board-progress")
        await pilot.pause()
        assert "▶ t1 explain money creation" in app.todos_pane.render().plain

        app.sink.tool_call("update_task", {
            "updates": [
                {"id": "t1", "resolution": "resolved"},
                '{"id":"t2","resolution":"resolved"}',
            ],
        }, call_id="board-resolve")
        app.sink.tool_result("update_task", "Updated ['t1', 't2'].", is_error=False,
                             call_id="board-resolve")
        await pilot.pause()

        board = app.todos_pane.render().plain
        assert f"{GLYPHS['ok']} t1 explain money creation" in board
        assert f"{GLYPHS['ok']} t2 provide a learning path" in board
        # The heading is what the user reads as "did it finish?".
        assert app.todos_pane.summary() == "2/2"


async def test_task_board_ignores_a_bare_object_where_a_list_belongs() -> None:
    """add_task rejects this shape outright, so the pane must add nothing —
    accepting it here would rebuild the pane/board divergence from the other
    direction, showing a task the board never registered."""
    app = FrontierAgentApp(_FakeSession())
    async with app.run_test(size=(120, 30)) as pilot:
        app.sink.begin_task()
        app.sink.tool_call("add_task", {"tasks": {"description": "not in a list"}},
                           call_id="board-bare")
        app.sink.tool_result("add_task", "Error: add_task expects a list of objects",
                             is_error=False, call_id="board-bare")
        await pilot.pause()
        assert "not in a list" not in app.todos_pane.render().plain
        assert app.todos_pane.summary() == ""


async def test_thinking_body_uses_quieter_tier_than_operational_detail() -> None:
    app = FrontierAgentApp(_FakeSession(), theme="dark")
    async with app.run_test() as pilot:
        await app.transcript.clear_all()
        app.sink.thinking_delta("considering the next step")
        await pilot.pause()

        body = app.transcript.query_one(".thinking-block-body", Static)
        assert str(body.content.style) == rich_style("dark", "subtle")
        assert rich_style("dark", "subtle") != rich_style("dark", "muted")


@pytest.mark.parametrize("theme", ["catppucin", "catppucin-latte", "tokyo-night", "tokyo-night-day", "dracula", "nord", "gruvbox", "gruvbox-light", "one-dark", "one-light", "solarized", "solarized-light"])
async def test_tui_supports_curated_themes(theme: str) -> None:
    app = FrontierAgentApp(_FakeSession(), theme=theme)
    async with app.run_test() as pilot:
        assert app.theme == theme
        app.query_one("#prompt", Input).value = f"/theme {theme}"
        await pilot.press("enter")
        await _wait_until(lambda: app.theme == theme)


async def test_gruvbox_uses_its_actual_palette_for_tui_and_rich_content() -> None:
    app = FrontierAgentApp(_FakeSession(), theme="gruvbox")
    async with app.run_test() as pilot:
        await pilot.pause()
        palette = app.get_theme(app.theme)
        # Gruvbox verbatim, on both halves of the UI — see
        # ``test_themes.py::test_semantic_colors_match_upstream_exactly``.
        assert palette.background == "#282828"
        assert palette.primary == "#83a598"
        assert palette.accent == "#fe8019"
        assert "#fabd2f" in app.sink._style("tool")
        # …and the warm text ramp, not a neutral grey.
        assert app.sink._style("muted") == "#bdae93"
        assert "#b8bb26" in app.sink._style("ok")


async def test_sidebar_toggle_persists_across_resize() -> None:
    app = FrontierAgentApp(_FakeSession())
    async with app.run_test(size=(120, 30)) as pilot:
        sidebar = app.query_one("#sidebar")
        assert sidebar.display is True
        await pilot.press("ctrl+b")
        assert sidebar.display is False

        await pilot.resize_terminal(80, 24)
        await pilot.resize_terminal(120, 30)
        assert sidebar.display is False

        await pilot.press("ctrl+b")
        assert sidebar.display is True


async def test_very_small_terminal_remains_operable() -> None:
    """A cramped SSH split should degrade layout without crashing or losing input."""
    app = FrontierAgentApp(_FakeSession())
    async with app.run_test(size=(40, 10)) as pilot:
        await pilot.pause()
        assert app.query_one("#sidebar").display is False
        prompt = app.query_one("#prompt", Input)
        assert prompt.display is True
        assert prompt.has_focus
        assert app.status.size.height == 1
        assert "\n" not in app.status.render().plain
        assert cell_len(app.status.render().plain) <= 38


@pytest.mark.parametrize(
    "width", [120, 99, 80, 60, 40],
)
async def test_status_bar_uses_responsive_information_tiers(
    width: int,
) -> None:
    session = _FakeSession()
    session.cfg.model = "a-very-long-demo-model-name"
    session.usage.total = 10
    app = FrontierAgentApp(session)
    async with app.run_test(size=(width, 10)) as pilot:
        app.sink.begin_task()
        app.sink.tool_call("bash", {"command": "echo a very long command for the footer"})
        app.presentation.transition(
            PresentationPhase.AWAITING_APPROVAL,
            tool=app.presentation.current_tool,
        )
        app._refresh_status()
        await pilot.pause()

        status = app.status.render().plain
        assert "approval" in status
        assert "a-very" not in status
        assert "0 tokens" not in status
        assert ("ctx 50%" in status) is (width >= 60)
        assert app.status.size.height == 1
        assert cell_len(status) <= max(1, width - 2)
        prompt = app.query_one("#prompt", Input)
        assert prompt.display is True


async def test_status_bar_keeps_all_wide_fields_at_100_columns() -> None:
    session = _FakeSession()
    session.cfg.model = "a-very-long-demo-model-name"
    session.usage.total = 10
    app = FrontierAgentApp(session)
    async with app.run_test(size=(100, 10)) as pilot:
        app.sink.begin_task()
        app.sink.tool_call("bash", {"command": "pwd"})
        app.presentation.transition(
            PresentationPhase.AWAITING_APPROVAL,
            tool=app.presentation.current_tool,
        )
        app._refresh_status()
        await pilot.pause()
        app._refresh_status()

        status = app.status.render().plain
        expected_fields = (
            "approval", "bash", "coding", "ctx 50%", "t 1", "q 0",
        )
        for expected in expected_fields:
            assert expected in status
        assert "a-very" not in status and "0 tokens" not in status
        assert cell_len(status) <= 98


async def test_resumed_app_replays_human_conversation() -> None:
    session = _FakeSession()
    session.history = [
        system_msg("private system prompt"),
        user_msg("continue the refactor"),
        assistant_msg(
            "<think>private reasoning</think>\n## Done\n\nUpdated the parser.",
            tool_calls=[{
                "id": "call-1", "type": "function",
                "function": {"name": "bash", "arguments": '{"command":"pwd"}'},
            }],
        ),
        {"role": "tool", "tool_call_id": "call-1", "content": "/tmp"},
    ]
    app = FrontierAgentApp(session, resumed=True)
    async with app.run_test() as pilot:
        await pilot.pause()
        assert len(app.transcript.query(".history-user")) == 1
        assert len(app.transcript.query(".history-assistant")) == 1
        assert len(app.transcript.query(".history-system")) == 0
        assistant_block = app.transcript.query_one(".history-assistant", Static)
        assistant_renderable = getattr(assistant_block.render(), "_renderable", None)
        assert "private reasoning" not in getattr(assistant_renderable, "markup", "")
        assert len(app.transcript.query(".history-tool-call")) == 1
        assert len(app.transcript.query(".tool-result")) == 1
        assert len(app.activity.records) == 1
        assert app.activity.records[0].name == "bash"
        assert app.activity.records[0].state == ActivityState.SUCCESS


async def test_resume_restores_grouped_coordinator_and_subagent_activity() -> None:
    session = _FakeSession()
    session.mode = "agent_team"
    session.history = [
        user_msg("research the market"),
        assistant_msg("Delegating research.", tool_calls=[{
            "id": "call-team", "type": "function",
            "function": {
                "name": "collect_reports",
                "arguments": '{"task_id":"task-1"}',
            },
        }]),
        {
            "role": "tool", "name": "collect_reports",
            "tool_call_id": "call-team", "content": "1 report ready",
            "duration_ms": 1250,
        },
        assistant_msg("Research complete."),
    ]
    session.tui_state = {
        "version": 1,
        "subagents": [
            {
                "session_id": "root::market_research",
                "name": "market_research",
                "role_id": "agent_team_sub",
                "status": "ready",
                "active": False,
                "elapsed_s": 1.1,
                "queued": 0,
                "completed": 1,
                "events": [{
                    "id": "event-1", "kind": "tool_call", "title": "web_search",
                    "detail": "market size", "turn": 1, "is_error": False,
                }],
            },
            {
                "session_id": "root::unfinished_writer",
                "name": "unfinished_writer",
                "status": "running",
                "active": True,
                "elapsed_s": 0.5,
            },
        ],
    }

    app = FrontierAgentApp(session, resumed=True)
    async with app.run_test() as pilot:
        await pilot.pause()

        processes = list(app.transcript.query(".process-group"))
        assert len(processes) == 1
        process = processes[0]
        assert process.collapsed is True
        assert "1 step" in str(process.title)
        assert len(list(process.query(".history-tool-call"))) == 1
        assert len(list(process.query(".tool-result"))) == 1

        assert [record.name for record in app.activity.records] == [
            "collect_reports",
        ]
        worker = app.activity.agent_records["root::market_research"]
        assert worker.state is ActivityState.SUCCESS
        assert app.activity.agent_records[
            "root::unfinished_writer"
        ].state is ActivityState.INTERRUPTED
        assert app.sink.snapshot_state()["subagents"][1]["status"] == "aborted"
        assert "root::market_research" in app.activity.agent_events
        assert next(iter(
            app.activity.agent_events["root::market_research"].values()
        )).event_kind == "tool_call"


async def test_resume_switches_chrome_and_replays_history() -> None:
    session = _FakeSession()
    app = FrontierAgentApp(session)
    state = {
        "session_id": "saved-session",
        "name": "Parser experiment",
        "mode": "research",
        "cwd": "/saved/project",
        "model": "other-model",
        "plan_active": True,
        "history": [user_msg("old question"), assistant_msg("old answer")],
    }
    async with app.run_test() as pilot:
        app.start_activity("old-call", "bash", "pwd")
        assert app._tools == 1
        assert await app._apply_resume_state("fallback", state) is True
        await pilot.pause()
        title = app.query_one("#title", Static).render().plain
        assert "research" in title and "Parser experiment" in title and "/saved/project" in title
        assert session.cfg.model == "other-model"
        assert "other-model" not in app.status.render().plain
        assert len(app.transcript.query(".history-user")) == 1
        assert len(app.transcript.query(".history-assistant")) == 1
        assert app.activity.records == []
        assert app._tools == 0
        assert app.query_one("#prompt", Input).has_focus


async def test_resume_refreshes_deliverables_from_new_session_directory(
    monkeypatch, tmp_path,
) -> None:
    first = tmp_path / "session-one"
    second = tmp_path / "session-two"
    first.mkdir()
    second.mkdir()
    (first / "old.txt").write_text("old")
    target = second / "deck.pptx"
    target.write_bytes(b"pptx")
    monkeypatch.setenv("FRONTIER_AGENT_OUTPUTS_DIR", str(first))
    session = _FakeSession()
    original_switch = session.switch_session

    def switch(state: dict, *, fallback_id: str = "") -> None:
        original_switch(state, fallback_id=fallback_id)
        monkeypatch.setenv("FRONTIER_AGENT_OUTPUTS_DIR", str(second))

    session.switch_session = switch  # type: ignore[method-assign]
    app = FrontierAgentApp(session)
    state = {
        "session_id": "session-two",
        "history": [user_msg("old question"), assistant_msg("old answer")],
    }

    async with app.run_test() as pilot:
        assert await app._apply_resume_state("session-two", state) is True
        await pilot.pause()
        assert app._deliverables_root == second
        assert app.deliverables._files == [target]


async def test_interrupt_finalizes_stream_and_restores_prompt_focus() -> None:
    session = _FakeSession()
    session.hold.clear()
    app = FrontierAgentApp(session)
    async with app.run_test() as pilot:
        prompt = app.query_one("#prompt", Input)
        prompt.value = "long task"
        await pilot.press("enter")
        await _wait_until(lambda: app.busy and "hello" in app.transcript._buf)

        await pilot.press("ctrl+c")
        await _wait_until(lambda: not app.busy)
        await pilot.pause()

        assert app.transcript._live is None
        assert app.transcript._live_kind is None
        assert app.presentation.phase is PresentationPhase.INTERRUPTED
        assert "interrupted" in app.status.render().plain
        assert prompt.has_focus


async def test_clear_command_resets_activity_timeline_and_tool_count() -> None:
    app = FrontierAgentApp(_FakeSession())
    async with app.run_test() as pilot:
        prompt = app.query_one("#prompt", Input)
        app.sink.tool_call("bash", {"command": "pwd"}, call_id="clear-me")
        app.sink.tool_result("bash", "/tmp", is_error=False, call_id="clear-me")
        await pilot.pause()
        assert app._tools == 1
        assert app.activity.records

        prompt.value = "/clear"
        await pilot.press("enter")
        await pilot.pause()
        assert app._tools == 0
        assert app.activity.records == []


@pytest.mark.parametrize(
    "key,check",
    [
        ("y", lambda o: o.decision.approved and not o.all_session and not o.decision.remember),
        ("n", lambda o: not o.decision.approved),
        ("a", lambda o: o.decision.approved and o.all_session),
    ],
)
async def test_approval_single_keys(key: str, check) -> None:
    app = FrontierAgentApp(_FakeSession())
    async with app.run_test() as pilot:
        result: dict[str, ApprovalOutcome] = {}
        app.push_screen(ApprovalScreen("bash", "run ls"), lambda o: result.update(o=o))
        await pilot.pause()
        await pilot.press(key)
        await _wait_until(lambda: "o" in result)
        assert check(result["o"])


async def test_approval_redirect_feedback() -> None:
    app = FrontierAgentApp(_FakeSession())
    async with app.run_test() as pilot:
        result: dict[str, ApprovalOutcome] = {}
        screen = ApprovalScreen("write_file", "overwrite")
        app.push_screen(screen, lambda o: result.update(o=o))
        await pilot.pause()
        await pilot.press("e")  # reveal the redirect input
        await pilot.pause()
        screen.query_one("#ap-input", Input).value = "edit it instead"
        await pilot.press("enter")
        await _wait_until(lambda: "o" in result)
        assert result["o"].decision.approved is False
        assert result["o"].decision.feedback == "edit it instead"


async def test_approval_dangerous_requires_yes() -> None:
    app = FrontierAgentApp(_FakeSession())
    async with app.run_test() as pilot:
        result: dict[str, ApprovalOutcome] = {}
        screen = ApprovalScreen("bash", "rm -rf", dangerous="destructive shell command")
        app.push_screen(screen, lambda o: result.update(o=o))
        await pilot.pause()
        screen.query_one("#ap-input", Input).value = "yes"
        await pilot.press("enter")
        await _wait_until(lambda: "o" in result)
        assert result["o"].decision.approved is True


async def test_approval_hotkey_letters_survive_markup() -> None:
    """Regression: the y/n/a/A/e hint letters must not be eaten as markup tags
    (the old ``[y]es · [n]o …`` markup string rendered as ``es · o …``)."""
    screen = ApprovalScreen("write_file", "overwrite")
    hint = screen._hint().plain
    for key in ("y", "n", "a", "A", "e"):
        assert key in hint, f"hotkey {key!r} missing from hint: {hint!r}"
    # The header renders the tool name literally (no markup mangling).
    assert "write_file" in screen._header().plain


async def test_approval_shows_diff_preview_in_modal() -> None:
    """The proposed diff is rendered *inside* the modal, not just the transcript."""
    app = FrontierAgentApp(_FakeSession())
    async with app.run_test() as pilot:
        screen = ApprovalScreen(
            "write_file", "overwrite",
            preview="--- a\n+++ b\n-old line\n+new line\n", preview_kind="diff",
        )
        app.push_screen(screen)
        await pilot.pause()
        # The preview widget is mounted inside the modal and carries the diff.
        assert screen.query_one("#ap-preview-body", Static) is not None
        plain = screen._preview_renderable().plain
        assert "old line" in plain and "new line" in plain


async def test_approval_defaults_to_no_and_summarizes_large_diff() -> None:
    app = FrontierAgentApp(_FakeSession())
    diff = "--- a\n+++ b\n" + "".join(f"-old {i}\n+new {i}\n" for i in range(80))
    async with app.run_test(size=(100, 30)) as pilot:
        screen = ApprovalScreen(
            "write_file", "overwrite", preview=diff, preview_kind="diff",
        )
        app.push_screen(screen)
        await pilot.pause()

        assert screen.query_one("#ap-opts", OptionList).highlighted == 1
        summary = screen.query_one("#ap-summary", Static).render().plain
        assert "+80 / -80" in summary

        preview = screen.query_one("#ap-preview")
        before = preview.scroll_y
        await pilot.press("ctrl+d")
        await pilot.pause()
        assert preview.scroll_y > before


async def test_model_screen_arrow_select() -> None:
    """The /model picker: arrow-keys move off the current model, Enter selects."""
    app = FrontierAgentApp(_FakeSession())
    async with app.run_test() as pilot:
        result: dict[str, str] = {}
        app.push_screen(ModelScreen(["m1", "m2", "m3"], current="m1"),
                        lambda v: result.update(v=v))
        await pilot.pause()
        await pilot.press("down")   # current m1 (idx 0) → m2
        await pilot.press("enter")
        await _wait_until(lambda: "v" in result)
        assert result["v"] == "m2"


async def test_slash_model_opens_picker() -> None:
    """Typing bare `/model` in the TUI opens the arrow-key picker modal."""
    app = FrontierAgentApp(_FakeSession())
    async with app.run_test() as pilot:
        app.query_one("#prompt", Input).value = "/model"
        await pilot.press("enter")
        await _wait_until(lambda: isinstance(app.screen, ModelScreen))
        assert isinstance(app.screen, ModelScreen)


async def test_slash_context_opens_usage_visualization() -> None:
    from apodex.usage import ContextBreakdown, Usage

    session = _FakeSession()
    session.cfg.context_window = 262_144
    session.cfg.max_tokens = 32_768
    session.usage = Usage(
        input=150_000, output=8_000, cached=90_000, last_input=143_280,
        compactions=1,
        breakdown=ContextBreakdown(
            system=12_000, conversation=40_000, tool_calls=5_000,
            tool_results=60_000, summarized=15_000, other=11_280,
            total=143_280,
        ),
    )
    app = FrontierAgentApp(session)
    async with app.run_test() as pilot:
        await pilot.pause()
        status = app.status.render().plain
        assert "ctx 143k/256k 55%" in status
        assert "fake-model" not in status
        assert "158,000 tokens" not in status
        app.query_one("#prompt", Input).value = "/context"
        await pilot.press("enter")
        await _wait_until(lambda: isinstance(app.screen, ContextScreen))
        rendered = app.screen.query_one("#context-body", Static).render().plain
        assert "143,280 / 262,144 tokens" in rendered
        assert "Conversation / history" in rendered
        assert "Tool calls & results" in rendered
        assert "Summarized history" in rendered
        await pilot.press("escape")
        await _wait_until(lambda: not isinstance(app.screen, ContextScreen))


async def test_context_capacity_is_always_present_before_first_response() -> None:
    from apodex.usage import Usage

    session = _FakeSession()
    session.cfg.context_window = 262_144
    session.usage = Usage()
    app = FrontierAgentApp(session)
    async with app.run_test() as pilot:
        await pilot.pause()
        assert "ctx --/256k" in app.status.render().plain


async def test_slash_theme_opens_picker_and_applies_selection() -> None:
    app = FrontierAgentApp(_FakeSession(), theme="catppuccin")
    async with app.run_test() as pilot:
        app.query_one("#prompt", Input).value = "/theme"
        await pilot.press("enter")
        await _wait_until(lambda: isinstance(app.screen, ThemeScreen))
        await pilot.press("down")
        await pilot.press("enter")
        await _wait_until(lambda: app.theme == "catppuccin-latte")


async def test_slash_workflow_opens_picker_and_switches_mode() -> None:
    session = _FakeSession()
    app = FrontierAgentApp(session)
    async with app.run_test() as pilot:
        app.query_one("#prompt", Input).value = "/workflow"
        await pilot.press("enter")
        await _wait_until(lambda: isinstance(app.screen, WorkflowScreen))
        await pilot.press("down")
        await pilot.press("enter")
        await _wait_until(lambda: session.slash_commands == ["/mode agent_team"])


async def test_menu_button_opens_unified_settings_and_keyboard_applies() -> None:
    session = _FakeSession()
    app = FrontierAgentApp(session, theme="catppuccin")
    async with app.run_test() as pilot:
        await pilot.click("#menu-button")
        await _wait_until(lambda: isinstance(app.screen, SettingsScreen))

        # Theme starts on catppuccin: down highlights latte, Enter applies it.
        await pilot.press("down")
        await pilot.press("enter")
        await _wait_until(lambda: app.theme == "catppuccin-latte")


async def test_workspace_diff_poll_can_update_behind_settings_modal() -> None:
    app = FrontierAgentApp(_FakeSession())
    async with app.run_test() as pilot:
        await pilot.press("f2")
        await _wait_until(lambda: isinstance(app.screen, SettingsScreen))

        # Background polls finish against the app while query_one() is scoped
        # to this modal screen, which does not contain the sidebar tabs.
        app._store_workspace_diff([("notes.md", 1, 0)], "+new note\n")
        assert app.diff_tab.display is True
        app._store_workspace_diff([], "")
        assert app.diff_tab.display is False


async def test_settings_arrow_keys_switch_section_and_apply_workflow() -> None:
    session = _FakeSession()
    app = FrontierAgentApp(session)
    async with app.run_test() as pilot:
        await pilot.press("f2")
        await _wait_until(lambda: isinstance(app.screen, SettingsScreen))
        await pilot.press("right")
        await pilot.press("down")
        await pilot.press("enter")
        await _wait_until(lambda: session.slash_commands == ["/mode agent_team"])


async def test_settings_behavior_and_permissions_toggles_apply_together() -> None:
    session = _FakeSession()
    app = FrontierAgentApp(session)
    async with app.run_test() as pilot:
        await pilot.press("f2")
        await _wait_until(lambda: isinstance(app.screen, SettingsScreen))

        await pilot.press("right", "right")  # Behavior
        await pilot.press("space")           # Plan mode on
        await pilot.press("down", "space")   # Full thinking off
        await pilot.press("right")           # Permissions
        await pilot.press("space")           # Auto-approve on
        await pilot.press("enter")

        await _wait_until(lambda: not isinstance(app.screen, SettingsScreen))
        assert session.plan_state.active is True
        assert session.verbose is False
        assert app.sink._verbose is False
        assert session.approver.auto_approve is True


async def test_settings_sessions_tab_stages_a_saved_session() -> None:
    app = FrontierAgentApp(_FakeSession())
    async with app.run_test() as pilot:
        result = {}
        app.push_screen(SettingsScreen(
            ("catppuccin",),
            "catppuccin",
            "react",
            sessions=(("saved-session", "Yesterday · react · 4 msgs · /work"),),
            current_session="current-session",
        ), lambda value: result.update(value=value))
        await pilot.pause()
        await pilot.press("right", "right", "right", "right")
        await pilot.press("down", "enter")
        await _wait_until(lambda: "value" in result)
        assert result["value"].resume_session_id == "saved-session"


async def test_model_screen_escape_cancels() -> None:
    app = FrontierAgentApp(_FakeSession())
    async with app.run_test() as pilot:
        result: dict[str, str | None] = {}
        app.push_screen(ModelScreen(["m1", "m2"], current="m1"),
                        lambda v: result.update(v=v))
        await pilot.pause()
        await pilot.press("escape")
        await _wait_until(lambda: "v" in result)
        assert result["v"] is None


async def test_approval_arrow_nav_selects_option() -> None:
    """Arrow-keys + Enter on the option list decide the call (not just hotkeys)."""
    app = FrontierAgentApp(_FakeSession())
    async with app.run_test() as pilot:
        result: dict[str, ApprovalOutcome] = {}
        screen = ApprovalScreen("bash", "run ls")
        app.push_screen(screen, lambda o: result.update(o=o))
        await pilot.pause()
        screen.query_one("#ap-opts", OptionList).highlighted = 1  # the "No" option
        await pilot.press("enter")
        await _wait_until(lambda: "o" in result)
        assert result["o"].decision.approved is False


# ── end-to-end: the real session + observer + a scripted LLM, via the TUI ──
def _real_session(ws: str, *, auto_approve: bool, script: list):
    from apodex.config import ModelConfig
    from apodex.render import Renderer
    from apodex.session import TerminalSession
    from apodex.tests.test_smoke import _ScriptedLLM

    s = TerminalSession(
        cfg=ModelConfig(model="fake", api_key="x", base_url=None),
        cwd=ws, renderer=Renderer(color=False),
        auto_approve=auto_approve, max_turns=8, interactive=False, mode="coding",
    )
    s.llm = _ScriptedLLM(script=list(script))
    return s


async def test_real_session_config_and_welcome_never_render_key_fragments(tmp_path) -> None:
    secret = "sk-demo-secret-9876"
    session = _real_session(str(tmp_path), auto_approve=True, script=[])
    session.cfg.api_key = secret
    app = FrontierAgentApp(session)
    async with app.run_test() as pilot:
        prompt = app.query_one("#prompt", Input)
        prompt.value = "/config"
        await pilot.press("enter")
        await _wait_until(
            lambda: any(
                "API key" in block.render().plain
                for block in app.transcript.query(".block")
            )
        )
        transcript = "\n".join(
            block.render().plain for block in app.transcript.query(".block")
        )

    assert "Ready for a local BYOK demo" in transcript
    assert "provider: openai" in transcript
    assert "API key (OPENAI_API_KEY): configured" in transcript
    for fragment in (secret, "sk-d", "9876"):
        assert fragment not in transcript


async def test_e2e_read_through_tui(tmp_path) -> None:
    """A safe (auto-run) tool flows engine → observer → sink → widgets."""
    from apodex.tests.test_smoke import _tc
    from frontier_agent.core.messages import assistant_msg, text_of

    (tmp_path / "hello.txt").write_text("MAGIC_42\n")
    script = [
        assistant_msg("reading it", tool_calls=[_tc("read_file", {"path": "hello.txt"}, 1)]),
        assistant_msg("The file says MAGIC_42."),
    ]
    session = _real_session(str(tmp_path), auto_approve=True, script=script)
    app = FrontierAgentApp(session)
    async with app.run_test() as pilot:
        app.query_one("#prompt", Input).value = "read hello.txt"
        await pilot.press("enter")
        await _wait_until(lambda: session.history and not app.busy, timeout=8)

    joined = " ".join(text_of(m.get("content")) for m in session.history)
    assert "MAGIC_42" in joined          # tool output reached the conversation
    assert app._tools >= 1               # a tool call drove the activity pane


async def test_e2e_write_needs_modal_approval(tmp_path) -> None:
    """A CONFIRM tool opens the approval modal; pressing 'y' lets it run."""
    from apodex.tests.test_smoke import _tc
    from apodex.tui.screens import ApprovalScreen as _AS
    from frontier_agent.core.messages import assistant_msg

    script = [
        assistant_msg("creating it", tool_calls=[
            _tc("write_file", {"path": "out.py", "content": "x = 2\n"}, 1)]),
        assistant_msg("Created out.py."),
    ]
    session = _real_session(str(tmp_path), auto_approve=False, script=script)
    app = FrontierAgentApp(session)
    async with app.run_test() as pilot:
        app.query_one("#prompt", Input).value = "create out.py"
        await pilot.press("enter")
        await _wait_until(lambda: isinstance(app.screen, _AS), timeout=8)
        await pilot.press("y")  # approve the write
        await _wait_until(lambda: not app.busy, timeout=8)

    out = tmp_path / "out.py"
    assert out.exists() and out.read_text().strip() == "x = 2"


# ── sidebar workspaces ────────────────────────────────────────────────────
async def test_sidebar_defaults_to_plan_and_gives_each_tab_the_full_pane() -> None:
    from apodex.todo import TodoItem

    app = FrontierAgentApp(_FakeSession())
    async with app.run_test(size=(120, 30)) as pilot:
        await pilot.pause()
        sidebar = app.query_one("#sidebar").size.height
        assert app._sidebar_tab == "plan"
        assert app.query_one("#sidebar-plan").display is True
        assert app.query_one("#sidebar-activity").display is False
        assert app.deliverables_box.display is False
        assert app.query_one("#sidebar-diff").display is False
        assert app.query_one("#sidebar-tab-diff", Tab).display is False
        assert app.todos_box.size.height > sidebar // 2

        app.show_todos([TodoItem(f"step {i}", "pending") for i in range(40)])
        await pilot.pause()
        assert app.todos_box.max_scroll_y > 0

        app.action_next_sidebar_tab()
        await pilot.pause()
        assert app._sidebar_tab == "activity"
        assert app.query_one("#sidebar-plan").display is False
        assert app.query_one("#sidebar-activity").display is True
        assert app.activity.size.height > sidebar // 2


async def test_plan_and_activity_follow_their_newest_row() -> None:
    from apodex.todo import TodoItem

    app = FrontierAgentApp(_FakeSession())
    async with app.run_test(size=(120, 30)) as pilot:
        await pilot.pause()
        app.show_todos([TodoItem(f"step {i}", "pending") for i in range(40)])
        for index in range(40):
            app.start_activity(f"call-{index}", "bash", f"command {index}")
            app.finish_activity(f"call-{index}", "bash", is_error=False, ms=index)
        await pilot.pause()

        assert app.todos_box.scroll_offset.y == app.todos_box.max_scroll_y
        app.action_next_sidebar_tab()
        await pilot.pause()
        assert app.activity.highlighted == len(app.activity.records) - 1
        await pilot.resize_terminal(100, 24)
        await pilot.pause()
        assert app.activity.highlighted == len(app.activity.records) - 1


async def test_deliverables_expose_host_path_and_session_in_heading(monkeypatch) -> None:
    monkeypatch.setenv("FRONTIER_AGENT_OUTPUTS_DIR", "/outputs")
    monkeypatch.setenv(
        "APODEX_HOST_OUTPUTS_DIR",
        "/host/project/.apodex/outputs/test-session",
    )
    monkeypatch.setenv(
        "APODEX_HOST_WORKSPACE_DIR",
        "/host/project/.apodex/runs/test-session/workspace",
    )
    app = FrontierAgentApp(_FakeSession())

    assert app._deliverables_location_lines() == [
        "Host: /host/project/.apodex/outputs/test-session",
        "Agent: /outputs",
        "Work: /host/project/.apodex/runs/test-session/workspace (intermediate)",
    ]


async def test_final_report_is_saved_as_a_markdown_deliverable(
    monkeypatch, tmp_path,
) -> None:
    monkeypatch.setenv("FRONTIER_AGENT_OUTPUTS_DIR", str(tmp_path))
    app = FrontierAgentApp(_FakeSession())
    async with app.run_test(size=(120, 30)) as pilot:
        app.sink.final("# Result\n\nThe generated report.")
        await pilot.pause()

        report = tmp_path / "final-report.md"
        assert report.read_text() == "# Result\n\nThe generated report.\n"
        assert app._sidebar_tab == "deliverables"
        assert app.focused is app.deliverables
        assert app.deliverables.selected_path == report

        app.sink.begin_task()
        await pilot.pause()
        assert app._sidebar_tab == "plan"


async def test_final_summary_does_not_pollute_an_explicit_output_manifest(
    monkeypatch, tmp_path,
) -> None:
    monkeypatch.setenv("FRONTIER_AGENT_OUTPUTS_DIR", str(tmp_path))
    deliverable = tmp_path / "report-zh.md"
    deliverable.write_text("requested artifact\n")
    app = FrontierAgentApp(_FakeSession())

    async with app.run_test(size=(120, 30)) as pilot:
        app.sink.final("Coordinator status summary only.")
        await pilot.pause()

        assert deliverable.read_text() == "requested artifact\n"
        assert not (tmp_path / "final-report.md").exists()
        assert app.transcript.final_text == "Coordinator status summary only."
        assert app.deliverables.selected_path == deliverable


async def test_incomplete_run_does_not_overwrite_a_final_report(
    monkeypatch, tmp_path,
) -> None:
    monkeypatch.setenv("FRONTIER_AGENT_OUTPUTS_DIR", str(tmp_path))
    report = tmp_path / "final-report.md"
    report.write_text("previous verified delivery\n")
    app = FrontierAgentApp(_FakeSession())

    async with app.run_test(size=(120, 30)) as pilot:
        app.sink.begin_task()
        app.sink.incomplete(
            "Fixing these before sign-off.",
            turns=150,
            stopped_by="wall_deadline",
        )
        await pilot.pause()

        assert report.read_text() == "previous verified delivery\n"
        block = app.query_one(".transcript-incomplete", Static)
        panel = block.render()._renderable
        assert isinstance(panel, Panel)
        assert "Incomplete output" in str(panel.title)
        assert "Fixing these before sign-off" in panel.renderable.markup
        assert "Final report" not in str(panel.title)
        assert app.presentation.phase == PresentationPhase.INCOMPLETE
        assert app._sidebar_tab == "plan"


async def test_incomplete_run_replaces_the_copyable_report_text(
    monkeypatch, tmp_path,
) -> None:
    """Ctrl-Y must never hand back the *previous* task's report.

    ``transcript.final_text`` only resets on a whole-transcript replacement, so
    an ``incomplete`` render that left it alone let ``action_copy_report`` copy
    the earlier run's answer and announce it as this one's.
    """
    monkeypatch.setenv("FRONTIER_AGENT_OUTPUTS_DIR", str(tmp_path))
    app = FrontierAgentApp(_FakeSession())

    async with app.run_test(size=(120, 30)) as pilot:
        app.sink.begin_task()
        app.sink.final("the delivered answer", turns=3)
        await pilot.pause()
        assert app.transcript.final_text == "the delivered answer"

        app.sink.begin_task()
        await pilot.pause()
        app.sink.incomplete("only got halfway", turns=150, stopped_by="max_turns")
        await pilot.pause()

        assert app.transcript.final_text == "only got halfway"
        # The pane is reloaded, but a run that stopped short never fronts it.
        assert app._sidebar_tab == "plan"


async def test_a_follow_up_task_restarts_the_status_bar_after_an_incomplete_run(
    monkeypatch, tmp_path,
) -> None:
    """``INCOMPLETE`` is terminal, so ``transition`` alone cannot leave it.

    ``run_task`` re-enters itself inside the same session worker to run queued
    steering input, so the next task starts before the finished one settles back
    to IDLE — with the status bar still labelled "incomplete" and its timer stuck.
    """
    monkeypatch.setenv("FRONTIER_AGENT_OUTPUTS_DIR", str(tmp_path))
    app = FrontierAgentApp(_FakeSession())

    async with app.run_test(size=(120, 30)) as pilot:
        app.sink.begin_task()
        app.sink.incomplete("partial", turns=9, stopped_by="wall_deadline")
        await pilot.pause()
        assert app.presentation.phase == PresentationPhase.INCOMPLETE

        app.sink.working_on()
        assert app.presentation.phase == PresentationPhase.THINKING
        assert app.presentation.task_finished_at is None


async def test_llm_configuration_failure_is_not_rendered_as_final_report(
    monkeypatch, tmp_path,
) -> None:
    monkeypatch.setenv("FRONTIER_AGENT_OUTPUTS_DIR", str(tmp_path))
    app = FrontierAgentApp(_FakeSession())
    async with app.run_test(size=(120, 30)) as pilot:
        app.sink.begin_task()
        app.sink.llm_failure(
            "Provider/model: openai/gpt-test\n"
            "Provider response: Error code: 401",
            configuration_error=True,
        )
        await pilot.pause()

        error_block = app.query_one(".transcript-error", Static)
        error_panel = error_block.render()._renderable
        assert isinstance(error_panel, Panel)
        assert "LLM configuration error" in str(error_panel.title)
        assert "Error code: 401" in error_panel.renderable.plain
        assert "Final report" not in str(error_panel.title)
        assert not (tmp_path / "final-report.md").exists()
        assert "issues found" in app.transcript.query_one(".process-group").title


async def test_completed_task_preselects_files_without_focusing_hidden_sidebar(
    monkeypatch, tmp_path,
) -> None:
    monkeypatch.setenv("FRONTIER_AGENT_OUTPUTS_DIR", str(tmp_path))
    app = FrontierAgentApp(_FakeSession())
    async with app.run_test(size=(80, 24)) as pilot:
        prompt = app.query_one("#prompt", Input)
        assert app.focused is prompt
        app.sink.final("done")
        await pilot.pause()

        assert app._sidebar_tab == "deliverables"
        assert app.query_one("#sidebar").display is False
        assert app.focused is prompt


async def test_completed_task_preselects_diff_when_files_changed(tmp_path) -> None:
    from apodex.changes import WorkspaceJournal

    source = tmp_path / "module.py"
    source.write_text("value = 1\n")
    session = _FakeSession()
    session.cwd = str(tmp_path)
    session.journal = WorkspaceJournal(str(tmp_path))
    session.journal.record_before("module.py")
    source.write_text("value = 2\n")

    app = FrontierAgentApp(session)
    async with app.run_test(size=(80, 24)):
        prompt = app.query_one("#prompt", Input)
        app.sink.final("done")
        await _wait_until(lambda: app._sidebar_tab == "diff")

        assert app.query_one("#sidebar-tab-diff", Tab).display is True
        assert app.query_one("#sidebar").display is False
        assert app.focused is prompt


async def test_space_previews_selected_deliverable_and_space_closes(
    monkeypatch, tmp_path,
) -> None:
    monkeypatch.setenv("FRONTIER_AGENT_OUTPUTS_DIR", str(tmp_path))
    source = tmp_path / "example.py"
    source.write_text("def answer():\n    return 42\n")
    app = FrontierAgentApp(_FakeSession())
    async with app.run_test(size=(120, 30)) as pilot:
        app.action_toggle_deliverables()
        await pilot.pause()

        assert app.focused is app.deliverables
        assert app.deliverables.selected_path == source
        await pilot.press("space")
        await pilot.pause()
        assert isinstance(app.screen, FilePreviewScreen)
        assert "return 42" in app.screen._renderable().code

        await pilot.press("space")
        await pilot.pause()
        assert not isinstance(app.screen, FilePreviewScreen)
        assert app.focused is app.deliverables

        await pilot.click("#deliverables", offset=(10, 0))
        await pilot.pause()
        assert isinstance(app.screen, FilePreviewScreen)


async def test_file_preview_adapters_and_fallbacks(tmp_path) -> None:
    import json

    # 1. Jupyter Notebook (.ipynb)
    nb_path = tmp_path / "test.ipynb"
    nb_path.write_text(json.dumps({
        "cells": [
            {"cell_type": "markdown", "source": ["# Title\n", "Hello notebook"]},
            {"cell_type": "code", "source": ["print(42)\n"], "outputs": [{"output_type": "stream", "text": ["42\n"]}]}
        ]
    }))
    screen_nb = FilePreviewScreen(nb_path, label="test.ipynb")
    res_nb = screen_nb._renderable()
    assert "Jupyter Notebook" in res_nb.plain
    assert "Hello notebook" in res_nb.plain
    assert "print(42)" in res_nb.plain

    # 2. PDB Structure (.pdb)
    pdb_path = tmp_path / "test.pdb"
    pdb_path.write_text(
        "HEADER    HYDROLASE                               06-AUG-26   1ABC\n"
        "TITLE     TEST PROTEIN STRUCTURE\n"
        "ATOM      1  N   MET A   1       1.000   2.000   3.000  1.00  85.00           N\n"
        "ATOM      2  CA  MET A   1       2.000   3.000   4.000  1.00  90.00           C\n"
    )
    screen_pdb = FilePreviewScreen(pdb_path, label="test.pdb")
    res_pdb = screen_pdb._renderable()
    assert "PDB Biological Structure" in res_pdb.plain
    assert "TEST PROTEIN STRUCTURE" in res_pdb.plain
    assert "Atoms: 2 (ATOM)" in res_pdb.plain
    assert "87.50" in res_pdb.plain

    # 3. 3D OBJ Mesh (.obj)
    obj_path = tmp_path / "test.obj"
    obj_path.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n")
    screen_obj = FilePreviewScreen(obj_path, label="test.obj")
    res_obj = screen_obj._renderable()
    assert "Wavefront OBJ" in res_obj.plain
    assert "Vertices: 3" in res_obj.plain
    assert "Faces: 1" in res_obj.plain

    # 4. Archive (.zip)
    zip_path = tmp_path / "test.zip"
    import zipfile
    with zipfile.ZipFile(zip_path, "w") as z:
        z.writestr("sub/hello.txt", "hello world")
    screen_zip = FilePreviewScreen(zip_path, label="test.zip")
    res_zip = screen_zip._renderable()
    assert "Archive" in res_zip.plain
    assert "sub/hello.txt" in res_zip.plain

    # 5. PDF Fallback or Render (.pdf)
    pdf_path = tmp_path / "test.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 header test")
    screen_pdf = FilePreviewScreen(pdf_path, label="test.pdf")
    res_pdf = screen_pdf._renderable()
    assert ("PDF Document" in res_pdf.plain) or ("Could not read PDF" in res_pdf.plain)

    # 6. DOCX Fallback or Render (.docx)
    docx_path = tmp_path / "test.docx"
    docx_path.write_bytes(b"PK\x03\x04 dummy docx")
    screen_docx = FilePreviewScreen(docx_path, label="test.docx")
    res_docx = screen_docx._renderable()
    assert ("Word Document" in getattr(res_docx, "plain", "")) or ("Word Document" in getattr(res_docx, "markup", "")) or ("Could not read Word" in getattr(res_docx, "plain", ""))



async def test_mouse_tabs_and_space_expand_activity_tool_calls() -> None:
    app = FrontierAgentApp(_FakeSession())
    async with app.run_test(size=(120, 30)) as pilot:
        app.start_activity("call-42", "bash", "python build.py --release")
        app.finish_activity("call-42", "bash", is_error=False, ms=37)
        await pilot.click("#sidebar-tab-activity")
        await pilot.pause()

        assert app._sidebar_tab == "activity"
        assert app.focused is app.activity
        assert app.activity.selected_record.call_id == "call-42"
        await pilot.press("space")
        await pilot.pause()

        assert isinstance(app.screen, ActivityDetailScreen)
        assert app.screen.call_id == "call-42"
        assert app.screen.summary == "python build.py --release"
        await pilot.press("space")
        await pilot.pause()
        assert app.focused is app.activity

        await pilot.click("#activity", offset=(2, 0))
        await pilot.pause()
        assert isinstance(app.screen, ActivityDetailScreen)


async def test_ctrl_tab_cycles_sidebar_workspaces() -> None:
    app = FrontierAgentApp(_FakeSession())
    async with app.run_test() as pilot:
        assert app._sidebar_tab == "plan"
        await pilot.press("ctrl+tab")
        assert app._sidebar_tab == "activity"
        await pilot.press("ctrl+tab")
        assert app._sidebar_tab == "deliverables"
        await pilot.press("ctrl+tab")
        assert app._sidebar_tab == "plan"


async def test_diff_tab_and_status_show_session_file_changes(tmp_path) -> None:
    from apodex.changes import WorkspaceJournal

    source = tmp_path / "module.py"
    source.write_text("value = 1\nold = True\n")
    session = _FakeSession()
    session.cwd = str(tmp_path)
    session.journal = WorkspaceJournal(str(tmp_path))
    session.journal.record_before("module.py")
    source.write_text("value = 2\nextra = 3\n")

    app = FrontierAgentApp(session)
    async with app.run_test(size=(120, 30)) as pilot:
        await _wait_until(lambda: app._workspace_diff_stats == [("module.py", 2, 2)])
        await pilot.pause()
        app._refresh_status()
        await pilot.click("#sidebar-tab-diff")
        await pilot.pause()

        assert app._sidebar_tab == "diff"
        assert app.query_one("#sidebar-diff").display is True
        rendered = app.diff_pane.render().plain
        assert "1 file changed  +2  -2" in rendered
        assert "--- a/module.py" in rendered
        assert "+value = 2" in rendered
        assert "1 file" in app.status.render().plain
        assert "+2" in app.status.render().plain
        assert "-2" in app.status.render().plain
        assert str(app.query_one("#sidebar-tab-diff", Tab).label) == "Diff  1"
        assert app.query_one("#sidebar-tab-diff", Tab).display is True

        await pilot.press("ctrl+shift+tab")
        assert app._sidebar_tab == "deliverables"
        await pilot.press("ctrl+tab")
        assert app._sidebar_tab == "diff"

        await pilot.press("escape")
        assert app._sidebar_tab == "plan"


async def test_diff_pane_scrolls_from_the_keyboard(tmp_path) -> None:
    """The tab focuses this pane, so the arrow keys have to move it.

    ``Static`` scrolls its own overflow with the mouse but ships no key
    bindings, which left a focused diff taller than the pane immovable.
    """
    from apodex.changes import WorkspaceJournal

    source = tmp_path / "long.py"
    source.write_text("".join(f"old {i}\n" for i in range(200)))
    session = _FakeSession()
    session.cwd = str(tmp_path)
    session.journal = WorkspaceJournal(str(tmp_path))
    session.journal.record_before("long.py")
    source.write_text("".join(f"new {i}\n" for i in range(200)))

    app = FrontierAgentApp(session)
    async with app.run_test(size=(120, 30)) as pilot:
        await _wait_until(lambda: bool(app._workspace_diff_stats))
        # The stats landing is not the tab appearing: ``display`` was just
        # flipped to True and until Textual re-lays-out the tab's region is
        # still (0, 0, 0, 0), so the click below would land on the screen
        # origin and focus the prompt instead.
        await pilot.pause()
        await pilot.click("#sidebar-tab-diff")
        await pilot.pause()
        assert app.focused is app.diff_scroll
        assert app.diff_scroll.max_scroll_y > 0     # not clipped to one screen

        await pilot.press("pagedown")
        await pilot.pause()
        assert app.diff_scroll.scroll_offset.y > 0

        await pilot.press("home")
        await pilot.pause()
        assert app.diff_scroll.scroll_offset.y == 0

        await pilot.press("escape")
        assert app._sidebar_tab == "plan"


async def test_change_counters_never_squeeze_out_the_queued_indicator() -> None:
    """Ungated, the counters sat ahead of every gated segment and the ellipsis
    ate the ``queued`` indicator that says a steer is still waiting.

    The bar is allowed to drop things a narrow terminal cannot hold — what it
    may not do is drop *more* once a session has changes than the same session
    would drop with none.
    """
    app = FrontierAgentApp(_FakeSession())
    async with app.run_test(size=(120, 30)):
        app.presentation.queued_steers = 2

        def rendered(width: int, changes: tuple[int, int, int]) -> str:
            # No ``pilot.pause()`` between the two: ``show`` sets the renderable
            # synchronously, and yielding would let the app's own 0.4 s status
            # refresh repaint the bar from the fake session instead.
            app.status.show(
                presentation=app.presentation, mode="coding",
                ctx="143k/256k 55%", tools=17, width=width,
                changes=changes,
            )
            return app.status.render().plain

        for width in (60, 80, 99, 100, 110, 119, 120):
            indicator = "queued 2" if width < 100 else "q 2"
            quiet = rendered(width, (0, 0, 0))
            busy = rendered(width, (9, 480, 260))
            if indicator in quiet:
                assert indicator in busy, f"counters ate the indicator at {width}"

        wide = rendered(120, (9, 480, 260))
        assert "9 files" in wide and "+480" in wide and "-260" in wide


async def test_cycling_off_a_diff_tab_that_just_vanished_does_not_crash(tmp_path) -> None:
    """``_store_workspace_diff`` can leave "diff" active with empty stats.

    Its follow-up branch returns before the correction below it, so the tab is
    gone from ``_available_sidebar_tabs`` while ``_sidebar_tab`` still names
    it. ``tuple.index`` raised straight out of the key handler.
    """
    from apodex.changes import WorkspaceJournal

    source = tmp_path / "module.py"
    source.write_text("value = 1\n")
    session = _FakeSession()
    session.cwd = str(tmp_path)
    session.journal = WorkspaceJournal(str(tmp_path))
    session.journal.record_before("module.py")
    source.write_text("value = 2\n")

    app = FrontierAgentApp(session)
    async with app.run_test(size=(120, 30)) as pilot:
        await _wait_until(lambda: bool(app._workspace_diff_stats))
        await pilot.pause()
        app._show_sidebar_tab("diff", focus=False)

        # Reproduce the window: completion arrives, stats come back empty, and
        # the follow-up branch returns without moving off the Diff tab.
        app._completed_workspace_pending = True
        app._completed_workspace_needs_followup = True
        app._store_workspace_diff([], "")
        assert app._sidebar_tab == "diff"
        assert "diff" not in app._available_sidebar_tabs()

        app.action_next_sidebar_tab()
        assert app._sidebar_tab == "plan"
        app._show_sidebar_tab("diff", focus=False)
        app.action_previous_sidebar_tab()
        assert app._sidebar_tab == "deliverables"


async def test_a_failed_diff_read_is_not_reported_as_a_clean_tree(tmp_path) -> None:
    """Rendering the failure as "no changes" hid a whole session's edits."""
    from apodex.changes import WorkspaceJournal

    source = tmp_path / "module.py"
    source.write_text("value = 1\n")
    session = _FakeSession()
    session.cwd = str(tmp_path)
    session.journal = WorkspaceJournal(str(tmp_path))
    session.journal.record_before("module.py")
    source.write_text("value = 2\n")

    app = FrontierAgentApp(session)
    async with app.run_test(size=(120, 30)) as pilot:
        await _wait_until(lambda: bool(app._workspace_diff_stats))
        await pilot.pause()
        good = list(app._workspace_diff_stats)

        app._store_workspace_diff([], "", None, "RecursionError: too deep")
        await pilot.pause()

        assert app._workspace_diff_error == "RecursionError: too deep"
        assert app._workspace_diff_stats == good      # last good reading kept
        assert app.query_one("#sidebar-tab-diff", Tab).display is True
        rendered = app.diff_pane.render().plain
        assert "could not read session changes" in rendered
        assert "RecursionError" in rendered
        assert "No session file changes" not in rendered


async def test_the_error_banner_clears_on_the_next_good_read(tmp_path) -> None:
    """A transient failure used to pin the banner for the rest of the session.

    ``show_error`` keeps the last good reading, and ``report()`` is memoised
    per fingerprint — so the read after a failure normally carries *identical*
    content, which the "repaint only when the content changed" guard skipped.
    """
    from apodex.changes import WorkspaceJournal

    source = tmp_path / "module.py"
    source.write_text("value = 1\n")
    session = _FakeSession()
    session.cwd = str(tmp_path)
    session.journal = WorkspaceJournal(str(tmp_path))
    session.journal.record_before("module.py")
    source.write_text("value = 2\n")

    app = FrontierAgentApp(session)
    async with app.run_test(size=(120, 30)) as pilot:
        await _wait_until(lambda: bool(app._workspace_diff_stats))
        await pilot.pause()
        stats = list(app._workspace_diff_stats)
        diff = app._workspace_diff_text

        app._store_workspace_diff([], "", None, "OSError: stale handle")
        await pilot.pause()
        assert "could not read session changes" in app.diff_pane.render().plain

        # The identical reading the poll returns a tick later, unchanged.
        app._store_workspace_diff(stats, diff, None, "")
        await pilot.pause()

        rendered = app.diff_pane.render().plain
        assert "could not read session changes" not in rendered
        assert "value = 2" in rendered


async def test_activity_rows_are_ellipsized_not_horizontally_scrolled() -> None:
    """A long command used to hang a horizontal scrollbar across the sidebar."""
    app = FrontierAgentApp(_FakeSession())
    async with app.run_test(size=(120, 30)) as pilot:
        app.start_activity("long", "bash", "grep -rn 'x' " + "very/deep/path/" * 12)
        app.action_next_sidebar_tab()
        await pilot.pause()

        width = app.activity.content_size.width
        row = app.activity._render_row(app.activity.records[-1], 0.0, width)
        assert cell_len(row.plain) <= width
        assert row.plain.endswith("…")
        # The state glyph, tool name and duration all survive truncation.
        assert row.plain.startswith("◐ bash ")
        assert app.activity.show_horizontal_scrollbar is False


async def test_sidebar_headings_carry_counts_for_offscreen_rows() -> None:
    from apodex.todo import TodoItem

    app = FrontierAgentApp(_FakeSession())
    async with app.run_test(size=(120, 30)) as pilot:
        await pilot.pause()
        assert str(app.query_one("#sidebar-tab-plan", Tab).label) == "Plan"
        assert str(app.query_one("#sidebar-tab-activity", Tab).label) == "Activity"

        app.show_todos([
            TodoItem("a", "completed"), TodoItem("b", "completed"), TodoItem("c", "pending"),
        ])
        app.start_activity("x", "bash", "pwd")
        app.finish_activity("x", "bash", is_error=True, ms=3)
        app.start_activity("y", "bash", "sleep")
        app._refresh_status()
        await pilot.pause()

        assert str(app.query_one("#sidebar-tab-plan", Tab).label) == "Plan  2/3"
        assert str(app.query_one("#sidebar-tab-activity", Tab).label) == "Activity  2 ◐ 1 ✗ 1"


async def test_empty_activity_pane_says_so() -> None:
    """A blank pane read as a rendering failure rather than an idle timeline."""
    app = FrontierAgentApp(_FakeSession())
    async with app.run_test() as pilot:
        await pilot.pause()
        assert app.activity.records == []
        assert app.activity.summary() == ""


async def test_scrollbars_are_one_cell_wide() -> None:
    """Textual's default 2-cell gutter is a visible slab at this sidebar width."""
    app = FrontierAgentApp(_FakeSession())
    async with app.run_test(size=(120, 30)) as pilot:
        await pilot.pause()
        for selector in (
            "#transcript", "#activity", "#todos-box", "#deliverables",
            "#sidebar-diff",
        ):
            assert app.query_one(selector).styles.scrollbar_size_vertical == 1, selector


async def test_scrollbar_thumb_is_quieter_than_body_text() -> None:
    """A scrollbar at full text contrast draws the eye harder than the content
    it scrolls past. The thumb sits between the trough and the subtle tier."""
    from apodex.tests.test_themes import contrast_ratio
    from apodex.tui.themes import palette

    app = FrontierAgentApp(_FakeSession(), theme="catppuccin")
    async with app.run_test() as pilot:
        await pilot.pause()
        spec = palette("catppuccin")
        thumb = app.query_one("#transcript").styles.scrollbar_color
        thumb_hex = f"#{thumb.rgb[0]:02x}{thumb.rgb[1]:02x}{thumb.rgb[2]:02x}"
        against_panel = contrast_ratio(thumb_hex, spec.panel)
        assert 1.5 <= against_panel < contrast_ratio(spec.subtle, spec.panel)


async def test_short_content_sits_at_the_top_not_the_bottom() -> None:
    """Regression: anchoring runs before the layout that would report the new
    content height, so a pane with less content than height parked at a negative
    scroll offset — the welcome note floated at the bottom of an empty pane."""
    app = FrontierAgentApp(_FakeSession())
    async with app.run_test(size=(120, 30)) as pilot:
        await pilot.pause()
        assert app.transcript.scroll_offset.y == 0
        assert app.todos_box.scroll_offset.y == 0

        app.sink.echo_user("hello")
        app.sink.content_delta("a short answer")
        await pilot.pause()
        await app.sink.finish_stream()
        await pilot.pause()
        assert app.transcript.max_scroll_y == 0        # nothing to scroll…
        assert app.transcript.scroll_offset.y == 0     # …so pinned to the top


async def test_streaming_block_grows_from_short_content_and_stays_at_tail() -> None:
    """Growing one live widget must update the scrollbar's real end.

    No new block is mounted after streaming starts, so mount-only following
    used to leave both the thumb and viewport at the top of a long response.
    """
    app = FrontierAgentApp(_FakeSession())
    async with app.run_test(size=(80, 20)) as pilot:
        await pilot.pause()
        await app.transcript.clear_all()
        app.sink.echo_user("write a long answer")
        for index in range(60):
            app.sink.content_delta(
                f"Paragraph {index} has enough text to occupy a visible row.\n\n"
            )
            await pilot.pause(0.005)
        await pilot.pause()
        await pilot.pause()

        assert app.transcript.max_scroll_y > 0
        assert app.transcript.scroll_offset.y == app.transcript.max_scroll_y


async def test_user_scroll_can_leave_and_restore_stream_following() -> None:
    app = FrontierAgentApp(_FakeSession())
    async with app.run_test(size=(80, 20)) as pilot:
        await pilot.pause()
        await app.transcript.clear_all()
        for index in range(50):
            app.sink.note(f"line {index}")
        await pilot.pause()
        await pilot.pause()

        transcript = app.transcript
        transcript.scroll_to(y=5, animate=False, immediate=True)
        await pilot.pause()
        app.sink.content_delta("new output\n\n" * 20)
        await pilot.pause()
        assert transcript.scroll_offset.y == 5

        transcript.scroll_end(animate=False)
        await pilot.pause()
        assert transcript._wants_tail is True
        app.sink.content_delta("tail output\n\n" * 20)
        await pilot.pause()
        await pilot.pause()
        assert transcript.scroll_offset.y == transcript.max_scroll_y


async def test_paging_down_at_the_bottom_keeps_following() -> None:
    """A scroll that cannot move is not the user leaving the newest content.

    Textual releases the anchor *before* the scroll lands, and releases it even
    for page-down at the very end. Since nothing then moves, no ``scroll_y``
    watcher fires to restore the intent, so treating the release as a departure
    stopped the transcript following for the rest of the session.
    """
    app = FrontierAgentApp(_FakeSession())
    async with app.run_test(size=(80, 20)) as pilot:
        await pilot.pause()
        transcript = app.transcript
        await transcript.clear_all()
        for index in range(60):
            app.sink.note(f"line {index}")
        await pilot.pause()
        await pilot.pause()
        assert transcript.scroll_offset.y == transcript.max_scroll_y

        transcript.focus()
        await pilot.pause()
        await pilot.press("pagedown")
        await pilot.pause()
        await pilot.pause()
        assert transcript._wants_tail is True

        for index in range(40):
            app.sink.note(f"more {index}")
            await pilot.pause(0.005)
        await pilot.pause()
        await pilot.pause()
        assert transcript.scroll_offset.y == transcript.max_scroll_y


async def test_content_shrinking_under_a_reader_does_not_resume_following() -> None:
    """Collapsing or pruning clamps ``scroll_y`` down onto the new end.

    That is the layout moving the view, not the user asking to be pulled along,
    so it must not re-arm following and yank them to the bottom.
    """
    app = FrontierAgentApp(_FakeSession())
    async with app.run_test(size=(80, 20)) as pilot:
        await pilot.pause()
        transcript = app.transcript
        await transcript.clear_all()
        for index in range(20):
            app.sink.note(f"head {index}")
        await pilot.pause()
        big = await transcript.add_collapsible(
            "thinking",
            "\n".join(f"deep {index}" for index in range(60)),
            classes="thinking-block",
            collapsed=False,
        )
        await pilot.pause()
        await pilot.pause()

        transcript.scroll_to(y=transcript.max_scroll_y - 3, animate=False, immediate=True)
        await pilot.pause()
        assert transcript._wants_tail is False
        resting = transcript.scroll_offset.y

        big.collapsed = True
        await pilot.pause()
        await pilot.pause()
        assert transcript.scroll_offset.y <= resting
        assert transcript._wants_tail is False

        for index in range(40):
            app.sink.note(f"more {index}")
            await pilot.pause(0.005)
        await pilot.pause()
        await pilot.pause()
        assert transcript.scroll_offset.y < transcript.max_scroll_y


async def test_clearing_the_transcript_re_arms_following() -> None:
    app = FrontierAgentApp(_FakeSession())
    async with app.run_test(size=(80, 20)) as pilot:
        await pilot.pause()
        transcript = app.transcript
        await transcript.clear_all()
        for index in range(60):
            app.sink.note(f"line {index}")
        await pilot.pause()
        await pilot.pause()

        transcript.scroll_to(y=4, animate=False, immediate=True)
        await pilot.pause()
        assert transcript._wants_tail is False

        await transcript.clear_all()
        await pilot.pause()
        assert transcript._wants_tail is True
        for index in range(60):
            app.sink.note(f"fresh {index}")
            await pilot.pause(0.005)
        await pilot.pause()
        await pilot.pause()
        assert transcript.scroll_offset.y == transcript.max_scroll_y


async def test_follow_tail_is_safe_before_the_pane_is_mounted() -> None:
    """``follow_tail`` is public and called from the app's render pump."""
    box = TailScroll()
    assert box._wants_tail is True
    box.follow_tail()  # must not raise


async def test_user_messages_mark_clear_turn_boundaries() -> None:
    app = FrontierAgentApp(_FakeSession())
    async with app.run_test() as pilot:
        await pilot.pause()
        await app.transcript.clear_all()
        app.sink.echo_user("first task")
        app.sink.echo_user("second task")
        await pilot.pause()

        starts = list(app.transcript.query(".turn-start"))
        assert len(starts) == 2
        assert all(block.has_class("user-message") for block in starts)
        assert starts[0].styles.border.top[0] == "dashed"
        assert starts[0].styles.border.left[0] == "outer"


async def test_review_cursor_stays_visible_on_a_user_prompt() -> None:
    """The turn rail and the review cursor must not paint the same cell alike."""
    app = FrontierAgentApp(_FakeSession())
    async with app.run_test() as pilot:
        await pilot.pause()
        await app.transcript.clear_all()
        app.sink.echo_user("first task")
        await pilot.pause()

        block = next(iter(app.transcript.query(".turn-start")))
        app.transcript.review_move(-1)
        await pilot.pause()

        assert block.has_class("review-active")
        # Same colour is fine — the same *glyph* would make the cursor vanish.
        assert block.styles.outline.left[0] != block.styles.border.left[0]


def test_no_block_rail_collides_with_the_review_cursor() -> None:
    """The cursor's outline overwrites the rail cell, so no rail may match it.

    Checked against the stylesheet rather than one rendered block: the point is
    that a *new* rule must not reintroduce the collision either.
    """
    cursor = re.search(
        r"\.review-active\s*\{[^}]*outline-left:\s*(\w+)\s+(\$[\w-]+)",
        FrontierAgentApp.CSS,
    )
    assert cursor is not None
    style, colour = cursor.group(1), cursor.group(2)
    rails = re.findall(r"border-left:\s*(\w+)\s+(\$[\w-]+)", FrontierAgentApp.CSS)
    assert rails, "expected the transcript to draw accent rails"
    assert (style, colour) not in rails


async def test_input_method_commit_reaches_the_prompt_intact() -> None:
    """End to end: the terminal's key report has to land in the prompt as the
    phrase the user chose. Asserting on the parser alone would miss the half of
    this that matters — whether the per-codepoint key events it produces are
    printable enough for ``Input`` to insert them, and in order."""
    from textual._xterm_parser import XTermParser

    from apodex.tui.ime import ime_commit_sequence

    app = FrontierAgentApp(_FakeSession())
    async with app.run_test() as pilot:
        prompt = app.query_one("#prompt", Input)
        prompt.focus()
        await pilot.pause()
        parser = XTermParser()
        phrase = "标称的跟实际有很大差别"
        for event in parser.feed(ime_commit_sequence(phrase)):
            if isinstance(event, events.Key):
                prompt.post_message(event)
        await pilot.pause()
        assert prompt.value == phrase


def test_input_method_commits_survive_the_kitty_keyboard_protocol() -> None:
    """A CJK input method does not send committed text as UTF-8 bytes: with
    ``KITTY_REPORT_ASSOCIATED_TEXT`` requested, the terminal reports one key
    event carrying the phrase as decimal codepoints. Textual's collector used to
    abandon the sequence at 32 characters — reached by the fifth character — and
    re-issue it as literal keys, so anything committed from a candidate window
    landed in the prompt as ``^[32;;26377:24456:…u``. See ``apodex.tui.ime``.
    """
    from textual._xterm_parser import XTermParser

    from apodex.tui.ime import ime_commit_sequence, widen_escape_sequence_limit

    def typed(text: str) -> str:
        parser = XTermParser()
        return "".join(
            event.character or ""
            for event in parser.feed(ime_commit_sequence(text))
            if isinstance(event, events.Key)
        )

    assert widen_escape_sequence_limit() >= 512
    # Four characters fit under the old ceiling and always worked; five is where
    # it broke, and the failure is a hard cutoff rather than a race.
    for phrase in ("有", "有很大差", "有很大差别", "标称的跟实际有很大差别"):
        assert typed(phrase) == phrase, phrase
    # Committing with return rather than space is the same shape.
    parser = XTermParser()
    committed = "".join(
        event.character or ""
        for event in parser.feed(ime_commit_sequence("这样也可以", key=13))
        if isinstance(event, events.Key)
    )
    assert committed == "这样也可以"


def test_the_tui_widens_the_escape_limit_before_reading_input() -> None:
    """The fix has to be installed by construction, not by importing a module:
    the driver starts reading stdin as soon as the app runs."""
    import textual._xterm_parser as xterm_parser

    from apodex.tui.ime import MIN_ESCAPE_SEQUENCE_LIMIT

    original = xterm_parser._MAX_SEQUENCE_SEARCH_THRESHOLD
    try:
        xterm_parser._MAX_SEQUENCE_SEARCH_THRESHOLD = 32
        FrontierAgentApp(_FakeSession())
        assert xterm_parser._MAX_SEQUENCE_SEARCH_THRESHOLD == MIN_ESCAPE_SEQUENCE_LIMIT
    finally:
        xterm_parser._MAX_SEQUENCE_SEARCH_THRESHOLD = original


async def test_transcript_process_rows_use_dense_vertical_spacing() -> None:
    app = FrontierAgentApp(_FakeSession())
    async with app.run_test(size=(120, 30)) as pilot:
        await pilot.pause()
        await app.transcript.begin_process()
        app.sink.tool_call("bash", {"command": "pwd"}, call_id="dense")
        app.sink.tool_result("bash", "/tmp", is_error=False, call_id="dense")
        await pilot.pause()

        call = app.transcript.query_one(".tool-call", Static)
        result = app.transcript.query_one(".tool-result", Collapsible)
        assert call.styles.margin.bottom == 0
        assert result.styles.margin.bottom == 0


async def test_transcript_follows_new_output_but_not_over_the_user() -> None:
    app = FrontierAgentApp(_FakeSession())
    async with app.run_test(size=(120, 30)) as pilot:
        await pilot.pause()
        for index in range(60):
            app.sink.note(f"line {index}")
        await pilot.pause()
        await pilot.pause()
        transcript = app.transcript
        assert transcript.max_scroll_y > 0
        assert transcript.scroll_offset.y == transcript.max_scroll_y

        # Reading further up must survive new output arriving.
        transcript.scroll_to(y=10, animate=False, immediate=True)
        await pilot.pause()
        app.sink.note("more output while scrolled up")
        await pilot.pause()
        await pilot.pause()
        assert transcript.scroll_offset.y == 10


async def test_task_board_persists_after_completion_until_next_query() -> None:
    from apodex.todo import TodoItem

    app = FrontierAgentApp(_FakeSession())
    async with app.run_test(size=(120, 30)) as pilot:
        await pilot.pause()
        # 1. Initially "no plan yet"
        assert "no plan yet" in app.todos_pane.render().plain

        # 2. Agent updates plan during task
        app.show_todos([TodoItem("Fix main bug", "completed"), TodoItem("Add tests", "in_progress")])
        await pilot.pause()
        rendered = app.todos_pane.render().plain
        assert "Fix main bug" in rendered
        assert "Add tests" in rendered

        # 3. Task completes and chrome refreshes (e.g. show_todos([]) called with empty todo)
        app._refresh_chrome()
        await pilot.pause()
        rendered_after_chrome = app.todos_pane.render().plain
        # Board MUST NOT collapse back to "no plan yet" after completion!
        assert "Fix main bug" in rendered_after_chrome
        assert "no plan yet" not in rendered_after_chrome

        # 4. Next query enters -> task board clears for new query
        app.clear_todos()
        await pilot.pause()
        assert "no plan yet" in app.todos_pane.render().plain


# ── the startup logo ──────────────────────────────────────────────────────
def test_logo_never_exceeds_the_width_it_was_fitted_to() -> None:
    """Every tier must fit, because the transcript would wrap what does not.

    A wrapped logo folds the peaks back under themselves, so the interesting
    widths are the ones on either side of each tier's threshold.
    """
    from apodex.tui.logo import (
        _MARK_WIDTH as MARK_WIDTH,
    )
    from apodex.tui.logo import (
        FULL_WIDTH,
        NAMED_WIDTH,
        ONE_LINE_WIDTH,
        render_logo,
    )

    edges = (ONE_LINE_WIDTH, MARK_WIDTH, NAMED_WIDTH, FULL_WIDTH)
    for width in (20, *(edge + step for edge in edges for step in (-1, 0)), 200):
        rendered = render_logo("catppuccin", width).plain.splitlines()
        widest = max(cell_len(line) for line in rendered)
        assert widest <= width, f"logo is {widest} cells wide at width {width}"
        # A logo that fills the transcript is as bad as one that wraps.
        assert len(rendered) <= 13


def test_logo_tiers_keep_the_wordmark_for_as_long_as_they_can() -> None:
    from apodex.tui.logo import (
        _MARK_WIDTH as MARK_WIDTH,
    )
    from apodex.tui.logo import (
        FULL_WIDTH,
        NAMED_WIDTH,
        render_logo,
    )

    def logo(width: int) -> list[str]:
        return render_logo("catppuccin", width).plain.splitlines()

    # Widest: the wordmark is set in the pixel face beside the mark. The face
    # has no lowercase, so plain "FrontierAgent" appearing means it degraded.
    assert "FrontierAgent" not in "".join(logo(FULL_WIDTH))
    assert "FRONTIER" not in "".join(logo(FULL_WIDTH))  # …it is pixels, not text

    # Then the name in plain text beside the mark, then below it, then alone.
    assert "FrontierAgent" in logo(FULL_WIDTH - 1)[4]
    assert logo(NAMED_WIDTH - 1)[-2].strip() == "FrontierAgent"
    assert logo(MARK_WIDTH - 1) == ["◭ FrontierAgent"]

    # The tagline survives every tier wide enough to hold the mark.
    for width in (FULL_WIDTH, FULL_WIDTH - 1, NAMED_WIDTH - 1, MARK_WIDTH):
        assert "self-evolving · by Apodex" in "".join(logo(width))


def test_logo_takes_every_colour_from_the_active_palette() -> None:
    """The logo's whole point is following the theme, so nothing may be fixed."""
    from apodex.tui.logo import render_logo

    for theme in ("catppuccin", "gruvbox-light", "nord", "solarized"):
        spec = palette(theme)
        used = {str(span.style).replace("bold ", "").replace("italic ", "")
                for span in render_logo(theme, 200).spans}
        assert used == {spec.primary, spec.accent, spec.foreground, spec.muted}


async def test_logo_opens_the_transcript_and_repaints_on_a_theme_switch() -> None:
    app = FrontierAgentApp(_FakeSession())
    async with app.run_test(size=(140, 30)) as pilot:
        await pilot.pause()
        logo = app.transcript.query_one(".startup-logo", Static)
        assert "self-evolving · by Apodex" in logo.render().plain
        # It is the transcript's first block — a banner below the welcome note
        # would not be a banner.
        blocks = [child for child in app.transcript.children
                  if child.has_class("block")]
        assert blocks[0] is logo

        def colours() -> set[str]:
            # Static normalizes hex to rgb() on the way out, so compare the
            # parsed colour rather than the style string.
            block = app.transcript.query_one(".startup-logo", Static)
            return {Style.parse(str(span.style)).color.triplet.hex
                    for span in block.render().spans}

        before = colours()
        app._apply_theme("gruvbox")
        await pilot.pause()
        after = colours()
        assert after != before
        assert palette("gruvbox").primary in after


async def test_logo_re_fits_itself_when_the_pane_changes_width() -> None:
    """Regression: the fit at mount is a guess — layout has not run yet.

    Fitting to the app's width instead of the pane's rendered an 86-column logo
    into a 74-column pane, and ``text-wrap: nowrap`` did not save it: Textual
    measures the wrapped height before the clip applies, so the block reserved
    18 rows and pushed the transcript into a scroll.
    """
    from apodex.tui.logo import FULL_WIDTH

    app = FrontierAgentApp(_FakeSession())
    async with app.run_test(size=(120, 30)) as pilot:
        await pilot.pause()
        logo = app.transcript.query_one(".startup-logo", Static)

        def fitted() -> tuple[int, int]:
            lines = logo.render().plain.splitlines()
            return max(cell_len(line) for line in lines), logo.size.height

        # The sidebar takes 45 columns, so 120 is not the width to fit to.
        widest, height = fitted()
        assert widest <= app.transcript.content_size.width
        assert height == len(logo.render().plain.splitlines())

        # Wide enough for the pixel wordmark, then narrow enough to lose it.
        await pilot.resize_terminal(FULL_WIDTH + 6, 30)
        await pilot.pause()
        assert fitted()[0] == FULL_WIDTH

        await pilot.resize_terminal(60, 30)
        await pilot.pause()
        widest, height = fitted()
        assert widest <= app.transcript.content_size.width
        assert height == len(logo.render().plain.splitlines())
