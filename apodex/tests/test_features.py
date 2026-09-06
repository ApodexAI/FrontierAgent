"""Unit tests for the UX features: diff preview, todo plan, theming."""

from __future__ import annotations

import asyncio

import pytest

from apodex.agent_tools import (
    RISK_CONFIRM,
    RISK_DENY,
    RISK_SAFE,
    assess_tool_risk,
    coding_tools,
    is_read_only_bash,
    localize_path_args,
)
from apodex.diff_preview import change_stats, proposed_change, unified_diff
from apodex.render import Renderer
from apodex.todo import clear_todos, get_todos, todo_write


# ── diff preview ──────────────────────────────────────────────────────────
def test_diff_preview_new_file(tmp_path):
    cwd = str(tmp_path)
    diff = unified_diff("file_editor_create", {"path": "new.py", "content": "a = 1\nb = 2\n"}, cwd)
    assert diff is not None
    assert "+a = 1" in diff and "+b = 2" in diff
    assert change_stats("file_editor_create", {"path": "new.py", "content": "a = 1\nb = 2\n"}, cwd) == (2, 0)


def test_diff_preview_str_replace(tmp_path):
    (tmp_path / "calc.py").write_text("def add(a, b):\n    return a - b\n")
    cwd = str(tmp_path)
    args = {"path": "calc.py", "old_str": "return a - b", "new_str": "return a + b"}
    diff = unified_diff("file_editor_str_replace", args, cwd)
    assert diff is not None
    assert "-    return a - b" in diff
    assert "+    return a + b" in diff


def test_diff_preview_no_change_returns_none(tmp_path):
    (tmp_path / "x.txt").write_text("same\n")
    cwd = str(tmp_path)
    # write the identical content → no diff
    assert unified_diff("write_file", {"path": "x.txt", "content": "same\n"}, cwd) is None
    # str_replace whose old_str isn't present → unpreviewable → None
    assert unified_diff("file_editor_str_replace", {"path": "x.txt", "old_str": "nope", "new_str": "y"}, cwd) is None
    # non-edit tool → None
    assert unified_diff("read_file", {"path": "x.txt"}, cwd) is None


def test_diff_preview_skips_binary(tmp_path):
    (tmp_path / "b.bin").write_bytes(b"\x89PNG\x00\x00\x01data")
    assert unified_diff("write_file", {"path": "b.bin", "content": "x"}, str(tmp_path)) is None


def test_diff_preview_str_replace_requires_exactly_once(tmp_path):
    (tmp_path / "m.py").write_text("x = 1\nx = 1\n")  # old_str appears twice
    cwd = str(tmp_path)
    # >1 occurrence → tool would error → not previewable
    assert unified_diff("file_editor_str_replace", {"path": "m.py", "old_str": "x = 1", "new_str": "y = 2"}, cwd) is None
    (tmp_path / "s.py").write_text("x = 1\n")  # exactly once → previewable
    assert unified_diff("file_editor_str_replace", {"path": "s.py", "old_str": "x = 1", "new_str": "y = 2"}, cwd) is not None


def test_diff_preview_normalizes_path(tmp_path):
    diff = unified_diff("file_editor_create", {"path": "sub/../t.py", "content": "z\n"}, str(tmp_path))
    assert diff is not None and "sub/../t.py" not in diff and "t.py" in diff


def test_write_append_preview(tmp_path):
    (tmp_path / "log.txt").write_text("line1\n")
    cwd = str(tmp_path)
    change = proposed_change("write_file", {"path": "log.txt", "content": "line2\n", "append": True}, cwd)
    assert change is not None
    _, old, new = change
    assert old == "line1\n" and new == "line1\nline2\n"


# ── todo plan ─────────────────────────────────────────────────────────────
def test_todo_write_and_store():
    clear_todos()
    items = [
        {"content": "read code", "status": "completed"},
        {"content": "fix bug", "status": "in_progress"},
        {"content": "run tests", "status": "pending"},
    ]
    out = asyncio.run(todo_write.ainvoke({"todos": items}))
    assert "3" in out
    todos = get_todos()
    assert [t.content for t in todos] == ["read code", "fix bug", "run tests"]
    assert [t.status for t in todos] == ["completed", "in_progress", "pending"]
    assert todos[0].glyph == "✓" and todos[1].glyph == "▶" and todos[2].glyph == "○"
    clear_todos()


def test_todo_write_rejects_empty():
    clear_todos()
    out = asyncio.run(todo_write.ainvoke({"todos": []}))
    assert "Error" in out
    assert get_todos() == []


def test_todo_in_coding_tools():
    assert "todo_write" in {t.name for t in coding_tools()}


# ── agent profiles (coding / research modes) ──────────────────────────────
def test_profiles_registered_and_distinct():
    from apodex.profiles import get_profile, profile_names

    assert set(profile_names()) >= {"coding", "research"}
    coding = {t.name for t in get_profile("coding").tools()}
    research = {t.name for t in get_profile("research").tools()}
    assert "file_editor_str_replace" in coding and "web_search" not in coding
    assert {"web_search", "web_fetch"} <= research and "file_editor_str_replace" not in research
    # both attach robustness observers
    obs = [type(o).__name__ for o in get_profile("coding").extra_observers(["bash"])]
    assert "TextRepetitionGuard" in obs


def test_profile_prompts_come_from_react_workflow():
    """The prompts reuse workflows/default_research (react_research) builders."""
    from apodex.profiles import get_profile

    coding_p = get_profile("coding").system_prompt("/tmp/repo")
    research_p = get_profile("research").system_prompt("/tmp/repo")
    assert "/tmp/repo" in coding_p and len(coding_p) > 500
    assert len(research_p) > 500 and "/tmp/repo" in research_p


def test_unknown_mode_raises():
    from apodex.profiles import get_profile

    with pytest.raises(KeyError):
        get_profile("nonexistent")


def test_web_tools_auto_approved():
    cwd = "/tmp"
    assert assess_tool_risk("web_search", {"q": "x"}, cwd).level == RISK_SAFE
    assert assess_tool_risk("web_fetch", {"url": "http://x"}, cwd).level == RISK_SAFE


def test_builtin_workflow_tools_auto_approved():
    cwd = "/tmp"
    for tool_name in ("add_task", "update_task", "finish_planning",
                      "create_subagent", "assign_task", "collect_reports",
                      "stop_subagent", "submit_report", "read_text", "view_image"):
        assert assess_tool_risk(tool_name, {}, cwd).level == RISK_SAFE, tool_name


# ── read-only bash auto-approve (reduces approval friction) ───────────────
def test_read_only_bash_autoapproved():
    cwd = "/tmp"
    for cmd in ("ls -la", "find . -name '*.py'", "grep -r foo src", "tree -L 2",
                "cat README.md", "git status", "git diff", "pwd && ls"):
        assert is_read_only_bash(cmd), cmd
        assert assess_tool_risk("bash", {"command": cmd}, cwd).level == RISK_SAFE, cmd


def test_mutating_bash_requires_confirm():
    cwd = "/tmp"
    for cmd in ("rm file.txt", "mv a b", "echo x > out.txt", "pytest -q",
                "pip install requests", "git commit -m x", "find . -delete",
                "cat x | tee y", "python script.py", "make build"):
        assert not is_read_only_bash(cmd), cmd
        assert assess_tool_risk("bash", {"command": cmd}, cwd).level == RISK_CONFIRM, cmd


def test_dangerous_bash_denied():
    assert assess_tool_risk("bash", {"command": "rm -rf /"}, "/tmp").level == RISK_DENY


def test_write_capable_commands_not_autoapproved():
    """Regression: commands that can write/exec must NOT be classified safe."""
    cwd = "/tmp"
    for cmd in (
        "sort -o out.txt in.txt",      # sort -o writes
        "env FOO=1 rm x",              # env execs an arbitrary command
        "ls & rm x",                   # background & hides a mutating command
        "find . -fprintf out '%p'",    # find write action
        "find . -name x -ok rm {} ;",  # find interactive exec
        "ls | xargs rm",               # xargs runs anything
    ):
        assert not is_read_only_bash(cmd), cmd
        # Either gate satisfies the intent: the command must not reach the
        # model unreviewed. ``deny`` is the stricter outcome — the bash
        # denylist refuses some of these outright (``xargs`` feeding ``rm``
        # cannot have its stdin targets validated), and that must not read
        # as a regression against ``confirm``.
        assert assess_tool_risk("bash", {"command": cmd}, cwd).level in (
            RISK_CONFIRM, RISK_DENY,
        ), cmd
    # genuinely read-only stays auto-approved
    assert is_read_only_bash("ls -la && cat x")
    assert is_read_only_bash("grep -rn foo src")


def test_outside_cwd_fail_closed():
    from apodex.agent_tools import _outside_cwd
    assert _outside_cwd("", "/tmp") is True            # empty → deny
    assert _outside_cwd("/etc/passwd", "/tmp") is True  # clearly outside


# ── search prune is repo-driven (.gitignore), NOT hardcoded to a project ──
def test_prune_set_is_universal_plus_gitignore(tmp_path):
    from apodex.local_tools import _prune_dir_names

    (tmp_path / ".gitignore").write_text("__pycache__/\nbuild/\nmyoutput/\n")
    prune = _prune_dir_names(str(tmp_path))
    # universal artifact dirs are always pruned
    assert {".git", ".venv", "node_modules", "__pycache__"} <= prune
    # this repo's OWN declared dirs are pruned
    assert "build" in prune and "myoutput" in prune
    # a name this repo does NOT ignore is NOT pruned (no overfitting)
    assert "results" not in prune and "src" not in prune


def test_local_glob_grep_skip_gitignored_dir(tmp_path, monkeypatch):
    import asyncio as _a

    from apodex.local_tools import glob_search, grep_search
    (tmp_path / ".gitignore").write_text("junk/\n")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "real.py").write_text("MARKER = 1\n")
    (tmp_path / "junk").mkdir()
    (tmp_path / "junk" / "ignored.py").write_text("MARKER = 2\n")
    monkeypatch.chdir(tmp_path)

    files = _a.run(glob_search.ainvoke({"pattern": "**/*.py"}))
    assert "src/real.py" in files and "junk" not in files  # gitignored dir pruned

    hits = _a.run(grep_search.ainvoke({"pattern": "MARKER"}))
    assert "src/real.py" in hits and "junk/ignored.py" not in hits


def test_optimize_find_injects_prune(tmp_path, monkeypatch):
    from apodex.local_tools import _optimize_find
    monkeypatch.chdir(tmp_path)
    cmd, changed = _optimize_find('find . -name "*.py" | head -5')
    assert changed and "-prune" in cmd and "*/.venv/*" in cmd
    # already-pruned / mutating finds are left alone
    assert _optimize_find("find . -name x -delete")[1] is False
    assert _optimize_find("ls -la")[1] is False


def test_optimize_find_quote_aware_split(tmp_path, monkeypatch):
    """A '|' inside a quoted -name must not be treated as a pipeline."""
    from apodex.local_tools import _optimize_find
    monkeypatch.chdir(tmp_path)
    out, changed = _optimize_find('find . -name "a|b.txt" | head')
    assert changed and "-prune" in out and out.rstrip().endswith("| head")


def test_glob_matches_root_and_nested(tmp_path, monkeypatch):
    import asyncio as _a

    from apodex.local_tools import glob_search
    (tmp_path / "root.py").write_text("x")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "deep.py").write_text("y")
    monkeypatch.chdir(tmp_path)
    files = _a.run(glob_search.ainvoke({"pattern": "**/*.py"})).split()
    assert "root.py" in files and "sub/deep.py" in files  # root file must match


def test_gitignore_path_anchored_not_overpruned(tmp_path):
    from apodex.local_tools import _gitignored_dir_names
    (tmp_path / ".gitignore").write_text("benchmarks/public/results/\nbuild/\nvenvthing\n")
    names = _gitignored_dir_names(str(tmp_path))
    assert "build" in names            # bare dir → pruned anywhere (git-correct)
    assert "results" not in names      # path-anchored → NOT pruned by leaf name
    assert "venvthing" in names


# ── path localization (abs→rel avoids the ~50s sandbox slow path) ─────────
def test_localize_absolute_path_inside_cwd(tmp_path):
    cwd = str(tmp_path)
    (tmp_path / "sub").mkdir()
    abspath = str(tmp_path / "sub" / "f.py")
    out = localize_path_args("read_file", {"path": abspath}, cwd)
    assert out is not None and out["path"] == "sub/f.py"


def test_localize_preserves_absolute_task_input_path(tmp_path, monkeypatch):
    inputs = tmp_path / ".apodex" / "inputs" / "run"
    inputs.mkdir(parents=True)
    attached = inputs / "brief.md"
    attached.write_text("brief")
    monkeypatch.setenv("FRONTIER_AGENT_INPUTS_DIR", str(inputs))

    out = localize_path_args(
        "grep_search", {"path": str(attached), "pattern": "brief"}, str(tmp_path),
    )

    assert out is None


def test_local_read_file_fast_and_scoped(tmp_path, monkeypatch):
    import asyncio as _a

    from apodex.local_tools import read_file
    (tmp_path / "README.md").write_text("# Hello\nworld\n")
    monkeypatch.chdir(tmp_path)
    # relative
    assert "Hello" in _a.run(read_file.ainvoke({"path": "README.md"}))
    # leading-slash repo-root confusion → falls back to cwd/README.md
    assert "Hello" in _a.run(read_file.ainvoke({"path": "/README.md"}))
    # outside the workspace → fast rejection, not a system-file leak
    outside = tmp_path.parent / f"{tmp_path.name}-outside.txt"
    outside.write_text("must not be readable\n")
    out = _a.run(read_file.ainvoke({"path": str(outside)}))
    assert "outside the working directory" in out
    # missing file → fast not-found
    assert "not found" in _a.run(read_file.ainvoke({"path": "nope.txt"}))


def test_localize_leaves_relative_and_outside(tmp_path):
    cwd = str(tmp_path)
    # already relative → no rewrite
    assert localize_path_args("read_file", {"path": "a.py"}, cwd) is None
    # absolute but outside cwd → no rewrite (left for the tool's own guard)
    assert localize_path_args("read_file", {"path": "/etc/hosts"}, cwd) is None
    # non-path tool → no rewrite
    assert localize_path_args("bash", {"command": "ls"}, cwd) is None


# ── theming / rendering does not crash ────────────────────────────────────
def test_renderer_themes_smoke(capsys):
    from apodex.todo import TodoItem
    from apodex.tui.themes import CLI_THEME_NAMES

    for theme in CLI_THEME_NAMES:
        r = Renderer(theme=theme)
        r.banner(model="m", cwd="/tmp", auto_approve=False)
        r.tool_call("file_editor_str_replace", {"path": "a.py"}, risk_reason="modifies a file")
        r.diff_preview("--- a/a.py\n+++ b/a.py\n@@ -1 +1 @@\n-old\n+new", stats=(1, 1))
        r.tool_result("bash", "ok", is_error=False, ms=12)
        r.todos([TodoItem("step 1", "completed"), TodoItem("step 2", "in_progress")])
        r.final("**done**", turns=3, tool_calls=2, stopped_by="no_tool")
    # If we got here without an exception, rendering is robust across themes.
    capsys.readouterr()


def test_explicit_compaction_compresses_tool_results_before_summary():
    from types import SimpleNamespace

    from frontier_agent.core.messages import assistant_msg, system_msg, tool_msg, user_msg
    from frontier_agent.core.runtime.loop.compact_llm import LLMSummaryCompactor

    class _SummaryLLM:
        prompt = ""

        async def chat(self, messages):
            self.prompt = messages[0]["content"]
            return SimpleNamespace(content="## Investigation so far\nDone.")

    llm = _SummaryLLM()
    history = [
        system_msg("system instructions"),
        user_msg("inspect the project"),
        assistant_msg(tool_calls=[{
            "id": "call-1", "type": "function",
            "function": {"name": "read_file", "arguments": "{}"},
        }]),
        tool_msg("start\n" + "x" * 3_000 + "\nhttps://example.test/source", "call-1"),
    ]

    compacted = asyncio.run(LLMSummaryCompactor(summary_llm=llm).compact(
        history, keep_recent=0, compress_all_tool_results=True,
    ))

    assert len(compacted) == 2
    assert compacted[1]["content"].startswith("[Compacted summary")
    assert "[Compressed tool result:" in llm.prompt
    assert "https://example.test/source" in llm.prompt
    assert "x" * 3_000 not in llm.prompt


def test_renderer_call_id_does_not_change_line_mode_output(capsys):
    renderer = Renderer(theme="mono")
    renderer.tool_call("bash", {"command": "pwd"})
    renderer.tool_result("bash", "/tmp", is_error=False, ms=12)
    baseline = capsys.readouterr().out

    renderer.tool_call("bash", {"command": "pwd"}, call_id="call-a")
    renderer.tool_result(
        "bash", "/tmp", is_error=False, ms=12, call_id="call-a",
    )
    assert capsys.readouterr().out == baseline


# ── human-in-the-loop approval (redirect with feedback) ────────────────────
def _turn_ctx():
    from frontier_agent.core.loop_types import TurnContext
    return TurnContext(turn=1, max_turns=10, task_id="", role_id="", ai_text="",
                       thinking="", tool_calls=[], messages=[], usage=None, metadata={})


def test_approver_decision_auto_and_noninteractive():
    from apodex.observers import Approver
    assert asyncio.run(Approver(auto_approve=True).confirm("bash", "x", "r")).approved is True
    d = asyncio.run(Approver(auto_approve=False, interactive=False).confirm("bash", "x", "r"))
    assert d.approved is False and d.feedback == ""  # no TTY → fail safe, no feedback


def test_on_tool_call_redirect_feeds_back_user_instruction(tmp_path):
    """A typed redirect at the gate becomes the declined call's result so the
    agent course-corrects (human-in-the-loop)."""
    from apodex.observers import Decision, TerminalObserver

    class _Redirect:
        auto_approve = False
        async def confirm(self, name, target, reason, **kw):
            return Decision(False, feedback="use ripgrep instead of rm")

    obs = TerminalObserver(Renderer(theme="mono"), _Redirect(), str(tmp_path))
    iv = asyncio.run(obs.on_tool_call(_turn_ctx(), {"name": "bash", "args": {"command": "rm -rf build"}}))
    assert iv is not None and iv.skip_with_result
    assert "use ripgrep instead of rm" in iv.skip_with_result
    assert "declined" in iv.skip_with_result.lower()


def test_plain_reject_stops_the_task(tmp_path):
    """A plain reject (no feedback) ends the task at on_turn_end so the model
    can't keep retrying the declined action."""
    from apodex.observers import Decision, TerminalObserver

    class _Reject:
        auto_approve = False
        async def confirm(self, name, target, reason, **kw):
            return Decision(False)  # plain reject, no feedback

    obs = TerminalObserver(Renderer(theme="mono"), _Reject(), str(tmp_path))
    iv = asyncio.run(obs.on_tool_call(_turn_ctx(), {"name": "bash", "args": {"command": "rm -rf build"}}))
    assert iv is not None and "task stopped" in (iv.skip_with_result or "")
    end = asyncio.run(obs.on_turn_end(_turn_ctx()))
    assert end is not None and end.stop_reason == "user_rejected"
    # one-shot: the abort flag is cleared after firing
    assert asyncio.run(obs.on_turn_end(_turn_ctx())) is None


def test_redirect_feedback_does_not_stop(tmp_path):
    """A typed redirect continues the task (no stop) — only plain reject stops."""
    from apodex.observers import Decision, TerminalObserver

    class _Redirect:
        auto_approve = False
        async def confirm(self, name, target, reason, **kw):
            return Decision(False, feedback="use grep instead")

    obs = TerminalObserver(Renderer(theme="mono"), _Redirect(), str(tmp_path))
    asyncio.run(obs.on_tool_call(_turn_ctx(), {"name": "bash", "args": {"command": "rm x"}}))
    assert asyncio.run(obs.on_turn_end(_turn_ctx())) is None  # feedback → keep going


def test_on_tool_result_suppresses_synthetic_markers(tmp_path):
    """Rejected/blocked synthetic results aren't re-rendered as ✓ panels."""
    from apodex.observers import Approver, TerminalObserver
    from frontier_agent.core.loop_types import ToolResult

    class _Rec(Renderer):
        def __init__(self):
            super().__init__(theme="mono")
            self.called = []
            self.completed = []
            self.shown = []
        def tool_call(self, name, args, risk_reason="", danger=False):
            self.called.append(name)
        def tool_result(self, name, result, *, is_error, ms=0):
            self.shown.append(name)
        def activity_result(
            self, name, *, call_id="", is_error, ms=0, outcome="",
        ):
            self.completed.append((name, call_id, outcome))

    rec = _Rec()
    obs = TerminalObserver(rec, Approver(auto_approve=True), str(tmp_path))
    asyncio.run(obs.on_tool_call(
        _turn_ctx(), {"id": "1", "name": "bash", "args": {"command": "pwd"}},
    ))
    synthetic = ToolResult(name="bash", args={}, result="[user rejected this bash call — choose a different approach]",
                           duration_ms=1, tool_call_id="1", is_error=False)
    asyncio.run(obs.on_tool_result(_turn_ctx(), synthetic))
    assert rec.called == ["bash"]  # pre-call-id custom signature remains compatible
    assert rec.completed == [("bash", "1", "skipped")]
    assert rec.shown == []  # suppressed
    real = ToolResult(name="bash", args={}, result="real output", duration_ms=1, tool_call_id="2", is_error=False)
    asyncio.run(obs.on_tool_result(_turn_ctx(), real))
    assert rec.shown == ["bash"]  # normal result still rendered


def test_observer_closes_the_narration_block_at_each_turn_boundary(tmp_path):
    """One assistant message is one block, even with no tool row between them."""
    from apodex.observers import Approver, TerminalObserver
    from frontier_agent.core.loop_types import LLMDeltaContext

    class _Rec(Renderer):
        def __init__(self):
            super().__init__(theme="mono")
            self.events = []

        def content_delta(self, s):
            self.events.append(("delta", s))

        def end_turn_text(self):
            self.events.append(("end", ""))

        def turn_text_fallback(self, ai_text, thinking):
            self.events.append(("fallback", ai_text))

    def delta_ctx(text):
        return LLMDeltaContext(turn=1, max_turns=10, task_id="", role_id="",
                               delta=text, accumulated_text=text, delta_index=0,
                               metadata={})

    rec = _Rec()
    obs = TerminalObserver(rec, Approver(auto_approve=True), str(tmp_path))
    asyncio.run(obs.on_llm_delta(delta_ctx("first.")))
    asyncio.run(obs.on_llm_response(_turn_ctx()))
    # A turn the provider never streamed keeps taking the fallback path only:
    # that renders a discrete block already, so there is nothing to close.
    asyncio.run(obs.on_llm_response(_turn_ctx()))
    assert rec.events == [("delta", "first."), ("end", ""), ("fallback", "")]


def test_line_mode_turn_boundary_leaves_the_thinking_ticker_alone():
    """Closing prose must not blank the only indicator during generation."""
    r = Renderer(theme="mono")
    r.content_delta("streamed prose")
    assert r._streaming_kind == "content"
    r.end_turn_text()
    assert r._streaming_kind is None

    r._streaming_kind = "thinking"
    r.end_turn_text()
    assert r._streaming_kind == "thinking"


def test_observer_forwards_call_identity_to_activity_aware_renderer(tmp_path):
    from apodex.observers import Approver, TerminalObserver
    from frontier_agent.core.loop_types import ToolResult

    class _Rec(Renderer):
        def __init__(self):
            super().__init__(theme="mono")
            self.events = []

        def tool_call(self, name, args, risk_reason="", danger=False, *, call_id=""):
            self.events.append(("call", name, call_id))

        def tool_result(self, name, result, *, is_error, ms=0, call_id=""):
            self.events.append(("result", name, call_id))

    rec = _Rec()
    obs = TerminalObserver(rec, Approver(auto_approve=True), str(tmp_path))
    asyncio.run(obs.on_tool_call(
        _turn_ctx(), {"id": "call-a", "name": "bash", "args": {"command": "pwd"}},
    ))
    asyncio.run(obs.on_tool_result(_turn_ctx(), ToolResult(
        name="bash", args={}, result="/tmp", duration_ms=3,
        tool_call_id="call-a", is_error=False,
    )))

    assert rec.events == [
        ("call", "bash", "call-a"),
        ("result", "bash", "call-a"),
    ]


# ── bash command intent (model description shown at approval) ────────────────
def test_bash_summary_prefers_description_then_comment_then_command():
    r = Renderer(theme="mono")
    # model-provided description wins
    assert r._summarize_args("bash", {"command": "ls -la", "description": "List files here"}) == "List files here"
    # else a leading '# comment' label
    assert r._summarize_args("bash", {"command": "# rebuild the index\nmake index"}) == "rebuild the index"
    # else the raw command (and a shebang is NOT treated as a label)
    assert r._summarize_args("bash", {"command": "git status"}) == "git status"
    assert r._summarize_args("bash", {"command": "#!/bin/sh\necho hi"}).startswith("#!/bin/sh")


def test_step_summary_leads_with_the_most_specific_argument():
    """A step row is scanned for *what* the agent is doing, so the summary has
    to be the search term or the URL — not the working directory it happens to
    also carry, and not a ``k=v`` dump truncated mid-word."""
    r = Renderer(theme="mono")
    # ``pattern`` beats ``path``: leading with the directory printed it twice
    # and dropped the pattern, which is the only part identifying the search.
    assert r._summarize_args(
        "grep_search", {"pattern": "def load_config", "path": "src"},
    ) == "def load_config  in src"
    # The web tools used to fall through to the ``k=v`` tail, which truncates at
    # 40 characters — mid-query for anything a model actually searches for.
    long_query = "how to configure sglang for a 35B model on a single 5090"
    assert r._summarize_args("web_search", {"query": long_query}) == long_query
    assert r._summarize_args(
        "web_fetch", {"url": "https://example.com/a", "info_to_extract": "size"},
    ) == "https://example.com/a"
    # A batched fetch: the list must not be stringified as Python source.
    assert r._summarize_args(
        "web_fetch", {"url": ["https://a.test/1", "https://b.test/2"]},
    ) == "https://a.test/1  +1 more"
    # A download names the source, not the destination path it also carries.
    assert r._summarize_args(
        "download_file", {"url": "https://x.test/p.pdf", "path": "out/p.pdf"},
    ) == "https://x.test/p.pdf"
    # A snippet leads with its first real line, not a comment, and says how much
    # more there is instead of truncating at 40 characters mid-expression.
    assert r._summarize_args(
        "run_python_code", {"code": "# load it\nimport pandas as pd\nprint(1)"},
    ) == "import pandas as pd  · 3 lines"
    assert r._summarize_args("run_python_code", {"code": "  "}) == "python snippet"
    # Task board tools carry a list of dicts; a stringified list is Python
    # source, not a summary.
    assert r._summarize_args(
        "add_task", {"tasks": [{"description": "measure the image"}]},
    ) == "1 item · measure the image"
    assert r._summarize_args(
        "update_task",
        {"updates": [{"id": "t1", "resolution": "resolved"}, {"id": "t2"}]},
    ) == "2 items · resolved"
    assert r._summarize_args("todo_write", {"todos": []}) == "0 items"


def test_bash_tool_call_shows_intent_and_exact_command(capsys):
    # With a description, the dim sub-line still surfaces the exact command.
    r = Renderer(theme="mono")
    r.tool_call("bash", {"command": "rm -rf build", "description": "Remove the build dir"},
                risk_reason="recursive/forced delete", danger=True)
    out = capsys.readouterr().out
    assert "Remove the build dir" in out and "rm -rf build" in out


# ── steering / type-ahead queue ──────────────────────────────────────────────
def test_steer_inbox_feed_and_drain():
    from apodex.steer import SteerInbox

    class _R:
        def queued(self, t):  # renderer stub
            pass

    box = SteerInbox(_R())
    box._feed("first line\npart")        # one full line; 'part' stays buffered
    assert box.queue == ["first line"]
    box._feed("ial\nsecond\n")            # completes 'partial', then 'second'
    assert box.drain() == ["first line", "partial", "second"]
    assert box.drain() == []              # cleared after drain


def test_steer_inbox_wakes_a_parked_coordinator():
    from apodex.steer import SteerInbox

    class _R:
        def queued(self, t):
            pass

    async def _exercise():
        box = SteerInbox(_R())
        waiter = asyncio.create_task(box.wait_for_input())
        await asyncio.sleep(0)
        assert not waiter.done()

        box.enqueue("change direction")

        assert await asyncio.wait_for(waiter, timeout=0.1) is True
        assert box.drain() == ["change direction"]

    asyncio.run(_exercise())


def test_collect_wait_intervention_is_injected_immediately(tmp_path):
    from apodex.observers import Approver, TerminalObserver
    from apodex.steer import SteerInbox
    from frontier_agent.core.loop_types import TurnContext

    async def _exercise():
        renderer = Renderer(theme="mono")
        inbox = SteerInbox(renderer)
        observer = TerminalObserver(
            renderer, Approver(auto_approve=True), str(tmp_path),
            steer_inbox=inbox,
        )
        ctx = TurnContext(
            turn=1, max_turns=10, task_id="", role_id="", ai_text="",
            thinking="", tool_calls=[{"name": "collect_reports"}],
            messages=[], usage=None, metadata={},
        )
        waiter = asyncio.create_task(observer.wait_for_tool_interrupt(
            ctx, {"name": "collect_reports"},
        ))
        await asyncio.sleep(0)
        inbox.enqueue("prioritize the new request")

        assert await asyncio.wait_for(waiter, timeout=0.1) is True
        intervention = await observer.on_tool_wait_interrupted(ctx)
        assert intervention is not None
        assert intervention.inject_messages == ["prioritize the new request"]
        assert inbox.queue == []

    asyncio.run(_exercise())


def test_on_turn_end_injects_steer_only_when_turn_had_tools(tmp_path):
    from apodex.observers import Approver, TerminalObserver
    from frontier_agent.core.loop_types import TurnContext

    class _Inbox:
        def __init__(self, items):
            self._items = items
        def drain(self):
            out, self._items = self._items, []
            return out

    def ctx(tool_calls):
        return TurnContext(turn=1, max_turns=10, task_id="", role_id="", ai_text="",
                           thinking="", tool_calls=tool_calls, messages=[], usage=None, metadata={})

    # turn WITH tool calls → loop continues → steer injected as next user message
    obs = TerminalObserver(Renderer(theme="mono"), Approver(auto_approve=True),
                           str(tmp_path), steer_inbox=_Inbox(["focus on the tests"]))
    iv = asyncio.run(obs.on_turn_end(ctx([{"name": "bash"}])))
    assert iv is not None and iv.inject_messages == ["focus on the tests"]

    # turn with NO tool calls (model is finishing) → NOT injected (would dangle);
    # left in the queue for the session to run as a follow-up
    obs2 = TerminalObserver(Renderer(theme="mono"), Approver(auto_approve=True),
                            str(tmp_path), steer_inbox=_Inbox(["do it later"]))
    assert asyncio.run(obs2.on_turn_end(ctx([]))) is None


def test_native_workflow_run_is_steerable(tmp_path, monkeypatch):
    """A native workflow (react / agent_team) must accept mid-run steering too.

    The workflow's main agent appends ``sdk_extra_observers`` to its own loop,
    so the session's TerminalObserver only needs a live SteerInbox — this test
    pins that wiring plus the leftover→follow-up handoff.
    """
    from apodex.config import ModelConfig
    from apodex.session import TerminalSession
    from frontier_agent.core.loop_types import TurnContext

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.chdir(tmp_path)

    class _Profile:
        workflow = "agent_team"
        workflow_profile = "tui"

    seen: dict = {}

    class _Runtime:
        async def __aenter__(self):
            return self
        async def __aexit__(self, *exc):
            return False

        async def run(self, task, *, meta, pipeline_id, extra_input=None):
            observer = meta["sdk_extra_observers"][0]
            usage_observer = meta["sdk_extra_observers"][1]
            seen["observer_types"] = [
                type(item).__name__ for item in meta["sdk_extra_observers"]
            ]
            seen["inbox"] = observer.steer_inbox
            seen["pipeline_id"] = pipeline_id
            seen["task"] = task
            seen["extra_input"] = extra_input
            # The user types while the coordinator works (TUI input box / stdin).
            observer.steer_inbox.queue.append("also check the changelog")
            ctx = TurnContext(turn=1, max_turns=10, task_id="", role_id="", ai_text="",
                              thinking="", tool_calls=[{"name": "assign_task"}],
                              messages=[], usage=None, metadata={})
            seen["intervention"] = await observer.on_turn_end(ctx)
            from frontier_agent.core.messages import assistant_msg, system_msg, user_msg
            await usage_observer.on_llm_response(TurnContext(
                turn=1, max_turns=10, task_id="", role_id="", ai_text="done",
                thinking="", tool_calls=[],
                messages=[
                    system_msg("system"), user_msg("question"), assistant_msg("done"),
                ],
                usage={"prompt_tokens": 800, "completion_tokens": 20}, metadata={},
            ))
            # A second line arrives too late to be injected → must not be lost.
            observer.steer_inbox.queue.append("and summarize it")
            return {"final_answer": "done"}

    monkeypatch.setattr("benchmarks.public.core.kernel_adapter.BenchmarkSession", _Runtime)

    session = TerminalSession(
        cfg=ModelConfig(model="fake", api_key="x", base_url=None), cwd=str(tmp_path),
        renderer=Renderer(theme="mono"), auto_approve=True, max_turns=5,
        interactive=False, mode="coding",
    )
    session.tui_mode = True  # don't touch stdin from the test
    follow_ups: list[str] = []

    async def _capture(task):
        follow_ups.append(task)

    asyncio.run(_drive_workflow(session, _Profile(), _capture))

    # the observer had a live inbox, and the typed line was injected mid-run
    assert seen["inbox"] is not None
    assert seen["intervention"] is not None
    assert seen["intervention"].inject_messages == ["also check the changelog"]
    assert seen["observer_types"] == [
        "TerminalObserver", "UsageObserver", "TraceObserver",
    ]
    assert session.usage.last_input == 800
    assert session.usage.breakdown is not None
    # the late line ran as a follow-up task instead of being dropped
    assert follow_ups == ["and summarize it"]
    # and the run released the inbox afterwards
    assert session._inbox is None


def _run_workflow_returning(state: dict, tmp_path, monkeypatch) -> None:
    """Drive ``_run_native_workflow`` against a workflow stubbed to return *state*."""
    from apodex.config import ModelConfig
    from apodex.render import Renderer
    from apodex.session import TerminalSession

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.chdir(tmp_path)

    class _Profile:
        workflow = "agent_team"
        workflow_profile = "tui"

    class _Runtime:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def run(self, task, *, meta, pipeline_id, extra_input=None):
            return state

    monkeypatch.setattr("benchmarks.public.core.kernel_adapter.BenchmarkSession", _Runtime)
    session = TerminalSession(
        cfg=ModelConfig(model="fake", api_key="x", base_url=None), cwd=str(tmp_path),
        renderer=Renderer(theme="mono"), auto_approve=True, max_turns=5,
        interactive=False, mode="coding",
    )
    asyncio.run(session._run_native_workflow("investigate", _Profile()))


def test_native_workflow_llm_error_is_not_reported_as_delivery(
    tmp_path, monkeypatch, capsys,
):
    # ``not_found`` is what the node reports for its deterministic placeholder.
    _run_workflow_returning({
        "final_answer": "## Best available result\nFallback prose",
        "answer_status": "not_found",
        "stopped_by": "llm_error",
        "llm_error": "Error code: 401 - invalid_api_key",
        "llm_error_reason": "non_transient",
    }, tmp_path, monkeypatch)

    out = capsys.readouterr().out
    assert "LLM configuration error" in out
    assert "Error code: 401" in out
    assert "Fallback prose" not in out
    assert "Final report" not in out
    assert "Result (" not in out


def test_native_workflow_keeps_a_salvaged_partial_answer(
    tmp_path, monkeypatch, capsys,
):
    """``best_effort`` prose came out of a real salvage call — do not discard it.

    ``stopped_by`` alone cannot distinguish it from the placeholder above, so
    dropping every llm_error answer would silently throw away real work.
    """
    _run_workflow_returning({
        "final_answer": "Partial finding: the 5090 profile needs moe_wna16.",
        "answer_status": "best_effort",
        "stopped_by": "llm_error",
        "llm_error": "Error code: 429 rate limit",
        "llm_error_reason": "transient",
    }, tmp_path, monkeypatch)

    out = capsys.readouterr().out
    assert "LLM call failed" in out
    assert "the 5090 profile needs moe_wna16" in out
    assert "Partial output produced before the failure" in out
    assert "Final report" not in out
    assert "Result (" not in out


def test_native_workflow_wall_deadline_is_not_reported_as_delivery(
    tmp_path, monkeypatch, capsys,
):
    _run_workflow_returning({
        "final_answer": "Fixing these before sign-off.",
        "answer_status": "best_effort",
        "final_answer_source": "existing_partial",
        "stopped_by": "wall_deadline",
        "react_steps": [{}] * 150,
    }, tmp_path, monkeypatch)

    out = capsys.readouterr().out
    assert "Incomplete output" in out
    assert "wall_deadline" in out
    assert "Fixing these before sign-off" in out
    assert "partial output was not saved as a final report" in out
    assert "Result (" not in out


def test_native_workflow_no_tool_stop_is_not_reported_as_delivery(
    tmp_path, monkeypatch, capsys,
):
    """Workflows nudge on a tool-less turn, so ``no_tool`` means truncated.

    Only the generic coding loop runs ``no_tool_behavior="stop"``, where a
    plain-text turn is the normal finish. Sharing one set across both paths
    presented a workflow that gave up mid-task as a delivered report.
    """
    _run_workflow_returning({
        "final_answer": "I was unable to finish the comparison.",
        "stopped_by": "no_tool",
        "react_steps": [{}] * 12,
    }, tmp_path, monkeypatch)

    out = capsys.readouterr().out
    assert "Incomplete output" in out
    assert "no_tool" in out
    assert "partial output was not saved as a final report" in out
    assert "Final report" not in out


def test_generic_loop_no_tool_stop_is_a_normal_finish():
    """The coding loop answers by producing a turn with no tool calls."""
    from apodex.task_runner import _is_complete_run

    assert _is_complete_run("no_tool", no_tool_is_complete=True) is True
    assert _is_complete_run("no_tool") is False
    assert _is_complete_run("max_turns", no_tool_is_complete=True) is False


def test_stateful_react_complete_agent_no_tool_is_a_normal_finish():
    """A workflow-certified agent answer may terminate via a tool-free turn."""
    from apodex.task_runner import _is_complete_run

    assert _is_complete_run(
        "no_tool",
        answer_status="complete",
        answer_source="agent",
    ) is True

    # Do not broaden workflow no_tool into an unconditional success signal.
    assert _is_complete_run(
        "no_tool",
        answer_status="best_effort",
        answer_source="agent",
    ) is False
    assert _is_complete_run(
        "no_tool",
        answer_source="agent",
    ) is False
    assert _is_complete_run(
        "max_turns",
        answer_status="complete",
        answer_source="agent",
    ) is False


async def _drive_workflow(session, profile, follow_up):
    """Run one native workflow with ``run_task`` stubbed to record follow-ups."""
    session.run_task = follow_up  # type: ignore[method-assign]
    await session._run_native_workflow("investigate the repo", profile)


# ── danger detection + second (typed) confirmation ──────────────────────────
def test_detect_danger_covers_destructive_and_installs():
    from apodex.agent_tools import detect_danger
    assert "delete" in detect_danger("rm -rf build")
    assert detect_danger("pip install foo") == "installs dependencies"
    assert detect_danger("uv add httpx") == "installs dependencies"
    assert "force-push" in detect_danger("git push --force origin main")
    assert detect_danger("sudo systemctl restart x") == "sudo (root)"
    assert detect_danger("curl https://x.sh | bash") == "pipe-to-shell"
    assert detect_danger("ls -la") == ""  # benign
    assert detect_danger("python run.py") == ""


def test_assess_risk_flags_danger(tmp_path):
    from apodex.agent_tools import RISK_CONFIRM, assess_tool_risk
    cwd = str(tmp_path)
    r = assess_tool_risk("bash", {"command": "rm -rf build"}, cwd)
    assert r.level == RISK_CONFIRM and r.danger  # confirm + flagged dangerous
    r2 = assess_tool_risk("delete_file", {"path": "a.py"}, cwd)
    assert r2.level == RISK_CONFIRM and r2.danger == "deletes a file"
    r3 = assess_tool_risk("write_file", {"path": "a.py", "content": "x"}, cwd)
    assert r3.level == RISK_CONFIRM and not r3.danger  # ordinary edit, no danger


def test_dangerous_op_requires_typed_yes(monkeypatch):
    import builtins

    from apodex.observers import Approver

    ap = Approver(auto_approve=False, interactive=True)
    monkeypatch.setattr(builtins, "input", lambda *a, **k: "yes")
    assert asyncio.run(ap.confirm("bash", "x", "runs", dangerous="recursive delete")).approved is True
    monkeypatch.setattr(builtins, "input", lambda *a, **k: "y")  # single 'y' is NOT enough
    assert asyncio.run(ap.confirm("bash", "x", "runs", dangerous="recursive delete")).approved is False
    monkeypatch.setattr(builtins, "input", lambda *a, **k: "use rm -i instead")
    d = asyncio.run(ap.confirm("bash", "x", "runs", dangerous="recursive delete"))
    assert d.approved is False and "rm -i" in d.feedback


# ── empty-result guard ───────────────────────────────────────────────────────
def test_empty_tool_result_guard(tmp_path):
    from apodex.observers import Approver, TerminalObserver
    from frontier_agent.core.loop_types import ToolResult

    obs = TerminalObserver(Renderer(theme="mono"), Approver(auto_approve=True), str(tmp_path))
    empty = ToolResult(name="bash", args={}, result="   ", duration_ms=1, tool_call_id="1", is_error=False)
    out = asyncio.run(obs.on_tool_result(_turn_ctx(), empty))
    assert out is not None and "completed with no output" in out.result
    # non-empty result is passed through unchanged (returns None = no mutation)
    full = ToolResult(name="bash", args={}, result="data", duration_ms=1, tool_call_id="2", is_error=False)
    assert asyncio.run(obs.on_tool_result(_turn_ctx(), full)) is None


def test_coding_prompt_has_verify_and_act_directives(tmp_path):
    from apodex.prompts import build_system_prompt
    p = build_system_prompt(str(tmp_path))
    assert "VERIFY" in p and "concise" in p.lower()
    # Claude-aligned "act, don't just narrate" guidance (no terminal tool):
    assert "SAME turn" in p and "no tool call" in p.lower()
    # Reinforce the schema-level bash `description` field so the model fills it
    # (shown at the approval prompt).
    assert "`description`" in p and "approval prompt" in p
    # Explicit language directive (respond in the user's language) — both modes.
    assert "SAME language" in p
    from apodex.prompts import build_research_prompt
    assert "SAME language" in build_research_prompt(str(tmp_path))


def test_engine_log_router_surfaces_only_recovery_notes(tmp_path):
    """Fix #1: engine logs go to a file; only recovery warnings become notes."""
    import logging

    from apodex.cli import _EngineLogRouter

    notes: list[str] = []

    class _R:
        def note(self, m):
            notes.append(m)

    fh = logging.FileHandler(str(tmp_path / "engine.log"), encoding="utf-8")
    router = _EngineLogRouter(_R(), fh)

    def rec(msg):
        return logging.LogRecord("frontier_agent", logging.WARNING, __file__, 1, msg, None, None)

    router.emit(rec("[LeakedToolCallRetry] turn=14 | leaked content, scheduling retry"))
    assert notes and "retry" in notes[-1].lower()  # surfaced as a clean note
    notes.clear()
    router.emit(rec("some unrelated library warning"))
    assert notes == []  # non-recovery warnings: file only, no UI noise
    fh.close()
    # both records were written to the log file
    assert "LeakedToolCallRetry" in (tmp_path / "engine.log").read_text()


def test_engine_log_router_follows_active_run(tmp_path, monkeypatch):
    import logging

    from apodex.cli import _EngineLogRouter

    monkeypatch.setenv("APODEX_RUNS_ROOT", str(tmp_path / "runs"))
    monkeypatch.setenv("APODEX_SESSION_ID", "run-one")
    router = _EngineLogRouter(object())
    record = logging.LogRecord(
        "frontier_agent", logging.WARNING, __file__, 1, "first", None, None,
    )
    router.emit(record)
    monkeypatch.setenv("APODEX_SESSION_ID", "run-two")
    record.msg = "second"
    router.emit(record)

    assert "first" in (tmp_path / "runs" / "run-one" / "engine.log").read_text()
    assert "second" in (tmp_path / "runs" / "run-two" / "engine.log").read_text()


# ── plan mode ───────────────────────────────────────────────────────────────
def test_is_mutating_tool():
    from apodex.agent_tools import is_mutating_tool
    assert is_mutating_tool("write_file", {"path": "a.py"}) is True
    assert is_mutating_tool("delete_file", {"path": "a.py"}) is True
    assert is_mutating_tool("bash", {"command": "rm -rf build"}) is True
    assert is_mutating_tool("bash", {"command": "ls -la"}) is False  # read-only bash
    assert is_mutating_tool("read_file", {"path": "a.py"}) is False
    assert is_mutating_tool("grep_search", {"pattern": "x"}) is False
    # The shipped workflows write every deliverable through create_file, so
    # plan mode has to lock it too — otherwise "edits are locked" is untrue.
    assert is_mutating_tool("create_file", {"path": "/outputs/report.md"}) is True
    assert is_mutating_tool("download_file", {"path": "/workspace/p.pdf"}) is True


def test_sandbox_write_tools_are_confirmed_with_a_visible_target():
    """They used to fall through to the unknown-tool branch: a confirm prompt
    naming no path, so the user approved a write they could not see."""
    from apodex.agent_tools import RISK_CONFIRM, assess_tool_risk

    risk = assess_tool_risk("create_file", {"path": "/outputs/report.docx"}, "/tmp/p")

    assert risk.level == RISK_CONFIRM
    assert risk.target == "/outputs/report.docx"


def test_sandbox_write_tools_are_not_denied_for_living_outside_cwd():
    """/outputs is outside the working directory by design, so the cwd deny
    that guards host writes must not apply to them."""
    from apodex.agent_tools import RISK_DENY, assess_tool_risk

    outside = assess_tool_risk("write_file", {"path": "/outputs/x.md"}, "/tmp/p")
    sandboxed = assess_tool_risk("create_file", {"path": "/outputs/x.md"}, "/tmp/p")

    assert outside.level == RISK_DENY
    assert sandboxed.level != RISK_DENY


def test_plan_mode_blocks_edits_but_not_reads(tmp_path):
    from apodex.observers import Approver, TerminalObserver
    from apodex.plan import PlanState
    from frontier_agent.core.loop_types import ToolResult

    class _Rec(Renderer):
        def __init__(self):
            super().__init__(theme="mono")
            self.completed = []
            self.results = []

        def activity_result(
            self, name, *, call_id="", is_error, ms=0, outcome="",
        ):
            self.completed.append((call_id, outcome))

        def tool_result(self, name, result, *, is_error, ms=0, call_id=""):
            self.results.append(name)

    plan = PlanState(active=True)
    renderer = _Rec()
    obs = TerminalObserver(renderer, Approver(auto_approve=True),
                           str(tmp_path), plan_state=plan)
    # a write is blocked (skipped, not executed) — and the task is NOT aborted
    iv = asyncio.run(obs.on_tool_call(_turn_ctx(), {
        "id": "plan-block", "name": "write_file",
        "args": {"path": "a.py", "content": "x"},
    }))
    assert iv is not None and "Plan mode is active" in (iv.skip_with_result or "")
    assert obs._abort_reason is None  # blocking a write must not stop the task
    asyncio.run(obs.on_tool_result(_turn_ctx(), ToolResult(
        name="write_file", args={"path": "a.py"}, result=iv.skip_with_result,
        duration_ms=0, tool_call_id="plan-block", is_error=False,
    )))
    assert renderer.completed == [("plan-block", "skipped")]
    assert renderer.results == []
    # a read passes through (returns None or a rewrite, never a skip)
    (tmp_path / "r.py").write_text("hi\n")
    iv2 = asyncio.run(obs.on_tool_call(_turn_ctx(), {"name": "read_file", "args": {"path": "r.py"}}))
    assert iv2 is None or getattr(iv2, "skip_with_result", None) is None


def test_synthetic_tracking_is_by_call_occurrence_when_ids_repeat(tmp_path):
    from apodex.observers import Approver, TerminalObserver
    from apodex.plan import PlanState
    from frontier_agent.core.loop_types import ToolResult

    class _Rec(Renderer):
        def __init__(self):
            super().__init__(theme="mono")
            self.completed = []
            self.results = []

        def activity_result(
            self, name, *, call_id="", is_error, ms=0, outcome="",
        ):
            self.completed.append((name, call_id, outcome))

        def tool_result(self, name, result, *, is_error, ms=0, call_id=""):
            self.results.append((name, call_id))

    (tmp_path / "read.txt").write_text("data")
    renderer = _Rec()
    obs = TerminalObserver(
        renderer, Approver(auto_approve=True), str(tmp_path),
        plan_state=PlanState(active=True),
    )
    shared_id = "reused"
    asyncio.run(obs.on_tool_call(_turn_ctx(), {
        "id": shared_id, "name": "read_file", "args": {"path": "read.txt"},
    }))
    blocked = asyncio.run(obs.on_tool_call(_turn_ctx(), {
        "id": shared_id, "name": "write_file",
        "args": {"path": "out.txt", "content": "x"},
    }))
    asyncio.run(obs.on_tool_result(_turn_ctx(), ToolResult(
        name="read_file", args={"path": "read.txt"}, result="data",
        duration_ms=1, tool_call_id=shared_id, is_error=False,
    )))
    asyncio.run(obs.on_tool_result(_turn_ctx(), ToolResult(
        name="write_file", args={"path": "out.txt"},
        result=blocked.skip_with_result, duration_ms=0,
        tool_call_id=shared_id, is_error=False,
    )))

    assert renderer.results == [("read_file", shared_id)]
    assert renderer.completed == [("write_file", shared_id, "skipped")]


def test_exit_plan_mode_approve_unlocks_edits(tmp_path):
    from apodex.observers import Approver, Decision, TerminalObserver
    from apodex.plan import PlanState

    plan = PlanState(active=True)
    obs = TerminalObserver(Renderer(theme="mono"), Approver(auto_approve=True),
                           str(tmp_path), plan_state=plan)
    iv = asyncio.run(obs.on_tool_call(_turn_ctx(), {"name": "exit_plan_mode", "args": {"plan": "1. edit foo"}}))
    assert iv is not None and "APPROVED" in (iv.skip_with_result or "")
    assert plan.active is False  # edits unlocked

    # rejection (with feedback) keeps plan mode on and feeds the revision back
    plan2 = PlanState(active=True)

    class _Revise:
        auto_approve = False
        async def confirm(self, name, target, reason, **kw):
            return Decision(False, feedback="split into two steps")

    obs2 = TerminalObserver(Renderer(theme="mono"), _Revise(), str(tmp_path), plan_state=plan2)
    iv2 = asyncio.run(obs2.on_tool_call(_turn_ctx(), {"name": "exit_plan_mode", "args": {"plan": "do it"}}))
    assert plan2.active is True  # still planning
    assert "split into two steps" in (iv2.skip_with_result or "")
    assert "revise" in (iv2.skip_with_result or "").lower()


# ── collapsed thinking + verbose toggle ────────────────────────────────────
def test_thinking_default_is_verbose():
    # Thinking is shown raw by default; ``/verbose`` toggles the collapse.
    assert Renderer(theme="mono")._verbose is True


def test_thinking_verbose_and_collapsed_no_crash(capsys):
    r = Renderer(theme="mono")
    # Verbose (default): streams raw thinking inline.
    r.thinking_delta("raw thought ")
    r.content_delta("answer")
    # Collapsed (opt-in): no running loop / mono → ticker is inert, must not raise.
    r.set_verbose(False)
    r.thinking_delta("step 1 ... ")
    r.thinking_delta("step 2 ... ")
    r.content_delta("final answer")
    capsys.readouterr()


# ── Tier 0: read-before-edit guard (fsguard) ─────────────────────────────────
def test_fsguard_read_before_edit_and_stale(tmp_path):
    import os
    import time

    from apodex import fsguard
    fsguard.clear()
    cwd = str(tmp_path)
    (tmp_path / "a.py").write_text("x\n")
    assert "has not been read" in (fsguard.check_can_edit("a.py", cwd) or "")  # existing+unread
    assert fsguard.check_can_edit("new.py", cwd) is None                       # new file is fine
    fsguard.record_read("a.py", cwd)
    assert fsguard.check_can_edit("a.py", cwd) is None                          # read → editable
    os.utime(str(tmp_path / "a.py"), (time.time() + 10, time.time() + 10))      # changed on disk
    assert "changed on disk" in (fsguard.check_can_edit("a.py", cwd) or "")
    fsguard.clear()


# ── Tier 2: persistent permission rules ──────────────────────────────────────
def test_permission_store_prefix_match_and_persist(tmp_path):
    from apodex.permissions import PermissionStore, rule_for
    p = str(tmp_path / "perm.json")
    s = PermissionStore(path=p)
    s.add_allow("bash", {"command": "npm test -- -k foo"})
    assert "Bash(npm test)" in s.allow
    assert s.allows("bash", {"command": "npm test --watch"})
    assert s.allows("bash", {"command": "cd /workspace && npm test --watch"})  # helper cd ignored
    assert not s.allows("bash", {"command": "npm publish"})            # different 2nd word
    assert not s.allows("bash", {"command": "npm test && rm -rf x"})   # compound: rm -rf not allowed
    s.deny.add("write_file")
    assert s.denies("write_file", {"path": "a"})
    assert "Bash(npm test)" in PermissionStore.load(p).allow           # round-trips
    assert rule_for("bash", {"command": "cd /workspace && git push origin main"}) == "Bash(git push)"


def test_assess_with_rules_layering(tmp_path):
    from apodex.agent_tools import RISK_DENY, RISK_SAFE, assess_with_rules
    from apodex.permissions import PermissionStore
    cwd = str(tmp_path)
    # allow downgrades confirm → safe
    assert assess_with_rules("bash", {"command": "make build"}, cwd,
                             PermissionStore(allow={"Bash(make build)"})).level == RISK_SAFE
    # deny forces a block
    assert assess_with_rules("bash", {"command": "make build"}, cwd,
                             PermissionStore(deny={"Bash(make build)"})).level == RISK_DENY
    # user saved allow rule works for confirm calls
    r = assess_with_rules("bash", {"command": "uv run pytest"}, cwd,
                          PermissionStore(allow={"Bash(uv run)"}))
    assert r.level == RISK_SAFE
    # auto_for_me mode treats confirm calls as safe
    r2 = assess_with_rules("bash", {"command": "uv run pytest"}, cwd, auto_for_me=True)
    assert r2.level == RISK_SAFE


def test_user_settings_save_and_load(tmp_path):
    from apodex.config import UserSettings
    p = str(tmp_path / "settings.json")
    u = UserSettings(
        theme="catppuccin",
        workflow="agent_team",
        auto_approve=True,
        auto_for_me=True,
        verbose=False,
        plan_mode=True,
        path=p,
    )
    u.save()

    loaded = UserSettings.load(p)
    assert loaded.theme == "catppuccin"
    assert loaded.workflow == "agent_team"
    assert loaded.auto_approve is True
    assert loaded.auto_for_me is True
    assert loaded.verbose is False
    assert loaded.plan_mode is True


# ── Tier 1: usage tracking ────────────────────────────────────────────────────
def test_usage_observer_accumulates_and_context_pct():
    from apodex.usage import Usage, UsageObserver
    from frontier_agent.core.loop_types import TurnContext
    u = Usage()
    obs = UsageObserver(u)

    def ctx(pt, ct):
        return TurnContext(turn=1, max_turns=10, task_id="", role_id="", ai_text="", thinking="",
                           tool_calls=[], messages=[],
                           usage={"prompt_tokens": pt, "completion_tokens": ct}, metadata={})

    asyncio.run(obs.on_llm_response(ctx(100, 20)))
    asyncio.run(obs.on_llm_response(ctx(150, 30)))
    assert u.input == 250 and u.output == 50 and u.total == 300 and u.last_input == 150
    assert u.context_pct_left(1000) == 85


def test_usage_observer_estimates_context_when_provider_omits_usage():
    from apodex.usage import Usage, UsageObserver
    from frontier_agent.core.loop_types import TurnContext
    from frontier_agent.core.messages import assistant_msg, system_msg, user_msg

    usage = Usage()
    observer = UsageObserver(usage)
    assert usage.context_status(262_144) == "--/256k"

    asyncio.run(observer.on_llm_response(TurnContext(
        turn=1, max_turns=10, task_id="", role_id="", ai_text="answer",
        thinking="", tool_calls=[],
        messages=[
            system_msg("system instructions"),
            user_msg("question with enough content to count"),
            assistant_msg("answer"),
        ],
        usage=None, metadata={},
    )))

    assert usage.estimated is True
    assert usage.last_input > 0
    assert usage.breakdown is not None
    assert usage.context_status(262_144).endswith("/256k 0%")


def test_usage_observer_builds_provider_calibrated_context_breakdown():
    from apodex.usage import Usage, UsageObserver
    from frontier_agent.core.loop_types import TurnContext
    from frontier_agent.core.messages import assistant_msg, system_msg, tool_msg, user_msg

    class _Tool:
        def to_openai_schema(self):
            return {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read a file",
                    "parameters": {"type": "object"},
                },
            }

    messages = [
        system_msg("system instructions"),
        user_msg("[Compacted summary of earlier turns]\nold decisions"),
        user_msg("inspect the implementation"),
        assistant_msg("", tool_calls=[{
            "id": "call-1", "type": "function",
            "function": {"name": "read_file", "arguments": '{"path":"x.py"}'},
        }]),
        tool_msg("file contents", "call-1"),
    ]
    usage = Usage()
    observer = UsageObserver(usage, tools=[_Tool()])

    def context(current_messages, provider_usage=None):
        return TurnContext(
            turn=1, max_turns=10, task_id="", role_id="", ai_text="",
            thinking="", tool_calls=[], messages=current_messages,
            usage=provider_usage, metadata={},
        )

    asyncio.run(observer.on_llm_response(context(
        [*messages, assistant_msg("done")],
        {"prompt_tokens": 1_000, "completion_tokens": 25},
    )))

    assert usage.last_input == 1_000
    assert usage.context_status(262_144) == "1.0k/256k 0%"
    assert usage.breakdown is not None and usage.breakdown.total == 1_000
    categories = dict(usage.breakdown.display_categories())
    assert sum(categories.values()) == 1_000
    assert categories["System & definitions"] > 0
    assert categories["Conversation / history"] > 0
    assert categories["Tool calls & results"] > 0
    assert categories["Summarized history"] > 0

    restored = Usage()
    restored.restore(usage.to_dict())
    assert restored.last_input == usage.last_input
    assert restored.breakdown == usage.breakdown
    restored.clear_context()
    assert restored.last_input == 0 and restored.breakdown is None
    assert restored.total == usage.total


# ── Tier 0: read_file line numbers + range ───────────────────────────────────
def test_read_file_line_numbers_and_range(tmp_path, monkeypatch):
    from apodex.local_tools import read_file
    monkeypatch.chdir(tmp_path)
    (tmp_path / "f.py").write_text("alpha\nbeta\ngamma\n")
    out = asyncio.run(read_file.ainvoke({"path": "f.py"}))
    assert "\t" in out and "alpha" in out                       # cat -n style
    out2 = asyncio.run(read_file.ainvoke({"path": "f.py", "start_line": 2, "end_line": 2}))
    assert "beta" in out2 and "alpha" not in out2 and "[lines 2-2 of 3]" in out2


def test_read_file_image_vision(tmp_path, monkeypatch):
    from apodex.local_tools import read_file
    monkeypatch.chdir(tmp_path)
    (tmp_path / "sample.png").write_bytes(b"\x89PNG\r\n\x1a\n\x00\x00")

    # Mock _vision_read returning text
    monkeypatch.setattr("plugins.tools._reader_core._vision_read", lambda data, mime: "Transcribed image content")
    out = asyncio.run(read_file.ainvoke({"path": "sample.png"}))
    assert "read via vision" in out
    assert "Transcribed image content" in out

    # Mock _vision_read returning None (unconfigured / failed)
    monkeypatch.setattr("plugins.tools._reader_core._vision_read", lambda data, mime: None)
    out_fail = asyncio.run(read_file.ainvoke({"path": "sample.png"}))
    assert "vision unavailable" in out_fail


def test_grep_context_lines(tmp_path, monkeypatch):
    from apodex.local_tools import grep_search
    monkeypatch.chdir(tmp_path)
    (tmp_path / "f.txt").write_text("one\nTARGET\nthree\n")
    out = asyncio.run(grep_search.ainvoke({"pattern": "TARGET", "context_lines": 1}))
    assert "one" in out and "TARGET" in out and "three" in out   # context window shown


# ── Tier 2: todo single-in_progress ──────────────────────────────────────────
def test_todo_single_in_progress():
    from apodex.todo import _parse
    items = _parse([{"content": "a", "status": "in_progress"},
                    {"content": "b", "status": "in_progress"}])
    assert [i.status for i in items] == ["in_progress", "pending"]


# ── Tier 1: environment block ─────────────────────────────────────────────────
def test_environment_section(tmp_path):
    from apodex.env import environment_section
    s = environment_section(str(tmp_path), "my-model")
    for needle in ("Working directory", "Is a git repository", "Today's date", "my-model"):
        assert needle in s


# ── Tier 2: prompt has the new safety/behaviour sections ─────────────────────
def test_prompt_tier2_sections(tmp_path):
    from apodex.prompts import build_research_prompt, build_system_prompt
    c = build_system_prompt(str(tmp_path))
    assert "Executing actions with care" in c and "reversibility" in c
    assert "do NOT reissue" in c                       # don't re-attempt a denied call
    assert "comment" in c.lower()                      # code-style discipline
    assert "untrusted" in build_research_prompt(str(tmp_path)).lower()


# ── Tier 2: [A] always-allow persists a rule ─────────────────────────────────
def test_approver_capital_a_remembers(monkeypatch):
    import builtins

    from apodex.observers import Approver
    ap = Approver(auto_approve=False, interactive=True)
    monkeypatch.setattr("apodex.observers._read_single_key", lambda: None)  # line path
    monkeypatch.setattr(builtins, "input", lambda *a, **k: "A")
    d = asyncio.run(ap.confirm("bash", "x", "runs a shell command"))
    assert d.approved and d.remember


def test_approval_prompt_names_the_destination():
    """The gate computed a target but neither prompt rendered it, so the
    question named the tool and not the file."""
    from apodex.agent_tools import assess_tool_risk
    from apodex.observers import _target_suffix

    risk = assess_tool_risk("create_file", {"path": "/outputs/report.docx"}, "/tmp/p")

    assert _target_suffix("create_file", risk.target) == " /outputs/report.docx"
    # bash is the exception: its target is the command, already shown verbatim
    # on the call line and in the preview.
    assert _target_suffix("bash", "pytest -q") == ""
    assert _target_suffix("create_file", "") == ""


def test_download_file_target_is_the_resolved_destination(monkeypatch, tmp_path):
    """``path`` is optional, its directory components are ignored, and a
    collision renames the file — so echoing the raw argument would name a file
    the tool never writes."""
    from apodex.agent_tools import assess_tool_risk

    monkeypatch.setenv("FRONTIER_AGENT_WORKSPACE_DIR", str(tmp_path))

    directory = assess_tool_risk(
        "download_file", {"url": "https://example.com/f.pdf"}, "/tmp/p",
    ).target
    named = assess_tool_risk(
        "download_file",
        {"url": "https://example.com/f.pdf", "path": "/elsewhere/p.pdf"},
        "/tmp/p",
    ).target

    assert directory.startswith(str(tmp_path / "downloads"))
    assert "from the URL" in directory          # no filename is known yet
    assert named.startswith(str(tmp_path / "downloads" / "p.pdf"))
    assert "/elsewhere/" not in named           # the requested directory is ignored
    assert "renamed" in named                   # collisions rename it
