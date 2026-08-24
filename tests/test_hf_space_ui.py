"""The Gradio layer, driven exactly as a browser event would drive it.

Only the handler functions are called — no server is started — so this stays a
fast unit test while still covering the wiring between the adapter's events and
the five panes the visitor sees.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("gradio", reason="the hf-space extra is not installed")

from deploy.huggingface.config import load_config
from deploy.huggingface.mock_llm import (
    MockLLMServer,
    error_turn,
    text_turn,
    tool_call_turn,
)

_ANSWER = "I saved notes.md describing the ReAct loop."


@pytest.fixture
def ui_env(tmp_path, monkeypatch):
    """See ``test_hf_space_runtime.demo_env`` for why the config is reloaded."""
    from frontier_agent.infra.config import get_config

    monkeypatch.setenv("SANDBOX_BACKEND", "native")
    monkeypatch.setenv("DEMO_RUNTIME_ROOT", str(tmp_path / "runtime"))
    monkeypatch.setenv("OPENAI_API_KEY", "sk-ui-test-key-000111222333")
    monkeypatch.setenv("DEMO_TASK_TIMEOUT_SECONDS", "120")
    monkeypatch.setenv("DEMO_MAX_TURNS", "6")
    monkeypatch.setenv("DEMO_REPORTER", "false")
    get_config(force_reload=True)
    yield monkeypatch
    monkeypatch.undo()
    get_config(force_reload=True)


def _handlers(demo) -> dict[str, object]:
    """The Blocks graph's Python callbacks, keyed by function name."""
    found: dict[str, object] = {}
    for dep in demo.fns.values():
        name = getattr(dep.fn, "__name__", "")
        if name:
            found[name] = dep.fn
    return found


def _build(monkeypatch, server: MockLLMServer):
    from deploy.huggingface.app import build_demo

    monkeypatch.setenv("OPENAI_BASE_URL", server.base_url)
    monkeypatch.setenv("OPENAI_MODEL", server.model)
    config = load_config()
    demo = build_demo(config)
    return config, demo, _handlers(demo)


def test_the_page_advertises_the_workflow_model_and_limits(ui_env) -> None:
    from deploy.huggingface.app import _header

    ui_env.setenv("HF_MODEL_ID", "apodex/Apodex-1.1-mini")
    header = _header(load_config())
    assert "FrontierAgent Demo" in header
    assert "`react`" in header
    assert "apodex/Apodex-1.1-mini" in header
    assert "huggingface.co/apodex/Apodex-1.1-mini" in header
    assert "`bash`" not in header, "a public demo must not advertise a shell"


def test_the_header_never_shows_the_api_key(ui_env) -> None:
    from deploy.huggingface.app import _header

    ui_env.setenv("OPENAI_API_KEY", "sk-should-never-render-4242")
    assert "sk-should-never-render-4242" not in _header(load_config())


def test_every_expected_control_is_wired(ui_env) -> None:
    with MockLLMServer(script=[text_turn("ok")]) as server:
        _config, _demo, handlers = _build(ui_env, server)
    for name in ("_run", "_stop", "_clear", "_new_session"):
        assert name in handlers, f"{name} is not wired to any control"


async def test_a_run_paints_status_activity_answer_and_files(ui_env) -> None:
    script = [
        tool_call_turn("write_file", {
            "path": "outputs/notes.md", "content": "# Notes\n\nReAct.\n",
        }),
        text_turn(_ANSWER),
    ]
    with MockLLMServer(script=script) as server:
        _config, _demo, handlers = _build(ui_env, server)
        frames = [
            frame async for frame in handlers["_run"]("Explain ReAct.", "")
        ]

    assert len(frames) > 2, "the UI must update progressively, not once"
    status, task_board, activity, answer, files, run_id, session_id = frames[-1]

    assert "Completed" in status
    assert "write_file" in activity
    assert "notes.md" in activity
    assert _ANSWER in answer
    assert files and files[0].endswith("notes.md")
    assert run_id and session_id
    assert "Tasks will appear here" in task_board

    # The first frame must already tell the visitor something is happening.
    assert "Queued" in frames[0][0] or "Running" in frames[0][0]


async def test_a_failing_endpoint_shows_guidance_not_a_traceback(ui_env) -> None:
    with MockLLMServer(script=[error_turn(401)]) as server:
        _config, _demo, handlers = _build(ui_env, server)
        frames = [frame async for frame in handlers["_run"]("Anything.", "")]

    status = frames[-1][0]
    assert "Failed" in status
    assert "OPENAI_API_KEY" in status
    assert "Traceback" not in status
    assert "sk-ui-test-key" not in status


async def test_an_empty_prompt_is_reported_in_the_status_pane(ui_env) -> None:
    with MockLLMServer(script=[text_turn("never reached")]) as server:
        _config, _demo, handlers = _build(ui_env, server)
        frames = [frame async for frame in handlers["_run"]("   ", "")]
        assert server.calls == 0

    assert "Failed" in frames[-1][0]
    assert "Enter a task" in frames[-1][0]


async def test_clear_wipes_the_panes_and_this_session_only(ui_env) -> None:
    script = [
        tool_call_turn("write_file", {"path": "outputs/a.md", "content": "a"}),
        text_turn("done"),
    ]
    with MockLLMServer(script=script) as server:
        _config, _demo, handlers = _build(ui_env, server)
        frames = [frame async for frame in handlers["_run"]("Write a.md", "")]
        session_id = frames[-1][6]
        assert frames[-1][4], "expected a file before clearing"

        status, task_board, activity, answer, files, run_id = handlers["_clear"](
            session_id,
        )

    assert "Idle" in status
    assert answer == ""
    assert files == []
    assert run_id == ""
    assert "No activity" in activity
    assert "Tasks will appear here" in task_board


async def test_task_tools_update_a_dedicated_board_not_activity(ui_env) -> None:
    script = [
        tool_call_turn("add_task", {
            "tasks": [{"description": "Research the ReAct loop"}],
        }),
        tool_call_turn("update_task", {
            "updates": [{"id": "t1", "resolution": "in_progress"}],
        }),
        tool_call_turn("update_task", {
            "updates": [{"id": "t1", "resolution": "resolved"}],
        }),
        text_turn("done"),
    ]
    with MockLLMServer(script=script) as server:
        _config, _demo, handlers = _build(ui_env, server)
        frames = [frame async for frame in handlers["_run"]("Research.", "")]

    _status, task_board, activity, _answer, _files, _run_id, _session_id = frames[-1]
    assert "Research the ReAct loop" in task_board
    assert "1/1" in task_board
    assert "task-resolved" in task_board
    assert "add_task" not in activity
    assert "update_task" not in activity


async def test_text_only_run_provides_an_answer_download(ui_env) -> None:
    with MockLLMServer(script=[text_turn("plain final answer")]) as server:
        _config, _demo, handlers = _build(ui_env, server)
        frames = [frame async for frame in handlers["_run"]("Answer.", "")]

    files = frames[-1][4]
    assert files and files[0].endswith("answer.md")


async def test_a_second_text_only_run_replaces_the_first_answer_download(
    ui_env,
) -> None:
    """The fallback is per-run, but the emptiness check that guards it is not.

    A second text-only ask in the same session found the *previous* run's
    ``answer.md`` still sitting in ``outputs/``, concluded the tree was not
    empty, and wrote nothing — leaving the visitor downloading the answer to
    their earlier question with nothing on screen to say so.
    """
    script = [text_turn("FIRST answer"), text_turn("SECOND answer")]
    with MockLLMServer(script=script) as server:
        _config, _demo, handlers = _build(ui_env, server)
        first = [frame async for frame in handlers["_run"]("q1", "")]
        session_id = first[-1][6]
        second = [frame async for frame in handlers["_run"]("q2", session_id)]

    files = second[-1][4]
    assert [Path(p).name for p in files] == ["answer.md"]
    assert Path(files[0]).read_text(encoding="utf-8").strip() == "SECOND answer"


async def test_a_generated_answer_is_retired_once_the_agent_writes_a_file(
    ui_env,
) -> None:
    """The stale fallback must not linger beside a genuine deliverable."""
    script = [
        text_turn("FIRST answer"),
        tool_call_turn("write_file", {
            "path": "outputs/notes.md", "content": "# Real deliverable\n",
        }),
        text_turn("SECOND answer"),
    ]
    with MockLLMServer(script=script) as server:
        _config, _demo, handlers = _build(ui_env, server)
        first = [frame async for frame in handlers["_run"]("q1", "")]
        session_id = first[-1][6]
        assert [Path(p).name for p in first[-1][4]] == ["answer.md"]
        second = [frame async for frame in handlers["_run"]("q2", session_id)]

    assert [Path(p).name for p in second[-1][4]] == ["notes.md"]


async def test_an_agent_authored_answer_md_is_never_overwritten(ui_env) -> None:
    """Ownership is tracked, not inferred from the filename.

    ``outputs/answer.md`` is a name the agent may pick itself, and that file is
    a deliverable — the fallback must keep its hands off it.
    """
    script = [
        tool_call_turn("write_file", {
            "path": "outputs/answer.md", "content": "# The agent's own file\n",
        }),
        text_turn("I wrote answer.md myself."),
    ]
    with MockLLMServer(script=script) as server:
        _config, _demo, handlers = _build(ui_env, server)
        frames = [frame async for frame in handlers["_run"]("Write it.", "")]

    files = frames[-1][4]
    assert [Path(p).name for p in files] == ["answer.md"]
    assert "the agent's own file" in Path(files[0]).read_text(
        encoding="utf-8",
    ).lower()


async def test_a_later_agent_authored_answer_md_replaces_the_fallback(ui_env) -> None:
    """Fallback ownership must not be inferred from the shared filename."""
    script = [
        text_turn("FIRST fallback"),
        tool_call_turn("write_file", {
            "path": "outputs/answer.md", "content": "# Agent deliverable\n",
        }),
        text_turn("SECOND final"),
    ]
    with MockLLMServer(script=script) as server:
        _config, _demo, handlers = _build(ui_env, server)
        first = [frame async for frame in handlers["_run"]("q1", "")]
        session_id = first[-1][6]
        second = [frame async for frame in handlers["_run"]("q2", session_id)]

    files = second[-1][4]
    assert [Path(p).name for p in files] == ["answer.md"]
    assert Path(files[0]).read_text(encoding="utf-8") == "# Agent deliverable\n"


def test_new_session_mints_a_different_unguessable_id(ui_env) -> None:
    with MockLLMServer(script=[text_turn("ok")]) as server:
        _config, _demo, handlers = _build(ui_env, server)
        first, first_label = handlers["_new_session"]("")
        second, _second_label = handlers["_new_session"](first)

    assert first != second
    assert len(first) >= 16
    # The label is a short prefix — never the addressable id.
    assert first not in first_label


def test_stop_on_an_idle_session_is_harmless(ui_env) -> None:
    with MockLLMServer(script=[text_turn("ok")]) as server:
        _config, _demo, handlers = _build(ui_env, server)
        assert "Idle" in handlers["_stop"]("")
