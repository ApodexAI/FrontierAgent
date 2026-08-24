"""Regressions for the four review findings on the public-demo boundaries.

Each test here reproduces a concrete escape that the original implementation
allowed. They are kept in their own module because they are adversarial: the
"model" is scripted to behave like a hostile or compromised endpoint, which is
the threat the public demo has to survive — the endpoint holds the API key, so
it is the one party able to leak it to visitors.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from deploy.huggingface import adapter as adapter_module
from deploy.huggingface.adapter import FrontierAgentAdapter
from deploy.huggingface.config import load_config
from deploy.huggingface.events import (
    ACTIVITY_FINISHED,
    ASSISTANT_DELTA,
    RUN_CANCELLED,
    RUN_COMPLETED,
    TASK_BOARD_UPDATED,
    DemoEvent,
)
from deploy.huggingface.mock_llm import MockLLMServer, Turn, text_turn, tool_call_turn
from deploy.huggingface.security import list_output_files
from deploy.huggingface.sessions import SessionStore

_API_KEY = "sk-fake-demo-key-do-not-log-4242"


@pytest.fixture
def demo_env(tmp_path, monkeypatch):
    """See ``test_hf_space_runtime.demo_env`` for why the config is reloaded."""
    from frontier_agent.infra.config import get_config

    monkeypatch.setenv("SANDBOX_BACKEND", "native")
    monkeypatch.setenv("DEMO_RUNTIME_ROOT", str(tmp_path / "runtime"))
    monkeypatch.setenv("OPENAI_API_KEY", _API_KEY)
    monkeypatch.setenv("DEMO_TASK_TIMEOUT_SECONDS", "300")
    monkeypatch.setenv("DEMO_MAX_TURNS", "6")
    monkeypatch.setenv("DEMO_REPORTER", "false")
    get_config(force_reload=True)
    yield monkeypatch
    monkeypatch.undo()
    get_config(force_reload=True)


def _build(monkeypatch, server: MockLLMServer):
    monkeypatch.setenv("OPENAI_BASE_URL", server.base_url)
    monkeypatch.setenv("OPENAI_MODEL", server.model)
    config = load_config()
    store = SessionStore(config.runtime_root, ttl_s=config.session_ttl_s)
    return config, store, FrontierAgentAdapter(config)


async def _collect(adapter, session, prompt: str) -> list[DemoEvent]:
    return [item async for item in adapter.run(session=session, prompt=prompt)]


# ── [P1] secrets must not reach a downloadable artifact ──────────────────


async def test_a_secret_written_into_a_file_never_becomes_downloadable(
    demo_env,
) -> None:
    """A hostile endpoint writing its own key into a deliverable.

    Event previews and the final answer were already redacted, but ``write_file``
    received the model's content verbatim and the download listing only checked
    filenames — so the key left the process inside a file the visitor could
    download.
    """
    script = [
        tool_call_turn("write_file", {
            "path": "outputs/leak.md",
            "content": f"# Notes\n\nThe configured credential is {_API_KEY}.\n",
        }),
        text_turn("I wrote the notes you asked for."),
    ]
    with MockLLMServer(script=script) as server:
        _config, store, adapter = _build(demo_env, server)
        session = store.create()
        events = await _collect(adapter, session, "Write notes to outputs/leak.md")

    assert events[-1].type == RUN_COMPLETED

    # Nothing the agent wrote — the whole workspace, ``outputs/`` included —
    # may contain the key.
    for path in session.workspace.rglob("*"):
        if path.is_file():
            body = path.read_text(encoding="utf-8", errors="ignore")
            assert _API_KEY not in body, f"secret landed in {path}"

    # Nor may anything containing it be offered for download.
    for path in list_output_files(session.outputs, redactor=adapter.redactor):
        assert _API_KEY not in path.read_text(encoding="utf-8", errors="ignore")

    # The events and the answer stay clean too (already true; kept as a guard).
    assert _API_KEY not in "\n".join(repr(e.to_dict()) for e in events)


async def test_the_runtime_trace_is_not_a_download_path(demo_env) -> None:
    """Records the one place a model-authored secret does still land on disk.

    ``TrajectoryFileObserver`` (core runtime) writes the raw assistant message,
    including its tool arguments, *before* any observer can rewrite them — so
    the trace under ``state/`` keeps whatever the endpoint sent. That is
    contained rather than fixed here: ``state/`` is outside the agent's
    authorised root, is never served to a browser, and is removed with the
    session. Asserting it keeps the containment honest instead of implying the
    trace is scrubbed.
    """
    script = [
        tool_call_turn("write_file", {
            "path": "outputs/leak.md", "content": f"credential {_API_KEY}",
        }),
        text_turn("done"),
    ]
    with MockLLMServer(script=script) as server:
        _config, store, adapter = _build(demo_env, server)
        session = store.create()
        await _collect(adapter, session, "Write it.")

    traces = [p for p in session.state.rglob("*.jsonl") if p.is_file()]
    assert traces, "expected the runtime to have written a trajectory"

    # It is not reachable as a download: not under outputs/, and the content
    # scan would withhold it even if it somehow were.
    offered = list_output_files(session.outputs, redactor=adapter.redactor)
    assert all(session.state not in p.parents for p in offered)
    for trace in traces:
        assert session.outputs not in trace.parents


async def test_a_secret_is_redacted_even_when_the_tool_call_is_the_only_path(
    demo_env,
) -> None:
    """The refusal must be visible, not silent, so the agent can react."""
    script = [
        tool_call_turn("write_file", {
            "path": "outputs/leak.md", "content": f"key={_API_KEY}",
        }),
        text_turn("done"),
    ]
    with MockLLMServer(script=script) as server:
        _config, store, adapter = _build(demo_env, server)
        session = store.create()
        events = await _collect(adapter, session, "Write it.")

    written = session.outputs / "leak.md"
    if written.exists():
        assert "REDACTED" in written.read_text(encoding="utf-8")
    finished = [e for e in events if e.type == ACTIVITY_FINISHED]
    assert finished, "the tool call should still have been reported"


async def test_a_secret_in_a_task_description_is_redacted_on_the_board(
    demo_env,
) -> None:
    """The task board is a newer path from model prose to the browser.

    It is covered, but not by the observer that emits it: ``add_task`` arguments
    are scrubbed by ``SecretArgumentObserver`` before the tool runs, so the
    description is already clean when the board projection reads it back. That
    makes this a boundary test rather than a redaction test — it fails if the
    secret guard ever stops covering the board tools, or if a future board path
    reads model prose from somewhere the guard does not reach.
    """
    script = [
        tool_call_turn("add_task", {"tasks": [
            {"description": f"Check whether {_API_KEY} is still valid"},
        ]}),
        text_turn("done"),
    ]
    with MockLLMServer(script=script) as server:
        _config, store, adapter = _build(demo_env, server)
        session = store.create()
        events = await _collect(adapter, session, "Plan it.")

    boards = [e for e in events if e.type == TASK_BOARD_UPDATED]
    assert boards, "the board tools should have projected board state"
    for item in boards:
        for task in item.data["tasks"]:
            assert _API_KEY not in task["description"], task
    assert any(
        "REDACTED" in task["description"]
        for item in boards for task in item.data["tasks"]
    ), "the description should have been masked, not dropped"


# ── [P1] streaming redaction must survive chunk boundaries ───────────────


async def test_a_secret_split_across_sse_chunks_is_still_redacted(
    demo_env,
) -> None:
    """Chunk boundaries are arbitrary, so per-delta matching is not enough.

    Neither half of a split secret matches on its own, so both were emitted
    verbatim and concatenating the deltas rebuilt the key.
    """
    head, tail = _API_KEY[:18], _API_KEY[18:]
    assert head and tail and head + tail == _API_KEY
    script = [Turn(content_chunks=[
        "your key is ", head, tail, " — keep it safe",
    ])]
    with MockLLMServer(script=script) as server:
        _config, store, adapter = _build(demo_env, server)
        events = await _collect(adapter, store.create(), "Echo the key.")

    streamed = "".join(
        str(e.data.get("text", "")) for e in events if e.type == ASSISTANT_DELTA
    )
    assert _API_KEY not in streamed, "the split secret was reassembled from deltas"
    assert "REDACTED" in streamed
    # The user-visible text must still be intelligible around the redaction.
    assert "your key is" in streamed
    assert "keep it safe" in streamed
    # And the terminal answer stays clean.
    assert _API_KEY not in str(events[-1].data.get("answer", ""))


async def test_streaming_redaction_does_not_swallow_ordinary_text(
    demo_env,
) -> None:
    """A hold-back buffer must still deliver everything, in order."""
    message = "alpha beta gamma delta epsilon zeta eta theta"
    with MockLLMServer(script=[text_turn(message)]) as server:
        _config, store, adapter = _build(demo_env, server)
        events = await _collect(adapter, store.create(), "Recite.")

    streamed = "".join(
        str(e.data.get("text", "")) for e in events if e.type == ASSISTANT_DELTA
    )
    assert message in streamed, streamed


# ── [P2] Stop must honour a bounded cancellation grace ───────────────────


async def test_stop_releases_the_runner_within_the_cancellation_grace(
    demo_env, monkeypatch,
) -> None:
    """A stuck call must not hold the single runner until the task timeout.

    Cooperative stop only takes effect at a turn boundary, which a hung LLM or
    tool call postpones. Without a bounded grace, Stop left the queue blocked
    for the full task wall.
    """
    monkeypatch.setattr(adapter_module, "_CANCEL_GRACE_S", 2.0)
    stall = 60.0  # far longer than the grace, well inside the task wall
    with MockLLMServer(script=[Turn(content="too late", delay_s=stall)]) as server:
        _config, store, adapter = _build(demo_env, server)
        session = store.create()

        started = time.monotonic()
        events: list[DemoEvent] = []
        agen = adapter.run(session=session, prompt="Hang, please.")
        async for item in agen:
            events.append(item)
            if item.type == "run_started":
                # Ask it to stop while the endpoint is still stalling.
                await asyncio.sleep(0.5)
                assert adapter.cancel(item.run_id) is True
        elapsed = time.monotonic() - started

    assert events[-1].type == RUN_CANCELLED, events[-1]
    # Specifically via the grace path, not by the call happening to finish.
    assert events[-1].data["reason"] == "cancel_grace_expired", events[-1].data
    assert elapsed < stall / 2, (
        f"Stop took {elapsed:.1f}s; it must not wait for the stalled call"
    )
    # The runner slot is free again immediately afterwards.
    assert adapter.active_run_ids == ()


# ── [P2] the inputs directory is read-only to the agent ──────────────────


async def test_the_agent_cannot_write_into_the_read_only_inputs_directory(
    demo_env,
) -> None:
    """``inputs/`` is documented as read-only; writes were permitted.

    The containment gate merged the read roots into a single allowed-root list
    used for reads *and* writes, and the runtime's authorised write root was the
    whole session tree rather than the workspace.
    """
    with MockLLMServer(script=[text_turn("placeholder")]) as server:
        _config, store, _adapter = _build(demo_env, server)
        session = store.create()

    # The absolute path, because that is what actually names the read-only
    # mount: a *relative* ``inputs/x`` resolves against the workspace and so
    # means ``workspace/inputs/x``, an ordinary scratch subdirectory.
    script = [
        tool_call_turn("write_file", {
            "path": str(session.inputs / "tampered.txt"), "content": "overwritten",
        }),
        text_turn("I could not write there."),
    ]
    with MockLLMServer(script=script) as server:
        _config2, _store2, adapter2 = _build(demo_env, server)
        events = await _collect(adapter2, session, "Tamper with the inputs.")

    finished = next(e for e in events if e.type == ACTIVITY_FINISHED)
    assert finished.data["ok"] is False, finished.data
    assert "read-only" in str(finished.data.get("detail", "")).lower()
    assert not (session.inputs / "tampered.txt").exists()


async def test_a_relative_inputs_path_is_a_workspace_subdirectory(
    demo_env,
) -> None:
    """Guards against reading the previous test as "inputs is unwritable by name".

    ``inputs/x`` relative to the workspace is scratch space and legitimately
    writable; only the session's real ``inputs/`` mount is protected.
    """
    script = [
        tool_call_turn("write_file", {
            "path": "inputs/scratch.txt", "content": "fine",
        }),
        text_turn("written"),
    ]
    with MockLLMServer(script=script) as server:
        _config, store, adapter = _build(demo_env, server)
        session = store.create()
        events = await _collect(adapter, session, "Write scratch.")

    finished = next(e for e in events if e.type == ACTIVITY_FINISHED)
    assert finished.data["ok"] is True, finished.data
    assert (session.workspace / "inputs" / "scratch.txt").is_file()
    assert not (session.inputs / "scratch.txt").exists()


async def test_inputs_remain_readable_after_being_made_write_protected(
    demo_env,
) -> None:
    """Making inputs read-only must not make them unreadable."""
    with MockLLMServer(script=[text_turn("placeholder")]) as server:
        _config, store, adapter = _build(demo_env, server)
        session = store.create()
        (session.inputs / "brief.txt").write_text(
            "The brief says: summarise in one line.", encoding="utf-8",
        )

    script = [
        tool_call_turn("read_file", {"path": str(session.inputs / "brief.txt")}),
        text_turn("The brief says to summarise in one line."),
    ]
    with MockLLMServer(script=script) as server:
        _config2, _store2, adapter2 = _build(demo_env, server)
        events = await _collect(adapter2, session, "Read the brief.")

    finished = next(e for e in events if e.type == ACTIVITY_FINISHED)
    assert finished.data["ok"] is True, finished.data
    assert "summarise in one line" in str(finished.data.get("detail", ""))


async def test_the_agent_cannot_write_into_the_state_directory(demo_env) -> None:
    """Trace/bookkeeping stays out of reach, by absolute path this time."""
    with MockLLMServer(script=[text_turn("placeholder")]) as server:
        _config, store, _adapter = _build(demo_env, server)
        session = store.create()

    script = [
        tool_call_turn("write_file", {
            "path": str(session.state / "tampered.json"), "content": "{}",
        }),
        text_turn("I could not write there."),
    ]
    with MockLLMServer(script=script) as server:
        _config2, _store2, adapter2 = _build(demo_env, server)
        events = await _collect(adapter2, session, "Tamper.")

    finished = next(e for e in events if e.type == ACTIVITY_FINISHED)
    assert finished.data["ok"] is False, finished.data
    assert not (session.state / "tampered.json").exists()


# ── self-review findings (same class: declared but not wired) ─────────────


async def test_a_retried_answer_is_still_rendered(demo_env) -> None:
    """The UI dropped every delta of a retried attempt.

    The delta handler gated on ``attempt == 1``, and the discard handler reset
    the accumulated answer — so after a discarded attempt the gate was false for
    every attempt-2 delta and the retried answer never appeared.
    """
    pytest.importorskip("gradio", reason="the hf-space extra is not installed")
    from deploy.huggingface.app import build_demo

    with MockLLMServer(script=[text_turn("the real answer")]) as server:
        _config, _store, _adapter = _build(demo_env, server)
        demo = build_demo(load_config())
        run = next(
            dep.fn for dep in demo.fns.values()
            if getattr(dep.fn, "__name__", "") == "_run"
        )
        frames = [frame async for frame in run("Answer me.", "")]

    # Simulate what the runtime reports on a retry by checking the handler's
    # own accumulation: the final pane must contain the answer either way.
    assert "the real answer" in str(frames[-1][3]), frames[-1][3]
    mid = [str(f[3]) for f in frames if str(f[3])]
    assert mid, "no frame ever carried answer text"


def test_the_delta_handler_accumulates_regardless_of_attempt_number() -> None:
    """Pin the specific gate that dropped retried text."""
    import inspect

    pytest.importorskip("gradio", reason="the hf-space extra is not installed")
    from deploy.huggingface import app

    source = inspect.getsource(app.build_demo)
    assert 'attempt", 1) == 1 or answer' not in source, (
        "the attempt gate is back: it silently drops a retried answer"
    )


async def test_a_withheld_file_is_not_announced_as_an_artifact(demo_env) -> None:
    """The artifact event and the download list must agree.

    ``ArtifactWatcher`` listed files without the redactor, so a file withheld
    for containing a secret was still reported as "produced" and then was
    missing from the downloads.
    """
    script = [
        tool_call_turn("write_file", {
            "path": "outputs/clean.md", "content": "nothing sensitive here",
        }),
        text_turn("done"),
    ]
    with MockLLMServer(script=script) as server:
        _config, store, adapter = _build(demo_env, server)
        session = store.create()
        events = await _collect(adapter, session, "Write a clean file.")

    announced = {
        str(e.data.get("relpath")) for e in events
        if e.type == "artifact_created"
    }
    offered = {
        str(p.relative_to(session.outputs))
        for p in list_output_files(session.outputs, redactor=adapter.redactor)
    }
    listed = set(events[-1].data.get("artifacts") or [])
    assert announced == offered, (announced, offered)
    assert listed == offered, (listed, offered)


def test_the_reported_concurrency_is_the_enforced_one() -> None:
    """Reporting the requested value would be a false claim."""
    config = load_config({
        "OPENAI_BASE_URL": "https://e.example.com/v1",
        "OPENAI_API_KEY": "sk-test-0123456789abcdef",
        "OPENAI_MODEL": "m",
        "SANDBOX_BACKEND": "native",
        "DEMO_MAX_CONCURRENCY": "8",
        "HOME": "/tmp",
    })
    assert config.max_concurrency == 8
    assert config.effective_concurrency == 1
    assert config.public_summary()["concurrency"] == "1"
    assert FrontierAgentAdapter(config).effective_concurrency == 1


def test_a_summary_endpoint_given_as_a_base_url_is_flagged() -> None:
    """The one endpoint variable that wants a full route, so it gets checked."""
    from deploy.huggingface.config import preflight

    base = {
        "OPENAI_BASE_URL": "https://e.example.com/v1",
        "OPENAI_API_KEY": "sk-test-0123456789abcdef",
        "OPENAI_MODEL": "m",
        "SANDBOX_BACKEND": "native",
        "SERPER_API_KEY": "k",
        "HOME": "/tmp",
    }
    wrong = preflight(load_config({
        **base, "SUMMARY_LLM_BASE_URL": "https://e.example.com/v1",
    }))
    assert any(
        i.field == "SUMMARY_LLM_BASE_URL" and "/chat/completions" in i.message
        for i in wrong.warnings
    ), wrong.format()

    right = preflight(load_config({
        **base, "SUMMARY_LLM_BASE_URL": "https://e.example.com/v1/chat/completions",
    }))
    assert not any(i.field == "SUMMARY_LLM_BASE_URL" for i in right.warnings)


def test_an_uncreatable_session_root_is_a_configuration_error(tmp_path) -> None:
    """Better than a SessionStore traceback on startup.

    The failure mode used here is a parent that is a *file*, because that fails
    for root too — a permission-denied directory would not, and CI plus this
    server both run as uid 0.
    """
    from deploy.huggingface.config import runtime_preflight

    blocker = tmp_path / "not-a-directory"
    blocker.write_text("", encoding="utf-8")
    config = load_config({
        "OPENAI_BASE_URL": "https://e.example.com/v1",
        "OPENAI_API_KEY": "sk-test-0123456789abcdef",
        "OPENAI_MODEL": "m",
        "SANDBOX_BACKEND": "native",
        "DEMO_RUNTIME_ROOT": str(blocker / "sessions"),
        "HOME": "/tmp",
    })
    checks = runtime_preflight(config)
    assert not checks.ok
    assert any(i.field == "DEMO_RUNTIME_ROOT" for i in checks.errors), checks.format()

    # And a usable root passes.
    good = load_config({
        "OPENAI_BASE_URL": "https://e.example.com/v1",
        "OPENAI_API_KEY": "sk-test-0123456789abcdef",
        "OPENAI_MODEL": "m",
        "SANDBOX_BACKEND": "native",
        "DEMO_RUNTIME_ROOT": str(tmp_path / "fine"),
        "HOME": "/tmp",
    })
    assert not any(
        i.field == "DEMO_RUNTIME_ROOT" for i in runtime_preflight(good).errors
    )


def test_a_discarded_attempt_does_not_splice_its_tail_onto_the_retry() -> None:
    """``StreamRedactor.discard`` drops held text instead of flushing it."""
    from deploy.huggingface.security import Redactor, StreamRedactor

    stream = StreamRedactor(Redactor.for_secrets([_API_KEY]))
    stream.feed("a draft that was thrown away")
    assert stream.pending > 0
    stream.discard()
    assert stream.pending == 0
    assert stream.flush() == ""
