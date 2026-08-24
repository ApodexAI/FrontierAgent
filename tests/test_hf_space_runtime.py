"""End-to-end demo runs against a fake OpenAI-compatible endpoint.

These exercise the *real* FrontierAgent ``react`` runtime — the same
``stateful-react-agent`` pipeline the CLI uses — with the model replaced by a
local HTTP server. No API token, no GPU, no network egress.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from pathlib import Path

import pytest

from deploy.huggingface.adapter import FrontierAgentAdapter
from deploy.huggingface.config import load_config
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
    DemoEvent,
)
from deploy.huggingface.mock_llm import (
    MockLLMServer,
    Turn,
    error_turn,
    text_turn,
    tool_call_turn,
)
from deploy.huggingface.security import resolve_download
from deploy.huggingface.sessions import SessionStore

_API_KEY = "sk-fake-demo-key-do-not-log-4242"

#: A relative path resolves against the session's authorised root, so a script
#: does not need to know the session id up front.
_OUTPUT_REL = "outputs/report.md"
_REPORT = "# Report\n\nReAct interleaves reasoning with tool use.\n"


@pytest.fixture
def demo_env(tmp_path, monkeypatch):
    """Isolate every process-global the runtime reads, per test.

    ``SANDBOX_BACKEND`` needs the explicit reload: the runtime resolves it via
    ``frontier_agent.infra.config.get_config()``, a singleton cached on first
    access, so whichever value was in the environment when some *earlier* test
    first touched that config would otherwise win — and this suite runs after
    tests that set ``bwrap``.
    """
    from frontier_agent.infra.config import get_config

    monkeypatch.setenv("SANDBOX_BACKEND", "native")
    monkeypatch.setenv("DEMO_RUNTIME_ROOT", str(tmp_path / "runtime"))
    monkeypatch.setenv("OPENAI_API_KEY", _API_KEY)
    monkeypatch.setenv("DEMO_TASK_TIMEOUT_SECONDS", "120")
    monkeypatch.setenv("DEMO_MAX_TURNS", "8")
    monkeypatch.setenv("DEMO_REPORTER", "false")
    get_config(force_reload=True)
    yield monkeypatch
    # Hand the cached config back to whatever the rest of the suite expects.
    monkeypatch.undo()
    get_config(force_reload=True)


def _write_report_script() -> list[Turn]:
    return [
        tool_call_turn("write_file", {"path": _OUTPUT_REL, "content": _REPORT}),
        text_turn("I wrote the report to outputs/report.md as requested."),
    ]


async def _collect(adapter, session, prompt: str) -> list[DemoEvent]:
    return [item async for item in adapter.run(session=session, prompt=prompt)]


def _types(events: list[DemoEvent]) -> list[str]:
    return [item.type for item in events]


def _build(monkeypatch, server: MockLLMServer):
    monkeypatch.setenv("OPENAI_BASE_URL", server.base_url)
    monkeypatch.setenv("OPENAI_MODEL", server.model)
    config = load_config()
    store = SessionStore(config.runtime_root, ttl_s=config.session_ttl_s)
    return config, store, FrontierAgentAdapter(config)


# ── the happy path ───────────────────────────────────────────────────────


async def test_prompt_runs_the_real_react_runtime_and_produces_an_artifact(
    demo_env,
) -> None:
    with MockLLMServer(script=_write_report_script()) as server:
        _config, store, adapter = _build(demo_env, server)
        session = store.create()
        events = await _collect(adapter, session, "Write a report on ReAct.")

    types = _types(events)
    assert types[0] == RUN_STARTED
    assert types[-1] == RUN_COMPLETED
    assert ACTIVITY_STARTED in types and ACTIVITY_FINISHED in types
    assert ASSISTANT_DELTA in types, "answer must stream, not appear at once"
    assert ARTIFACT_CREATED in types

    # A real tool ran: the tool call reached the real tool implementation.
    started = next(e for e in events if e.type == ACTIVITY_STARTED)
    assert started.data["activity"] == "write_file"
    finished = next(e for e in events if e.type == ACTIVITY_FINISHED)
    assert finished.data["ok"] is True

    # The final answer came from the runtime's own state, not from the UI.
    completed = events[-1]
    assert "outputs/report.md" in completed.data["answer"]
    assert completed.data["turns"] >= 2
    assert completed.data["tool_calls"] == 1

    # The artifact really exists in this session's outputs, and is downloadable.
    written = session.outputs / "report.md"
    assert written.read_text(encoding="utf-8") == _REPORT
    assert completed.data["artifacts"] == ["report.md"]
    assert resolve_download(session.outputs, "report.md") == written


async def test_streamed_deltas_reassemble_into_the_final_answer(demo_env) -> None:
    with MockLLMServer(script=[text_turn("alpha beta gamma delta epsilon")]) as server:
        _config, store, adapter = _build(demo_env, server)
        events = await _collect(adapter, store.create(), "Say the greek letters.")

    streamed = "".join(
        str(e.data["text"]) for e in events if e.type == ASSISTANT_DELTA
    )
    assert len(streamed) > 1, "a single chunk would not prove streaming"
    assert "alpha beta gamma delta epsilon" in streamed
    assert events[-1].data["answer"].strip().startswith("alpha")


async def test_events_are_ordered_start_then_activity_then_terminal(demo_env) -> None:
    """Ordering is load-bearing: the observer must be a *critical* one."""
    with MockLLMServer(script=_write_report_script()) as server:
        _config, store, adapter = _build(demo_env, server)
        events = await _collect(adapter, store.create(), "Write a report.")

    types = _types(events)
    assert types.index(RUN_STARTED) == 0
    assert types.index(ACTIVITY_STARTED) < types.index(ACTIVITY_FINISHED)
    assert all(not e.is_terminal for e in events[:-1])
    assert events[-1].is_terminal
    # Every event carries the identity the UI keys on.
    run_ids = {e.run_id for e in events}
    assert len(run_ids) == 1 and run_ids != {""}


# ── session isolation ────────────────────────────────────────────────────


async def test_two_sessions_cannot_see_or_download_each_others_output(
    demo_env,
) -> None:
    with MockLLMServer(script=_write_report_script() * 2) as server:
        _config, store, adapter = _build(demo_env, server)
        first = store.create()
        second = store.create()
        await _collect(adapter, first, "Write a report.")

    assert (first.outputs / "report.md").is_file()
    assert not (second.outputs / "report.md").exists()
    assert first.session_id != second.session_id
    assert first.root != second.root

    # B cannot reach A's file, even knowing its name and A's directory layout.
    from deploy.huggingface.security import DownloadDenied

    with pytest.raises(DownloadDenied):
        resolve_download(second.outputs, "report.md")
    with pytest.raises(DownloadDenied):
        resolve_download(second.outputs, f"../../{first.session_id}/outputs/report.md")


async def test_clearing_one_session_leaves_the_other_untouched(demo_env) -> None:
    with MockLLMServer(script=_write_report_script() * 2) as server:
        _config, store, adapter = _build(demo_env, server)
        first, second = store.create(), store.create()
        await _collect(adapter, first, "Write a report.")
        await _collect(adapter, second, "Write a report.")

    assert (first.outputs / "report.md").is_file()
    assert (second.outputs / "report.md").is_file()
    store.clear(first.session_id)
    assert not (first.outputs / "report.md").exists()
    assert (second.outputs / "report.md").is_file()


async def test_session_ids_are_unpredictable_and_validated(demo_env) -> None:
    from deploy.huggingface.sessions import InvalidSessionId

    with MockLLMServer(script=[text_turn("ok")]) as server:
        config, store, _adapter = _build(demo_env, server)

    ids = {store.create().session_id for _ in range(20)}
    assert len(ids) == 20
    assert all(len(sid) >= 16 for sid in ids)
    for malformed in ("../escape", "short", "with/slash", ""):
        with pytest.raises(InvalidSessionId):
            store.get(malformed)
    # An unusable id yields a *new* session rather than somebody else's.
    assert store.get_or_create("../escape").session_id not in {"", "../escape"}
    assert config.runtime_root.is_dir()


@pytest.mark.parametrize("template", [
    "../../../../tmp/{probe}",
    "../{probe}",
    "/tmp/{probe}",
    "/etc/{probe}",
])
async def test_the_agent_cannot_write_outside_its_own_session(
    demo_env, template: str,
) -> None:
    """Containment, exercised through a real tool call rather than asserted.

    ``write_file`` treats a path-guard denial as a cue to retry through the
    sandbox writer, which does no validation — so this must be blocked before
    the tool runs.
    """
    # A unique probe per run: a shared filename would let one leaked write make
    # every later run of this test fail (or, worse, pass for the wrong reason).
    probe = f"frontier-demo-escape-{uuid.uuid4().hex[:12]}.txt"
    escape = template.format(probe=probe)
    script = [
        tool_call_turn("write_file", {"path": escape, "content": "should not land"}),
        text_turn("I could not write there."),
    ]
    with MockLLMServer(script=script) as server:
        _config, store, adapter = _build(demo_env, server)
        session = store.create()
        events = await _collect(adapter, session, "Try to escape.")

    finished = next(e for e in events if e.type == ACTIVITY_FINISHED)
    assert finished.data["ok"] is False, finished.data
    assert "refused" in str(finished.data.get("detail", "")).lower()

    # Nothing was written anywhere outside the session's own workspace.
    assert not (Path("/tmp") / probe).exists()
    assert not (Path("/etc") / probe).exists()
    # The working directory especially: ``write_file``'s sandbox fallback
    # resolves a relative path against the *process* cwd, so that is where an
    # escape actually lands — and checking only /tmp and the runtime root would
    # let it through unnoticed.
    assert not (Path.cwd() / probe).exists()
    assert not (Path.cwd().parent / probe).exists()
    escaped = [
        path for path in session.root.parent.parent.rglob(probe)
        if session.workspace not in path.parents
    ]
    assert escaped == [], escaped


async def test_reads_outside_the_session_are_refused(demo_env) -> None:
    script = [
        tool_call_turn("read_file", {"path": "/etc/passwd"}),
        text_turn("I could not read that."),
    ]
    with MockLLMServer(script=script) as server:
        _config, store, adapter = _build(demo_env, server)
        events = await _collect(adapter, store.create(), "Read /etc/passwd.")

    finished = next(e for e in events if e.type == ACTIVITY_FINISHED)
    assert finished.data["ok"] is False
    detail = str(finished.data.get("detail", ""))
    assert "refused" in detail.lower()
    assert "root:" not in detail


async def test_the_session_state_directory_is_not_agent_writable(demo_env) -> None:
    """Trace/bookkeeping files must be out of the agent's reach."""
    script = [
        tool_call_turn("write_file", {
            "path": "../state/tampered.json", "content": "{}",
        }),
        text_turn("I could not write there."),
    ]
    with MockLLMServer(script=script) as server:
        _config, store, adapter = _build(demo_env, server)
        session = store.create()
        await _collect(adapter, session, "Tamper with the trace.")

    assert not (session.state / "tampered.json").exists()
    # Nor anywhere else the sandbox fallback might have resolved it to.
    assert not (Path.cwd() / "state" / "tampered.json").exists()
    assert not (Path.cwd().parent / "state" / "tampered.json").exists()


# ── limits: cancel, timeout, queue ───────────────────────────────────────


async def test_stop_lands_the_run_cleanly_instead_of_killing_it(demo_env) -> None:
    """Cancellation is cooperative: the agent stops at a turn boundary."""
    script = [
        tool_call_turn("write_file", {"path": f"outputs/step{i}.md", "content": "x"})
        for i in range(6)
    ] + [text_turn("done")]
    with MockLLMServer(script=script) as server:
        _config, store, adapter = _build(demo_env, server)
        session = store.create()

        events: list[DemoEvent] = []
        async for item in adapter.run(session=session, prompt="Write six files."):
            events.append(item)
            if item.type == ACTIVITY_FINISHED and len(
                [e for e in events if e.type == ACTIVITY_FINISHED]
            ) == 1:
                assert adapter.cancel(item.run_id) is True

    assert _types(events)[-1] == RUN_CANCELLED
    # It stopped early rather than running the whole script…
    assert len([e for e in events if e.type == ACTIVITY_STARTED]) < 6
    # …and it stopped *cleanly*: the runtime reported a paused loop.
    assert events[-1].data.get("stopped_by") in ("paused", "")


async def test_abandoning_a_queued_run_leaves_no_bookkeeping_behind(
    demo_env,
) -> None:
    """A visitor who closes the tab while queued must not leak a slot."""
    with MockLLMServer(script=[Turn(content="slow", delay_s=3)]) as server:
        _config, store, adapter = _build(demo_env, server)

        async def _drive(session):
            return await _collect(adapter, session, "Take your time.")

        held = asyncio.create_task(_drive(store.create()))
        await asyncio.sleep(0.5)                     # occupy the single slot
        abandoned = asyncio.create_task(_drive(store.create()))
        await asyncio.sleep(0.3)                     # let it start queueing
        abandoned.cancel()
        with pytest.raises(asyncio.CancelledError):
            await abandoned
        await held

    assert adapter.active_run_ids == ()
    assert adapter.busy_session_ids() == set()
    # The freed slot is genuinely reusable.
    with MockLLMServer(script=[text_turn("after")]) as server:
        _config, store, adapter2 = _build(demo_env, server)
        events = await _collect(adapter2, store.create(), "Again.")
    assert events[-1].type == RUN_COMPLETED


async def test_cancelling_an_unknown_run_is_reported_not_raised(demo_env) -> None:
    with MockLLMServer(script=[text_turn("ok")]) as server:
        _config, _store, adapter = _build(demo_env, server)
    assert adapter.cancel("run-does-not-exist") is False


async def test_a_slow_endpoint_fails_with_a_timeout_not_a_hang(demo_env) -> None:
    demo_env.setenv("DEMO_TASK_TIMEOUT_SECONDS", "30")
    with MockLLMServer(script=[Turn(content="too late", delay_s=90)]) as server:
        _config, store, adapter = _build(demo_env, server)
        events = await asyncio.wait_for(
            _collect(adapter, store.create(), "Wait for me."), timeout=120,
        )

    assert events[-1].type == RUN_FAILED
    assert events[-1].data["reason"] in ("timeout", "upstream_timeout")
    assert events[-1].data["message"]


async def test_a_full_queue_is_refused_with_a_terminal_event(demo_env) -> None:
    demo_env.setenv("DEMO_QUEUE_SIZE", "1")
    with MockLLMServer(script=[Turn(content="slow", delay_s=3)]) as server:
        _config, store, adapter = _build(demo_env, server)

        async def _drive(session):
            return await _collect(adapter, session, "Take your time.")

        held = asyncio.create_task(_drive(store.create()))
        await asyncio.sleep(0.5)          # let it occupy the single slot
        queued = asyncio.create_task(_drive(store.create()))
        await asyncio.sleep(0.2)          # …and fill the one queue place
        refused = await _collect(adapter, store.create(), "Me too.")
        results = await asyncio.gather(held, queued)

    assert refused[-1].type == RUN_FAILED
    assert refused[-1].data["reason"] == "queue_full"
    assert len(refused) == 1, "a rejected run must not emit run_started"
    # The runs that were admitted still completed, and one of them had to wait.
    assert all(events[-1].type == RUN_COMPLETED for events in results)
    assert any(QUEUED in _types(events) for events in results)


async def test_runs_are_serialised_even_when_submitted_together(demo_env) -> None:
    with MockLLMServer(script=[Turn(content="one at a time", delay_s=1)]) as server:
        _config, store, adapter = _build(demo_env, server)
        sessions = [store.create() for _ in range(3)]
        results = await asyncio.gather(*(
            _collect(adapter, s, "Go.") for s in sessions
        ))

    assert all(events[-1].type == RUN_COMPLETED for events in results)
    # With one runner slot, at least two of the three had to queue.
    assert sum(QUEUED in _types(events) for events in results) >= 2


# ── input validation ─────────────────────────────────────────────────────


@pytest.mark.parametrize(("prompt", "reason"), [
    ("", "empty_prompt"),
    ("   ", "empty_prompt"),
])
async def test_an_empty_prompt_is_refused_before_any_model_call(
    demo_env, prompt: str, reason: str,
) -> None:
    with MockLLMServer(script=[text_turn("never reached")]) as server:
        _config, store, adapter = _build(demo_env, server)
        events = await _collect(adapter, store.create(), prompt)
        assert server.calls == 0

    assert _types(events) == [RUN_FAILED]
    assert events[-1].data["reason"] == reason


async def test_an_oversized_prompt_is_refused(demo_env) -> None:
    demo_env.setenv("DEMO_MAX_PROMPT_CHARS", "50")
    with MockLLMServer(script=[text_turn("never reached")]) as server:
        _config, store, adapter = _build(demo_env, server)
        events = await _collect(adapter, store.create(), "x" * 200)
        assert server.calls == 0

    assert events[-1].data["reason"] == "prompt_too_long"
    assert "50" in events[-1].data["message"]


# ── upstream failures reach the user as guidance ─────────────────────────


@pytest.mark.parametrize(("status", "needle"), [
    (401, "OPENAI_API_KEY"),
    (404, "OPENAI_BASE_URL"),
])
async def test_endpoint_rejection_surfaces_an_actionable_message(
    demo_env, status: int, needle: str,
) -> None:
    demo_env.setenv("DEMO_TASK_TIMEOUT_SECONDS", "60")
    with MockLLMServer(script=[error_turn(status)]) as server:
        _config, store, adapter = _build(demo_env, server)
        events = await _collect(adapter, store.create(), "Anything.")

    terminal = events[-1]
    assert terminal.type in (RUN_FAILED, RUN_COMPLETED)
    text = f"{terminal.data.get('message', '')} {terminal.data.get('answer', '')}"
    assert needle in text or "endpoint" in text.lower(), text


# ── secrets stay inside the process ──────────────────────────────────────


async def test_the_api_key_never_appears_in_events_or_logs(
    demo_env, caplog,
) -> None:
    caplog.set_level(logging.DEBUG)
    with MockLLMServer(script=_write_report_script()) as server:
        _config, store, adapter = _build(demo_env, server)
        session = store.create()
        events = await _collect(adapter, session, "Write a report.")

    rendered = "\n".join(repr(item.to_dict()) for item in events)
    assert _API_KEY not in rendered
    assert _API_KEY not in caplog.text
    for path in session.outputs.rglob("*"):
        if path.is_file():
            assert _API_KEY not in path.read_text(encoding="utf-8", errors="ignore")


async def test_a_secret_echoed_by_the_endpoint_is_redacted_in_events(
    demo_env,
) -> None:
    """A hostile or sloppy endpoint must not be able to print the key back."""
    with MockLLMServer(script=[text_turn(f"your key is {_API_KEY} ok")]) as server:
        _config, store, adapter = _build(demo_env, server)
        events = await _collect(adapter, store.create(), "Echo the key.")

    assert _API_KEY not in "\n".join(repr(e.to_dict()) for e in events)
    assert "REDACTED" in events[-1].data["answer"]


# ── rolled-back turns ────────────────────────────────────────────────────


def _attempt(turn: int, call_id: str, *, outcome: str = "accepted") -> object:
    from frontier_agent.core.loop_types import LLMAttemptContext

    return LLMAttemptContext(
        turn=turn, max_turns=6, task_id="t", role_id="main",
        call_id=call_id, attempt_id=f"{call_id}_attempt_01", attempt_index=1,
        phase="finished", outcome=outcome,
    )


def _observer() -> tuple[object, object]:
    from deploy.huggingface.adapter import EventChannel, StructuredEventObserver
    from deploy.huggingface.security import Redactor

    channel = EventChannel()
    observer = StructuredEventObserver(
        channel=channel, session_id="s", run_id="r", redactor=Redactor(),
    )
    return observer, channel


def _pushed(channel) -> list[DemoEvent]:
    return list(channel._items)  # noqa: SLF001 - inspecting the transport under test


async def test_a_rolled_back_turn_tells_the_ui_to_drop_the_draft() -> None:
    """A rollback observer pops an assistant message the browser already has.

    ``DuplicateQueryRollbackObserver`` rejects the turn in ``on_llm_response``,
    i.e. after its deltas streamed and after the attempt was delivered — so no
    ``ATTEMPT_DISCARDED`` fires and nothing else would tell the UI that what it
    is showing was thrown away. A second logical call under an unchanged turn
    number is the signal.
    """
    observer, channel = _observer()
    await observer.on_llm_attempt(_attempt(3, "llm_aaa"))
    await observer.on_llm_attempt(_attempt(3, "llm_bbb"))  # same turn, re-run

    resets = [
        e for e in _pushed(channel)
        if e.data.get("discard_stream") and e.data.get("reason") == "turn_rolled_back"
    ]
    assert len(resets) == 1
    assert resets[0].data["turn"] == 3


async def test_a_normal_turn_boundary_is_not_mistaken_for_a_rollback() -> None:
    """Advancing to the next turn, and retrying inside one call, are both fine.

    Without this the UI would blank the answer at every turn boundary — the
    signal has to be a *new call for the same turn*, nothing else.
    """
    observer, channel = _observer()
    await observer.on_llm_attempt(_attempt(1, "llm_aaa"))
    await observer.on_llm_attempt(_attempt(2, "llm_bbb"))
    await observer.on_llm_attempt(_attempt(3, "llm_ccc"))
    # A second attempt within one logical call (provider retry): same call_id.
    await observer.on_llm_attempt(_attempt(3, "llm_ccc"))

    assert not [
        e for e in _pushed(channel)
        if e.data.get("reason") == "turn_rolled_back"
    ]


async def test_a_degraded_reply_is_explained_rather_than_silently_empty() -> None:
    """The runaway guard's last resort returns a reasoning-only reply.

    It is delivered, not discarded, so the answer pane simply gets nothing for
    that turn. Emit the reason instead of leaving an unexplained empty step.
    """
    from frontier_agent.core.loop_types import ATTEMPT_ACCEPTED_DEGRADED

    observer, channel = _observer()
    await observer.on_llm_attempt(_attempt(
        1, "llm_aaa", outcome=ATTEMPT_ACCEPTED_DEGRADED,
    ))

    warnings = [e for e in _pushed(channel) if e.data.get("reason")]
    assert warnings, "a degraded acceptance must not pass silently"
    # Bytes were delivered, so the draft must NOT be dropped.
    assert not any(e.data.get("discard_stream") for e in warnings)
