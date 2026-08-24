"""Site-3 truncation must be recoverable from the trajectory.

A tool result is cut in three places. The first two (``tool_exec.py:244`` and
``_apply_aggregate_budget``) cut ``result_str`` *before* the ``ToolResult`` is
built, so the observer records an already-truncated body and the spill store is
what holds the original. The third — the post-processor applied at
``agent_loop.py:762`` — is the opposite: ``notify_tool_result`` runs first, so the
trajectory keeps the full body while the message the model sees is cut by the
post-processor's own budget — 6_000 by default, bash 4_000 under agent-team — with
nothing persisted and no pointer. (Not 15_000: that is
``SwarmSubagentRuntime.tool_result_max_chars``, the site-1/2 per-tool cap, which
runs earlier and is a different gate. An earlier version of this docstring named
it here and so described the wrong site.) That content exists only in the
trajectory, which is not sandbox-visible, so only an in-process tool can reach it.
"""

from __future__ import annotations

import ast
import asyncio
import json
from pathlib import Path

import pytest

from frontier_agent.components.observers.trajectory import TrajectoryFileObserver
from frontier_agent.core.execution_context import (
    ExecutionScope,
    reset_current_execution_scope,
    set_current_execution_scope,
)
from frontier_agent.core.llm import LLMResponse
from frontier_agent.core.loop_types import LoopConfig, ToolResult, TurnContext
from frontier_agent.core.runtime.loop.agent_loop import (
    _with_recovery_handle,
    run_agent_loop,
)
from frontier_agent.core.tool import tool
from plugins.tools.recover_result import _find_record, recover_result

BIG = "HEAD-" + ("payload " * 8_000) + "-TAILMARKER"


@tool
async def _bulky(q: str) -> str:
    """Returns a body far past the site-3 cap.

    Args:
        q: ignored.
    """
    del q
    return BIG


@pytest.fixture
def scope():
    sc = ExecutionScope(task_id="t1", role_id="react")
    token = set_current_execution_scope(sc)
    try:
        yield sc
    finally:
        reset_current_execution_scope(token)


def _ctx(turn: int) -> TurnContext:
    return TurnContext(
        turn=turn, max_turns=4, task_id="t1", role_id="react", ai_text="",
        thinking="", tool_calls=[], messages=[], usage=None, metadata={},
    )


# ── The observer publishes where its file is ──────────────────────────────


def test_the_path_is_published_across_the_hooks_spawned_task(tmp_path, scope) -> None:
    """``notify_observers`` dispatches non-critical hooks as separate tasks, so a
    contextvar set in ``on_loop_start`` would be invisible to the loop. The scope
    object is shared by reference, so its dict is not — that is why the path
    travels this way and not as a contextvar."""
    from frontier_agent.core.loop_types import notify_observers

    observer = TrajectoryFileObserver(tmp_path, filename="probe", formats=["jsonl"])

    async def drive() -> None:
        await notify_observers([observer], "on_loop_start", LoopConfig(task_id="t1"))
        await asyncio.sleep(0.05)

    asyncio.run(drive())
    published = scope.metadata.get(TrajectoryFileObserver.SCOPE_KEY)
    assert published == str(tmp_path / "probe.jsonl")


def test_no_path_is_published_when_jsonl_is_off(tmp_path, scope) -> None:
    """Absence of the key is the signal that no handle can be minted."""
    observer = TrajectoryFileObserver(tmp_path, filename="probe", formats=["json"])
    asyncio.run(observer.on_loop_start(LoopConfig(task_id="t1")))
    assert TrajectoryFileObserver.SCOPE_KEY not in scope.metadata


def test_publishing_does_not_create_the_directory(tmp_path, scope) -> None:
    """``_path()`` mkdirs; answering "where is my trajectory" must not."""
    target = tmp_path / "not-yet"
    observer = TrajectoryFileObserver(target, filename="probe", formats=["jsonl"])
    observer._publish_jsonl_path()
    assert scope.metadata[TrajectoryFileObserver.SCOPE_KEY]
    assert not target.exists(), "publishing created the directory"


# ── The footer ────────────────────────────────────────────────────────────


def _result(body: str, call_id: str = "call_7") -> ToolResult:
    return ToolResult(
        name="bash", args={}, result=body, duration_ms=1,
        tool_call_id=call_id, is_error=False,
    )


def test_footer_appears_only_when_content_was_actually_cut() -> None:
    full = "x" * 1_000
    assert "recover_result" in _with_recovery_handle(
        "x" * 400, _result(full), 3, enabled=True,
    )
    # Nothing cut -> nothing to advertise.
    assert _with_recovery_handle(full, _result(full), 3, enabled=True) == full


def test_no_footer_when_the_tool_is_not_bound() -> None:
    """A footer naming a tool the agent cannot call is worse than no footer."""
    out = _with_recovery_handle("x" * 400, _result("x" * 1_000), 3, enabled=False)
    assert "recover_result" not in out


def test_no_footer_without_a_call_id() -> None:
    """The handle is (turn, call_id); an empty id resolves to nothing."""
    out = _with_recovery_handle(
        "x" * 400, _result("x" * 1_000, call_id=""), 3, enabled=True,
    )
    assert "recover_result" not in out


def test_footer_names_the_turn_and_id_without_call_syntax() -> None:
    """The values must be exact, and the shape must not look like source.

    This asserted ``recover_result(turn=17, call_id="call_abc")`` verbatim until a
    live run showed the cost of that shape: the model read the callable form as
    code and reproduced it inside a ```bash block rather than emitting a tool
    call, with ``LeakedToolCallRetryObserver`` firing twice. The values are still
    pinned; the parenthesised call is now pinned OUT.
    """
    out = _with_recovery_handle(
        "x" * 10, _result("x" * 900, call_id="call_abc"), 17, enabled=True,
    )
    assert "recover_result" in out
    assert "turn 17" in out
    assert "call_abc" in out
    assert "recover_result(" not in out, "footer reads as a callable expression"


def test_no_footer_when_the_body_already_names_a_spill_file() -> None:
    """Two routes to the same bytes is worse than one, and this one is redundant.

    Gate ① spills the FULL pre-cut output and leaves a path in the body, so the
    spill file is a strict superset of whatever site 3 removed — verified on a live
    agent-team run: 42,770 chars in the file behind an 8,000-char body, with the
    elided middle present in it. Measured cost of advertising both: 43 footers,
    zero ``recover_result`` calls, and the agent running ``cat`` on the spill path
    after writing "Let me call recover_result".
    """
    from plugins.tools._sandbox import _DEFAULT_SPILL_DIR

    body = "kept output\n[Full output saved to " + _DEFAULT_SPILL_DIR + "/s/9f.md]"
    out = _with_recovery_handle(body, _result("x" * 90_000), 4, enabled=True)

    assert out == body, "footer competed with a pointer that already covers the cut"


def test_the_footer_still_fires_when_nothing_else_covers_the_cut() -> None:
    """The suppression must not swallow the case the tool exists for: a result cut
    below gate ① never reached the store, so no path names it."""
    body = "kept output with no pointer at all"
    out = _with_recovery_handle(body, _result("x" * 9_000), 4, enabled=True)

    assert "recover_result" in out


def test_a_lookalike_directory_does_not_suppress_the_footer() -> None:
    """``"/spill" in body`` also fires on ``/spillover``; a real pointer always
    names a file UNDER the store, so the separator is what distinguishes them."""
    body = "see /spillover/notes.md for context"
    out = _with_recovery_handle(body, _result("x" * 9_000), 4, enabled=True)

    assert "recover_result" in out


# ── The scan ──────────────────────────────────────────────────────────────


def _write(path: Path, records: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n",
        encoding="utf-8",
    )


def test_an_empty_id_never_matches(tmp_path) -> None:
    """Pre-8a trajectories have no ``tool_call_id`` field. Treating the absent
    field as "" made a scan return the WRONG body while reporting success — this
    is that failure, pinned."""
    f = tmp_path / "t.jsonl"
    _write(f, [
        {"t": "start"},
        {"t": "result", "turn": 8, "name": "web_fetch", "result": "WRONG"},
        {"t": "result", "turn": 8, "name": "bash", "result": "ALSO WRONG"},
    ])
    assert _find_record(f, 8, "") is None
    assert _find_record(f, 8, "call_1") is None


def test_the_last_attempt_of_the_current_run_wins(tmp_path) -> None:
    """The file is opened in append mode under a deterministic stem, so a re-run
    of the same task accumulates both runs; and ``agent_loop`` decrements the turn
    counter on ``continue_to_next_turn``, so a turn can be attempted twice with
    colliding synthetic ids."""
    f = tmp_path / "t.jsonl"
    _write(f, [
        {"t": "start"},
        {"t": "result", "turn": 2, "tool_call_id": "call_2_0", "result": "PREVIOUS RUN"},
        {"t": "end"},
        {"t": "start"},
        {"t": "result", "turn": 2, "tool_call_id": "call_2_0", "result": "FIRST ATTEMPT"},
        {"t": "result", "turn": 2, "tool_call_id": "call_2_0", "result": "RETRY"},
    ])
    record = _find_record(f, 2, "call_2_0")
    assert record is not None
    assert record["result"] == "RETRY"


def test_an_unparseable_line_is_skipped(tmp_path) -> None:
    """Bodies reach 150K against an 8KB buffer, so a large record is flushed to
    disk in pieces and a partial line can appear mid-file."""
    f = tmp_path / "t.jsonl"
    f.write_text(
        json.dumps({"t": "start"}) + "\n"
        + '{"t": "result", "turn": 1, "tool_call_id": "a", "result": "trunc'  # partial
        + "\n"
        + json.dumps({
            "t": "result", "turn": 1, "tool_call_id": "call_x", "result": "GOOD",
        }) + "\n",
        encoding="utf-8",
    )
    record = _find_record(f, 1, "call_x")
    assert record is not None and record["result"] == "GOOD"


# ── The tool ──────────────────────────────────────────────────────────────


def test_the_tool_never_imports_the_sandbox() -> None:
    """That import is the only thing that makes a tool sandboxed, and the
    trajectory is not mounted anywhere. Mirrors the AST guard added for
    ``_writer_core`` after the writer regression."""
    src = Path("plugins/tools/recover_result.py").read_text()
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.ImportFrom):
            assert "_sandbox" not in (node.module or "")
        elif isinstance(node, ast.Import):
            for alias in node.names:
                assert "_sandbox" not in alias.name


def test_tool_reports_plainly_when_there_is_no_transcript(scope) -> None:
    out = asyncio.run(recover_result.func(turn=1, call_id="call_1"))
    assert "unavailable" in out.lower()
    # Must not send the agent round the same loop again.
    assert "recover_result(" not in out


def test_tool_refuses_an_empty_call_id(tmp_path, scope) -> None:
    scope.metadata[TrajectoryFileObserver.SCOPE_KEY] = str(tmp_path / "t.jsonl")
    out = asyncio.run(recover_result.func(turn=1, call_id=""))
    assert "call_id is required" in out


def test_tool_paginates_and_points_at_the_next_offset(tmp_path, scope) -> None:
    f = tmp_path / "t.jsonl"
    body = "".join(f"{i:06d}." for i in range(4_000))     # 28_000 chars
    _write(f, [
        {"t": "start"},
        {"t": "result", "turn": 5, "name": "bash", "tool_call_id": "c1",
         "result": body},
    ])
    scope.metadata[TrajectoryFileObserver.SCOPE_KEY] = str(f)

    first = asyncio.run(recover_result.func(turn=5, call_id="c1"))
    assert body[:200] in first
    assert "chars remain" in first
    assert "offset=8000" in first

    second = asyncio.run(recover_result.func(turn=5, call_id="c1", offset=8_000))
    assert body[8_000:8_200] in second

    tail = asyncio.run(recover_result.func(turn=5, call_id="c1", offset=24_000))
    assert body[-50:] in tail
    assert "[end of result]" in tail


def test_tool_rejects_an_offset_past_the_end(tmp_path, scope) -> None:
    f = tmp_path / "t.jsonl"
    _write(f, [
        {"t": "start"},
        {"t": "result", "turn": 1, "tool_call_id": "c1", "result": "short"},
    ])
    scope.metadata[TrajectoryFileObserver.SCOPE_KEY] = str(f)
    out = asyncio.run(recover_result.func(turn=1, call_id="c1", offset=999))
    assert "past the end" in out


# ── End to end, through the real loop ─────────────────────────────────────


@pytest.mark.asyncio
async def test_round_trip_recovers_what_site_3_cut(tmp_path) -> None:
    """The whole point, driven through ``run_agent_loop``: a body cut at site 3 is
    fetched back verbatim from the trajectory the observer wrote."""
    observer = TrajectoryFileObserver(tmp_path, filename="probe", formats=["jsonl"])

    class _LLM:
        model = "stub"

        def __init__(self) -> None:
            self.n = 0

        async def chat(self, messages, **kwargs):
            del kwargs
            self.n += 1
            if self.n == 1:
                return LLMResponse(
                    content="",
                    tool_calls=[{
                        "id": "call_probe", "type": "function",
                        "function": {
                            "name": "_bulky",
                            "arguments": json.dumps({"q": "x"}),
                        },
                    }],
                    finish_reason="tool_calls",
                )
            self.seen = messages
            return LLMResponse(content="done", tool_calls=[], finish_reason="stop")

    llm = _LLM()
    result = await run_agent_loop(
        system_prompt="s", user_message="go", llm=llm, tools=[_bulky, recover_result],
        config=LoopConfig(max_turns=2, tool_result_max_chars=2_000),
        observers=[observer],
    )
    observer._close_jsonl()

    tool_messages = [m for m in result.messages if m.get("role") == "tool"]
    assert tool_messages, "no tool message reached history"
    shown = str(tool_messages[0].get("content") or "")
    assert len(shown) < len(BIG), "site 3 did not truncate"
    assert "recover_result" in shown, shown[-300:]
    assert "call_probe" in shown, shown[-300:]
    assert "-TAILMARKER" not in shown, "the tail should have been cut"

    # The trajectory kept the whole body, and the handle finds it.
    record = _find_record(tmp_path / "probe.jsonl", 1, "call_probe")
    assert record is not None
    assert record["result"] == BIG, "the trajectory did not keep the full body"

    scope = ExecutionScope(task_id="t1", role_id="react")
    scope.metadata[TrajectoryFileObserver.SCOPE_KEY] = str(tmp_path / "probe.jsonl")
    token = set_current_execution_scope(scope)
    try:
        tail = await recover_result.func(
            turn=1, call_id="call_probe", offset=len(BIG) - 40,
        )
    finally:
        reset_current_execution_scope(token)
    assert "-TAILMARKER" in tail, "recovery did not return the cut tail"


# ── A sub-agent's handle must resolve to its OWN transcript ────────────────
#
# This is the only path that matters in practice: recover_result is bound in
# ``sub_agent_tools`` and deliberately NOT in ``main_agent_tools``, so every real
# call comes from a sub-agent. ``subagent_runtime`` notes twice that "AgentBus
# builds the sub-agent's ExecutionScope without our scope_metadata" — a fresh
# scope, not an inherited one — which is what makes this safe, and what makes it
# worth pinning: if a sub-agent ever read the parent's scope it would recover the
# COORDINATOR's tool output while reporting success, the same
# wrong-body-with-a-success-message failure an empty call_id used to cause.


def test_a_subagents_handle_resolves_to_its_own_transcript(tmp_path) -> None:
    main_dir, sub_dir = tmp_path / "traj", tmp_path / "traj" / "subagents"
    main_scope = ExecutionScope(task_id="main", role_id="coordinator")
    sub_scope = ExecutionScope(task_id="main", role_id="researcher")

    published: dict[str, str] = {}
    for label, scope_obj, directory, stem in (
        ("main", main_scope, main_dir, "main_agent"),
        ("sub", sub_scope, sub_dir, "wiki_mechanics.t01"),
    ):
        token = set_current_execution_scope(scope_obj)
        try:
            observer = TrajectoryFileObserver(
                directory, filename=stem, formats=["jsonl"],
            )
            observer._publish_jsonl_path()
            published[label] = scope_obj.metadata[TrajectoryFileObserver.SCOPE_KEY]
        finally:
            reset_current_execution_scope(token)

    assert published["main"] != published["sub"]
    # Exact paths, not substrings: pytest derives ``tmp_path`` from the test name,
    # which contains "subagents" here, so a substring check passes on the tmp dir
    # rather than on the structure under test.
    assert published["main"] == str(main_dir / "main_agent.jsonl")
    assert published["sub"] == str(sub_dir / "wiki_mechanics.t01.jsonl")


def test_a_subagent_recovers_the_body_its_own_scope_names(tmp_path) -> None:
    """Two transcripts carry the SAME turn and call id with different bodies, so
    only the scope's path can pick between them. This pins that the scan reads the
    path its scope names and discovers files by no other route — a fallback or a
    glob would surface the coordinator's body while reporting success. It does not
    test scope inheritance: only one scope is active here, and the fresh-scope
    guarantee is ``subagent_runtime``'s, covered by the sibling test above."""
    main_file = tmp_path / "main_agent.jsonl"
    sub_file = tmp_path / "subagents" / "researcher.t01.jsonl"
    sub_file.parent.mkdir(parents=True)
    for path, body in ((main_file, "COORDINATOR-BODY"), (sub_file, "SUBAGENT-BODY")):
        path.write_text(
            json.dumps({"t": "start"}) + "\n"
            + json.dumps({
                "t": "result", "turn": 2, "tool": "bash",
                "tool_call_id": "call-7", "result": body,
            }) + "\n",
            encoding="utf-8",
        )

    sub_scope = ExecutionScope(task_id="main", role_id="researcher")
    sub_scope.metadata[TrajectoryFileObserver.SCOPE_KEY] = str(sub_file)
    token = set_current_execution_scope(sub_scope)
    try:
        out = asyncio.run(recover_result.ainvoke({"turn": 2, "call_id": "call-7"}))
    finally:
        reset_current_execution_scope(token)

    assert "SUBAGENT-BODY" in out
    assert "COORDINATOR-BODY" not in out
