"""The JSONL trajectory must be addressable by ``(turn, tool_call_id)``.

That pair is the handle a recovery tool needs: a turn can hold several results
from the same tool because ``parallel_tool_calls`` is enabled, so the tool name
does not identify one of them.

Scope note, because the obvious reading of these tests is wrong: they show that
this observer does not clip what it is *handed*. They do NOT show that the
trajectory holds the original tool output. Sites 1 and 2 cut ``result_str``
before the ``ToolResult`` is built (``tool_exec.py:243-247``), so for those the
recorded body is already a preview. Only the site-3 cut — the post-processor at
``agent_loop.py:762``, applied after ``notify_tool_result`` — leaves the
trajectory holding something the model cannot see. See
docs/context-offloading-followups.md item 8.
"""

from __future__ import annotations

import asyncio
import json

from frontier_agent.components.observers.trajectory import (
    _BODY_MAX_CHARS,
    TrajectoryFileObserver,
)
from frontier_agent.core.loop_types import ToolResult, TurnContext


def _ctx(turn: int) -> TurnContext:
    return TurnContext(
        turn=turn, max_turns=8, task_id="task", role_id="role",
        ai_text="", thinking="", tool_calls=[], messages=[],
        usage=None, metadata={},
    )


def _results(path) -> list[dict]:
    records = [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]
    return [r for r in records if r.get("t") == "result"]


def _observer(tmp_path) -> TrajectoryFileObserver:
    return TrajectoryFileObserver(tmp_path, filename="agent", formats=["jsonl"])


def test_parallel_calls_of_one_tool_stay_distinguishable(tmp_path) -> None:
    observer = _observer(tmp_path)
    for call_id, body in (("call_a", "first"), ("call_b", "second")):
        asyncio.run(observer.on_tool_result(_ctx(3), ToolResult(
            name="search", args={}, result=body, duration_ms=1,
            tool_call_id=call_id, is_error=False,
        )))
    observer._close_jsonl()

    results = _results(tmp_path / "agent.jsonl")
    # Same turn, same tool name: only the id separates them.
    assert [r["turn"] for r in results] == [3, 3]
    assert [r["name"] for r in results] == ["search", "search"]
    by_id = {r["tool_call_id"]: r["result"] for r in results}
    assert by_id == {"call_a": "first", "call_b": "second"}


def test_the_observer_does_not_clip_what_it_is_handed(tmp_path) -> None:
    """``_BODY_MAX_CHARS`` bounds the JSON snapshot only. This says nothing
    about whether the body reaching the observer was already truncated
    upstream — see the module docstring."""
    observer = _observer(tmp_path)
    body = "x" * (_BODY_MAX_CHARS + 5_000)
    asyncio.run(observer.on_tool_result(_ctx(1), ToolResult(
        name="read_file", args={}, result=body, duration_ms=1,
        tool_call_id="call_big", is_error=False,
    )))
    observer._close_jsonl()

    (record,) = _results(tmp_path / "agent.jsonl")
    assert record["result"] == body, "the observer clipped a body it was handed"
    assert record["tool_call_id"] == "call_big"


def test_missing_runtime_id_is_recorded_empty_not_synthesised(tmp_path) -> None:
    """A synthesised id would match nothing outside the JSON snapshot, so an
    absent id is reported as absent rather than invented."""
    observer = _observer(tmp_path)
    asyncio.run(observer.on_tool_result(_ctx(2), ToolResult(
        name="bash", args={}, result="out", duration_ms=1,
        tool_call_id="", is_error=False,
    )))
    observer._close_jsonl()

    (record,) = _results(tmp_path / "agent.jsonl")
    assert record["tool_call_id"] == ""


def test_json_snapshot_pairing_is_unaffected(tmp_path) -> None:
    """The JSON branch owns a separate id-synthesis path that advances
    ``_tool_results_seen``. Recording the id in JSONL must not drive it."""
    observer = TrajectoryFileObserver(
        tmp_path, filename="agent", formats=["json", "jsonl"],
    )
    asyncio.run(observer.on_tool_result(_ctx(1), ToolResult(
        name="search", args={}, result="a", duration_ms=1,
        tool_call_id="call_a", is_error=False,
    )))
    observer._close_jsonl()

    # A result carrying its own id must not consume a synthesis slot.
    assert observer._tool_results_seen.get(1, 0) == 0
