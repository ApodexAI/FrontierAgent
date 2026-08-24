"""A streamed tool-call that never received a name must not reach history.

``_stream_llm_response`` creates an accumulator slot for ANY tool-call delta
carrying an index, pre-filled with ``name=""``. Two real cases leave it empty:
a content-free chunk that merely opens a tool-call block, and a stream cut
before the name delta arrives.

Emitting such a call is worse than dropping it. It cannot be executed, and the
loop records it in durable history as ``name=""`` — after which some chat
templates fail to render that history at all (``can only concatenate str (not
"NoneType") to str``, HTTP 400 from apodex-1.1). Every later request in the
session replays the same history and is rejected identically, so one malformed
delta ends the run, and ends it looking like an ordinary empty submission
rather than the infrastructure fault it is.
"""
from __future__ import annotations

import logging

from frontier_agent.core.llm import StreamDelta
from frontier_agent.core.runtime.loop._streaming import _stream_llm_response


class _FakeLLM:
    """Yields a fixed StreamDelta script, like a provider adapter would."""

    def __init__(self, deltas: list[StreamDelta]) -> None:
        self._deltas = deltas

    async def stream(self, messages, timeout=None):
        for delta in self._deltas:
            yield delta


async def _noop(*args, **kwargs) -> None:
    return None


async def _run(deltas: list[StreamDelta]):
    return await _stream_llm_response(
        _FakeLLM(deltas), messages=[], timeout=30, on_delta=_noop,
    )


async def test_index_only_chunk_does_not_become_a_tool_call() -> None:
    """The content-free chunk that opens a tool-call block carries no name."""
    resp = await _run([
        StreamDelta(tool_call_deltas=[{"index": 0}]),
        StreamDelta(content="thinking about it"),
    ])
    assert resp.tool_calls == []
    assert resp.content == "thinking about it"


async def test_stream_cut_before_the_name_arrives_is_dropped() -> None:
    """Arguments streamed, then the stream ended — the call is unexecutable."""
    resp = await _run([
        StreamDelta(tool_call_deltas=[{"index": 0, "id": "call_1"}]),
        StreamDelta(tool_call_deltas=[{"index": 0, "arguments": '{"command":'}]),
    ])
    assert resp.tool_calls == []


async def test_a_named_call_still_survives() -> None:
    resp = await _run([
        StreamDelta(tool_call_deltas=[
            {"index": 0, "id": "call_1", "name": "bash"},
        ]),
        StreamDelta(tool_call_deltas=[
            {"index": 0, "arguments": '{"command": "ls"}'},
        ]),
    ])
    assert len(resp.tool_calls) == 1
    assert resp.tool_calls[0]["function"]["name"] == "bash"
    assert resp.tool_calls[0]["function"]["arguments"] == '{"command": "ls"}'
    assert resp.tool_calls[0]["id"] == "call_1"


async def test_a_named_call_to_a_nonexistent_tool_is_kept() -> None:
    """Deliberate: it reaches the executor and comes back as "unknown tool",
    which the model can read and correct. Only NAMELESS calls are dropped."""
    resp = await _run([
        StreamDelta(tool_call_deltas=[
            {"index": 0, "id": "c1", "name": "no_such_tool", "arguments": "{}"},
        ]),
    ])
    assert [tc["function"]["name"] for tc in resp.tool_calls] == ["no_such_tool"]


async def test_only_the_nameless_slot_is_dropped_from_a_mixed_turn() -> None:
    """A real parallel-tool turn must not lose its good calls to a bad sibling."""
    resp = await _run([
        StreamDelta(tool_call_deltas=[
            {"index": 0, "id": "c0", "name": "bash", "arguments": "{}"},
            {"index": 1},
            {"index": 2, "id": "c2", "name": "read_file", "arguments": "{}"},
        ]),
    ])
    assert [tc["function"]["name"] for tc in resp.tool_calls] == ["bash", "read_file"]


async def test_the_drop_is_logged_not_silent(caplog) -> None:
    """A provider emitting these consistently is an upstream defect, and this
    is the only place left that can see it."""
    with caplog.at_level(logging.WARNING):
        await _run([StreamDelta(tool_call_deltas=[{"index": 0}])])
    assert any("no function name" in r.getMessage() for r in caplog.records)
    assert any("dropped 1 " in r.getMessage() for r in caplog.records)


async def test_a_clean_turn_logs_nothing(caplog) -> None:
    with caplog.at_level(logging.WARNING):
        resp = await _run([
            StreamDelta(tool_call_deltas=[{"index": 0, "id": "c", "name": "bash"}]),
        ])
    assert len(resp.tool_calls) == 1
    assert not [r for r in caplog.records if "function name" in r.getMessage()]
