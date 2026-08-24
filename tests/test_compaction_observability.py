"""Compaction must leave a durable trace, including what it kept.

Compaction rewrites ``messages`` in place, so a trajectory built from the
post-compaction history shows the rollup with the replaced turns already gone.
Nothing recorded what the summariser wrote, which made compaction quality
unauditable after a run: the log line reported how many tokens were freed but
never whether what survived was worth keeping.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from frontier_agent.components.observers.trajectory import TrajectoryFileObserver
from frontier_agent.core.llm import LLMResponse
from frontier_agent.core.loop_types import CompactionEvent
from frontier_agent.core.runtime.loop.tiered_compact import TieredCompactor
from frontier_agent.core.tool import tool

_SUMMARY = "The user asked about X. Sources found: https://example.com/a."


@tool
async def _probe(q: str) -> str:
    """Tool stub whose result is long enough to be worth compacting.

    Args:
        q: Query.
    """
    return "finding " * 400 + q


class _SummaryLLM:
    """Summariser that returns a known body, so the record can be checked."""

    def __init__(self, text: str = _SUMMARY) -> None:
        self.text = text
        self.calls = 0

    async def chat(self, messages, **kwargs):
        self.calls += 1
        return LLMResponse(content=self.text)


def _tool_history(body: str) -> list[dict]:
    return [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "research"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "call-1", "name": "web_fetch", "args": {}}],
        },
        {"role": "tool", "tool_call_id": "call-1", "content": body},
        {"role": "user", "content": "continue"},
    ]


# ── The compactor reports what it selected ───────────────────────────────


def test_tier1_reports_an_event_with_no_summary() -> None:
    """Tier 1 blanks tool bodies; no summariser runs, so ``summary`` is empty —
    which must not be confused with a summariser that produced nothing."""
    compactor = TieredCompactor(
        keep_tool_result=0, summary_llm=_SummaryLLM(), relief_target=10**9,
    )
    asyncio.run(compactor.compact(_tool_history("x" * 5_000), keep_recent=1))

    event = compactor.last_event
    assert event is not None
    assert event.selected == "tier1"
    assert event.summary == ""
    assert event.attempts == 0
    assert event.relief_met is True
    assert event.tokens_before > event.tokens_after


def test_tier2_reports_the_summary_it_produced() -> None:
    llm = _SummaryLLM()
    compactor = TieredCompactor(
        keep_tool_result=0, summary_llm=llm, relief_target=1,
    )
    asyncio.run(compactor.compact(_tool_history("x" * 5_000), keep_recent=1))

    assert llm.calls == 1
    event = compactor.last_event
    assert event is not None
    assert event.selected == "tier2"
    assert _SUMMARY in event.summary
    assert event.attempts == 1


def test_a_losing_tier2_does_not_attach_its_summary_to_another_tier() -> None:
    """Tier 2 can run and still lose to a cheaper candidate. The stash would
    hold its summary either way, so the event keys off the selected label."""
    llm = _SummaryLLM("y" * 40_000)   # a summary far larger than the fallbacks
    compactor = TieredCompactor(
        keep_tool_result=0, summary_llm=llm, relief_target=1,
    )
    asyncio.run(compactor.compact(_tool_history("x" * 5_000), keep_recent=1))

    event = compactor.last_event
    assert event is not None
    assert llm.calls == 1, "Tier 2 ran"
    assert event.selected != "tier2", "and lost"
    assert event.summary == "", "so its summary is not this event's"


def test_a_stale_summary_is_not_reported_on_a_later_compaction() -> None:
    """Second compaction settles on Tier 1; the first one's summary must not
    resurface as though this turn had produced it."""
    llm = _SummaryLLM()
    compactor = TieredCompactor(
        keep_tool_result=0, summary_llm=llm, relief_target=1,
    )
    asyncio.run(compactor.compact(_tool_history("x" * 5_000), keep_recent=1))
    assert compactor.last_event is not None
    assert compactor.last_event.selected == "tier2"

    compactor._relief_target = 10**9        # now Tier 1 alone is enough
    asyncio.run(compactor.compact(_tool_history("x" * 5_000), keep_recent=1))
    assert compactor.last_event is not None
    assert compactor.last_event.selected == "tier1"
    assert compactor.last_event.summary == ""


# ── The observer writes it down ──────────────────────────────────────────


def _compaction_records(path) -> list[dict]:
    return [
        record
        for line in path.read_text().splitlines() if line.strip()
        for record in [json.loads(line)]
        if record.get("t") == "compaction"
    ]


def test_the_observer_writes_the_summary_whole(tmp_path) -> None:
    observer = TrajectoryFileObserver(
        tmp_path, filename="agent", formats=["jsonl"],
    )
    long_summary = "s" * 40_000
    asyncio.run(observer.on_compaction(CompactionEvent(
        turn=7, seq=2, selected="tier2", tokens_before=9_000,
        tokens_after=3_000, relief_met=True, spill_refs=4,
        attempts=3, summary=long_summary,
    )))
    observer._close_jsonl()

    (record,) = _compaction_records(tmp_path / "agent.jsonl")
    assert record["turn"] == 7
    assert record["seq"] == 2
    assert record["selected"] == "tier2"
    assert record["tokens_before"] == 9_000
    assert record["tokens_after"] == 3_000
    assert record["relief_met"] is True
    assert record["spill_refs"] == 4
    assert record["attempts"] == 3
    # Not clipped: the body bound exists for tool output, not for this.
    assert record["summary"] == long_summary


def test_observers_without_the_hook_are_unaffected() -> None:
    """The hook is opt-in; ``notify_observers`` skips observers lacking it."""
    from frontier_agent.core.loop_types import notify_observers

    class _Bare:
        critical = False

    event = CompactionEvent(
        turn=1, seq=1, selected="tier1", tokens_before=10,
        tokens_after=5, relief_met=True, spill_refs=0,
    )
    assert asyncio.run(notify_observers([_Bare()], "on_compaction", event)) == []


@pytest.mark.asyncio
async def test_a_failing_hook_does_not_break_the_run(tmp_path) -> None:
    """Compaction has already happened when the hook fires, so a broken
    observer must not be able to take the loop down with it."""
    from frontier_agent.core.loop_types import notify_observers

    class _Broken:
        critical = False

        async def on_compaction(self, event):
            raise RuntimeError("boom")

    event = CompactionEvent(
        turn=1, seq=1, selected="tier1", tokens_before=10,
        tokens_after=5, relief_met=True, spill_refs=0,
    )
    assert await notify_observers([_Broken()], "on_compaction", event) == []


# ── The loop connects the two ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_the_loop_stamps_turn_and_sequence_and_notifies() -> None:
    """The compactor knows neither the turn nor how many compactions preceded
    it, so the loop stamps both before broadcasting.

    The stub calls a tool every turn on purpose: compaction runs at turn end,
    and a turn that answers with ``finish_reason="stop"`` returns before ever
    getting there.
    """
    from frontier_agent.core.loop_types import LoopConfig
    from frontier_agent.core.runtime.loop.agent_loop import run_agent_loop

    class _AlwaysCompact:
        def should_compact(self, turn, messages, estimated_tokens):
            return True

    class _Recorder:
        critical = False

        def __init__(self) -> None:
            self.events: list[CompactionEvent] = []

        async def on_compaction(self, event: CompactionEvent) -> None:
            # Copied: the compactor reuses one mutable event object per call.
            self.events.append(CompactionEvent(**vars(event)))

    class _SearchingLLM:
        model = "stub"

        def __init__(self) -> None:
            self.calls = 0

        async def chat(self, messages, **kwargs):
            del messages, kwargs
            self.calls += 1
            return LLMResponse(
                content="",
                tool_calls=[{
                    "id": f"call_{self.calls}",
                    "type": "function",
                    "function": {
                        "name": "_probe",
                        "arguments": json.dumps({"q": "x" * 40}),
                    },
                }],
                finish_reason="tool_calls",
            )

    recorder = _Recorder()
    compactor = TieredCompactor(
        keep_tool_result=0, summary_llm=_SummaryLLM(), relief_target=10**9,
    )
    await run_agent_loop(
        system_prompt="system",
        user_message="go",
        llm=_SearchingLLM(),
        tools=[_probe],
        config=LoopConfig(
            max_turns=3,
            compactor=compactor,
            compaction_policy=_AlwaysCompact(),
            keep_recent=1,
        ),
        observers=[recorder],
    )

    assert recorder.events, "the loop never reported a compaction"
    assert [e.seq for e in recorder.events] == list(
        range(1, len(recorder.events) + 1),
    ), "sequence must count compactions, starting at 1"
    assert all(e.turn >= 1 for e in recorder.events)
    assert all(e.selected for e in recorder.events)
    # Distinct turns, in order — the stamp is the loop's, not a constant.
    assert [e.turn for e in recorder.events] == sorted(
        {e.turn for e in recorder.events},
    )


@pytest.mark.asyncio
async def test_a_compactor_without_an_event_is_simply_not_reported() -> None:
    """``last_event`` is opt-in; the default compactor exposes none and must not
    make the loop raise."""
    from frontier_agent.core.loop_types import LoopConfig
    from frontier_agent.core.runtime.loop.agent_loop import run_agent_loop

    class _AlwaysCompact:
        def should_compact(self, turn, messages, estimated_tokens):
            return True

    class _Recorder:
        critical = False

        def __init__(self) -> None:
            self.events: list[CompactionEvent] = []

        async def on_compaction(self, event: CompactionEvent) -> None:
            self.events.append(event)

    class _PlainCompactor:
        def compact(self, messages, keep_recent):
            return messages

    class _ProbingLLM:
        model = "stub"

        async def chat(self, messages, **kwargs):
            del messages, kwargs
            return LLMResponse(
                content="",
                tool_calls=[{
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "_probe",
                        "arguments": json.dumps({"q": "x"}),
                    },
                }],
                finish_reason="tool_calls",
            )

    recorder = _Recorder()
    await run_agent_loop(
        system_prompt="system",
        user_message="go",
        llm=_ProbingLLM(),
        tools=[_probe],
        config=LoopConfig(
            max_turns=1,
            compactor=_PlainCompactor(),
            compaction_policy=_AlwaysCompact(),
        ),
        observers=[recorder],
    )
    assert recorder.events == []


def test_a_record_arriving_after_the_stream_closed_still_lands(tmp_path) -> None:
    """``on_compaction`` is passive, so it runs as a background task, and
    ``on_loop_end`` — which closes the stream — is scheduled the same way. Both
    end up in one ``drain_background_observers`` gather with no ordering between
    them, so a final-turn compaction can arrive after the close.
    """
    observer = TrajectoryFileObserver(
        tmp_path, filename="agent", formats=["jsonl"],
    )
    asyncio.run(observer.on_compaction(CompactionEvent(
        turn=1, seq=1, selected="tier2", tokens_before=9, tokens_after=3,
        relief_met=True, spill_refs=0, summary="first",
    )))
    observer._close_jsonl()
    asyncio.run(observer.on_compaction(CompactionEvent(
        turn=2, seq=2, selected="tier1", tokens_before=9, tokens_after=3,
        relief_met=True, spill_refs=0, summary="",
    )))
    observer._close_jsonl()

    records = _compaction_records(tmp_path / "agent.jsonl")
    assert [r["seq"] for r in records] == [1, 2], "the late record was dropped"


# ── A failed summariser is not the same as no summariser ─────────────────


def _long_history(n_turns: int, body_len: int = 800) -> list[dict]:
    """History where the slice a failed summariser falls back to genuinely wins.

    Tier 1 blanks tool bodies but keeps every message, including the assistant
    reasoning; the slice drops the whole middle. Past a few turns the slice is
    the smaller of the two, so the rollback is selected under the ``tier2``
    label — which is the only way ``rollback_reason`` is ever reached.
    """
    messages: list[dict] = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "research"},
    ]
    for i in range(n_turns):
        messages.append({
            "role": "assistant",
            "content": "reasoning " * 60,
            "tool_calls": [{"id": f"c{i}", "name": "web_fetch", "args": {}}],
        })
        messages.append({
            "role": "tool", "tool_call_id": f"c{i}", "content": "x" * body_len,
        })
    messages.append({"role": "user", "content": "continue"})
    return messages


def test_a_rolled_back_summariser_is_distinguishable_from_none() -> None:
    class _FailingLLM:
        def __init__(self) -> None:
            self.calls = 0

        async def chat(self, messages, **kwargs):
            self.calls += 1
            raise RuntimeError("summariser down")

    llm = _FailingLLM()
    compactor = TieredCompactor(
        keep_tool_result=0, summary_llm=llm, relief_target=1,
    )
    asyncio.run(compactor.compact(_long_history(10), keep_recent=1))

    assert llm.calls == 1, "the summariser ran"
    event = compactor.last_event
    assert event is not None
    assert event.selected == "tier2", "the rollback slice won"
    assert event.summary == ""
    assert event.rollback_reason == "llm_error", (
        "a failed summariser must not look like one that never ran"
    )


def test_an_empty_summary_rolls_back_with_its_own_reason() -> None:
    class _EmptyLLM:
        async def chat(self, messages, **kwargs):
            return LLMResponse(content="")

    compactor = TieredCompactor(
        keep_tool_result=0, summary_llm=_EmptyLLM(), relief_target=1,
    )
    asyncio.run(compactor.compact(_long_history(10), keep_recent=1))

    event = compactor.last_event
    assert event is not None
    assert event.selected == "tier2"
    assert event.summary == ""
    assert event.rollback_reason == "empty_summary"


def test_a_successful_summary_carries_no_rollback_reason() -> None:
    compactor = TieredCompactor(
        keep_tool_result=0, summary_llm=_SummaryLLM(), relief_target=1,
    )
    asyncio.run(compactor.compact(_long_history(10), keep_recent=1))

    event = compactor.last_event
    assert event is not None
    assert event.selected == "tier2"
    assert _SUMMARY in event.summary
    assert event.rollback_reason == ""


def test_tiers_that_never_summarise_carry_no_rollback_reason() -> None:
    compactor = TieredCompactor(
        keep_tool_result=0, summary_llm=_SummaryLLM(), relief_target=10**9,
    )
    asyncio.run(compactor.compact(_tool_history("x" * 5_000), keep_recent=1))

    event = compactor.last_event
    assert event is not None
    assert event.selected == "tier1"
    assert event.summary == ""
    assert event.rollback_reason == ""
