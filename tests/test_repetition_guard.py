"""Coverage for RepetitionGuard — identical tool calls on consecutive turns."""

from __future__ import annotations

import pytest

from frontier_agent.components.observers.repetition_guard import (
    REPEATED_TOOL_CALLS_STOP_REASON,
    RepetitionGuard,
)
from frontier_agent.core.loop_types import Intervention, LoopConfig, TurnContext


def _ctx(turn: int, tool_calls: list[dict]) -> TurnContext:
    return TurnContext(
        turn=turn,
        max_turns=100,
        task_id="task",
        role_id="role",
        ai_text="",
        thinking="",
        tool_calls=tool_calls,
        messages=[],
        usage=None,
        metadata={},
    )


def _fetch(url: str = "https://example.test/dead") -> dict:
    return {"name": "web_fetch", "args": {"url": url}}


@pytest.mark.asyncio
async def test_hint_fires_on_the_third_identical_turn() -> None:
    guard = RepetitionGuard()

    assert await guard.on_turn_end(_ctx(1, [_fetch()])) is None
    assert await guard.on_turn_end(_ctx(2, [_fetch()])) is None

    intervention = await guard.on_turn_end(_ctx(3, [_fetch()]))
    assert intervention is not None
    assert intervention.inject_messages
    assert "web_fetch" in intervention.inject_messages[0]
    # Hint-only: the tool call itself still runs.
    assert intervention.stop_reason is None
    assert not intervention.skip_tool_execution


@pytest.mark.asyncio
async def test_hint_fires_once_per_streak_not_every_later_turn() -> None:
    guard = RepetitionGuard()

    for turn in (1, 2):
        await guard.on_turn_end(_ctx(turn, [_fetch()]))
    assert await guard.on_turn_end(_ctx(3, [_fetch()])) is not None
    assert await guard.on_turn_end(_ctx(4, [_fetch()])) is None
    assert await guard.on_turn_end(_ctx(5, [_fetch()])) is None
    assert await guard.on_turn_end(_ctx(6, [_fetch()])) is not None


@pytest.mark.asyncio
async def test_differing_arguments_break_the_streak() -> None:
    guard = RepetitionGuard()

    await guard.on_turn_end(_ctx(1, [_fetch()]))
    await guard.on_turn_end(_ctx(2, [_fetch()]))
    await guard.on_turn_end(_ctx(3, [_fetch("https://example.test/other")]))
    assert await guard.on_turn_end(_ctx(4, [_fetch()])) is None


@pytest.mark.asyncio
async def test_a_tool_free_turn_breaks_the_streak() -> None:
    guard = RepetitionGuard()

    await guard.on_turn_end(_ctx(1, [_fetch()]))
    await guard.on_turn_end(_ctx(2, [_fetch()]))
    assert await guard.on_turn_end(_ctx(3, [])) is None
    assert await guard.on_turn_end(_ctx(4, [_fetch()])) is None


@pytest.mark.asyncio
async def test_the_whole_batch_is_the_signature() -> None:
    """A turn calling A+B differs from one calling A alone."""
    guard = RepetitionGuard()
    batch = [_fetch(), {"name": "read_file", "args": {"path": "/tmp/x"}}]

    await guard.on_turn_end(_ctx(1, batch))
    await guard.on_turn_end(_ctx(2, [_fetch()]))
    assert await guard.on_turn_end(_ctx(3, batch)) is None

    # Turns 3-5 are then three identical batches in a row.
    await guard.on_turn_end(_ctx(4, batch))
    intervention = await guard.on_turn_end(_ctx(5, batch))
    assert intervention is not None
    assert "read_file, web_fetch" in intervention.inject_messages[0]


@pytest.mark.asyncio
async def test_unserialisable_arguments_do_not_raise() -> None:
    circular: dict[str, object] = {}
    circular["self"] = circular
    call = {"name": "bash", "args": circular}
    guard = RepetitionGuard()

    for turn in (1, 2):
        assert await guard.on_turn_end(_ctx(turn, [call])) is None
    assert await guard.on_turn_end(_ctx(3, [call])) is not None


@pytest.mark.asyncio
async def test_loop_start_clears_state_and_threshold_has_a_floor() -> None:
    guard = RepetitionGuard(threshold=1)
    assert guard.threshold == 2

    await guard.on_turn_end(_ctx(1, [_fetch()]))
    await guard.on_loop_start(LoopConfig())
    assert await guard.on_turn_end(_ctx(1, [_fetch()])) is None


# ── stop_after: the only guard that can end the duplicate-search loop ─


@pytest.mark.asyncio
async def test_stop_after_ends_the_streak_and_is_off_by_default() -> None:
    hint_only = RepetitionGuard()
    stopping = RepetitionGuard(stop_after=4)

    for turn in range(1, 9):
        assert (await hint_only.on_turn_end(_ctx(turn, [_fetch()]))
                or Intervention()).stop_reason is None

    outcomes = [await stopping.on_turn_end(_ctx(t, [_fetch()])) for t in (1, 2, 3, 4)]
    assert outcomes[0] is None
    assert outcomes[1] is None
    assert outcomes[2] is not None and outcomes[2].inject_messages
    assert outcomes[3] is not None
    assert outcomes[3].stop_reason == REPEATED_TOOL_CALLS_STOP_REASON


@pytest.mark.asyncio
async def test_stop_after_never_precedes_the_first_hint() -> None:
    """A stop_after below threshold is raised so the model is warned first."""
    guard = RepetitionGuard(threshold=3, stop_after=2)
    assert guard.stop_after == 4

    assert await guard.on_turn_end(_ctx(1, [_fetch()])) is None
    assert await guard.on_turn_end(_ctx(2, [_fetch()])) is None
    third = await guard.on_turn_end(_ctx(3, [_fetch()]))
    assert third is not None
    assert third.inject_messages and third.stop_reason is None
    fourth = await guard.on_turn_end(_ctx(4, [_fetch()]))
    assert fourth is not None and fourth.stop_reason == REPEATED_TOOL_CALLS_STOP_REASON


def test_the_stop_reason_is_rescued_by_fan_in() -> None:
    """An unlisted reason classifies as incomplete but loses force_final_answer."""
    from frontier_agent.components.agent_bus.fan_in import (
        _INCOMPLETE_NOTES,
        INCOMPLETE_STOP_REASONS,
    )

    assert REPEATED_TOOL_CALLS_STOP_REASON in INCOMPLETE_STOP_REASONS
    assert REPEATED_TOOL_CALLS_STOP_REASON in _INCOMPLETE_NOTES
