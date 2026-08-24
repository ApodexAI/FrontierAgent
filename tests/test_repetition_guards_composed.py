"""What the three repetition guards do when mounted together.

The question a downstream reviewer asked before running this on 800 trials:
each guard is covered in isolation, but the sub-agent stack mounts all three,
and only their *composition* decides whether a looping agent terminates.

The model here is deliberately the pathological one measured in production: a
byte-identical tool call every turn (reproduced 4/4 at temperature 0.0, 0.3 and
0.7 against a real endpoint), with its deliberation in the reasoning channel and
no visible prose. That last detail is what makes composition matter — see
``test_the_text_guard_alone_cannot_end_this_loop``.
"""

from __future__ import annotations

import json

import pytest

from frontier_agent.components.observers.duplicate_query_rollback import (
    DuplicateQueryRollbackObserver,
)
from frontier_agent.components.observers.repetition_guard import (
    REPEATED_TOOL_CALLS_STOP_REASON,
    RepetitionGuard,
)
from frontier_agent.components.observers.text_repetition_guard import (
    TextRepetitionGuard,
)
from frontier_agent.core.llm import LLMResponse
from frontier_agent.core.loop_types import LoopConfig, LoopPolicy
from frontier_agent.core.runtime.loop.agent_loop import run_agent_loop
from frontier_agent.core.tool import tool

_QUERY = "who is the doctor youtuber born june 24"


@tool
async def _web_search(q: str) -> str:
    """Search stub.

    Args:
        q: Query.
    """
    return f"## Search Results\n[1] **irrelevant hit** — https://example.test/{len(q)}"


class _StuckLLM:
    """Emits one identical tool call forever, reasoning only, no visible text."""

    model = "stuck"

    def __init__(self) -> None:
        self.calls = 0

    async def chat(self, messages: list, **kwargs: object) -> LLMResponse:
        del messages, kwargs
        self.calls += 1
        return LLMResponse(
            content="",
            reasoning_content=(
                "The searches are not yielding the answer. Let me think about "
                "this differently. " * 40
            ),
            tool_calls=[{
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": "_web_search",
                    "arguments": json.dumps({"q": _QUERY}),
                },
            }],
            finish_reason="tool_calls",
        )


def _sub_agent_stack() -> list[object]:
    """The observers an agent-team sub-agent actually mounts, in the same order."""
    return [
        RepetitionGuard(stop_after=6),
        TextRepetitionGuard(enable_stop=True),
        DuplicateQueryRollbackObserver(tool_names={"_web_search"}),
    ]


async def _run(observers: list[object], *, max_turns: int = 60):
    return await run_agent_loop(
        system_prompt="test",
        user_message="research this",
        llm=_StuckLLM(),
        tools=[_web_search],
        config=LoopConfig(
            max_turns=max_turns,
            loop_policy=LoopPolicy(
                terminal_tool_names=("submit_report",),
                no_tool_behavior="nudge",
            ),
        ),
        observers=observers,
    )


@pytest.mark.asyncio
async def test_the_composed_stack_terminates_a_deterministic_loop() -> None:
    result = await _run(_sub_agent_stack())

    assert result.stopped_by == REPEATED_TOOL_CALLS_STOP_REASON
    # Well inside the budget: the rollback absorbs the first few repeats
    # without spending turns, then the stop lands.
    assert result.turns_used < 12
    assert result.tool_calls_count < 12


@pytest.mark.asyncio
async def test_without_a_stop_capable_guard_the_loop_runs_to_max_turns() -> None:
    """The gap this composition closes.

    The rollback exhausts ``max_consecutive_rollbacks`` and then lets every
    further duplicate through permanently, and a hint-only RepetitionGuard
    never ends anything — so the run burns its whole budget.
    """
    result = await _run([
        RepetitionGuard(),
        TextRepetitionGuard(enable_stop=True),
        DuplicateQueryRollbackObserver(tool_names={"_web_search"}),
    ], max_turns=20)

    assert result.stopped_by == "max_turns"
    assert result.turns_used == 20


@pytest.mark.asyncio
async def test_the_text_guard_alone_cannot_end_this_loop() -> None:
    """Why the tool-channel guard has to be the one that stops.

    TextRepetitionGuard compares ``ai_text``. This model's repetition is all
    in the reasoning channel, so there is nothing for it to fingerprint and it
    never fires, however long the loop runs.
    """
    result = await _run([TextRepetitionGuard(enable_stop=True)], max_turns=15)

    assert result.stopped_by == "max_turns"


@pytest.mark.asyncio
async def test_the_rollback_alone_only_delays_the_loop() -> None:
    result = await _run(
        [DuplicateQueryRollbackObserver(tool_names={"_web_search"})],
        max_turns=15,
    )

    assert result.stopped_by == "max_turns"
    assert result.turns_used == 15
