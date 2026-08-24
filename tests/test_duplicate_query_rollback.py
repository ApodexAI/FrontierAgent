"""Coverage for DuplicateQueryRollbackObserver and the rollback pop it relies on."""

from __future__ import annotations

import json

import pytest

from frontier_agent.components.observers.duplicate_query_rollback import (
    DuplicateQueryRollbackObserver,
)
from frontier_agent.core.llm import LLMResponse
from frontier_agent.core.loop_types import (
    Intervention,
    LoopConfig,
    LoopPolicy,
    ToolResult,
    TurnContext,
)
from frontier_agent.core.messages import assistant_msg, tool_msg
from frontier_agent.core.runtime.loop.agent_loop import (
    _pop_last_assistant_turn,
    run_agent_loop,
)
from frontier_agent.core.tool import tool


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


def _search(*queries: str, **extra: object) -> dict:
    q: object = list(queries) if len(queries) != 1 else queries[0]
    return {"name": "web_search", "args": {"q": q, **extra}}


_OK_RESULT = "## Search Results\n[1] **a** — https://example.test/a"


def _result(tool_call: dict, *, text: str = _OK_RESULT) -> ToolResult:
    return ToolResult(
        name=str(tool_call["name"]),
        args=dict(tool_call["args"]),
        result=text,
        duration_ms=1,
        tool_call_id="call_1",
        is_error=False,
    )


async def _turn(
    guard: DuplicateQueryRollbackObserver,
    turn: int,
    tool_calls: list[dict],
    *,
    result_text: str = _OK_RESULT,
) -> Intervention | None:
    """Drive one whole turn in loop order: LLM response, then tool results.

    Bookkeeping lives in ``on_tool_result``, so a test that only drove
    ``on_llm_response`` would never record anything. The loop reaches the
    results hook only for a turn that was not rolled back, which is what
    makes "a failed search is not remembered" expressible.
    """
    ctx = _ctx(turn, tool_calls)
    intervention = await guard.on_llm_response(ctx)
    if intervention is not None:
        return intervention
    for tool_call in tool_calls:
        await guard.on_tool_result(ctx, _result(tool_call, text=result_text))
    return None


@pytest.mark.asyncio
async def test_repeat_of_an_executed_query_is_rolled_back() -> None:
    guard = DuplicateQueryRollbackObserver()

    assert await _turn(guard, 1, [_search("qing dynasty seals")]) is None

    intervention = await guard.on_llm_response(
        _ctx(2, [_search("qing dynasty seals")]),
    )
    assert intervention is not None
    assert intervention.pop_last_message
    assert intervention.continue_to_next_turn


@pytest.mark.asyncio
async def test_a_new_query_passes_and_rearms_the_budget() -> None:
    guard = DuplicateQueryRollbackObserver()

    await _turn(guard, 1, [_search("a")])
    assert await _turn(guard, 2, [_search("a")]) is not None
    # A clean turn resets the streak…
    assert await _turn(guard, 3, [_search("b")]) is None
    # …and detection is still armed for the next repeat.
    assert await _turn(guard, 4, [_search("b")]) is not None


@pytest.mark.asyncio
async def test_batch_composition_is_the_dedup_unit() -> None:
    """``[a, b]`` and ``[a, c]`` are different searches — query evolution."""
    guard = DuplicateQueryRollbackObserver()

    await _turn(guard, 1, [_search("a", "b")])
    assert await _turn(guard, 2, [_search("a", "c")]) is None
    assert await _turn(guard, 3, [_search("a", "b")]) is not None


@pytest.mark.asyncio
async def test_pagination_is_not_a_duplicate() -> None:
    """The aligned web_search exposes ``page``; paging is a new result set."""
    guard = DuplicateQueryRollbackObserver()

    await _turn(guard, 1, [_search("a", page=1)])
    assert await _turn(guard, 2, [_search("a", page=2)]) is None
    assert await _turn(guard, 3, [_search("a", page=2)]) is not None


@pytest.mark.asyncio
async def test_row_count_alone_is_not_a_new_search() -> None:
    guard = DuplicateQueryRollbackObserver()

    await _turn(guard, 1, [_search("a", num_results=10)])
    assert await guard.on_llm_response(
        _ctx(2, [_search("a", num_results=50)]),
    ) is not None


@pytest.mark.asyncio
async def test_budget_lets_a_duplicate_through_and_stays_latched() -> None:
    guard = DuplicateQueryRollbackObserver(max_consecutive_rollbacks=3)

    await _turn(guard, 1, [_search("a")])
    assert await _turn(guard, 2, [_search("a")]) is not None
    assert await _turn(guard, 3, [_search("a")]) is not None
    # Budget spent: the duplicate is allowed through instead of livelocking.
    assert await _turn(guard, 4, [_search("a")]) is None
    # Still latched — a let-through is not a clean turn.
    assert await _turn(guard, 5, [_search("a")]) is None
    # One genuinely clean turn re-arms it.
    assert await _turn(guard, 6, [_search("new")]) is None
    assert await _turn(guard, 7, [_search("a")]) is not None


@pytest.mark.asyncio
async def test_untracked_tools_and_empty_turns_are_ignored() -> None:
    guard = DuplicateQueryRollbackObserver()
    fetch = {"name": "web_fetch", "args": {"url": "https://example.test"}}

    assert await _turn(guard, 1, [fetch]) is None
    # Retrying the same fetch is legitimate after a transient render failure.
    assert await _turn(guard, 2, [fetch]) is None
    assert await _turn(guard, 3, []) is None
    # A malformed search (no usable ``q``) is untrackable, not a duplicate.
    blank = {"name": "web_search", "args": {"q": "   "}}
    assert await _turn(guard, 4, [blank]) is None
    assert await _turn(guard, 5, [blank]) is None


@pytest.mark.asyncio
async def test_loop_start_clears_state_between_runs() -> None:
    guard = DuplicateQueryRollbackObserver()

    await _turn(guard, 1, [_search("a")])
    await guard.on_loop_start(LoopConfig())
    assert await _turn(guard, 1, [_search("a")]) is None


# ── Review findings: what must NOT be rolled back or remembered ──────


@pytest.mark.asyncio
async def test_a_batch_carrying_a_terminal_tool_is_never_popped() -> None:
    """Popping would discard the report: the loop returns before tool exec."""
    guard = DuplicateQueryRollbackObserver()
    report = {"name": "submit_report", "args": {"report": "Scope/Finding"}}

    await _turn(guard, 1, [_search("a")])
    assert await _turn(guard, 2, [_search("a"), report]) is None
    # And the let-through is not silently recorded as a fresh execution
    # either — a later solo repeat is still caught.
    assert await _turn(guard, 3, [_search("a")]) is not None


@pytest.mark.asyncio
async def test_the_terminal_set_follows_the_loop_policy() -> None:
    guard = DuplicateQueryRollbackObserver()
    await guard.on_loop_start(
        LoopConfig(loop_policy=LoopPolicy(terminal_tool_names=("wrap_up",))),
    )
    wrap_up = {"name": "wrap_up", "args": {}}

    await _turn(guard, 1, [_search("a")])
    assert await _turn(guard, 2, [_search("a"), wrap_up]) is None


@pytest.mark.parametrize(
    "failed_text",
    [
        "[ERROR]: SERPER_API_KEY environment variable not set.",
        "[ERROR]: Unexpected error: connection reset",
        "No search results found.",
        "No results found for: qing dynasty seals",
        "",
    ],
)
@pytest.mark.asyncio
async def test_a_failed_search_is_not_remembered(failed_text: str) -> None:
    """A transient upstream failure is the one case where a retry is right."""
    guard = DuplicateQueryRollbackObserver()

    await _turn(guard, 1, [_search("a")], result_text=failed_text)

    assert await _turn(guard, 2, [_search("a")]) is None
    # The retry that DID return content is remembered.
    assert await _turn(guard, 3, [_search("a")]) is not None


@pytest.mark.asyncio
async def test_an_errored_tool_result_is_not_remembered() -> None:
    guard = DuplicateQueryRollbackObserver()
    call = _search("a")
    ctx = _ctx(1, [call])
    await guard.on_llm_response(ctx)
    failed = _result(call)
    failed.is_error = True
    await guard.on_tool_result(ctx, failed)

    assert await _turn(guard, 2, [_search("a")]) is None


@pytest.mark.asyncio
async def test_a_stringified_query_array_keys_like_a_real_one() -> None:
    """Models emit the same batch in both encodings; the tool coerces, so we must."""
    guard = DuplicateQueryRollbackObserver()
    as_list = {"name": "web_search", "args": {"q": ["alpha", "beta"]}}
    as_str = {"name": "web_search", "args": {"q": '["alpha", "beta"]'}}

    await _turn(guard, 1, [as_list])
    assert await _turn(guard, 2, [as_str]) is not None

    guard2 = DuplicateQueryRollbackObserver()
    await _turn(guard2, 1, [as_str])
    assert await _turn(guard2, 2, [as_list]) is not None


@pytest.mark.asyncio
async def test_a_plain_string_query_is_untouched_by_the_coercion() -> None:
    guard = DuplicateQueryRollbackObserver()
    bracketless = {"name": "web_search", "args": {"q": "who wrote [sic]"}}

    await _turn(guard, 1, [bracketless])
    assert await _turn(guard, 2, [bracketless]) is not None


# ── The pop primitive the rollback depends on ────────────────────────


def test_pop_removes_the_assistant_message() -> None:
    messages = [assistant_msg("keep"), assistant_msg("drop")]

    _pop_last_assistant_turn(messages)

    assert [m["content"] for m in messages] == ["keep"]


def test_pop_takes_trailing_tool_answers_with_the_assistant_message() -> None:
    """A capped / dropped tool call answers itself AFTER the assistant message.

    Popping one message there would strip the answer and leave an unanswered
    ``tool_call_id`` behind, which providers reject.
    """
    messages = [
        assistant_msg("earlier"),
        assistant_msg("rejected turn"),
        tool_msg("[tool call skipped] over the per-turn cap", "call_1"),
    ]

    _pop_last_assistant_turn(messages)

    assert [m["content"] for m in messages] == ["earlier"]


# ── Loop-level: the rollback actually breaks the repeat ──────────────


@tool
async def _fake_search(q: str) -> str:
    """Search stub.

    Args:
        q: Query.
    """
    return f"results for {q}"


def _tool_call_response(query: str) -> LLMResponse:
    return LLMResponse(
        content="",
        tool_calls=[{
            "id": f"call_{query}",
            "type": "function",
            "function": {
                "name": "_fake_search",
                "arguments": json.dumps({"q": query}),
            },
        }],
        finish_reason="tool_calls",
    )


class _ScriptedLLM:
    """Emits the same query forever, then one different query."""

    model = "scripted"

    def __init__(self, repeats: int) -> None:
        self._remaining_repeats = repeats
        self.calls = 0

    async def chat(self, messages: list, **kwargs: object) -> LLMResponse:
        del messages, kwargs
        self.calls += 1
        if self._remaining_repeats > 0:
            self._remaining_repeats -= 1
            return _tool_call_response("same query")
        return LLMResponse(content="final answer", finish_reason="stop")


@pytest.mark.asyncio
async def test_loop_rolls_back_repeats_without_spending_turns() -> None:
    llm = _ScriptedLLM(repeats=4)

    result = await run_agent_loop(
        system_prompt="test",
        user_message="research this",
        llm=llm,
        tools=[_fake_search],
        config=LoopConfig(
            max_turns=6,
            loop_policy=LoopPolicy(no_tool_behavior="stop"),
        ),
        observers=[DuplicateQueryRollbackObserver(tool_names={"_fake_search"})],
    )

    # The first search runs; the three repeats are popped before their tool
    # call, so only one tool call is ever executed.
    assert result.tool_calls_count == 1
    assert llm.calls == 5
    # Rolled-back turns don't consume the max_turns budget, so the run still
    # reaches its plain-text answer well inside six turns.
    assert result.stopped_by == "no_tool"
    assert result.final_content == "final answer"
    assert result.turns_used == 2


@pytest.mark.asyncio
async def test_without_the_guard_the_same_repeats_burn_every_turn() -> None:
    """Control group: the pathology this observer exists to stop."""
    llm = _ScriptedLLM(repeats=4)

    result = await run_agent_loop(
        system_prompt="test",
        user_message="research this",
        llm=llm,
        tools=[_fake_search],
        config=LoopConfig(
            max_turns=4,
            loop_policy=LoopPolicy(no_tool_behavior="stop"),
        ),
    )

    assert result.tool_calls_count == 4
    assert result.stopped_by == "max_turns"
    assert result.final_content != "final answer"
