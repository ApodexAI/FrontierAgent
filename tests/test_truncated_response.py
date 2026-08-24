"""A reply the output cap cut off is not a finished turn.

`finish_reason="length"` with **empty** visible text was already detected — that
is the reasoning runaway, and it gets resampled at a smaller cap. The same
finish_reason with text present was detected by nothing: it fell through as an
ordinary turn, reached `if not parsed_calls`, and under
`no_tool_behavior="stop"` ended the run on a sentence cut mid-token. Two measured
trials ended on "The shell ate my \\`" and "The replacement text got mangled by
the shell (the \\`", both with hours of work behind them, both scoring zero.

Truncation and "the model chose to stop talking" are opposite signals arriving in
the same shape, so the point of these cases is that they no longer share an exit.
"""

from __future__ import annotations

import pytest

from frontier_agent.core.llm import LLMResponse
from frontier_agent.core.loop_types import LoopConfig, LoopPolicy
from frontier_agent.core.runtime.loop._runaway import (
    _is_runaway_response,
    is_truncated_with_text,
)
from frontier_agent.core.runtime.loop.agent_loop import run_agent_loop
from frontier_agent.core.tool import tool

# ── the detector ─────────────────────────────────────────────────────────────


def test_truncation_with_text_is_recognised() -> None:
    """The case that used to be invisible: the exact shape of both trials."""
    resp = LLMResponse(content="The shell ate my `", finish_reason="length")
    assert is_truncated_with_text(resp) is True


def test_an_empty_truncation_stays_with_the_runaway_detector() -> None:
    """The two halves must not both claim the same response — a runaway needs a
    smaller cap, a truncation-with-text needs to be continued."""
    resp = LLMResponse(content="", finish_reason="length")
    assert is_truncated_with_text(resp) is False
    assert _is_runaway_response(resp) is True


def test_a_truncation_that_still_produced_a_tool_call_is_not_it() -> None:
    """The tool call is actionable, so the turn proceeds normally."""
    resp = LLMResponse(
        content="calling now",
        tool_calls=[{"id": "1", "function": {"name": "bash", "arguments": "{}"}}],
        finish_reason="length",
    )
    assert is_truncated_with_text(resp) is False


def test_a_normal_stop_is_not_truncation() -> None:
    resp = LLMResponse(content="Here is my complete answer.", finish_reason="stop")
    assert is_truncated_with_text(resp) is False


def test_thinking_only_content_does_not_count_as_visible_text() -> None:
    """Reasoning that filled the budget is a runaway, not a truncated answer —
    stripping thinking is what keeps the two apart."""
    resp = LLMResponse(content="<think>" + "reasoning " * 200 + "</think>", finish_reason="length")
    assert is_truncated_with_text(resp) is False


def test_a_long_completion_without_finish_reason_is_not_assumed_truncated() -> None:
    """Deliberately no completion-token fallback here, unlike the runaway
    detector. That heuristic reads "a big completion with nothing visible cannot
    be an empty reply" — sound only while the text is empty. With text present, a
    big completion is what a long legitimate answer looks like, so the same
    heuristic would declare every one of them truncated.
    """
    resp = LLMResponse(
        content="a long, complete and perfectly finished answer",
        finish_reason="",
        usage={"completion_tokens": 30_000},
    )
    assert is_truncated_with_text(resp) is False


# ── the loop ─────────────────────────────────────────────────────────────────


class _TruncatingLLM:
    """Truncates for the first `truncate_times` calls, then answers."""

    model = "stub"

    def __init__(self, truncate_times: int) -> None:
        self._truncate_times = truncate_times
        self.calls = 0
        self.seen: list[list[dict]] = []

    async def chat(self, messages: list[dict], **_kw: object) -> LLMResponse:
        self.calls += 1
        self.seen.append([dict(m) for m in messages])
        if self.calls <= self._truncate_times:
            return LLMResponse(
                content=f"partial fragment {self.calls} — the shell ate my `",
                finish_reason="length",
            )
        return LLMResponse(content="the complete answer", finish_reason="stop")


@pytest.mark.asyncio
async def test_a_truncated_turn_no_longer_ends_a_stop_mode_run() -> None:
    """The regression itself. `no_tool_behavior="stop"` is what
    `stateful_react_agent` sets, and it is why one cut sentence was fatal."""
    llm = _TruncatingLLM(truncate_times=1)

    result = await run_agent_loop(
        system_prompt="system",
        user_message="go",
        llm=llm,
        tools=[],
        config=LoopConfig(
            max_turns=5,
            loop_policy=LoopPolicy(no_tool_behavior="stop"),
        ),
    )

    assert llm.calls == 2, "the truncated turn must be continued, not terminal"
    assert result.stopped_by == "no_tool", (
        "the continuation answered normally, so the run ends the ordinary way"
    )
    assert "the complete answer" in (result.final_content or "")


@pytest.mark.asyncio
async def test_the_partial_text_is_kept_and_the_model_told_to_resume() -> None:
    """The fragment is hours of work in the worst case. Restarting from scratch
    would be a second way to lose it."""
    llm = _TruncatingLLM(truncate_times=1)

    await run_agent_loop(
        system_prompt="system",
        user_message="go",
        llm=llm,
        tools=[],
        config=LoopConfig(
            max_turns=5, loop_policy=LoopPolicy(no_tool_behavior="stop"),
        ),
    )

    continuation_request = llm.seen[1]
    assert any(
        "partial fragment 1" in str(m.get("content", ""))
        for m in continuation_request
    ), "the truncated text must survive in history"
    assert any(
        "cut off mid-sentence" in str(m.get("content", ""))
        for m in continuation_request
    ), "and the model must be told to continue rather than restart"


@pytest.mark.asyncio
async def test_persistent_truncation_stops_with_its_own_reason() -> None:
    """`no_tool` was the misdiagnosis that made this invisible: the trajectory
    showed a short sentence and a clean-looking stop, which reads as a finished
    run. A model that truncates every continuation is a different fault."""
    llm = _TruncatingLLM(truncate_times=99)

    result = await run_agent_loop(
        system_prompt="system",
        user_message="go",
        llm=llm,
        tools=[],
        config=LoopConfig(
            max_turns=10,
            truncation_max_continuations=2,
            loop_policy=LoopPolicy(no_tool_behavior="stop"),
        ),
    )

    assert result.stopped_by == "response_truncated"
    assert llm.calls == 3, "one initial reply plus two continuations"


@pytest.mark.asyncio
async def test_truncation_does_not_spend_the_no_tool_nudge_budget() -> None:
    """The two budgets are separate because the two signals are opposite. Sharing
    one would let a truncation consume the allowance a genuinely tool-less turn
    needs, and vice versa.
    """

    class _TruncateThenGoQuiet:
        model = "stub"

        def __init__(self) -> None:
            self.calls = 0

        async def chat(self, _messages: list[dict], **_kw: object) -> LLMResponse:
            self.calls += 1
            if self.calls == 1:
                return LLMResponse(content="cut off `", finish_reason="length")
            return LLMResponse(content="just chatting", finish_reason="stop")

    llm = _TruncateThenGoQuiet()

    result = await run_agent_loop(
        system_prompt="system",
        user_message="go",
        llm=llm,
        tools=[],
        config=LoopConfig(
            max_turns=10,
            no_tool_max_retries=2,
            truncation_max_continuations=2,
            loop_policy=LoopPolicy(no_tool_behavior="nudge"),
        ),
    )

    # 1 truncation + 2 tool-less turns (the second exhausts no_tool_max_retries).
    assert llm.calls == 3
    assert result.stopped_by == "no_tool"


@pytest.mark.asyncio
async def test_the_continuation_budget_resets_after_a_productive_turn() -> None:
    """A single truncation early in a long run must not leave the loop one
    truncation away from stopping hours later."""

    class _TruncateTwiceApart:
        model = "stub"

        def __init__(self) -> None:
            self.calls = 0

        async def chat(self, _messages: list[dict], **_kw: object) -> LLMResponse:
            self.calls += 1
            if self.calls in (1, 3):
                return LLMResponse(content=f"cut {self.calls} `", finish_reason="length")
            if self.calls == 2:
                return LLMResponse(
                    content="working",
                    tool_calls=[{
                        "id": "1",
                        "function": {"name": "noop", "arguments": "{}"},
                    }],
                    finish_reason="tool_calls",
                )
            return LLMResponse(content="done", finish_reason="stop")

    @tool
    async def noop() -> str:
        """A tool that does nothing."""
        return "ok"

    llm = _TruncateTwiceApart()
    result = await run_agent_loop(
        system_prompt="system",
        user_message="go",
        llm=llm,
        tools=[noop],
        config=LoopConfig(
            max_turns=10,
            truncation_max_continuations=1,
            loop_policy=LoopPolicy(no_tool_behavior="stop"),
        ),
    )

    assert result.stopped_by != "response_truncated", (
        "the productive turn between the two truncations must reset the budget"
    )
    assert llm.calls == 4
