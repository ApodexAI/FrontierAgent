from __future__ import annotations

from frontier_agent.core.llm import LLMResponse
from frontier_agent.core.runtime.loop.agent_loop import _answer_dropped_tool_calls
from frontier_agent.core.runtime.loop.model_profile import (
    DefaultThinkingParser,
    HistoryPolicy,
    ModelProfile,
    NativeMessageNormalizer,
)
from frontier_agent.core.runtime.loop.tool_call_parser import DefaultToolCallParser


def _native_call(name: str, call_id: str) -> dict:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": "{}"},
    }


def test_native_parser_drops_unknown_companion_when_known_call_exists() -> None:
    response = LLMResponse(tool_calls=[
        _native_call("glob_search", "known"),
        _native_call("read_file_stub", "unknown"),
    ])

    calls = DefaultToolCallParser().parse(response, {"glob_search"})

    assert [call["name"] for call in calls] == ["glob_search"]


def test_native_parser_keeps_all_unknown_calls_for_explicit_correction() -> None:
    response = LLMResponse(tool_calls=[
        _native_call("read_file_stub", "unknown"),
    ])

    calls = DefaultToolCallParser().parse(response, {"glob_search"})

    assert [call["name"] for call in calls] == ["read_file_stub"]


def _assistant_turn(response: LLMResponse) -> dict:
    """The history message the agent loop writes before it parses."""
    profile = ModelProfile(model_id="gpt-4o", provider="openai")
    thinking = DefaultThinkingParser().extract(response, profile)
    return NativeMessageNormalizer().to_history(
        response, thinking, HistoryPolicy(), profile.thinking_format,
    )


def test_a_dropped_companion_call_still_gets_a_tool_response() -> None:
    """An orphan ``tool_call_id`` is a hard HTTP 400 on Azure and others.

    The assistant history message is written from the raw response, so it keeps
    the dropped call's id; without a matching tool message the very next
    request is malformed and the run dies instead of recovering.
    """
    response = LLMResponse(tool_calls=[
        _native_call("glob_search", "known"),
        _native_call("read_file_stub", "unknown"),
    ])
    tool_names = {"glob_search"}
    history = _assistant_turn(response)
    messages = [history]
    parsed = DefaultToolCallParser().parse(response, tool_names)

    _answer_dropped_tool_calls(messages, history, parsed, tool_names)

    executed = {call["id"] for call in parsed}
    answered = {
        message["tool_call_id"] for message in messages
        if message.get("role") == "tool"
    }
    recorded = {call["id"] for call in history["tool_calls"]}
    assert recorded <= executed | answered
    assert answered == {"unknown"}
    # The correction the executor would have produced is not lost either, so
    # the model learns the name is wrong instead of reissuing it every turn.
    assert "read_file_stub" in messages[-1]["content"]


def test_executed_calls_are_left_for_the_executor_to_answer() -> None:
    response = LLMResponse(tool_calls=[_native_call("glob_search", "known")])
    tool_names = {"glob_search"}
    history = _assistant_turn(response)
    messages = [history]
    parsed = DefaultToolCallParser().parse(response, tool_names)

    _answer_dropped_tool_calls(messages, history, parsed, tool_names)

    assert messages == [history]
