from __future__ import annotations

import asyncio
from types import SimpleNamespace

from frontier_agent.core.runtime.session_history import (
    SessionCompactionConfig,
    SessionHistoryCompactor,
    build_session_turn,
    messages_to_session_turns,
    render_session_history,
)


def test_render_replays_turns_and_tool_results_in_order() -> None:
    prompt = render_session_history(
        [{
            "messages": [
                {"role": "user", "content": "Find the release date"},
                {"role": "assistant", "content": "I will search."},
                {
                    "role": "tool",
                    "name": "web_search",
                    "content": "Official page: released 2026-04-03",
                },
                {"role": "assistant", "content": "It was April 3."},
            ],
        }],
        "What source confirmed that?",
    )

    assert prompt.index("Find the release date") < prompt.index(
        "Official page: released 2026-04-03"
    )
    assert prompt.index("Official page: released 2026-04-03") < prompt.index(
        "It was April 3."
    )
    assert prompt.index("It was April 3.") < prompt.index(
        "What source confirmed that?"
    )
    assert "Query/tool result (web_search)" in prompt


def test_build_turn_is_non_recursive_and_fills_missing_results() -> None:
    turn = build_session_turn(
        "current follow-up",
        [
            {"role": "system", "content": "system prompt"},
            {
                "role": "user",
                "content": "earlier replay envelope + current follow-up",
            },
            {"role": "tool", "name": "search", "content": "result 42"},
        ],
        "final 42",
        steps=[
            {"tool_name": "search", "tool_result": "result 42"},
            {"tool_name": "fetch", "tool_result": "supporting detail"},
        ],
    )

    assert turn["messages"][0] == {"role": "user", "content": "current follow-up"}
    assert turn["messages"][1]["content"] == "result 42"
    assert turn["messages"][2]["content"] == "supporting detail"
    assert turn["messages"][-1] == {"role": "assistant", "content": "final 42"}
    assert all("earlier replay envelope" not in str(message) for message in turn["messages"])


def test_execution_scoped_task_board_is_not_replayed_across_turns() -> None:
    turn = build_session_turn(
        "fix the first bug",
        [
            {"role": "user", "content": "fix the first bug"},
            {
                "role": "tool",
                "name": "add_task",
                "content": "[task board] 0/2 resolved",
            },
            {
                "role": "tool",
                "name": "read_file",
                "content": "relevant source evidence",
            },
        ],
        "fixed",
        steps=[
            {"tool_name": "update_task", "tool_result": "t1 resolved"},
            {"tool_name": "finish_planning", "tool_result": "execution unlocked"},
            {"tool_name": "web_search", "tool_result": "supporting evidence"},
        ],
    )

    prompt = render_session_history([turn], "now fix a different bug")

    assert "add_task" not in prompt
    assert "task board" not in prompt
    assert "update_task" not in prompt
    assert "execution unlocked" not in prompt
    assert "relevant source evidence" in prompt
    assert "supporting evidence" in prompt


def test_render_filters_task_board_from_legacy_resumed_turns() -> None:
    prompt = render_session_history(
        [{"messages": [
            {"role": "user", "content": "old query"},
            {
                "role": "tool",
                "name": "add_task",
                "content": "[task board] old plan",
            },
            {"role": "assistant", "content": "old answer"},
        ]}],
        "new query",
    )

    assert "old plan" not in prompt
    assert "old answer" in prompt
    assert "new query" in prompt


def test_compactor_drops_old_results_before_recent_turns() -> None:
    class _LLM:
        calls = 0

        async def chat(self, messages):
            self.calls += 1
            return SimpleNamespace(content="should not be needed")

    llm = _LLM()
    recent = [
        {"messages": [
            {"role": "user", "content": f"recent query {i}"},
            {"role": "tool", "name": "search", "content": f"recent result {i}"},
            {"role": "assistant", "content": f"recent answer {i}"},
        ]}
        for i in range(5)
    ]
    compactor = SessionHistoryCompactor(
        summary_llm=llm,
        config=SessionCompactionConfig(
            context_window=20_000,
            max_completion_tokens=0,
        ),
    )
    result = asyncio.run(compactor.compact([{
        "messages": [
            {"role": "user", "content": "old query"},
            {"role": "tool", "name": "search", "content": "x" * 100_000},
            {"role": "assistant", "content": "old answer"},
        ],
    }, *recent], "next"))

    assert llm.calls == 0
    assert result.changed and result.tool_results_removed
    assert all(
        message["role"] != "tool" for message in result.turns[0]["messages"]
    )
    assert any(
        message["role"] == "tool" for message in result.turns[-1]["messages"]
    )


def test_compactor_never_compacts_the_current_query() -> None:
    class _LLM:
        calls = 0

        async def chat(self, messages):
            self.calls += 1
            return SimpleNamespace(content="unused")

    llm = _LLM()
    compactor = SessionHistoryCompactor(
        summary_llm=llm,
        config=SessionCompactionConfig(context_window=4_096),
    )
    result = asyncio.run(compactor.compact([], "current " + "q" * 100_000))

    assert result.turns == []
    assert result.changed is False
    assert llm.calls == 0


def test_compactor_summarizes_old_turns_and_keeps_last_five() -> None:
    class _LLM:
        calls = 0

        async def chat(self, messages):
            self.calls += 1
            return SimpleNamespace(content="old decisions and evidence rollup")

    llm = _LLM()
    old = [
        {"messages": [
            {"role": "user", "content": f"old query {i} " + "q" * 10_000},
            {"role": "assistant", "content": f"old answer {i}"},
        ]}
        for i in range(2)
    ]
    recent = [
        {"messages": [
            {"role": "user", "content": f"recent query {i}"},
            {"role": "assistant", "content": f"recent answer {i}"},
        ]}
        for i in range(5)
    ]
    compactor = SessionHistoryCompactor(
        summary_llm=llm,
        config=SessionCompactionConfig(
            context_window=20_000,
            max_completion_tokens=0,
        ),
    )
    result = asyncio.run(compactor.compact([*old, *recent], "next"))

    assert llm.calls == 1
    assert result.changed and result.summarized
    assert result.turns[0]["summary"].endswith(
        "old decisions and evidence rollup"
    )
    assert len(result.turns[1:]) == 5
    assert result.turns[-1]["messages"][0]["content"] == "recent query 4"


def test_legacy_flat_messages_upgrade_to_turns() -> None:
    turns = messages_to_session_turns([
        {"role": "system", "content": "ignored"},
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "q2"},
        {"role": "assistant", "content": "a2"},
    ])

    assert len(turns) == 2
    assert turns[0]["messages"][0]["content"] == "q1"
    assert turns[1]["messages"][-1]["content"] == "a2"
