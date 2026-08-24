"""EAS session affinity must be present in the request URL query."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from frontier_agent.core.messages import user_msg
from frontier_agent.infra.openai_client import OpenAIClient
from frontier_agent.infra.session_context import (
    eas_session_headers,
    mirror_session_query,
    set_task_session_id,
)


def _completion() -> SimpleNamespace:
    return SimpleNamespace(
        choices=[SimpleNamespace(
            message=SimpleNamespace(content="ok", tool_calls=None),
            finish_reason="stop",
        )],
        usage=None,
        model="test-model",
    )


def test_construction_time_session_header_mirrors_into_query() -> None:
    client = OpenAIClient(
        "test-model",
        api_key="test-key",
        default_headers={
            "HTTP-Referer": "frontier-agent",
            "X-Upstream-Session-Id": "task-42",
        },
    )

    # Statically configured (no task context at build time) — always applies.
    assert client._session_query(None) == {"x-upstream-session-id": "task-42"}
    assert mirror_session_query(None) == {}
    assert mirror_session_query({"other": "value"}) == {}


def test_cached_client_drops_the_previous_tasks_session_query(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A per-profile-cached client must not pin later tasks to task 1's worker."""
    monkeypatch.delenv("FRONTIER_AGENT_LLM_STICKY_SESSION", raising=False)
    set_task_session_id("task-1")
    client = OpenAIClient(
        "test-model", api_key="test-key",
        default_headers=eas_session_headers(),
    )
    assert client._session_query(None) == {"x-upstream-session-id": "task-1"}

    # Same client object, next task, and no per-call binding (sticky switched
    # off, or a call path that skips ``bind_session_id``).
    set_task_session_id("task-2")
    assert client._session_query(None) == {}
    # A per-call binding still wins outright.
    assert client._session_query({"x-upstream-session-id": "task-2"}) == {
        "x-upstream-session-id": "task-2",
    }
    set_task_session_id("")


def test_sticky_kill_switch_suppresses_construction_time_affinity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FRONTIER_AGENT_LLM_STICKY_SESSION", "0")
    set_task_session_id("task-9")
    try:
        assert eas_session_headers() == {}
        client = OpenAIClient(
            "test-model", api_key="test-key",
            default_headers=eas_session_headers(),
        )
        assert client._session_query(None) == {}
    finally:
        set_task_session_id("")


@pytest.mark.asyncio
async def test_chat_mirrors_per_call_session_into_extra_query() -> None:
    client = OpenAIClient("test-model", api_key="test-key")
    raw_response = SimpleNamespace(parse=lambda: _completion())
    create = AsyncMock(return_value=raw_response)
    client._client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(
                with_raw_response=SimpleNamespace(create=create),
            ),
        ),
    )

    await client.chat(
        [user_msg("hello")],
        extra_headers={"x-upstream-session-id": "task-7:sub"},
    )

    kwargs = create.call_args.kwargs
    assert kwargs["extra_headers"] == {
        "x-upstream-session-id": "task-7:sub",
    }
    assert kwargs["extra_query"] == {
        "x-upstream-session-id": "task-7:sub",
    }


@pytest.mark.asyncio
async def test_stream_mirrors_session_but_ordinary_headers_do_not() -> None:
    client = OpenAIClient("test-model", api_key="test-key")

    async def fake_stream():
        yield SimpleNamespace(
            choices=[SimpleNamespace(
                delta=SimpleNamespace(
                    content="ok", reasoning_content=None, tool_calls=None,
                ),
                finish_reason="stop",
            )],
            usage=None,
            model="test-model",
        )

    create = AsyncMock(return_value=fake_stream())
    client._client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create)),
    )

    _ = [
        delta async for delta in client.stream(
            [user_msg("hello")],
            extra_headers={"x-upstream-session-id": "task-8:dag"},
        )
    ]
    kwargs = create.call_args.kwargs
    assert kwargs["extra_query"] == {
        "x-upstream-session-id": "task-8:dag",
    }

    create.reset_mock()
    _ = [
        delta async for delta in client.stream(
            [user_msg("hello")], extra_headers={"X-Title": "test"},
        )
    ]
    assert "extra_query" not in create.call_args.kwargs
