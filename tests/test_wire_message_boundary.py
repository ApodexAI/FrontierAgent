"""What may leave the process on a message, enforced at the one place it can.

``OpenAIClient`` hands each ``Message`` dict to the SDK verbatim, so any key on a
message is a wire key. Two of them must survive the filter for provider reasons
and one class of them must not survive at all; both directions are pinned here.
"""

from __future__ import annotations

from types import SimpleNamespace

from frontier_agent.core.messages import (
    WIRE_MESSAGE_KEYS,
    Message,
    assistant_msg,
    for_wire,
    system_msg,
    tool_msg,
    user_msg,
)


def test_in_process_keys_are_dropped() -> None:
    """The TUI's presentation metadata is not part of any chat API."""
    messages: list[Message] = [
        {
            "role": "tool",
            "name": "bash",
            "tool_call_id": "c1",
            "content": "ok",
            "duration_ms": 1200,
            "is_error": False,
        },
    ]

    out = for_wire(messages)

    assert out[0] == {
        "role": "tool", "name": "bash", "tool_call_id": "c1", "content": "ok",
    }
    # The caller's list is not mutated: retry paths reuse it.
    assert "duration_ms" in messages[0]


def test_reasoning_content_survives() -> None:
    """DeepSeek-V4 / o-series proxies REQUIRE it on prior assistant turns, so the
    decision belongs to assistant_msg_with_reasoning, not to this filter."""
    messages = [assistant_msg("done", reasoning="thought about it")]

    assert for_wire(messages)[0]["reasoning_content"] == "thought about it"
    assert "reasoning_content" in WIRE_MESSAGE_KEYS


def test_content_blocks_are_not_inspected() -> None:
    """Anthropic prompt caching puts cache_control INSIDE a content block. A
    message-level filter must leave the block alone."""
    cached: list[Message] = [
        {
            "content": [
                {"type": "text", "text": "sys", "cache_control": {"type": "ephemeral"}},
            ],
            "role": "system",
        },
    ]

    assert for_wire(cached)[0]["content"][0]["cache_control"] == {"type": "ephemeral"}


def test_a_clean_list_is_returned_identical_not_merely_equal() -> None:
    """Some served checkpoints are byte-shape sensitive, so the common path must
    not rebuild anything."""
    messages = [system_msg("s"), user_msg("u"), assistant_msg("a")]

    out = for_wire(messages)

    assert out is messages
    assert all(a is b for a, b in zip(out, messages, strict=True))


def test_key_order_is_preserved() -> None:
    """``content`` before ``role`` is deliberate in the builders; rebuilding in a
    fixed order would change the serialized bytes."""
    messages: list[Message] = [
        {"content": "x", "role": "assistant", "is_error": True},
    ]

    assert list(for_wire(messages)[0]) == ["content", "role"]


def test_every_builder_already_produces_wire_only_messages() -> None:
    built = [
        system_msg("s"),
        user_msg("u"),
        assistant_msg("a", tool_calls=[{
            "id": "c1", "type": "function",
            "function": {"name": "bash", "arguments": "{}"},
        }]),
        tool_msg("out", "c1"),
    ]

    for message in built:
        assert set(message) <= WIRE_MESSAGE_KEYS, message


async def test_the_openai_client_filters_before_the_sdk_call(monkeypatch) -> None:
    """End to end at the only client that passes message dicts through verbatim.

    Patches ``with_raw_response.create`` because that is the path ``chat`` takes;
    the plain ``create`` is the streaming one.
    """
    from frontier_agent.infra.openai_client import OpenAIClient

    seen: dict = {}

    class _Raw:
        @staticmethod
        def parse():
            return SimpleNamespace(
                choices=[SimpleNamespace(
                    message=SimpleNamespace(
                        content="hi", tool_calls=None, reasoning_content=None,
                        reasoning=None,
                    ),
                    finish_reason="stop",
                )],
                usage=None,
                model="m",
            )

    async def _create(**kwargs):
        seen.update(kwargs)
        return _Raw()

    client = OpenAIClient(model="m", api_key="k", base_url="http://localhost")
    monkeypatch.setattr(
        client._client.chat.completions.with_raw_response, "create", _create,
        raising=False,
    )

    dirty: list[Message] = [
        {"content": "q", "role": "user"},
        {
            "content": "r", "role": "tool", "tool_call_id": "c1",
            "duration_ms": 5, "is_error": False,
        },
    ]

    await client.chat(dirty)

    assert seen, "the SDK was never called"
    for message in seen["messages"]:
        assert set(message) <= WIRE_MESSAGE_KEYS, message
    # The caller's list still carries what the TUI needs.
    assert dirty[1]["duration_ms"] == 5
