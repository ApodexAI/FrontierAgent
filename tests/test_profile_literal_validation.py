"""Contract tests for the YAML-sourced Literal guards.

``thinking_format`` and ``llm.protocol`` come from user-authored profile YAML but
feed Literal-typed ``ModelProfile`` fields, so they are validated on the way in.
The contract is that an unusable value is warned about and replaced by a
fallback — never that it aborts profile loading.

These pin the shape check specifically. A frozenset membership test raises
``TypeError: unhashable type`` for a list or dict, and ``str.lower()`` raises
``AttributeError`` for a non-string, so both guards need an isinstance check
before they touch the value. Both profile loaders share the code path, so both
are exercised here rather than only one.
"""

from __future__ import annotations

import importlib

import pytest

from frontier_agent.core.runtime.loop.model_profile import (
    ThinkingFormat,
    WireProtocol,
    is_thinking_format,
    is_wire_protocol,
)
from frontier_agent.infra.protocol_client import protocol_of

# Values a YAML parser can legitimately produce for a scalar field. The list and
# dict cases are the ones that used to raise rather than fall back; the set is
# included because CPython special-cases set-in-frozenset and so masked the bug.
UNUSABLE = [
    ["tag"],
    {"format": "tag"},
    {"tag"},
    5,
    3.5,
    True,
    None,
    "",
    "not-a-real-format",
]

PROFILE_LOADERS = [
    "workflows.agent_team.profile",
    "workflows.stateful_react_agent.profile",
]


@pytest.mark.parametrize("value", UNUSABLE)
def test_thinking_format_guard_rejects_without_raising(value: object) -> None:
    assert is_thinking_format(value) is False


@pytest.mark.parametrize("value", UNUSABLE)
def test_wire_protocol_guard_rejects_without_raising(value: object) -> None:
    assert is_wire_protocol(value) is False


@pytest.mark.parametrize("value", ["tag", "content_block", "reasoning_content", "none"])
def test_thinking_format_guard_accepts_every_declared_member(value: str) -> None:
    assert is_thinking_format(value) is True


@pytest.mark.parametrize(
    "value", ["chat_completions", "anthropic", "responses", "bedrock"],
)
def test_wire_protocol_guard_accepts_every_declared_member(value: str) -> None:
    assert is_wire_protocol(value) is True


@pytest.mark.parametrize("value", UNUSABLE)
def test_protocol_of_falls_back_for_unusable_config(value: object) -> None:
    """``protocol_of`` lowercases before narrowing, so it needs its own check."""
    assert protocol_of({"protocol": value}) == "chat_completions"


def test_protocol_of_normalises_case_and_honours_valid_values() -> None:
    assert protocol_of({"protocol": "ANTHROPIC"}) == "anthropic"
    assert protocol_of({}) == "chat_completions"


@pytest.mark.parametrize("module_path", PROFILE_LOADERS)
@pytest.mark.parametrize("value", UNUSABLE)
def test_profile_loaders_survive_unusable_thinking_format(
    module_path: str, value: object,
) -> None:
    """An unusable YAML value must not abort loading — it falls back instead."""
    resolve = importlib.import_module(module_path)._resolve_thinking_format
    result = resolve({"agent": {"thinking_format": value}, "llm": {}})
    assert is_thinking_format(result), f"fallback {result!r} is not a ThinkingFormat"


@pytest.mark.parametrize("module_path", PROFILE_LOADERS)
def test_profile_loaders_honour_a_valid_explicit_format(module_path: str) -> None:
    resolve = importlib.import_module(module_path)._resolve_thinking_format
    profile = {"agent": {"thinking_format": "reasoning_content"}, "llm": {}}
    assert resolve(profile) == "reasoning_content"


@pytest.mark.parametrize("module_path", PROFILE_LOADERS)
def test_profile_loaders_survive_unusable_protocol(module_path: str) -> None:
    """``llm.protocol`` reaches the same ModelProfile field via protocol_of."""
    resolve = importlib.import_module(module_path)._resolve_thinking_format
    result = resolve({"agent": {}, "llm": {"protocol": ["anthropic"]}})
    assert is_thinking_format(result)


def test_guards_stay_in_sync_with_their_literal_aliases() -> None:
    """A new member added to either Literal must be accepted by its guard."""
    from typing import get_args
    for member in get_args(ThinkingFormat):
        assert is_thinking_format(member), member
    for member in get_args(WireProtocol):
        assert is_wire_protocol(member), member
