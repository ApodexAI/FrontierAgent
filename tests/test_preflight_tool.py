"""Regression tests for the benchmark preflight helper."""

from __future__ import annotations

from typing import Any

import pytest

from tools.preflight import check_kernel_llm, check_workflow_llm


class _RecordingLLM:
    def __init__(self) -> None:
        self.calls: list[tuple[list[dict[str, str]], dict[str, Any]]] = []

    async def chat(self, messages: list[dict[str, str]], **kwargs: Any) -> None:
        self.calls.append((messages, kwargs))


@pytest.mark.parametrize(
    ("pipeline", "module_name", "loader_name", "builder_name"),
    [
        (
            "stateful-react-agent",
            "workflows.stateful_react_agent.profile",
            "load_react_profile",
            "create_react_llm",
        ),
        (
            "agent_team",
            "workflows.agent_team.profile",
            "load_swarm_profile",
            "create_swarm_llm",
        ),
    ],
)
async def test_workflow_check_uses_current_profile_api(
    monkeypatch: pytest.MonkeyPatch,
    pipeline: str,
    module_name: str,
    loader_name: str,
    builder_name: str,
) -> None:
    module = __import__(module_name, fromlist=[loader_name, builder_name])
    llm = _RecordingLLM()
    loaded: list[str] = []

    def load_profile(name: str) -> dict[str, str]:
        loaded.append(name)
        return {"profile": name}

    def create_llm(profile: dict[str, str]) -> _RecordingLLM:
        assert profile == {"profile": "benchmark"}
        return llm

    monkeypatch.setattr(module, loader_name, load_profile)
    monkeypatch.setattr(module, builder_name, create_llm)

    assert await check_workflow_llm(pipeline, "benchmark") is None
    assert loaded == ["benchmark"]
    assert llm.calls == [([{"role": "user", "content": "hi"}], {"max_tokens": 1})]


async def test_workflow_check_rejects_unknown_pipeline() -> None:
    error = await check_workflow_llm("unknown-workflow", "default")

    assert error is not None
    assert "unknown pipeline 'unknown-workflow'" in error
    assert "stateful-react-agent" in error
    assert "agent_team" in error


async def test_kernel_check_returns_client_construction_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from frontier_agent.infra import llm_adapter

    def fail_to_create(_config: Any) -> None:
        raise RuntimeError("cannot build client")

    monkeypatch.setattr(llm_adapter, "create_llm", fail_to_create)

    assert await check_kernel_llm() == "RuntimeError: cannot build client"
