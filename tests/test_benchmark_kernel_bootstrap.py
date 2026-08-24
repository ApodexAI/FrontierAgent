from __future__ import annotations

import pytest

from benchmarks.public.core.kernel_adapter import BenchmarkSession
from benchmarks.public.core.registry import REGISTRY
from frontier_agent.components.agent_bus import AgentBus
from frontier_agent.components.agent_bus.spawn_guard import SpawnGuard
from frontier_agent.core.runtime import registry
from frontier_agent.core.runtime.resources.manager import ResourceManager
from frontier_agent.scheduling.pipeline_registry import PipelineRegistry
from frontier_agent.scheduling.process_manager import ProcessManager


@pytest.mark.asyncio
async def test_benchmark_session_bootstraps_team_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://127.0.0.1:9/v1")
    monkeypatch.setenv("OPENAI_MODEL", "test")
    before = registry.snapshot()

    async with BenchmarkSession():
        pipelines = registry.get(PipelineRegistry)
        for pipeline_id in (
            "stateful-react-agent",
            "agent_team",
            "agent_team_report",
            "agent-team",
            "agent-team-report",
        ):
            assert pipelines.has(pipeline_id)
        for dataset, config in REGISTRY.items():
            assert pipelines.has(config.default_pipeline), (
                f"{dataset} references unregistered pipeline "
                f"{config.default_pipeline!r}"
            )
        assert len(registry.get(ResourceManager).all_tools) == 23
        assert registry.get(AgentBus) is not None
        assert registry.get(SpawnGuard) is not None
        task = await registry.get(ProcessManager).create_task("runtime contract")
        assert task.request.question == "runtime contract"

    assert registry.snapshot() == before
