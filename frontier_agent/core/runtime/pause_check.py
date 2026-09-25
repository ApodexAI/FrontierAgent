"""Product task-store adapter for AgentCore pause polling."""
from __future__ import annotations

from agent_core.runtime.pause_check import PauseCheckFn, pause_check_from_state
from agent_core.runtime.pause_check import make_task_pause_check as _make_pause_check

from frontier_agent.core.errors import TaskNotFoundError
from frontier_agent.core.runtime.registries import services as registry
from frontier_agent.core.types import TaskId


def make_task_pause_check(task_id: str | TaskId) -> PauseCheckFn:
    async def load_status(value: str) -> object:
        from frontier_agent.scheduling.process_manager import ProcessManager
        manager = registry.get_optional(ProcessManager)
        return None if manager is None else await manager.get_task(TaskId(value))
    return _make_pause_check(str(task_id), load_status, missing_exceptions=(TaskNotFoundError,))

__all__ = ["PauseCheckFn", "make_task_pause_check", "pause_check_from_state"]
