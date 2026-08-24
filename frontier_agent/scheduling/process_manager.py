"""Task lifecycle management — in-memory only."""

from __future__ import annotations

import logging
from typing import Any

from frontier_agent.core.errors import InvalidStateTransition, TaskNotFoundError
from frontier_agent.core.events import EventType
from frontier_agent.core.types import TaskId, TaskStatus
from frontier_agent.models.task import ResearchRequest, Task
from frontier_agent.state.event_store.sqlite import EventStore

logger = logging.getLogger("frontier_agent.scheduling.process_manager")

# Valid state transitions.
_TRANSITIONS: dict[TaskStatus, set[TaskStatus]] = {
    TaskStatus.CREATED: {TaskStatus.RUNNING, TaskStatus.FAILED, TaskStatus.ABORTED},
    TaskStatus.RUNNING: {
        TaskStatus.SUSPENDED, TaskStatus.COMPLETED,
        TaskStatus.FAILED, TaskStatus.ABORTED,
    },
    TaskStatus.SUSPENDED: {TaskStatus.RUNNING, TaskStatus.FAILED, TaskStatus.ABORTED},
    TaskStatus.COMPLETED: {TaskStatus.RUNNING},
    TaskStatus.FAILED: {TaskStatus.CREATED, TaskStatus.RUNNING, TaskStatus.COMPLETED},
    TaskStatus.ABORTED: set(),
}


class ProcessManager:
    """In-memory task registry — the only PM surface bench/scheduler use."""

    def __init__(
        self,
        event_store: EventStore,
    ) -> None:
        self._event_store = event_store
        self._tasks: dict[TaskId, Task] = {}

    async def create_task(
        self,
        question: str,
        config: dict[str, Any] | None = None,
    ) -> Task:
        """Create a new task and record TASK_CREATED."""
        cfg = config or {}
        task = Task(request=ResearchRequest(question=question))
        task.thread_id = f"frontier_agent-{task.id}"
        task.context = dict(cfg)

        self._tasks[task.id] = task

        await self._event_store.append(
            task_id=task.id,
            event_type=EventType.TASK_CREATED,
            payload={"question": question, "config": cfg},
        )
        return task

    async def get_task(self, task_id: TaskId | str) -> Task:
        """Look up a task by id; raise TaskNotFoundError if unknown."""
        tid = TaskId(task_id) if isinstance(task_id, str) else task_id
        task = self._tasks.get(tid)
        if task is None:
            raise TaskNotFoundError(str(tid))
        return task

    async def update_status(self, task_id: TaskId | str, status: TaskStatus) -> None:
        """Validated status transition."""
        tid = TaskId(task_id) if isinstance(task_id, str) else task_id
        task = await self.get_task(tid)

        if status not in _TRANSITIONS.get(task.status, set()):
            raise InvalidStateTransition(str(tid), task.status.value, status.value)

        old_status = task.status
        task.set_status(status)

        await self._event_store.append(
            task_id=tid,
            event_type=EventType.TASK_STATUS_CHANGED,
            payload={"old_status": old_status.value, "new_status": status.value},
        )

    async def set_error(self, task_id: TaskId | str, error: str) -> None:
        """Mark a task FAILED with an error message."""
        tid = TaskId(task_id) if isinstance(task_id, str) else task_id
        task = await self.get_task(tid)
        task.error = error
        task.set_status(TaskStatus.FAILED)

        await self._event_store.append(
            task_id=tid,
            event_type=EventType.ERROR,
            payload={"error": error},
        )

    async def abort_task(
        self,
        task_id: TaskId | str,
        reason: str = "Task aborted",
    ) -> Task:
        """Abort a task; terminal races are idempotent no-ops."""
        tid = TaskId(task_id) if isinstance(task_id, str) else task_id
        task = await self.get_task(tid)
        if task.status in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.ABORTED):
            return task

        old_status = task.status
        task.error = reason
        task.set_status(TaskStatus.ABORTED)

        await self._event_store.append(
            task_id=tid,
            event_type=EventType.TASK_STATUS_CHANGED,
            payload={
                "old_status": old_status.value,
                "new_status": TaskStatus.ABORTED.value,
                "message": reason,
            },
        )
        return task
