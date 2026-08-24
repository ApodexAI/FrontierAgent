"""In-memory ``EventStore`` compatibility sink for OSS benchmark runs.

The historical module path is preserved, but persistence remains out of the
trimmed distribution; benchmark artifacts are written through ``result.json``.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any


class EventStore:
    """No-op event store with the scheduler's asynchronous API shape."""

    async def append(
        self,
        task_id: Any = "",
        event_type: Any = None,
        payload: dict[str, Any] | None = None,
        agent_role: str = "system",
    ) -> None:
        return None

    async def replay(self, task_id: Any) -> AsyncIterator[Any]:
        for event in ():
            yield event

    async def get_events(
        self,
        task_id: Any,
        event_type: Any = None,
        after_id: int = 0,
        limit: int | None = None,
    ) -> list[Any]:
        return []

    async def get_events_for_agent(
        self,
        to_agent: str,
        after_id: int = 0,
        limit: int = 50,
        *,
        task_id: Any = None,
    ) -> list[Any]:
        return []
