"""Persist typed inter-agent messages and optionally publish them live."""

from __future__ import annotations

import asyncio
import logging
from enum import StrEnum
from typing import Protocol, runtime_checkable

from frontier_agent.core.events import EventType
from frontier_agent.core.protocols import EventReader, EventSink
from frontier_agent.core.runtime.events.bus import EventBus
from frontier_agent.core.types import TaskId
from frontier_agent.models.agent_message import AgentMessage
from frontier_agent.models.event import KernelEvent

logger = logging.getLogger(__name__)


@runtime_checkable
class _AgentCommEventStore(EventSink, EventReader, Protocol):
    """AgentComm needs both append (EventSink) and cursored reads
    (EventReader) — combined here because Python lacks an intersection
    type. Public Protocols stay minimal in ``core.protocols``.
    """


class DeliveryMode(StrEnum):
    """Message delivery mode.

    TRIGGER: persist + hot queue + EventBus broadcast.
        Use when receiver must act immediately (e.g., assertion → critic).
    QUEUE: persist + hot queue only (no broadcast).
        Use for status updates the receiver pulls when ready.
    """

    TRIGGER = "trigger"
    QUEUE = "queue"


class AgentComm:
    """Sends inter-agent messages and logs them as events.

    All messages are persisted to EventStore (truth source).
    Hot queues are optional in-memory acceleration.
    DeliveryMode controls whether EventBus broadcast fires.
    """

    def __init__(
        self,
        event_store: _AgentCommEventStore,
        event_bus: EventBus,
    ) -> None:
        self._event_store = event_store
        self._event_bus = event_bus
        self._cursors: dict[tuple[str, str | None], int] = {}
        self._hot_queues: dict[str, asyncio.Queue[KernelEvent]] = {}

    async def send(
        self,
        msg: AgentMessage,
        mode: DeliveryMode = DeliveryMode.QUEUE,
    ) -> KernelEvent:
        """Send an agent message.

        Always persists to EventStore (truth source).
        TRIGGER mode additionally broadcasts via EventBus for immediate wakeup.
        """
        event = KernelEvent(
            task_id=TaskId(msg.task_id),
            event_type=EventType.AGENT_MESSAGE,
            from_agent=msg.from_agent,
            to_agent=msg.to_agent,
            message_type=msg.message_type,
            correlation_id=msg.content.get("correlation_id"),
            payload={
                "message_id": msg.id,
                "from_agent": msg.from_agent,
                "to_agent": msg.to_agent,
                "message_type": msg.message_type,
                "content": msg.content,
                "parent_id": msg.parent_id,
                "correlation_id": msg.content.get("correlation_id"),
                "delivery_mode": mode.value,
            },
        )

        # 1. Persist (always — EventStore is truth source)
        persisted = await self._event_store.append(event)

        # 2. Hot queue (always — non-durable acceleration)
        queue = self._hot_queues.get(msg.to_agent)
        if queue is not None:
            try:
                queue.put_nowait(persisted)
            except asyncio.QueueFull:
                logger.debug(
                    "Hot queue for %s is full; consumer uses EventStore",
                    msg.to_agent,
                )

        # 3. Broadcast (TRIGGER only — immediate wakeup signal)
        if mode == DeliveryMode.TRIGGER:
            await self._event_bus.publish(
                event.event_type, event.payload,
            )

        logger.debug(
            "Agent message %s: %s → %s [%s] mode=%s",
            msg.id, msg.from_agent, msg.to_agent,
            msg.message_type, mode.value,
        )
        return persisted

    async def consume(
        self,
        agent_id: str,
        *,
        task_id: str | None = None,
        limit: int = 50,
    ) -> list[KernelEvent]:
        """Consume undelivered messages for an agent using an idempotent cursor.

        Cursor is per (agent_id, task_id). Restarting from 0 replays all.
        """
        cursor_key = (agent_id, task_id)
        after_id = self._cursors.get(cursor_key, 0)
        events = await self._event_store.get_events_for_agent(
            agent_id,
            after_id=after_id,
            limit=limit,
            task_id=task_id,
        )
        if events:
            self._cursors[cursor_key] = int(events[-1].id)
        return events

    def reset_cursor(
        self, agent_id: str, task_id: str | None = None,
    ) -> None:
        """Reset cursor for an agent (e.g., after recovery)."""
        self._cursors.pop((agent_id, task_id), None)

    def hot_queue_for(
        self, agent_id: str,
    ) -> asyncio.Queue[KernelEvent]:
        """Return the non-durable hot queue for an agent."""
        return self._hot_queues.setdefault(
            agent_id, asyncio.Queue(maxsize=256),
        )
