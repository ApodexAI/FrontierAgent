"""Async pub/sub event bus for kernel-level communication.

Bridges kernel-side event publishing with the OS event system and
frontend streaming layer.
"""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from collections.abc import Awaitable, Callable
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)

Handler = Callable[[dict[str, Any]], Awaitable[None]]


def _coerce_key(event_type: str | Enum) -> str:
    """Reduce any str/str-Enum event type to its string handler key."""
    if isinstance(event_type, Enum):
        value = getattr(event_type, "value", None)
        if isinstance(value, str):
            return value
        return str(value)
    return str(event_type)


class EventBus:
    """Typed async pub/sub for kernel events."""

    def __init__(self) -> None:
        self._handlers: dict[str, list[Handler]] = defaultdict(list)
        self._global_handlers: list[Handler] = []

    def subscribe(self, event_type: str | Enum, handler: Handler) -> None:
        """Subscribe a handler to a specific event type."""
        self._handlers[_coerce_key(event_type)].append(handler)

    def subscribe_all(self, handler: Handler) -> None:
        """Subscribe a handler to ALL event types (for logging/streaming)."""
        self._global_handlers.append(handler)

    async def publish(self, event_type: str | Enum, payload: dict[str, Any]) -> None:
        """Publish an event to all subscribed handlers."""
        key = _coerce_key(event_type)
        event_data = {"event_type": key, **payload}

        handlers = self._handlers.get(key, []) + self._global_handlers
        if not handlers:
            return

        tasks = [self._safe_call(h, event_data) for h in handlers]
        await asyncio.gather(*tasks)

    @staticmethod
    async def _safe_call(handler: Handler, data: dict[str, Any]) -> None:
        try:
            await handler(data)
        except Exception:
            logger.exception("Event handler failed: %s", handler.__name__)
