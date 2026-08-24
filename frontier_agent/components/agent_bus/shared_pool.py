"""Shared artifact pool for parallel sub-agent execution.

Holds dict-shaped artifacts produced concurrently by sub-agents and dedups
by ``id`` field when present. Domain-specific subclasses live in their
owning workflow — see its own ``shared_pool`` module.
"""

from __future__ import annotations

import asyncio
from typing import Any


class SharedArtifactPool:
    """Process-local artifact pool shared across parallel sub-agents.

    Items are stored as shallow copies; if an item carries an ``id`` field,
    later duplicates with the same id are silently dropped.
    """

    def __init__(self) -> None:
        self._items: list[dict[str, Any]] = []
        self._seen_ids: set[str] = set()
        self._lock = asyncio.Lock()

    async def add(self, items: list[dict[str, Any]]) -> None:
        if not items:
            return

        async with self._lock:
            for item in items:
                if not isinstance(item, dict):
                    continue
                item_id = str(item.get("id", "")).strip()
                if item_id and item_id in self._seen_ids:
                    continue
                if item_id:
                    self._seen_ids.add(item_id)
                self._items.append(dict(item))

    def get_all(self) -> list[dict[str, Any]]:
        return [dict(item) for item in self._items]

    def count(self) -> int:
        return len(self._items)
