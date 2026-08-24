"""Shared helpers for bus-scoped tool execution."""

from __future__ import annotations

from typing import Any

SWARM_SCOPE_KEY = "swarm_subagent_runtime"


def resolve_bus_task_id(scope: Any) -> str:
    """Return the AgentBus-scope id for tool calls.

    Heavy mode injects a synthetic per-run id via
    ``scope.metadata["bus_task_id"]``; everywhere else falls back to
    ``scope.task_id``.
    """
    metadata = getattr(scope, "metadata", None) or {}
    return str(metadata.get("bus_task_id") or scope.task_id)


def resolve_root_task_id(scope: Any) -> str:
    """Return the original root task_id (SSE / event-store scope).

    Heavy mode optionally stashes the root id under
    ``scope.metadata["root_task_id"]`` so tools that need to emit
    user-facing events can use it; absent it, ``scope.task_id`` is
    already the root.
    """
    metadata = getattr(scope, "metadata", None) or {}
    return str(metadata.get("root_task_id") or scope.task_id)


__all__ = ["SWARM_SCOPE_KEY", "resolve_bus_task_id", "resolve_root_task_id"]
