"""Caller-owned execution observers inherited by child asyncio tasks."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

_observers: ContextVar[tuple[Any, ...]] = ContextVar("execution_observers", default=())


def execution_observers() -> list[Any]:
    return list(_observers.get())


@contextmanager
def observe_execution(observers: list[Any]) -> Iterator[None]:
    """Apply reliable host observers to a workflow and its delegated loops."""
    token = _observers.set(tuple(observers))
    try:
        yield
    finally:
        _observers.reset(token)
