"""Service Registry — simplified dependency injection container.

All kernel services register themselves here. Other layers resolve
dependencies through the registry instead of importing singletons.
"""

from __future__ import annotations

from typing import Any, TypeVar

from frontier_agent.core.errors import ServiceNotRegistered

T = TypeVar("T")

_services: dict[type, Any] = {}


def register[T](service_type: type[T], instance: T) -> None:
    """Register a service instance by its type."""
    _services[service_type] = instance


def get[T](service_type: type[T]) -> T:
    """Retrieve a registered service. Raises ServiceNotRegistered if missing."""
    instance = _services.get(service_type)
    if instance is None:
        raise ServiceNotRegistered(service_type)
    return instance


def get_optional[T](service_type: type[T]) -> T | None:
    """Retrieve a service or None if not registered."""
    instance = _services.get(service_type)
    if instance is not None:
        return instance

    # Protocol keys such as ``EventSink`` / ``PhaseMiddlewareChain`` are
    # intentionally registered structurally by some composition roots during
    # the migration. Fall back to a runtime-checkable Protocol scan so lower
    # layers can resolve capabilities without importing concrete component or
    # state classes as registry keys.
    for candidate in _services.values():
        try:
            if isinstance(candidate, service_type):
                return candidate
        except TypeError:
            # ``service_type`` is not runtime-checkable (or not a class).
            break
    return None


def get_optional_by_type_name(type_name: str) -> Any | None:
    """Return the first service registered under a class with ``type_name``.

    Compatibility helper for migration shims: callers can support an old
    concrete registration key without importing that concrete class across
    layer boundaries.
    """
    for registered_type, instance in _services.items():
        if getattr(registered_type, "__name__", "") == type_name:
            return instance
    return None


def clear() -> None:
    """Clear all registrations (useful for testing)."""
    _services.clear()


def is_registered(service_type: type) -> bool:
    return service_type in _services


def snapshot() -> dict[type, Any]:
    """Return a shallow copy of the current registration map.

    Pair with ``restore`` to scope service registrations inside a
    ``with``/``async with`` block — used by ``BenchmarkSession`` so test
    runs do not leak service instances across sessions.
    """
    return dict(_services)


def restore(snapshot_map: dict[type, Any]) -> None:
    """Replace the registration map with ``snapshot_map``.

    Counterpart to :func:`snapshot`; ownership of ``snapshot_map`` is
    not retained — the registry stores its own copy.
    """
    _services.clear()
    _services.update(snapshot_map)
