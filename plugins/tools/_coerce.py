"""Coerce tool arguments that some models serialize as JSON strings."""

from __future__ import annotations

import json
from typing import Any


def coerce_json_list(value: Any) -> Any:
    """Return ``value`` unchanged, or parse it if it's a JSON-list string.

    ``value`` is returned as-is when it's already a list, isn't a string,
    doesn't look like JSON, or fails to parse. A bare JSON object is
    wrapped in a single-element list so the caller sees a uniform shape.
    """
    if not isinstance(value, str):
        return value
    s = value.strip()
    if not (s.startswith("[") or s.startswith("{")):
        return value
    try:
        parsed = json.loads(s)
    except (json.JSONDecodeError, ValueError):
        return value
    if isinstance(parsed, dict):
        return [parsed]
    return parsed


def coerce_json_object(value: Any) -> dict[str, Any] | None:
    """Return ``value`` as an object, or ``None`` if it isn't usable.

    Tolerates a double-encoded JSON-object string (``'{"description": "..."}'``)
    the same way :func:`coerce_json_list` tolerates a double-encoded list at the
    top level — some models serialise each list element as its own JSON string.
    A plain (non-JSON) string is NOT coerced: callers surface it as an error so
    the model re-sends the documented shape rather than silently losing the item.
    """
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        parsed = coerce_json_list(value)
        if isinstance(parsed, list) and len(parsed) == 1 and isinstance(parsed[0], dict):
            return parsed[0]
    return None


__all__ = ["coerce_json_list", "coerce_json_object"]
