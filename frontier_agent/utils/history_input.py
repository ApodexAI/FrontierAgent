"""Helpers for inputs produced by legacy multi-turn clients."""

from __future__ import annotations

_CURRENT_QUERY_MARKERS = (
    "\nanswer the query: ",
    "\nanswer the query：",
)


def extract_current_query(text: str) -> str:
    """Return the current query from a legacy wrapped history prompt."""
    if not text:
        return ""
    best_idx = -1
    best_len = 0
    for marker in _CURRENT_QUERY_MARKERS:
        idx = text.rfind(marker)
        if idx > best_idx:
            best_idx = idx
            best_len = len(marker)
    if best_idx < 0:
        return text
    return text[best_idx + best_len:].strip()
