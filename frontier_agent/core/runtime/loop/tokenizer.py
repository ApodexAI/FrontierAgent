"""Provide non-blocking access to tiktoken encoders.

Cache misses may perform network I/O, so initialization runs on a daemon
thread and callers use a character heuristic until it completes.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

logger = logging.getLogger(__name__)

# Sentinel for "this name has never been requested" — distinct from the
# ``None`` we store to mark a load that is in flight.
_MISSING = object()

# Per-encoding cache. State machine for a given name:
#   absent (==_MISSING) → never requested
#   None                → background init in flight; use the heuristic for now
#   False               → tiktoken unavailable / bad name; terminal, heuristic forever
#   <Encoding object>   → ready
_encoders: dict[str, Any] = {}
_lock = threading.Lock()


def _load(name: str) -> None:
    """Blocking tiktoken init — only ever runs on a daemon thread."""
    enc: Any
    try:
        import tiktoken
        enc = tiktoken.get_encoding(name)
    except Exception:
        enc = False
        logger.debug("tiktoken encoding %r unavailable; using heuristic", name)
    with _lock:
        _encoders[name] = enc


def get_encoding_nonblocking(name: str = "cl100k_base") -> Any | None:
    """Return the cached tiktoken encoder for ``name`` without ever blocking.

    The first call schedules a daemon-thread init and returns ``None``;
    later calls return the encoder once it has loaded, or ``None`` while
    it is still loading. Returns ``None`` permanently when tiktoken is
    unavailable — callers MUST fall back to a heuristic on ``None``.
    """
    enc = _encoders.get(name, _MISSING)
    if enc is not _MISSING:
        return enc or None  # None (loading) and False (failed) both collapse to None
    with _lock:
        if _encoders.get(name, _MISSING) is _MISSING:  # still unclaimed under the lock
            _encoders[name] = None  # mark loading so concurrent callers don't re-spawn
            threading.Thread(
                target=_load,
                args=(name,),
                name=f"tiktoken-init-{name}",
                daemon=True,
            ).start()
    return None
