"""The structured event vocabulary the Web UI consumes.

The Web layer must never parse ANSI/stdout to learn what the agent is doing.
Everything it needs arrives as one of these events, produced by
``adapter.py`` (lifecycle) and by the runtime observer it installs
(streaming / tool activity).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

# ── Lifecycle ────────────────────────────────────────────────────────────
SESSION_CREATED = "session_created"
QUEUED = "queued"
RUN_STARTED = "run_started"
RUN_COMPLETED = "run_completed"
RUN_FAILED = "run_failed"
RUN_CANCELLED = "run_cancelled"

# ── Progress ─────────────────────────────────────────────────────────────
ASSISTANT_DELTA = "assistant_delta"
ACTIVITY_STARTED = "activity_started"
ACTIVITY_FINISHED = "activity_finished"
ARTIFACT_CREATED = "artifact_created"
TASK_BOARD_UPDATED = "task_board_updated"
WARNING = "warning"

EVENT_TYPES: tuple[str, ...] = (
    SESSION_CREATED,
    QUEUED,
    RUN_STARTED,
    ASSISTANT_DELTA,
    ACTIVITY_STARTED,
    ACTIVITY_FINISHED,
    ARTIFACT_CREATED,
    TASK_BOARD_UPDATED,
    WARNING,
    RUN_FAILED,
    RUN_CANCELLED,
    RUN_COMPLETED,
)

#: A run is over — exactly one of these ends every run.
TERMINAL_EVENT_TYPES: frozenset[str] = frozenset(
    {RUN_COMPLETED, RUN_FAILED, RUN_CANCELLED},
)


@dataclass(frozen=True)
class DemoEvent:
    """One immutable, JSON-serialisable step of a demo run."""

    type: str
    session_id: str
    run_id: str = ""
    ts: float = field(default_factory=time.time)
    data: dict[str, Any] = field(default_factory=dict)

    @property
    def is_terminal(self) -> bool:
        return self.type in TERMINAL_EVENT_TYPES

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "session_id": self.session_id,
            "run_id": self.run_id,
            "ts": self.ts,
            **({"data": dict(self.data)} if self.data else {}),
        }


def event(
    type_: str,
    *,
    session_id: str,
    run_id: str = "",
    **data: Any,
) -> DemoEvent:
    """Build a :class:`DemoEvent`, keeping call sites short and uniform."""
    return DemoEvent(
        type=type_, session_id=session_id, run_id=run_id, data=dict(data),
    )


__all__ = [
    "ACTIVITY_FINISHED",
    "ACTIVITY_STARTED",
    "ARTIFACT_CREATED",
    "ASSISTANT_DELTA",
    "EVENT_TYPES",
    "QUEUED",
    "RUN_CANCELLED",
    "RUN_COMPLETED",
    "RUN_FAILED",
    "RUN_STARTED",
    "SESSION_CREATED",
    "TASK_BOARD_UPDATED",
    "TERMINAL_EVENT_TYPES",
    "WARNING",
    "DemoEvent",
    "event",
]
