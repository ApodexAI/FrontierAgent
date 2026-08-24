"""Pure presentation state for the Textual front end."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum


class PresentationPhase(StrEnum):
    IDLE = "idle"
    THINKING = "thinking"
    RESPONDING = "responding"
    RUNNING_TOOL = "running_tool"
    AWAITING_APPROVAL = "awaiting_approval"
    DONE = "done"
    INCOMPLETE = "incomplete"
    INTERRUPTED = "interrupted"
    ERROR = "error"


_TERMINAL_PHASES = frozenset({
    PresentationPhase.DONE,
    PresentationPhase.INCOMPLETE,
    PresentationPhase.INTERRUPTED,
    PresentationPhase.ERROR,
})


@dataclass
class TuiPresentationState:
    """Small, UI-only task lifecycle shared by the sink and status bar."""

    phase: PresentationPhase = PresentationPhase.IDLE
    task_started_at: float | None = None
    task_finished_at: float | None = None
    current_tool: str = ""
    queued_steers: int = 0
    idle_after: float | None = None
    terminal_hold_seconds: float = 2.0
    clock: Callable[[], float] = field(default=time.monotonic, repr=False, compare=False)

    @property
    def terminal(self) -> bool:
        return self.phase in _TERMINAL_PHASES

    def begin_task(self) -> None:
        now = self.clock()
        self.phase = PresentationPhase.THINKING
        self.task_started_at = now
        self.task_finished_at = None
        self.current_tool = ""
        self.queued_steers = 0
        self.idle_after = None

    def transition(self, phase: PresentationPhase, *, tool: str = "") -> None:
        if self.terminal:
            return
        if self.task_started_at is None and phase != PresentationPhase.IDLE:
            self.task_started_at = self.clock()
        self.phase = phase
        self.current_tool = tool if phase in {
            PresentationPhase.RUNNING_TOOL,
            PresentationPhase.AWAITING_APPROVAL,
        } else ""

    def finish(self, phase: PresentationPhase = PresentationPhase.DONE) -> None:
        if self.terminal:
            return
        if phase not in _TERMINAL_PHASES:
            raise ValueError(f"not a terminal presentation phase: {phase}")
        now = self.clock()
        if self.task_started_at is None:
            self.task_started_at = now
        self.phase = phase
        self.task_finished_at = now
        self.current_tool = ""
        self.idle_after = now + self.terminal_hold_seconds

    def interrupt(self) -> None:
        """Record user cancellation even if a final render raced just ahead of it."""
        now = self.clock()
        if self.task_started_at is None:
            self.task_started_at = now
        self.phase = PresentationPhase.INTERRUPTED
        self.task_finished_at = now
        self.current_tool = ""
        self.idle_after = now + self.terminal_hold_seconds

    def resume_after_error(self) -> None:
        """Continue the same task after a recoverable error was rendered."""
        if self.phase != PresentationPhase.ERROR:
            return
        self.phase = PresentationPhase.THINKING
        self.task_finished_at = None
        self.current_tool = ""
        self.idle_after = None

    def set_queued_steers(self, count: int) -> None:
        self.queued_steers = max(0, int(count))

    def elapsed_seconds(self) -> int | None:
        if self.task_started_at is None:
            return None
        end = self.task_finished_at if self.task_finished_at is not None else self.clock()
        return max(0, int(end - self.task_started_at))

    def settle(self) -> bool:
        """Return to idle after the terminal result has remained visible."""
        if self.idle_after is None or self.clock() < self.idle_after:
            return False
        self.reset()
        return True

    def reset(self) -> None:
        self.phase = PresentationPhase.IDLE
        self.task_started_at = None
        self.task_finished_at = None
        self.current_tool = ""
        self.queued_steers = 0
        self.idle_after = None


def format_elapsed(seconds: int | None) -> str:
    if seconds is None:
        return ""
    if seconds < 60:
        return f"{seconds}s"
    minutes, remainder = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes}m{remainder:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m"


__all__ = ["PresentationPhase", "TuiPresentationState", "format_elapsed"]
