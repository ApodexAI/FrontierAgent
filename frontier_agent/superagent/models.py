"""Business tasks outlive individual workflow attempts and client connections."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


class Conflict(ValueError):
    """A command conflicts with current task state or its idempotency key."""


class Fenced(Conflict):
    """An executor no longer owns the right to advance a task."""


@dataclass(frozen=True)
class Budget:
    seconds: int = 3600
    model_calls: int = 200
    attempts: int = 3
    parallel_agents: int = 4

    def __post_init__(self) -> None:
        for value in (self.seconds, self.model_calls, self.attempts, self.parallel_agents):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError("All budget limits must be positive integers")


@dataclass(frozen=True)
class Ticket:
    id: str
    title: str
    goal: str
    workflow: str
    status: str
    budget: Budget
    seconds_used: float
    calls_used: int
    tokens_used: int
    revision: int
    attempts: int
    result: str
    reason: str
    created_at: float
    updated_at: float


@dataclass(frozen=True)
class Assignment:
    id: str
    ticket_id: str
    owner: str
    revision: int
    started_at: float
    lease_until: float
    deadline: float
    status: str


Json = dict[str, Any]
