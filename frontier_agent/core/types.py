"""Base types for FrontierAgent kernel and application layers."""

from __future__ import annotations

from enum import StrEnum
from typing import NewType
from uuid import uuid4

# ── Identity types ──────────────────────────────────────────────────────────

TaskId = NewType("TaskId", str)
EventId = NewType("EventId", str)
SessionId = NewType("SessionId", str)
AgentSessionId = NewType("AgentSessionId", str)  # 12-char hex
PromptId = NewType("PromptId", str)
StepId = NewType("StepId", str)


def new_task_id() -> TaskId:
    return TaskId(uuid4().hex[:12])


def new_session_id() -> SessionId:
    return SessionId(uuid4().hex[:12])


def new_prompt_id() -> PromptId:
    return PromptId(uuid4().hex[:12])


def new_step_id() -> StepId:
    return StepId(uuid4().hex[:10])


# ── Enumerations ────────────────────────────────────────────────────────────


class TaskStatus(StrEnum):
    CREATED = "created"
    RUNNING = "running"
    SUSPENDED = "suspended"
    COMPLETED = "completed"
    FAILED = "failed"
    ABORTED = "aborted"


# Generic agent role identifier — use a free-form string, resolved through
# AgentRegistry at runtime. Workflows and tests register the concrete roles
# they need; the kernel stays role-agnostic.
AgentRoleId = NewType("AgentRoleId", str)
