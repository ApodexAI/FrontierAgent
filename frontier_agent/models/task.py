"""Task and research request models."""

from __future__ import annotations

from datetime import UTC, datetime

from pydantic import BaseModel, Field

from frontier_agent.core.types import TaskId, TaskStatus, new_task_id


class ResearchRequest(BaseModel):
    """User-submitted research question with optional configuration."""

    question: str
    mode: str = "deep"  # quick | deep | heavy_duty
    depth: str = "standard"  # standard | deep
    max_sources: int = 20
    language: str = "auto"  # auto | en | zh
    pipeline_id: str = "auto"


class Task(BaseModel):
    """A research task — the OS 'process' abstraction."""

    id: TaskId = Field(default_factory=new_task_id)
    thread_id: str = ""  # runtime checkpoint/resume thread id
    parent_task_id: str | None = None
    status: TaskStatus = TaskStatus.CREATED
    current_phase: str = ""
    pipeline_id: str = "auto"
    request: ResearchRequest
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    error: str | None = None
    context: dict = Field(default_factory=dict)

    # Result pointers
    report_id: str | None = None
    evidence_count: int = 0
    assertion_count: int = 0

    # Pin / favorite
    pinned: bool = False
    pinned_at: datetime | None = None

    # Title (user-editable; NULL → UI falls back to input_text preview)
    title: str | None = None

    # Archive (soft delete)
    archived: bool = False
    archived_at: datetime | None = None

    def set_status(self, status: TaskStatus) -> None:
        self.status = status
        self.updated_at = datetime.now(UTC)
