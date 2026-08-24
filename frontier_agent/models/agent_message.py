"""AgentMessage — typed inter-agent communication protocol.

All fields use str (not Enum) for maximum extensibility —
any agent can define its own message_type values.
"""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

from pydantic import BaseModel, Field


class AgentMessage(BaseModel):
    """A message between two agents, logged as a KernelEvent."""

    id: str = Field(default_factory=lambda: f"msg-{uuid4().hex[:8]}")
    task_id: str
    from_agent: str          # role_id of sender
    to_agent: str            # role_id of recipient
    message_type: str        # free-form: "assertion", "dispute", "delegation", etc.
    content: dict = Field(default_factory=dict)  # payload varies by message_type
    parent_id: str | None = None  # links to triggering message for chain tracing
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
