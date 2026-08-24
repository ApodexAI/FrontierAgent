"""Kernel-generic event identifiers."""

from __future__ import annotations

from enum import StrEnum


class EventType(StrEnum):
    """Framework-mechanical events. Workflow domains add their own."""

    # Task lifecycle
    TASK_CREATED = "task_created"
    TASK_STATUS_CHANGED = "task_status_changed"
    PHASE_TRANSITION = "phase_transition"
    # Agent actions (generic — works for any agent role)
    AGENT_ACTION = "agent_action"
    AGENT_MESSAGE = "agent_message"      # inter-agent communication
    AGENT_TOOL_CALL = "agent_tool_call"  # agent invokes a tool
    # Generic tool invocation
    TOOL_CALLED = "tool_called"
    TOOL_RESULT = "tool_result"
    ERROR = "error"
    # Output / memory lifecycle (consumed by the scheduler + WorkingMemory,
    # so they are framework mechanics, not workflow vocabulary)
    REPORT_GENERATED = "report_generated"
    WORKING_MEMORY_SNAPSHOT = "working_memory_snapshot"
