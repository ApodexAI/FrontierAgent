"""Pluggable message trimmers for sub-agent session history."""

from __future__ import annotations

import logging
from typing import Protocol, runtime_checkable

from frontier_agent.core.messages import Message, is_assistant_msg

logger = logging.getLogger(__name__)

# A task span: (start_index, end_index_or_None). ``None`` marks the
# currently running task — everything from ``start`` to the tail belongs
# to it.
TaskBoundary = tuple[int, int | None]


@runtime_checkable
class MessageTrimmer(Protocol):
    """Protocol for trimming sub-agent session history."""

    def trim(
        self,
        messages: list[Message],
        boundaries: list[TaskBoundary] | None = None,
    ) -> list[Message]:
        """Return the slice of ``messages`` to seed the next loop run with."""
        ...


class NullTrimmer:
    """Pass through — return messages unchanged.

    Default for sub-agent sessions; matches the current one-shot behavior
    where no trimming happens.
    """

    def trim(
        self,
        messages: list[Message],
        boundaries: list[TaskBoundary] | None = None,
    ) -> list[Message]:
        return list(messages)


class TaskBoundaryTrimmer:
    """Task-boundary-aware trimming for a reused, multi-task session.

    For a session that has completed N tasks and is running task N+1:
    returns
        [for each completed task: task_user_prompt, final_assistant_report]
        + [current_task_user_prompt, *everything_after_in_current_task]

    Tool calls, intermediate assistant turns, and tool results from
    *completed* tasks are dropped. This is the context compression that
    lets a reused sub-agent keep its task-level memory without blowing up
    token usage.

    The trimmer never returns system messages — the caller prepends the
    ``SystemMessage`` externally. The indices in ``boundaries`` refer to
    ``messages`` (no system offset).
    """

    def trim(
        self,
        messages: list[Message],
        boundaries: list[TaskBoundary] | None = None,
    ) -> list[Message]:
        if not boundaries or not messages:
            return list(messages)

        # If every boundary but the last is open, this is still the first
        # task — no compaction needed.
        completed = [b for b in boundaries if b[1] is not None]
        if not completed:
            return list(messages)

        trimmed: list[Message] = []
        for start, end in boundaries:
            if start < 0 or start >= len(messages):
                continue
            if end is None:
                # In-flight task: include everything from start to tail.
                trimmed.extend(messages[start:])
                continue

            # Completed task: keep the user task prompt, drop intermediate
            # turns, keep only the final assistant message (no tool_calls).
            trimmed.append(messages[start])
            final_ai = find_final_assistant(messages, start + 1, end)
            if final_ai is not None:
                trimmed.append(final_ai)

        logger.debug(
            "TaskBoundaryTrimmer: %d messages → %d (%d completed boundaries)",
            len(messages), len(trimmed), len(completed),
        )
        return trimmed


def find_final_assistant(
    messages: list[Message],
    start: int,
    end: int,
) -> Message | None:
    """Scan backwards from ``end`` to ``start`` for the last AIMessage
    that isn't just a tool-call stub (i.e. has actual text content).

    Public — used by ``TaskBoundaryTrimmer`` to find a completed task's
    final report and by ``AgentBus`` to enforce the
    ``SubAgentSession`` "every closed boundary has a clean final
    AIMessage" invariant.
    """
    end = min(end, len(messages) - 1)
    for j in range(end, start - 1, -1):
        msg = messages[j]
        if not is_assistant_msg(msg):
            continue
        if msg.get("tool_calls"):
            continue
        return msg
    return None


def trim_and_remap_boundaries(
    messages: list[Message],
    boundaries: list[TaskBoundary],
) -> tuple[list[Message], list[TaskBoundary]]:
    """Atomic trim + boundary index remap for eager post-task compression.

    For each *completed* boundary, keeps only [task_prompt, final_ai].
    Open (in-flight) boundaries keep everything from their start to the tail.
    Boundary indices are remapped atomically so the returned pair is always
    internally consistent — safe to write back to ``SubAgentSession`` without
    exposing any intermediate state.

    Returns the original lists unchanged (same references) when there are no
    completed boundaries (first task still running, or NullTrimmer semantics).
    """
    if not boundaries or not messages:
        return messages, boundaries

    if not any(b[1] is not None for b in boundaries):
        return messages, boundaries

    new_messages: list[Message] = []
    new_boundaries: list[TaskBoundary] = []

    for start, end in boundaries:
        if start < 0 or start >= len(messages):
            continue
        new_start = len(new_messages)

        if end is None:
            # Open boundary: keep everything from start to tail.
            new_messages.extend(messages[start:])
            new_boundaries.append((new_start, None))
        else:
            # Completed boundary: task_prompt + final assistant message only.
            new_messages.append(messages[start])
            final_ai = find_final_assistant(messages, start + 1, end)
            if final_ai is not None:
                new_messages.append(final_ai)
                new_end = new_start + 1
            else:
                # Boundary invariant guarantees a final_ai exists; keep start
                # as both start and end as a conservative fallback.
                new_end = new_start
            new_boundaries.append((new_start, new_end))

    return new_messages, new_boundaries
