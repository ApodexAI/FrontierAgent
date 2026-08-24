"""finalize_answer — terminal tool that ends the main-agent loop."""

from __future__ import annotations

import logging

from frontier_agent.components.agent_bus import AgentBus
from frontier_agent.core.execution_context import get_current_execution_scope
from frontier_agent.core.runtime.registries import services as registry
from frontier_agent.core.tool import tool
from plugins.tools._bus_scope import resolve_bus_task_id

logger = logging.getLogger(__name__)


def _unassigned_agent_names(task_id: str) -> list[str]:
    """Names of sub-agents created in this task but never assigned a task."""
    bus = registry.get_optional(AgentBus)
    if bus is None:
        return []
    return [
        s.name for s in bus.list_sessions_for_task(task_id)
        if s.total_task_count == 0
    ]


def finalize_gate(
    task_id: str,
    content: str,
    *,
    bus_task_id: str | None = None,
) -> str | None:
    """Shared pre-submission gate for the main agent's FINAL answer.

    Returns an error string when finalizing must be BLOCKED, else ``None``.
    Identical regardless of HOW the agent signals "done" — used by the
    ``finalize_answer`` tool and by agent-team's
    bare-text terminator (``BareTextFinalizeObserver``).

    Checks (in order): empty answer (blocks) · sub-agents created but never
    assigned (WARN-only, never blocks) · task-board items still open/in_progress
    (blocks) · (Planning Mode only) solo submission (blocks) · (Planning Mode
    only) answer not independently verified (blocks). The board / planning
    checks are no-ops outside agent-team (no board → no pending items; planning
    checks gated on ``planning_enabled``).
    """
    if not (content or "").strip():
        return "Your answer is empty — produce the full answer text."
    agent_bus_task_id = bus_task_id or task_id
    idle = _unassigned_agent_names(agent_bus_task_id)
    if idle:
        # Do NOT hard-block on idle sub-agents. This check used to return an
        # error to force the agent to assign them, but that can trap the run:
        # when an idle agent cannot actually be assigned (every session hit the
        # 5-task limit, or a name the model can no longer reach), the
        # coordinator spins on futile create/assign retries instead of
        # delivering the answer it already has. Emit a warning for
        # observability and let the finish proceed — the loop-time
        # UnassignedAgentNudge still surfaces genuinely-forgotten work early so
        # it gets assigned in time, and (in Planning Mode) the "no sub-agent did
        # any work" check below still blocks a truly empty solo submission.
        logger.warning(
            "finalize_gate: %d sub-agent(s) created but never assigned "
            "(%s) — allowing finish anyway, not blocking",
            len(idle), idle,
        )
    from plugins.tools.task_board import (
        in_planning,
        planning_enabled,
        unresolved_task_ids,
    )
    pending = unresolved_task_ids(task_id)
    if pending:
        return (
            f"Cannot finish: task board has unresolved item(s) {pending}. For "
            "each, call update_task(id, resolution='resolved') once it is "
            "answered AND corroborated, or resolution='cancelled' if no longer "
            "needed — then deliver your answer."
        )
    if in_planning(task_id):
        return (
            "Cannot finish: you are still in PLANNING MODE. Decompose the "
            "question into the task board, then call finish_planning to start "
            "the team — you cannot submit an answer during planning."
        )
    if planning_enabled(task_id):
        bus = registry.get_optional(AgentBus)
        sessions = bus.list_sessions_for_task(agent_bus_task_id) if bus else []
        worked = [s for s in sessions if getattr(s, "total_task_count", 0) > 0]
        if not worked:
            return (
                "Cannot finish: no sub-agent has done any work — this is a solo "
                "submission, which Planning Mode forbids. Delegate the task-board "
                "items to sub-agents (assign_task) and let them complete first."
            )
        verifiers = [s for s in worked if "verif" in (s.name or "").lower()]
        if not verifiers:
            return (
                "Cannot finish: the answer has not been independently verified. "
                "Spawn a `final_verifier` sub-agent to rigorously re-derive and "
                "check every value against sources; finish only once it confirms."
            )
    return None


@tool
async def finalize_answer(content: str, confidence: float = 0.7) -> str:
    """Submit the final answer and end the research loop.

    Call this when you have gathered enough evidence and are ready to deliver
    the answer. After this call, the solver loop terminates — no further
    tool calls will execute.

    Args:
        content: The complete answer in Markdown. Include inline ``[N]``
            citations; the pipeline appends a References section automatically.
        confidence: Self-assessed confidence in [0, 1] that the answer is
            correct and well-grounded.

    Returns:
        Short acknowledgement. The actual answer bytes live in loop metadata.
    """
    text = (content or "").strip()
    if not text:
        return (
            "finalize_answer rejected: `content` was empty. "
            "Pass the full answer in `content`."
        )

    # Must ``raise`` rather than ``return`` — FinalizeAnswerObserver only
    # skips latching when ``result.is_error`` is True. A string return here
    # would silently lock in the answer despite the "blocked" message. Gate
    # logic is shared with agent-team's bare-text terminator via finalize_gate.
    scope = get_current_execution_scope()
    if scope is not None:
        err = finalize_gate(
            scope.task_id,
            text,
            bus_task_id=resolve_bus_task_id(scope),
        )
        if err:
            raise ValueError(err)

    try:
        conf = max(0.0, min(1.0, float(confidence)))
    except (TypeError, ValueError):
        conf = 0.7

    logger.info(
        "finalize_answer accepted: len=%d chars, confidence=%.2f",
        len(text), conf,
    )
    return (
        f"Final answer accepted ({len(text)} chars, confidence={conf:.2f}). "
        "Solver loop will exit after this turn."
    )
