"""Adapt AgentLoopResult to legacy dict format expected by callers."""
from __future__ import annotations

from typing import Any

from frontier_agent.core.loop_types import AgentLoopResult


def adapt_result(result: AgentLoopResult) -> dict[str, Any]:
    """Convert AgentLoopResult -> legacy orchestrator return format.

    Legacy format keys: evidence_cards, assertions, react_steps,
    skills_used, messages (activity-log style), final_content,
    clarified_questions.
    """
    meta = result.metadata
    react_steps = meta.get("react_steps", [])

    # Build activity-log style messages from react_steps
    activity_log: list[dict[str, Any]] = []
    for step in react_steps:
        activity_log.append({
            "agent": meta.get("role_id", "react_solver"),
            "action": "tool_call",
            "detail": (
                f"Turn {step.get('turn', '?')}: "
                f"{step.get('tool_name', '?')}"
                f"({step.get('tool_args', '')[:80]})"
            ),
        })

    # When `finalize_answer` was called, prefer its payload — the last
    # AIMessage may be empty (the tool-call turn itself) or a terser
    # "I'll now submit" note rather than the real answer.
    final_content = meta.get("final_answer") or result.final_content

    return {
        "evidence_cards": meta.get("evidence_cards", []),
        "assertions": meta.get("assertions", []),
        "react_steps": react_steps,
        "skills_used": meta.get("skills_used", []),
        "messages": activity_log,
        "final_content": final_content,
        "clarified_questions": meta.get("clarified_questions", []),
    }
