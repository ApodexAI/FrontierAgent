"""Mechanism shared by workflows that finalize gracefully at execution limits."""

from __future__ import annotations

from frontier_agent.components.finalization.budget import (
    TASK_WALL_TIME_ENV,
    ResearchWall,
    check_wall_feasibility,
    nonnegative_seconds,
    positive_seconds,
    remaining_phase_budget_s,
    resolve_research_wall,
    soft_wall_deadline_s,
)
from frontier_agent.components.finalization.recovery import (
    COMMON_RECOVERY_NUDGE_PREFIXES,
    RECOVERY_CONTEXT_MAX_CHARS,
    RECOVERY_ITEM_MAX_CHARS,
    build_recovery_context,
    chat_with_fallback_budget,
    fallback_leg_count,
    has_malformed_tool_protocol,
    is_recovery_nudge,
    minimal_best_effort_answer,
    truncate_text_to_tokens,
)

__all__ = [
    "COMMON_RECOVERY_NUDGE_PREFIXES",
    "RECOVERY_CONTEXT_MAX_CHARS",
    "RECOVERY_ITEM_MAX_CHARS",
    "TASK_WALL_TIME_ENV",
    "ResearchWall",
    "build_recovery_context",
    "chat_with_fallback_budget",
    "check_wall_feasibility",
    "fallback_leg_count",
    "has_malformed_tool_protocol",
    "is_recovery_nudge",
    "minimal_best_effort_answer",
    "nonnegative_seconds",
    "positive_seconds",
    "remaining_phase_budget_s",
    "resolve_research_wall",
    "soft_wall_deadline_s",
    "truncate_text_to_tokens",
]
