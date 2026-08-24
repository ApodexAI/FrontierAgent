"""Budget limits for the orchestrator runtime."""

from __future__ import annotations

from pydantic import BaseModel, Field


class TaskBudget(BaseModel):
    """Allocated budget for a single research task.

    Uses soft limits (max_debate_rounds=0 means "prefer not to",
    not "absolutely forbidden") to avoid recreating the old
    topology-commit problem.
    """

    max_tokens: int = 500_000
    max_cost_usd: float | None = None
    max_wall_time_s: int = 300
    max_parallel: int = 3
    max_depth: int = 1  # v1: always 1 (no true recursion)
    max_search_calls: int = 20
    max_verify_passes: int = 5
    max_debate_rounds: int = 0
    default_model_tier: str = "medium"  # "light" | "medium" | "strong"
    role_tiers: dict[str, str] = Field(default_factory=dict)
