"""Warnings for token budgets that cannot hold at the same time."""

from __future__ import annotations

from typing import Any

from agent_core.runtime.loop.budget_consistency import (
    check_context_budget as _check_context_budget,
)

from frontier_agent.core.runtime.loop.tiered_compact import compaction_trigger_ratio


def check_context_budget(**kwargs: Any) -> list[str]:
    """Check against the same trigger ratio the product's compactor uses."""
    kwargs.setdefault("ratio", compaction_trigger_ratio())
    return _check_context_budget(**kwargs)


__all__ = ["check_context_budget"]
