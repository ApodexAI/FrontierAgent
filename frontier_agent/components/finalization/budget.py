"""Wall-clock arithmetic for workflows with a research-only deadline."""

from __future__ import annotations

import logging
import math
import os
import time
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

TASK_WALL_TIME_ENV = "FRONTIER_AGENT_TASK_WALL_TIME_S"


def positive_seconds(raw: object, *, label: str) -> float | None:
    """Parse a positive duration; invalid/disabled values contribute no cap."""
    if raw is None or raw == "":
        return None
    try:
        value = float(raw)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        logger.warning("Invalid %s=%r; ignoring it", label, raw)
        return None
    if not math.isfinite(value):
        logger.warning("Non-finite %s=%r; ignoring it", label, raw)
        return None
    return value if value > 0 else None


def nonnegative_seconds(raw: object, *, default: float, label: str) -> float:
    """Parse a non-negative duration with a tolerant profile fallback."""
    if raw is None or raw == "":
        return float(default)
    try:
        value = float(raw)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        logger.warning("Invalid %s=%r; using %.0fs", label, raw, default)
        return float(default)
    if not math.isfinite(value) or value < 0:
        logger.warning("Invalid non-negative %s=%r; using %.0fs", label, raw, default)
        return float(default)
    return value


def soft_wall_deadline_s(total_s: float, reserve_s: float) -> float:
    """Convert a hard task wall into a research deadline with finalize grace.

    The ``total_s * 0.5`` floor keeps a generous reserve from starving research
    on a short wall. When the floor binds, the reserve that actually survives is
    smaller than ``reserve_s`` — callers MUST pair this with
    :func:`remaining_phase_budget_s` so the finalization stage shrinks to the
    time that is really left instead of assuming it got its full reserve.
    """
    return max(total_s - reserve_s, total_s * 0.5)


@dataclass(frozen=True)
class ResearchWall:
    """Resolved research deadline plus the external ceiling it came from."""

    research_deadline_s: float
    """When the in-loop observer stops research. ``0`` = no in-loop wall."""

    hard_total_s: float
    """Externally enforced whole-task ceiling. ``0`` = none known."""


def resolve_research_wall(
    agent_cfg: dict[str, Any],
    *,
    reserve_s: float,
    label_prefix: str,
    env_var: str = TASK_WALL_TIME_ENV,
) -> ResearchWall:
    """Resolve the research deadline from profile budget and platform wall.

    ``research_wall_time_s`` is already a research-only budget and is not
    shortened. The legacy ``wall_deadline_s`` and the operational env value are
    total-task ceilings, so finalize grace is subtracted before comparing them —
    and they are also what :attr:`ResearchWall.hard_total_s` reports.
    """
    research_budget = "research_wall_time_s" in agent_cfg
    if research_budget:
        profile_raw = agent_cfg.get("research_wall_time_s")
    elif "wall_deadline_s" in agent_cfg:
        profile_raw = agent_cfg.get("wall_deadline_s")
    else:
        profile_raw = None
    profile_s = positive_seconds(
        profile_raw,
        label=f"{label_prefix} research wall deadline",
    )
    env_s = positive_seconds(os.environ.get(env_var), label=env_var)

    profile_deadline_s = profile_s
    if profile_deadline_s is not None and not research_budget:
        profile_deadline_s = soft_wall_deadline_s(profile_deadline_s, reserve_s)
    env_deadline_s = (
        soft_wall_deadline_s(env_s, reserve_s) if env_s is not None else None
    )
    candidates = [
        value
        for value in (profile_deadline_s, env_deadline_s)
        if value is not None
    ]

    # Only total-task values are hard ceilings; a research-only profile budget
    # is not one, because the reporter is deliberately outside it.
    hard_candidates = [
        value
        for value in (None if research_budget else profile_s, env_s)
        if value is not None
    ]
    return ResearchWall(
        research_deadline_s=min(candidates) if candidates else 0.0,
        hard_total_s=min(hard_candidates) if hard_candidates else 0.0,
    )


def remaining_phase_budget_s(
    requested_s: float,
    deadline_monotonic_s: float | None,
    *,
    minimum_s: float = 1.0,
) -> float:
    """Clamp a finalization phase ceiling to the time that is actually left.

    ``deadline_monotonic_s`` is a :func:`time.monotonic` instant, normally
    ``node_start + hard_total_s``. Returns ``requested_s`` unchanged when no
    external ceiling is known. ``minimum_s`` keeps the phase from being handed a
    zero/negative timeout — it still gets one short attempt and then fails open
    to its baseline answer, which is strictly better than being cancelled by the
    external ceiling with nothing to show.
    """
    if deadline_monotonic_s is None:
        return requested_s
    remaining = deadline_monotonic_s - time.monotonic()
    return max(min(float(requested_s), remaining), float(minimum_s))


def check_wall_feasibility(
    *,
    hard_total_s: float,
    research_deadline_s: float,
    tool_timeout_s: float,
    landing_budget_s: float,
    label_prefix: str,
) -> bool:
    """Warn when no schedule can honour the external ceiling; return feasibility.

    :func:`remaining_phase_budget_s` shrinks the finalization phase to fit, but
    it cannot shrink a tool call that is already running. When
    ``research_deadline_s + tool_timeout_s`` alone exceeds ``hard_total_s``, a
    single tool started just before the research deadline blows the wall no
    matter what the finalization stage does — the config itself is the problem
    (usually ``tool_timeout_s`` larger than half the wall). Say so loudly rather
    than letting the run get killed with no answer and no explanation.
    """
    if hard_total_s <= 0 or research_deadline_s <= 0:
        return True
    worst_case_s = research_deadline_s + max(tool_timeout_s, 0.0)
    if worst_case_s <= hard_total_s:
        return True
    logger.warning(
        "%s: wall-time config cannot guarantee a final answer — research stops "
        "at %.0fs and one late tool call may run to %.0fs, past the %.0fs hard "
        "ceiling, leaving no room for the %.0fs finalization phase. Lower "
        "tool_timeout_s (to at most half the wall) or raise the wall.",
        label_prefix,
        research_deadline_s,
        worst_case_s,
        hard_total_s,
        landing_budget_s,
    )
    return False


__all__ = [
    "TASK_WALL_TIME_ENV",
    "ResearchWall",
    "check_wall_feasibility",
    "nonnegative_seconds",
    "positive_seconds",
    "remaining_phase_budget_s",
    "resolve_research_wall",
    "soft_wall_deadline_s",
]
