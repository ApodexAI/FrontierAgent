"""Workflow-scoped scheduler defaults for stateful-react-agent."""

from __future__ import annotations

# The profile research budget / env-derived soft deadline is enforced by
# WallClockDeadlineObserver around research only. The scheduler uses this much
# larger ceiling solely as a catastrophic whole-graph backstop.
TASK_WALL_TIME_MODE = "soft_research"
TASK_WALL_TIME_S: int = 86400
