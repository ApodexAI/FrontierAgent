"""Workflow-scoped defaults for agent_team."""

from __future__ import annotations

# Per-task wall-time ceiling. 24h chosen to bound only catastrophic
# stuck-state cases — small models on multi-constraint BC200 questions
# can legitimately spend 8-12h in retry/fan-out loops, and we'd rather
# let them finish than abort. Combined with main_max_turns / sub_max_turns
# this still prevents true infinite loops.
TASK_WALL_TIME_S: int = 86400

# The workflow consumes its research budget / env-derived soft deadline inside
# the coordinator loop. Once that loop stops, the downstream reporter must be
# allowed to complete. Scheduler callers can still supply an explicit
# ``wall_time_s`` hard ceiling.
TASK_WALL_TIME_MODE = "soft_research"
