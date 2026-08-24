"""UnassignedAgentNudge — reminder when sub-agents were created but not assigned."""

from __future__ import annotations

import logging

from frontier_agent.components.agent_bus import AgentBus
from frontier_agent.core.loop_types import (
    BaseObserver,
    Intervention,
    TurnContext,
)
from frontier_agent.core.runtime import registry

logger = logging.getLogger(__name__)


class UnassignedAgentNudge(BaseObserver):
    critical: bool = True

    def __init__(self, *, cooldown_turns: int = 2) -> None:
        self._cooldown_turns = max(1, int(cooldown_turns))
        self._last_fired_turn: int = -1

    async def on_turn_end(self, ctx: TurnContext) -> Intervention | None:
        if ctx.turn - self._last_fired_turn < self._cooldown_turns:
            return None

        bus = registry.get_optional(AgentBus)
        if bus is None:
            return None

        sessions = bus.list_sessions_for_task(ctx.task_id)
        if not sessions:
            return None

        # Don't nudge when this turn already acted on it: (a) the agent
        # created agents this turn — give it the next turn to assign
        # naturally; (b) the agent DID call assign_task but the harness
        # rejected it (e.g. an output_paths contract violation),
        # so dispatched stayed 0. In case (b) the tool error already carries
        # the precise fix; stacking a generic "you never assigned them tasks"
        # on top is factually wrong (it literally just tried) and pulls
        # attention away from the real fix.
        attempted = {
            tc.get("name") for tc in (ctx.tool_calls or []) if isinstance(tc, dict)
        }
        if attempted & {"assign_task", "create_subagent"}:
            return None

        # Don't nag while the main agent is legitimately waiting on work.
        if any(s.current_job_id is not None for s in sessions):
            return None

        idle = [s.name for s in sessions if s.total_task_count == 0]
        if not idle:
            return None

        self._last_fired_turn = ctx.turn
        names = ", ".join(sorted(idle))
        logger.info("UnassignedAgentNudge: %s", idle)
        return Intervention(inject_messages=[
            f"You created agent(s) **{names}** but never assigned them "
            "tasks. Call `assign_task` for each before calling "
            "`finalize_answer` — idle agents block finalization and "
            "sub-agents cannot be silently dismissed. If you no longer "
            "need an agent's output, give it a throwaway task like "
            "\"Confirm no further info needed; submit an empty report\"."
        ])


__all__ = ["UnassignedAgentNudge"]
