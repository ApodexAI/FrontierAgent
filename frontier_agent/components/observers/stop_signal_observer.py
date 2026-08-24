"""Cooperative stop-signal observer for sub-agents."""

from __future__ import annotations

import logging

from frontier_agent.components.agent_bus.stop_signal import (
    SubAgentStopRegistry,
    get_stop_registry,
)
from frontier_agent.core.loop_types import BaseObserver, Intervention, TurnContext

logger = logging.getLogger(__name__)

# Fixed message injected into the sub-agent's context on a stop request.
# Intent: halt exploration immediately; submit_report only if there is
# genuinely valuable information, otherwise stop without reporting.
STOP_SIGNAL_PROMPT = (
    "[stop signal] The coordinator has asked you to STOP immediately. "
    "Do not start any new searches, fetches, code runs, or other "
    "exploration. If you already have genuinely valuable findings, call "
    "`submit_report` now with a brief report of what you have so far. If "
    "you have nothing worth reporting, do NOT call submit_report — just "
    "stop here."
)


class StopSignalObserver(BaseObserver):
    """Interrupt a sub-agent the moment a cooperative stop is requested.

    Bound to a single sub-agent via ``session_id`` so the main agent can
    stop one sub-agent without affecting its siblings.

    Fires on ``on_llm_response`` (LLM responded, tools not yet run) rather
    than ``on_turn_end`` so the stop takes effect *immediately* — we don't
    let the sub-agent burn one more exploration round before reacting. The
    only delay is physical: a stop requested while the sub-agent is mid
    tool-call lands at its next LLM response.
    """

    critical = True

    def __init__(
        self,
        session_id: str,
        registry: SubAgentStopRegistry | None = None,
    ) -> None:
        self._session_id = str(session_id)
        self._registry = registry or get_stop_registry()

    async def on_llm_response(self, ctx: TurnContext) -> Intervention | None:
        if not self._registry.consume(self._session_id):
            return None
        logger.info(
            "Stop signal -> interrupting session=%s at turn=%d "
            "(rolling back this turn, injecting stop prompt)",
            self._session_id,
            ctx.turn,
        )
        # Rollback mode (agent_loop Step 6.5): pop this turn's LLM response so
        # the exploration tool_calls it just decided on are never executed
        # (and no dangling tool_calls are left to 400 the next request),
        # inject the stop prompt, and jump straight to the next turn. If the
        # sub-agent instead chose submit_report this turn, FinalizeAnswer
        # Observer's stop_reason fires first and the loop ends before here.
        return Intervention(
            inject_messages=[STOP_SIGNAL_PROMPT],
            pop_last_message=True,
            continue_to_next_turn=True,
        )


__all__ = ["STOP_SIGNAL_PROMPT", "StopSignalObserver"]
