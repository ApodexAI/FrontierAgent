"""PlanningGateObserver — enforce Planning Mode inside a single loop."""

from __future__ import annotations

import logging

from frontier_agent.core.execution_context import get_current_execution_scope
from frontier_agent.core.loop_types import (
    BaseObserver,
    Intervention,
    ToolCallIntervention,
    TurnContext,
)
from plugins.tools._bus_scope import resolve_bus_task_id

logger = logging.getLogger(__name__)


class PlanningGateObserver(BaseObserver):
    critical = True

    def __init__(self, planning_max_turns: int = 40) -> None:
        self._cap = max(1, int(planning_max_turns))
        self._capped = False

    async def on_tool_call(
        self, ctx: TurnContext, tool_call: dict,
    ) -> ToolCallIntervention | None:
        from plugins.tools.task_board import in_planning, planning_block_message

        if not in_planning(ctx.task_id):
            return None
        name = str(tool_call.get("name", ""))
        block = planning_block_message(ctx.task_id, name)
        if block:
            return ToolCallIntervention(skip_with_result=block)
        return None

    async def on_turn_end(self, ctx: TurnContext) -> Intervention | None:
        from plugins.tools.task_board import (
            force_finish_planning,
            in_planning,
            render_board,
        )

        if self._capped or not in_planning(ctx.task_id):
            return None
        # ctx.turn is 0-based, so turn index (cap-1) is the cap'th turn.
        if ctx.turn + 1 < self._cap:
            return None
        self._capped = True
        force_finish_planning(ctx.task_id)
        logger.info(
            "PlanningGate: planning_max_turns=%d reached (task=%s) — "
            "auto-finishing planning, entering execution", self._cap, ctx.task_id,
        )
        scope = get_current_execution_scope()
        bus_task_id = resolve_bus_task_id(scope) if scope is not None else None
        return Intervention(inject_messages=[
            "Planning budget exhausted — you are now in EXECUTION mode. Stop "
            "planning and BUILD THE TEAM: create_subagent + assign_task to "
            "dispatch every task-board item to sub-agents, then collect, "
            "verify, and finalize. Do NOT solve the tasks yourself.\n"
            + render_board(ctx.task_id, bus_task_id=bus_task_id)
        ])
