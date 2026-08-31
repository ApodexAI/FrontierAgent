"""Periodic task-board reminder shared by coordinator workflows."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from frontier_agent.core.execution_context import get_current_execution_scope
from frontier_agent.core.loop_types import BaseObserver, Intervention, TurnContext

BoardSize = Callable[[str], int]
BoardRenderer = Callable[[str, str | None], str]
BusTaskResolver = Callable[[Any], str | None]


class TaskBoardObserver(BaseObserver):
    """Re-inject a non-empty task board after a configurable cooldown."""

    # This observer returns an Intervention.  Non-critical observer hooks are
    # fire-and-forget and their return values are deliberately not collected by
    # notify_observers, so this must remain critical for reminders to reach the
    # next LLM turn.
    critical = True

    def __init__(
        self,
        *,
        board_size: BoardSize,
        render_board: BoardRenderer,
        resolve_bus_task_id: BusTaskResolver,
        cooldown_turns: int = 5,
    ) -> None:
        self._board_size = board_size
        self._render_board = render_board
        self._resolve_bus_task_id = resolve_bus_task_id
        self._cooldown = max(1, int(cooldown_turns))
        self._last_fired = -10_000

    async def on_turn_end(self, ctx: TurnContext) -> Intervention | None:
        if self._board_size(ctx.task_id) == 0:
            return None
        if ctx.turn - self._last_fired < self._cooldown:
            return None
        self._last_fired = ctx.turn
        scope = get_current_execution_scope()
        bus_task_id = (
            self._resolve_bus_task_id(scope) if scope is not None else None
        )
        return Intervention(inject_messages=[
            "Current task board (keep it current via add_task / update_task; "
            "finalize only once every task is resolved):\n"
            + self._render_board(ctx.task_id, bus_task_id)
        ])


__all__ = ["TaskBoardObserver"]
