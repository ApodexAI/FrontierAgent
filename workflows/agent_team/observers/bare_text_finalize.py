"""Terminate the main-agent loop on a plain-text, no-tool turn."""
from __future__ import annotations

import logging

from frontier_agent.core.execution_context import get_current_execution_scope
from frontier_agent.core.loop_types import BaseObserver, Intervention, TurnContext
from plugins.tools._bus_scope import resolve_bus_task_id
from plugins.tools.finalize_answer import finalize_gate

logger = logging.getLogger(__name__)


class BareTextFinalizeObserver(BaseObserver):
    critical = True

    async def on_llm_response(self, ctx: TurnContext) -> Intervention | None:
        if ctx.tool_calls:
            return None
        text = (ctx.ai_text or "").strip()
        if not text:
            # Nothing said and no tool — let the loop's no-tool recovery nudge.
            return None

        scope = get_current_execution_scope()
        err = finalize_gate(
            ctx.task_id,
            text,
            bus_task_id=resolve_bus_task_id(scope) if scope is not None else None,
        )
        # On the very last turn tools are stripped (LastTurnForcer), so the only
        # way to emit anything is plain text; accept it rather than lose the
        # answer to a max_turns stop even if the gate would otherwise block.
        last_turn = ctx.turn >= ctx.max_turns - 1
        if err and not last_turn:
            return Intervention(continue_to_next_turn=True, inject_messages=[err])

        if isinstance(ctx.metadata, dict):
            ctx.metadata["final_answer"] = text
            ctx.metadata["final_answer_confidence"] = 1.0
        logger.info(
            "BareTextFinalizeObserver latched: turn=%d, len=%d, gate=%s",
            ctx.turn, len(text), "bypassed(last_turn)" if err else "passed",
        )
        return Intervention(stop_reason="final_answer")
