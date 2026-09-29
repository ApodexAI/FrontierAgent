"""UnknownAgentAssignmentGuard — deterministic recovery hint when assign_task
targets a sub-agent that was never created.

Real, added 2026-09-29. Root-caused live: a qwen3:14b agent_team trial hit
assign_task's existing "Unknown agent 'subagent1'" error -- assign_task.py
already has TWO deterministic, per-spec checks for this exact case
(``_unknown_agent_validation_errors`` and the main-loop check at the
``session is None`` branch), and both already return a clear, actionable
message telling the model to call create_subagent first. The model never
called create_subagent, never called assign_task again either -- it
spiraled into extended, unproductive internal reasoning for the rest of its
turn budget and ran out the 300s clock without ever resolving. The existing
error text alone was not enough to induce recovery in that trial.

This observer does not add a THIRD "Unknown agent" check -- the two that
exist already fire correctly and already say the right thing. It closes the
recovery gap the same way LeakedToolCallRetryObserver and
UnassignedAgentNudge close theirs: by rewriting the tool result the model
actually reads, in the SAME turn the error occurs, into an unmissable,
example-driven recovery instruction with the exact call shape inline --
so the model never gets a turn boundary in which to reason its way into
the stall instead of just fixing it.
"""
from __future__ import annotations

import logging
import re

from frontier_agent.core.loop_types import BaseObserver, ToolResult, TurnContext

logger = logging.getLogger(__name__)

_UNKNOWN_AGENT_RE = re.compile(r"Unknown agent [\"']([^\"']+)[\"']")


class UnknownAgentAssignmentGuard(BaseObserver):
    """Turn assign_task's "Unknown agent" error into an actionable recovery
    hint, in-place, before the model gets a chance to reason about it wrong.

    Args:
        max_hints_per_task: cap on how many times this rewrites a result
            for one loop run. After the cap, results pass through
            unmodified -- if the model is still failing after repeated,
            explicit, example-driven correction, more of the same text
            will not help and existing stall/no-progress guards should take
            over instead.
    """

    critical: bool = True

    def __init__(self, *, max_hints_per_task: int = 3) -> None:
        self._max_hints = max(1, int(max_hints_per_task))
        self._count = 0

    async def on_tool_result(
        self, ctx: TurnContext, result: ToolResult,
    ) -> ToolResult | None:
        if result.name != "assign_task":
            return None
        if not result.result or "Unknown agent" not in result.result:
            return None

        match = _UNKNOWN_AGENT_RE.search(result.result)
        bad_name = match.group(1) if match else "that agent"

        self._count += 1
        if self._count > self._max_hints:
            logger.info(
                "[UnknownAgentAssignmentGuard] Exhausted hint budget (%d) — "
                "leaving result unmodified.",
                self._max_hints,
            )
            return None

        hint = (
            "\n\nRECOVERY (read this before doing anything else): "
            f"{bad_name!r} does not exist — no sub-agent by that name was "
            "ever created in this run. Do NOT retry assign_task with a "
            "different guess at the name, and do not invent an unrelated "
            "task instead. Call create_subagent first, exactly like this:\n"
            f'  create_subagent(agents=[{{"name": "{bad_name}", '
            '"system_prompt": "<what this agent should do>"}])\n'
            "Then call assign_task again using that SAME name. If you do "
            "not actually need a sub-agent for this, do the work yourself "
            "directly instead of calling assign_task at all."
        )
        logger.warning(
            "[UnknownAgentAssignmentGuard] turn=%d | assign_task referenced "
            "unknown agent %r — appending recovery hint (%d/%d)",
            ctx.turn, bad_name, self._count, self._max_hints,
        )
        return ToolResult(
            name=result.name,
            args=result.args,
            result=result.result + hint,
            duration_ms=result.duration_ms,
            tool_call_id=result.tool_call_id,
            is_error=result.is_error,
            interrupted=result.interrupted,
        )


__all__ = ["UnknownAgentAssignmentGuard"]
