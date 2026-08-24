"""Warn the LLM one turn before the loop closes."""

from __future__ import annotations

from frontier_agent.core.loop_types import (
    BaseObserver,
    Intervention,
    TurnContext,
)


class LastTurnForcer(BaseObserver):
    critical = True

    def __init__(self, terminal_tool: str = "finalize_answer") -> None:
        self._terminal = terminal_tool
        self._fired = False

    async def on_turn_end(self, ctx: TurnContext) -> Intervention | None:
        if self._fired or ctx.turn != ctx.max_turns - 1:
            return None
        self._fired = True
        # Stash the strip-tools flag for the NEXT turn (= ``max_turns``).
        # Mirrors how ``LeakedToolCallRetryObserver`` plants
        # ``_llm_temp_override`` in metadata for one-shot consumption by
        # ``agent_loop`` at the top of the following turn.
        ctx.metadata["_llm_strip_tools"] = True
        # ``terminal_tool=""`` → no terminal tool (e.g. agent-team finishes by
        # ending a turn with a plain-text answer, not a tool call).
        if self._terminal:
            instruction = f"Call `{self._terminal}` now with your complete answer"
        else:
            instruction = "Deliver your COMPLETE answer as plain text now (no tool call)"
        return Intervention(inject_messages=[
            f"This is your penultimate turn. {instruction} — no more search "
            "rounds will be accepted. The next turn will run without tools, so "
            "any remaining work must land in this response."
        ])
