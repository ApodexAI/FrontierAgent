"""Reserve several tool-enabled turns for deliverables and final synthesis."""

from __future__ import annotations

from frontier_agent.core.loop_types import BaseObserver, Intervention, TurnContext

DEFAULT_FINALIZATION_MESSAGE = (
    "The execution budget is entering its finalization reserve. Stop starting "
    "new exploration. Use the remaining tool-enabled turns to finish and save "
    "the best currently achievable deliverables, run only essential checks, "
    "and then provide a complete plain-text answer. If some requested work "
    "cannot be completed, still preserve the existing artifacts and answer "
    "with the best supported result instead of returning no answer."
)


class FinalizationReserveObserver(BaseObserver):
    """Inject a one-shot finalization instruction before ``max_turns``.

    The reserve never fires on the first turn and is skipped entirely when the
    loop is too small to leave a tool-enabled turn before
    :class:`LastTurnForcer` takes over. The observer stamps
    ``finalization_phase`` in shared loop metadata, allowing workflow
    tools/observers to notice the transition later without coupling this
    generic observer to them.
    """

    critical = True

    def __init__(
        self,
        *,
        reserve_turns: int = 8,
        message: str = DEFAULT_FINALIZATION_MESSAGE,
    ) -> None:
        self._reserve_turns = max(1, int(reserve_turns))
        self._message = message.strip() or DEFAULT_FINALIZATION_MESSAGE
        self._fired = False

    async def on_turn_end(self, ctx: TurnContext) -> Intervention | None:
        # A reserve message injected at ``max_turns - 1`` would compete with
        # LastTurnForcer while the next turn has tools stripped. Small smoke
        # profiles (notably max_turns=3) therefore rely on LastTurnForcer only.
        trigger_turn = max(2, ctx.max_turns - self._reserve_turns)
        if trigger_turn >= ctx.max_turns - 1:
            return None
        if self._fired or ctx.turn < trigger_turn:
            return None
        self._fired = True
        ctx.metadata["finalization_phase"] = True
        ctx.metadata["finalization_reserve_turns"] = max(
            0, ctx.max_turns - ctx.turn,
        )
        return Intervention(inject_messages=[self._message])


__all__ = ["DEFAULT_FINALIZATION_MESSAGE", "FinalizationReserveObserver"]
