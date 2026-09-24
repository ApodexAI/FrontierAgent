from __future__ import annotations

import asyncio

from frontier_agent.core.loop_types import TurnContext, notify_observers
from plugins.tools import task_board as tb
from workflows.agent_team.observers.bare_text_finalize import (
    BareTextFinalizeObserver,
)

_TASK = "task"
_ANSWER = "Deployment complete; 100% of attacks blocked."


def _seed_board(resolutions: dict[str, str]) -> None:
    tb._BOARDS[_TASK] = {
        "seq": len(resolutions),
        "tasks": {
            tid: {
                "description": f"work {tid}",
                "resolution": resolution,
                "owners": [],
            }
            for tid, resolution in resolutions.items()
        },
    }


def _context(turn: int, max_turns: int = 20) -> TurnContext:
    return TurnContext(
        turn=turn,
        max_turns=max_turns,
        task_id=_TASK,
        role_id="coordinator",
        ai_text=_ANSWER,
        thinking="",
        tool_calls=[],
        messages=[],
        usage=None,
        metadata={},
    )


def _dispatch(ctx: TurnContext) -> list:
    try:
        return asyncio.run(
            notify_observers(
                [BareTextFinalizeObserver()],
                "on_llm_response",
                ctx,
            ),
        )
    finally:
        tb._BOARDS.pop(_TASK, None)


def test_mid_run_unfinished_board_still_blocks() -> None:
    """Away from the last turn the gate wins: rejected, never latched."""
    _seed_board({"t1": "open"})
    ctx = _context(turn=5)

    interventions = _dispatch(ctx)

    (intervention,) = interventions
    assert intervention.continue_to_next_turn is True
    assert intervention.stop_reason is None
    assert "final_answer" not in ctx.metadata


def test_final_turn_bypass_marks_unfinished_work() -> None:
    """The last-turn bypass must deliver WITH a visible unfinished-work note.

    ``finalize_gate`` blocks while board items are open, but the final turn
    accepts the answer anyway (never lose it to max_turns). That bypass used
    to be silent — an unfinished run looked like a clean success. It must
    latch the answer, flag ``finalize_gate_bypassed``, and append a note
    naming the still-open items (issue #19).
    """
    _seed_board({"t1": "open", "t2": "open", "t3": "open"})
    ctx = _context(turn=19)

    interventions = _dispatch(ctx)

    (intervention,) = interventions
    assert intervention.stop_reason == "final_answer"
    latched = ctx.metadata["final_answer"]
    assert "unfinished" in str(latched).casefold()
    for task_id in ("t1", "t2", "t3"):
        assert task_id in str(latched)
    assert ctx.metadata.get("finalize_gate_bypassed")


def test_final_turn_clean_board_latches_untouched() -> None:
    """Gate passes → the answer is latched verbatim, no note, no marker."""
    _seed_board({"t1": "resolved", "t2": "cancelled"})
    ctx = _context(turn=19)

    interventions = _dispatch(ctx)

    (intervention,) = interventions
    assert intervention.stop_reason == "final_answer"
    assert ctx.metadata["final_answer"] == _ANSWER
    assert "finalize_gate_bypassed" not in ctx.metadata
