"""RepetitionGuard — hint when consecutive turns repeat the same tool call.

The generic counterpart to the two narrower guards:

* :class:`~frontier_agent.components.observers.duplicate_query_rollback.DuplicateQueryRollbackObserver`
  knows what a *search* is: it dedupes ``web_search`` requests over the
  whole loop and pops the turn instead of hinting.
* :class:`~frontier_agent.components.observers.text_repetition_guard.TextRepetitionGuard`
  compares assistant *prose* across turns, so it only sees a loop the
  model narrates.

This guard compares exact tool-call signatures on *consecutive* turns and
covers everything the other two do not — ``web_fetch`` on one dead URL,
the same ``bash`` command, the same ``read_file`` — where the repetition
is visible in the arguments and nowhere else.

``stop_after`` (opt-in, off by default) is what makes it a stop-loss rather
than a suggestion, and it exists because the other two guards cannot end
this particular loop:

* The duplicate-query rollback spends its ``max_consecutive_rollbacks``
  budget and then lets every further duplicate through, permanently, until
  a genuinely new request appears. On a model deterministic enough to
  reproduce a byte-identical tool call at temperature 0.7, that new request
  never comes.
* ``TextRepetitionGuard`` needs ~60 characters of near-identical *visible*
  text. Under ``thinking_format: tag`` the model's deliberation lands in
  ``thinking``, not ``ai_text``, so a turn that is "2900 characters of
  identical reasoning plus one tool call" leaves it nothing to compare and
  it never fires.

Which left nothing to terminate the pathology it was measured in: 71% of
sub-agents that exhausted their turn budget did so inside a run of ten or
more consecutive byte-identical calls, median 87, worst case 198 of 200.
Enable ``stop_after`` where stopping is affordable — a sub-agent whose
partial report still reaches fan-in — and leave it off for an agent that
IS the run.
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any

from frontier_agent.core.loop_types import (
    BaseObserver,
    Intervention,
    LoopConfig,
    TurnContext,
)

logger = logging.getLogger(__name__)

__all__ = ["REPEATED_TOOL_CALLS_STOP_REASON", "RepetitionGuard"]


def _turn_signature(tool_calls: list[dict[str, Any]]) -> str:
    """Return a stable signature for a turn's whole tool-call batch.

    No-raise by contract — this runs on the critical observer path, where
    an exception costs the current turn. ``default=str`` covers values
    ``json.dumps`` rejects (``set``, ``datetime``, custom objects); a
    circular reference falls through to the ``repr`` fallback.
    """
    parts: list[str] = []
    for tool_call in tool_calls:
        name = str(tool_call.get("name") or "")
        args = tool_call.get("args")
        try:
            payload = json.dumps(args, sort_keys=True, default=str)
        except (TypeError, ValueError):
            payload = repr(args)
        digest = hashlib.blake2b(payload.encode(), digest_size=8).hexdigest()
        parts.append(f"{name}:{digest}")
    return "|".join(parts)


#: ``stop_reason`` for a ``stop_after`` stop. Must stay a member of
#: ``fan_in.INCOMPLETE_STOP_REASONS`` — the sub-agent rescue path is an
#: allowlist, so an unlisted reason silently loses its forced-final answer.
REPEATED_TOOL_CALLS_STOP_REASON = "repeated_tool_calls"


class RepetitionGuard(BaseObserver):
    """Hint after ``threshold`` identical turns; optionally stop after more.

    ``critical`` so the loop awaits the hook and collects the returned
    ``Intervention`` — a non-critical observer's return value is dropped.
    """

    critical: bool = True

    def __init__(self, threshold: int = 3, *, stop_after: int = 0) -> None:
        # 2 is the floor: a threshold of 1 would fire on every tool call.
        self.threshold = max(2, int(threshold))
        # 0 disables stopping. Anything positive is raised past ``threshold``
        # so the model always gets at least one hint, and one turn to act on
        # it, before the loop ends under it.
        self.stop_after = (
            max(self.threshold + 1, int(stop_after)) if stop_after else 0
        )
        self._last_signature = ""
        self._streak = 0

    async def on_loop_start(self, config: LoopConfig) -> None:
        del config
        self._reset()

    def _reset(self) -> None:
        self._last_signature = ""
        self._streak = 0

    async def on_turn_end(self, ctx: TurnContext) -> Intervention | None:
        if not ctx.tool_calls:
            # A tool-free turn breaks the streak: whatever the model is
            # doing now, it is not re-issuing the same call.
            self._reset()
            return None

        signature = _turn_signature(ctx.tool_calls)
        if signature == self._last_signature:
            self._streak += 1
        else:
            self._last_signature = signature
            self._streak = 1
            return None

        names = ", ".join(
            sorted({str(tc.get("name") or "") for tc in ctx.tool_calls}),
        )

        if self.stop_after and self._streak >= self.stop_after:
            logger.warning(
                "RepetitionGuard turn=%d: %s repeated %d times with identical "
                "arguments — stopping the loop.",
                ctx.turn, names, self._streak,
            )
            return Intervention(stop_reason=REPEATED_TOOL_CALLS_STOP_REASON)

        # Re-hint every ``threshold`` turns rather than once per streak: a
        # single message is easy for the model to scroll past, and a streak
        # this long has no natural end without the reminder.
        if self._streak < self.threshold or self._streak % self.threshold:
            return None

        logger.info(
            "RepetitionGuard turn=%d: %s repeated %d times with identical "
            "arguments — injecting a corrective hint.",
            ctx.turn, names, self._streak,
        )
        return Intervention(inject_messages=[
            f"You have now called {names} {self._streak} times in a row "
            "with identical arguments, and the results are not changing. "
            "Repeating it again will not help. Change the arguments, use a "
            "different tool, or work with what you already have."
        ])
