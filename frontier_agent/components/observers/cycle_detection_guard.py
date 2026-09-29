"""CycleDetectionGuard — hint/stop when tool calls fall into a stable,
short, alternating cycle across NON-consecutive-identical turns, INCLUDING
a "near-identical churn" variant where a volatile argument (like an edit
op's exact shape) differs each turn while the real target (path/url) stays
fixed.

Real, added 2026-09-08, hardened twice. Directly motivated by three real,
confirmed gaps found live, in order:

1. RepetitionGuard (exact match, consecutive turns only) missed a stable,
   short, alternating N-tool cycle (read_file/create_file/read_file/...)
   that ran 35+ turns -- fixed by this guard's period-N EXACT-signature
   cycle detection (stage 1, below).

2. That fix still missed a real, adjacent variant, confirmed live: the
   SAME sub-agent repeatedly called create_file on the SAME real path,
   but the argument shape changed turn to turn (a "create" op, then a
   "replace_text" op, then another "replace_text" op with a different
   internal structure) -- so no two turns' EXACT signatures ever
   matched, and neither RepetitionGuard nor stage 1 of this guard fired.
   This is "near-identical churn": the real, stable thing is WHICH
   resource is being hammered, not the literal argument bytes.

Stage 2 (coarse signature) directly targets this: for known tools with a
real, identifiable "target" argument (a file path, a URL), it hashes ONLY
that target, ignoring the rest of the (volatile) argument shape. A long
run of coarse-identical calls -- even with different literal argument
bytes -- means "hammering the same resource without making real
progress", which is the same real distress signal, independent of the
model happening to vary an inner op field each time.
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

__all__ = [
    "CYCLE_DETECTED_STOP_REASON",
    "CHURN_DETECTED_STOP_REASON",
    "CycleDetectionGuard",
]

CYCLE_DETECTED_STOP_REASON = "cycle_detected"
CHURN_DETECTED_STOP_REASON = "argument_churn_detected"

# Real, direct, per-tool map of which argument key holds the real "target"
# resource for stage 2's coarse signature. Deliberately small and explicit
# rather than a generic heuristic -- these are the exact tools confirmed,
# live, to exhibit near-identical churn (create_file) or that share the
# same real shape (read_file, web_fetch), not a guess at every possible
# tool's schema.
_TARGET_ARG_KEYS: dict[str, str] = {
    "read_file": "path",
    "create_file": "path",
    "web_fetch": "url",
}


def _turn_signature(tool_calls: list[dict[str, Any]]) -> str:
    """Real, direct reuse of RepetitionGuard's own signature logic --
    same, real, stable per-turn hash, so both guards agree on what
    counts as "the same turn" without duplicating divergent logic."""
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


def _turn_target_signature(tool_calls: list[dict[str, Any]]) -> str | None:
    """Real, coarse signature: tool name + ONLY its real target argument
    (from _TARGET_ARG_KEYS), ignoring everything else. Returns None for a
    turn with no tool call, or where no call names a tool with a known
    target key -- a turn with no coarse signature cannot be part of a
    churn streak (there is nothing stable to compare it against)."""
    if len(tool_calls) != 1:
        # Real, deliberate restriction to single-tool-call turns: a
        # multi-call turn's "target" is ambiguous, and every real churn
        # example observed live was one tool call per turn.
        return None
    tool_call = tool_calls[0]
    name = str(tool_call.get("name") or "")
    key = _TARGET_ARG_KEYS.get(name)
    if key is None:
        return None
    args = tool_call.get("args") or {}
    target = args.get(key)
    if not isinstance(target, str) or not target:
        return None
    return f"{name}:{target}"


class CycleDetectionGuard(BaseObserver):
    """Detects two real, distinct patterns RepetitionGuard cannot see:

    Stage 1 -- a stable, short, EXACT period-N cycle (e.g. A,B,A,B,...),
    which RepetitionGuard's consecutive-identical-turn check cannot see
    since no single turn repeats the one immediately before it.

    Stage 2 -- "near-identical churn": many consecutive turns calling the
    same tool on the same real target (same file path / URL), even
    though the full argument bytes differ turn to turn (e.g. a changing
    edit-op shape) and so never form an exact-match cycle at all.

    ``critical`` so the loop awaits the hook and collects the returned
    Intervention -- a non-critical observer's return value is dropped
    (the real, confirmed root cause of an earlier bug this same session).
    """

    critical: bool = True

    def __init__(
        self,
        *,
        max_period: int = 4,
        min_repeats: int = 3,
        stop_after_repeats: int = 0,
        churn_hint_after: int = 6,
        churn_stop_after: int = 10,
    ) -> None:
        self.max_period = max(2, int(max_period))
        self.min_repeats = max(2, int(min_repeats))
        self.stop_after_repeats = (
            max(self.min_repeats + 1, int(stop_after_repeats))
            if stop_after_repeats else 0
        )
        # Real, stage-2 thresholds: consecutive turns hitting the SAME
        # coarse target signature. Looser than stage 1's min_repeats,
        # since a real, legitimate multi-step edit of one file can
        # genuinely take a few turns -- this only fires on a real, long
        # run against one target with no progress signal in between.
        self.churn_hint_after = max(3, int(churn_hint_after))
        self.churn_stop_after = max(
            self.churn_hint_after + 1, int(churn_stop_after),
        )
        self._history: list[str] = []
        self._hinted_at_len = -1
        self._churn_target: str | None = None
        self._churn_streak = 0
        self._churn_hinted_at = -1

    async def on_loop_start(self, config: LoopConfig) -> None:
        del config
        self._history = []
        self._hinted_at_len = -1
        self._churn_target = None
        self._churn_streak = 0
        self._churn_hinted_at = -1

    def _detect_cycle(self) -> tuple[int, int] | None:
        best: tuple[int, int] | None = None
        n = len(self._history)
        for period in range(2, self.max_period + 1):
            if n < period * 2:
                continue
            block = self._history[n - period:]
            repeats = 1
            start = n - period
            while start - period >= 0 and self._history[start - period:start] == block:
                repeats += 1
                start -= period
            if repeats >= self.min_repeats and (best is None or repeats > best[1]):
                best = (period, repeats)
        return best

    async def on_turn_end(self, ctx: TurnContext) -> Intervention | None:
        if not ctx.tool_calls:
            self._history = []
            self._hinted_at_len = -1
            self._churn_target = None
            self._churn_streak = 0
            self._churn_hinted_at = -1
            return None

        # Real, stage 2 first (coarse target churn) -- checked before stage
        # 1 so a long churn run against one target hits its own, real
        # stop reason rather than silently satisfying an unrelated exact
        # cycle check first.
        target_sig = _turn_target_signature(ctx.tool_calls)
        if target_sig is not None and target_sig == self._churn_target:
            self._churn_streak += 1
        else:
            self._churn_target = target_sig
            self._churn_streak = 1 if target_sig is not None else 0

        if self._churn_target is not None:
            if self._churn_streak >= self.churn_stop_after:
                logger.warning(
                    "CycleDetectionGuard turn=%d: %d consecutive turns "
                    "hammering the same real target (%s) with no progress "
                    "-- stopping the loop.",
                    ctx.turn, self._churn_streak, self._churn_target,
                )
                return Intervention(stop_reason=CHURN_DETECTED_STOP_REASON)
            if (
                self._churn_streak >= self.churn_hint_after
                and self._churn_hinted_at != self._churn_streak
            ):
                self._churn_hinted_at = self._churn_streak
                logger.info(
                    "CycleDetectionGuard turn=%d: %d consecutive turns on "
                    "the same real target (%s) -- injecting a corrective "
                    "hint.",
                    ctx.turn, self._churn_streak, self._churn_target,
                )
                return Intervention(inject_messages=[
                    f"You have called the same tool on the same target "
                    f"({self._churn_target}) {self._churn_streak} times in "
                    "a row without finishing. Stop retrying variations of "
                    "the same edit -- either submit_report with what you "
                    "have, or take a genuinely different, real action."
                ])

        self._history.append(_turn_signature(ctx.tool_calls))
        max_len = self.max_period * (max(self.min_repeats, self.stop_after_repeats or 0) + 1)
        if len(self._history) > max_len:
            self._history = self._history[-max_len:]

        found = self._detect_cycle()
        if found is None:
            return None
        period, repeats = found

        if self.stop_after_repeats and repeats >= self.stop_after_repeats:
            logger.warning(
                "CycleDetectionGuard turn=%d: detected a period-%d cycle "
                "repeated %d times -- stopping the loop.",
                ctx.turn, period, repeats,
            )
            return Intervention(stop_reason=CYCLE_DETECTED_STOP_REASON)

        if len(self._history) == self._hinted_at_len:
            return None
        self._hinted_at_len = len(self._history)

        logger.info(
            "CycleDetectionGuard turn=%d: detected a period-%d cycle "
            "repeated %d times -- injecting a corrective hint.",
            ctx.turn, period, repeats,
        )
        return Intervention(inject_messages=[
            f"You appear to be stuck in a repeating cycle of {period} "
            f"alternating tool calls, repeated {repeats} times, without "
            "making real progress toward completing your task. Stop "
            "repeating this pattern -- either call submit_report with "
            "what you have already found, or take a genuinely different "
            "action."
        ])
