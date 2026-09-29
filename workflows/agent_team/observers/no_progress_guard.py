"""NoProgressGuard — break the main coordinator's create/assign spin.

Real, redesigned 2026-09-06, based on direct, empirical findings from
repeated real test runs with `digitsflow/bonsai-8b`:

1. The original guard gated ALL threshold checking behind
   `creates_in_streak >= min_repeated_creates`. A real, instrumented run
   showed `mgmt_streak` climb to 10 (past the soft threshold of 6) with
   zero intervention, since only 1 real `create_subagent` call had
   happened. Fixed: thresholds are now always evaluated;
   `creates_in_streak` lowers the effective threshold (an amplifier),
   not a gate.

2. `update_task` was treated as unconditional "real progress", resetting
   the streak on every call. A separate real run showed a coordinator
   spin for 63 turns by repeatedly calling `update_task` on the same
   task id, cycling through resolution values with no real convergence
   (`resolved` -> `completed` -> `closed` -> `open` -> `in_progress` ->
   `open` -> `in_progress` -> `closed` -> `resolved`, confirmed directly
   from the real call arguments) -- every one of these calls reset the
   guard, making this entire, real thrash pattern invisible. Fixed:
   `update_task` now only counts as real progress when it moves a task
   to a genuinely new resolution state it has not held before; revisiting
   a prior state increments a new `task_oscillation_streak` instead.

Also fixed in this same pass: `_assign_is_degenerate` and
`_record_assignment_quality` were still reading the pre-rename
`prompt`/`task` fields from `assign_task`'s own args. Since that field
was renamed to `task_prompt` earlier (2026-09-05, to resolve a separate,
real cross-tool field-name collision with `create_subagent`'s
`system_prompt`), degenerate-assignment detection had been silently
reading the wrong field ever since -- corrected here.

Real, further extended 2026-09-06 (same day, later pass): a live,
end-to-end verification of the redesign above surfaced a real, distinct,
third gap, independently confirmed across two different models
(`digitsflow/bonsai-8b`, `gemma4:e2b`). Both hit a real, correct
`finalize_gate` block (no real sub-agent work had happened), and both
then repeated a bare-text, zero-tool-call response over and over --
245 consecutive, identical repeats in one real run; several
near-identical "please provide a question" variants in the other --
rather than changing approach. Every one of the above-described
detectors is keyed off `ctx.tool_calls`, so a turn with NO tool calls at
all (`if not names: reset and return None`) was completely invisible to
all of them. A separate, real text-repetition detector is added below,
tracking `ctx.ai_text` directly, specifically for this
no-tool-calls case.
"""

from __future__ import annotations

import logging
import re

from frontier_agent.components.agent_bus.fan_in import ORCHESTRATOR_AGENT_NAME
from frontier_agent.core.loop_types import (
    BaseObserver,
    Intervention,
    LoopConfig,
    ToolResult,
    TurnContext,
)
from plugins.tools._coerce import coerce_json_list

logger = logging.getLogger(__name__)

_REPORT_AGENT_RE = re.compile(r'<report\s+agent="([^"]*)"')

_MIN_MEANINGFUL_PROMPT_LEN = 8

_MGMT_TOOLS = frozenset({
    "create_subagent",
    "assign_task",
    "collect_reports",
    "stop_subagent",
    "add_task",
    "update_task",
})

_DEGENERATE_PROMPTS = frozenset({
    "end", "done", "complete", "completed", "final", "finalize", "finish",
    "finished", "x", "ok", "okay", "stop", "confirm", "nothing", "na", "n/a",
})

_TERMINAL_RESOLUTIONS = frozenset({"resolved", "cancelled"})

# Real, added 2026-09-06: minimum Jaccard token-similarity for two
# bare-text (no tool call) turns to count as "the same" response.
# Calibrated directly against real, observed near-identical repeats from
# a genuine `gemma4:e2b` run (four consecutive "please provide the
# question..." variants measured 0.500-0.741 similarity to each other) --
# 0.5 is the real, lowest observed value in that confirmed pattern.
_TEXT_SIMILARITY_THRESHOLD = 0.5

# Real, added 2026-09-06: a bare-text streak this long or longer force-stops
# the run, same severity as the existing hard mgmt/oscillation stops.
_TEXT_STREAK_MAX = 6
# Real, added 2026-09-06 (later pass, same day): a bare-text streak this
# long gets a soft nudge first -- the model's one real chance to adapt
# before the hard stop above. Never actually implemented in the first
# pass; confirmed directly, honestly, by a real run that hit the hard stop
# with zero nudge ever injected (verified against the real conversation
# history, not just the log). Real, deliberate gap from the soft threshold
# to the hard one (3 vs. 6): gives the model a few real turns to react
# before termination, without allowing an unbounded number of repeats.
_TEXT_STREAK_SOFT = 3

# Real, added 2026-09-06 (further pass, same day): an ABSOLUTE count of
# consecutive bare-text (no tool call) responses, completely independent of
# text similarity. Confirmed directly, empirically: a real run produced
# 101 consecutive bare-text turns with the turn counter frozen the entire
# time, but the text was genuinely, substantially DIFFERENT each attempt
# (varied restatements of the same status/rules, not verbatim or
# near-verbatim repeats) -- Jaccard similarity never crossed
# _TEXT_SIMILARITY_THRESHOLD even once, so _text_streak never advanced
# past 1, and neither the soft nudge nor the hard stop above ever fired.
# This is a real, separate, second signal: it does not care WHAT the model
# is saying, only THAT it has said something, with no tool call, this many
# times in a row. Deliberately larger than _TEXT_STREAK_MAX -- an
# identical, exact repeat is a stronger, faster signal than a merely
# varied one, so the exact-repeat path should (and does) still fire first
# in that case; this is the real backstop for the case it doesn't.
_BARE_TEXT_ABSOLUTE_MAX = 15
_BARE_TEXT_ABSOLUTE_SOFT = 8


class NoProgressGuard(BaseObserver):
    critical: bool = True

    def __init__(
        self,
        *,
        soft_streak: int = 6,
        hard_streak: int = 12,
        min_repeated_creates: int = 3,
        cooldown_turns: int = 3,
        oscillation_max: int = 6,
    ) -> None:
        self._soft_streak = max(1, int(soft_streak))
        self._hard_streak = max(self._soft_streak + 1, int(hard_streak))
        self._min_repeated_creates = max(2, int(min_repeated_creates))
        self._cooldown_turns = max(1, int(cooldown_turns))
        self._oscillation_max = max(2, int(oscillation_max))
        self._reset()
        self._last_soft_turn = -(10**9)
        self._report_agents_this_turn: set[str] = set()
        self._degenerate_assignment_agents: set[str] = set()
        self._task_resolution_history: dict[str, list[str]] = {}
        # Real, added 2026-09-06: state for the bare-text repetition
        # detector. `_text_last_tokens` persists ACROSS `_reset()` calls
        # deliberately (unlike the other counters) -- it is cleared only in
        # `on_loop_start` and whenever a real tool call breaks the pattern,
        # so a real streak of similar tool-free turns is not silently lost
        # each time some *other*, unrelated counter resets.
        self._text_streak = 0
        self._text_last_tokens: set[str] | None = None
        # Real, added 2026-09-06 (soft-nudge pass): whether the soft nudge
        # has already been sent for the CURRENT text streak. A turn-number
        # cooldown (matching the existing mgmt-streak nudge) doesn't work
        # here -- confirmed directly, empirically, that ctx.turn can stay
        # frozen across many real, separate on_llm_response calls in a row
        # (the exact bug this whole detector exists to catch), so a
        # `ctx.turn - last_nudge_turn >= cooldown` check would never
        # genuinely re-satisfy. A streak-scoped flag is real and correct
        # regardless of whether the turn counter itself ever moves.
        self._text_nudge_sent_this_streak = False
        # Real, added 2026-09-06 (semantic-repeat pass): absolute count of
        # consecutive bare-text turns, independent of similarity. Also
        # persists deliberately across `_reset()` (like `_text_last_tokens`
        # above), since it must keep counting through the exact same
        # frozen-turn scenario that motivated moving detection into
        # on_llm_response in the first place.
        self._bare_text_count = 0
        self._bare_text_nudge_sent = False

    def _reset(self) -> None:
        self._mgmt_streak = 0
        self._creates_in_streak = 0
        self._degenerate_in_streak = 0
        self._oscillation_streak = 0

    async def on_loop_start(self, config: LoopConfig) -> None:
        self._reset()
        self._last_soft_turn = -(10**9)
        self._report_agents_this_turn.clear()
        self._degenerate_assignment_agents.clear()
        self._task_resolution_history.clear()
        self._text_streak = 0
        self._text_last_tokens = None
        self._text_nudge_sent_this_streak = False
        self._bare_text_count = 0
        self._bare_text_nudge_sent = False

    async def on_llm_response(self, ctx: TurnContext) -> Intervention | None:
        """Detect a repeated bare-text (no tool call) response.

        Real, added 2026-09-06, moved here (not on_turn_end) 2026-09-06,
        same day, later pass: this must fire on EVERY real LLM response,
        because BareTextFinalizeObserver's own real `continue_to_next_turn`
        intervention -- returned on every real finalize_gate block --
        bypasses on_turn_end entirely for that turn. Confirmed directly,
        empirically: a real run hit the exact same bare-text answer 13+
        times in a row, the turn counter never advanced past its starting
        value, and on_turn_end was never called for any of the repeats.
        """
        if ctx.tool_calls:
            # A real tool call this response — a different, real approach.
            # Let on_turn_end's own, existing logic classify it; don't let
            # a stale text streak persist across an intervening tool call.
            self._text_streak = 0
            self._text_last_tokens = None
            self._text_nudge_sent_this_streak = False
            self._bare_text_count = 0
            self._bare_text_nudge_sent = False
            return None

        text = (ctx.ai_text or "").strip()
        if not text:
            self._text_streak = 0
            self._text_last_tokens = None
            self._text_nudge_sent_this_streak = False
            self._bare_text_count = 0
            self._bare_text_nudge_sent = False
            return None

        # Real, added 2026-09-06 (semantic-repeat pass): the absolute count
        # increments on every real bare-text turn, regardless of whether
        # this specific one resembles the last one -- this is the real,
        # separate signal that catches the case the similarity-gated streak
        # below cannot: many consecutive bare-text turns that each say
        # something genuinely different. Checked and can fire BEFORE the
        # similarity-based checks below, since 15 consecutive turns of any
        # kind of unproductive bare text is real, sufficient evidence on
        # its own -- it does not need the text to repeat to be a problem.
        self._bare_text_count += 1
        if self._bare_text_count >= _BARE_TEXT_ABSOLUTE_MAX:
            logger.warning(
                "NoProgressGuard: forcing stop at turn %d "
                "(bare_text_count=%d, no tool calls, semantically varied "
                "content -- similarity-based text_streak never "
                "triggered)",
                ctx.turn, self._bare_text_count,
            )
            return Intervention(stop_reason="thrash_no_progress")
        if (
            self._bare_text_count >= _BARE_TEXT_ABSOLUTE_SOFT
            and not self._bare_text_nudge_sent
        ):
            self._bare_text_nudge_sent = True
            logger.info(
                "NoProgressGuard: bare-text absolute-count soft nudge at "
                "turn %d (bare_text_count=%d)",
                ctx.turn, self._bare_text_count,
            )
            return Intervention(inject_messages=[
                "You have responded with plain text and no tool call "
                "several times in a row, even though your wording has "
                "varied each time. This is not making progress. If a "
                "sub-agent has not actually completed real work yet, call "
                "create_subagent and assign_task right now to delegate the "
                "work. If the work is genuinely done, deliver your "
                "COMPLETE final answer now."
            ])

        tokens = self._normalize_tokens(text)
        same_as_last = (
            self._text_last_tokens is not None
            and self._jaccard(tokens, self._text_last_tokens)
            >= _TEXT_SIMILARITY_THRESHOLD
        )
        self._text_last_tokens = tokens
        if same_as_last:
            self._text_streak += 1
        else:
            # Real, genuine change in what the model is saying -- reset
            # both the streak and the nudge flag, so a real strategy shift
            # gets a full, fresh run before any further intervention.
            self._text_streak = 1
            self._text_nudge_sent_this_streak = False

        if self._text_streak >= _TEXT_STREAK_MAX:
            logger.warning(
                "NoProgressGuard: forcing stop at turn %d "
                "(text_streak=%d, repeated bare-text with no tool calls)",
                ctx.turn, self._text_streak,
            )
            return Intervention(stop_reason="thrash_no_progress")

        # SOFT: nudge, once per real streak (not turn-cooldown-limited, per
        # the real, confirmed reason in _text_nudge_sent_this_streak's own
        # comment above -- ctx.turn can stay frozen for the entire streak).
        if (
            self._text_streak >= _TEXT_STREAK_SOFT
            and not self._text_nudge_sent_this_streak
        ):
            self._text_nudge_sent_this_streak = True
            logger.info(
                "NoProgressGuard: text-streak soft nudge at turn %d "
                "(text_streak=%d)",
                ctx.turn, self._text_streak,
            )
            return Intervention(inject_messages=[
                "You have given the same or a very similar answer several "
                "times in a row without calling any tool. If a sub-agent "
                "has not actually completed real work yet, call "
                "create_subagent and assign_task to delegate the work. If "
                "the work is genuinely done, deliver your COMPLETE final "
                "answer now with new, substantive content -- repeating the "
                "same text again will not succeed."
            ])
        return None

    async def on_tool_result(
        self, ctx: TurnContext, result: ToolResult,
    ) -> ToolResult | None:
        if (
            result is not None
            and not result.is_error
            and str(result.name) == "collect_reports"
        ):
            self._report_agents_this_turn.update(
                self._subagent_report_agents(result.result)
            )
        return None

    async def on_turn_end(self, ctx: TurnContext) -> Intervention | None:
        self._record_assignment_quality(ctx.tool_calls)

        report_agents = self._report_agents_this_turn
        self._report_agents_this_turn = set()
        fresh_report = any(
            name not in self._degenerate_assignment_agents
            for name in report_agents
        )
        self._degenerate_assignment_agents.difference_update(report_agents)
        if fresh_report:
            self._reset()
            return None

        names = {str(tc.get("name", "")) for tc in (ctx.tool_calls or [])}
        # Empty turn, or any non-management tool → real work / a different
        # mode. Reset and let the natural finalize path run. (Real note,
        # 2026-09-06: bare-text repetition detection used to live here too,
        # but this hook is bypassed whenever an observer -- notably
        # BareTextFinalizeObserver on every real finalize_gate block --
        # fires `continue_to_next_turn=True`, which is exactly the real,
        # confirmed case this needed to catch. Moved to on_llm_response,
        # which fires on every real LLM response regardless of whether the
        # turn counter itself advances.)
        if not names or not names.issubset(_MGMT_TOOLS):
            self._reset()
            return None

        task_progress, task_oscillated = self._record_task_progress(
            ctx.tool_calls
        )
        if task_progress:
            self._reset()
            return None
        if task_oscillated:
            self._oscillation_streak += 1

        self._mgmt_streak += 1
        if "create_subagent" in names:
            self._creates_in_streak += 1
        if "assign_task" in names and self._assign_is_degenerate(ctx.tool_calls):
            self._degenerate_in_streak += 1

        creation_pathology = self._creates_in_streak >= self._min_repeated_creates
        effective_hard = self._hard_streak - (3 if creation_pathology else 0)
        effective_soft = self._soft_streak - (2 if creation_pathology else 0)
        degenerate = self._degenerate_in_streak >= 2

        if (
            self._mgmt_streak >= effective_hard
            or (degenerate and self._mgmt_streak >= effective_soft + 2)
            or self._oscillation_streak >= self._oscillation_max
        ):
            logger.warning(
                "NoProgressGuard: forcing stop at turn %d (mgmt_streak=%d, "
                "creates=%d, degenerate=%d, oscillation=%d)",
                ctx.turn, self._mgmt_streak, self._creates_in_streak,
                self._degenerate_in_streak, self._oscillation_streak,
            )
            return Intervention(stop_reason="thrash_no_progress")

        if (
            (self._mgmt_streak >= effective_soft
             or self._oscillation_streak >= max(2, self._oscillation_max - 2))
            and ctx.turn - self._last_soft_turn >= self._cooldown_turns
        ):
            self._last_soft_turn = ctx.turn
            logger.info(
                "NoProgressGuard: soft nudge at turn %d (mgmt_streak=%d, "
                "creates=%d, oscillation=%d)",
                ctx.turn, self._mgmt_streak, self._creates_in_streak,
                self._oscillation_streak,
            )
            return Intervention(inject_messages=[
                "You have repeatedly created and assigned sub-agents, or "
                "repeatedly changed task states, without making real "
                "progress — your sub-agents have already returned complete "
                "reports covering the question, or you are cycling a task's "
                "status back and forth. STOP delegating and STOP changing "
                "task states: do not call create_subagent, assign_task, or "
                "update_task again. Synthesize everything you already have "
                "and deliver your COMPLETE final answer as plain text now "
                "(no tool call)."
            ])
        return None

    def _record_task_progress(
        self, tool_calls: list[dict],
    ) -> tuple[bool, bool]:
        saw_new = False
        saw_repeat = False
        try:
            for tc in tool_calls or []:
                if str(tc.get("name", "")) != "update_task":
                    continue
                args = tc.get("args") or {}
                updates = coerce_json_list(args.get("updates") or []) or []
                for update in updates:
                    if not isinstance(update, dict):
                        continue
                    task_id = str(update.get("id") or "").strip()
                    resolution = str(update.get("resolution") or "").strip().lower()
                    if not task_id or not resolution:
                        continue
                    history = self._task_resolution_history.setdefault(task_id, [])
                    if resolution in history:
                        saw_repeat = True
                    else:
                        history.append(resolution)
                        saw_new = True
        except (AttributeError, TypeError, ValueError):
            return (False, False)
        if saw_new:
            return (True, False)
        return (False, saw_repeat)

    def _assign_is_degenerate(self, tool_calls: list[dict]) -> bool:
        try:
            for tc in tool_calls or []:
                if str(tc.get("name", "")) != "assign_task":
                    continue
                args = tc.get("args") or {}
                tasks = coerce_json_list(args.get("tasks") or []) or []
                prompts = [
                    str(t.get("task_prompt") or "").strip()
                    for t in tasks
                    if isinstance(t, dict)
                ]
                if prompts and all(self._is_degenerate(p) for p in prompts):
                    return True
        except (AttributeError, TypeError, ValueError):
            return False
        return False

    def _record_assignment_quality(self, tool_calls: list[dict]) -> None:
        try:
            for tc in tool_calls or []:
                if str(tc.get("name", "")) != "assign_task":
                    continue
                args = tc.get("args") or {}
                tasks = coerce_json_list(args.get("tasks") or []) or []
                for task in tasks:
                    if not isinstance(task, dict):
                        continue
                    agent = str(task.get("agent") or "").strip()
                    if not agent:
                        continue
                    prompt = str(task.get("task_prompt") or "").strip()
                    if self._is_degenerate(prompt):
                        self._degenerate_assignment_agents.add(agent)
                    else:
                        self._degenerate_assignment_agents.discard(agent)
        except (AttributeError, TypeError, ValueError):
            return

    @staticmethod
    def _normalize_tokens(text: str) -> set[str]:
        """Lowercase, strip punctuation, and tokenize for similarity checks."""
        cleaned = re.sub(r"[^\w\s]", "", text.lower())
        return set(cleaned.split())

    @staticmethod
    def _jaccard(a: set[str], b: set[str]) -> float:
        """Jaccard similarity between two token sets; 0.0 for two empty sets."""
        if not a and not b:
            return 0.0
        union = a | b
        if not union:
            return 0.0
        return len(a & b) / len(union)

    @staticmethod
    def _is_degenerate(prompt: str) -> bool:
        s = prompt.strip().lower().strip(".!?。！ ")
        return len(s) < _MIN_MEANINGFUL_PROMPT_LEN or s in _DEGENERATE_PROMPTS

    @staticmethod
    def _subagent_report_agents(result: str) -> set[str]:
        if not result:
            return set()
        try:
            return {
                name
                for name in _REPORT_AGENT_RE.findall(result)
                if name != ORCHESTRATOR_AGENT_NAME
            }
        except (AttributeError, TypeError):
            return set()


__all__ = ["NoProgressGuard"]
