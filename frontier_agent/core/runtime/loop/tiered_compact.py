"""Tiered, threshold-triggered context compaction for long tool-heavy loops."""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from typing import Any

from frontier_agent.core.loop_types import (
    BaseObserver,
    CompactionEvent,
    LoopConfig,
    TurnContext,
)
from frontier_agent.core.messages import Message, is_tool_msg, text_of, user_msg
from frontier_agent.core.runtime.loop.compact import (
    INPUT_ESTIMATE_KEY,
    KeepLastNToolResultsCompactor,
    _tool_names_by_call_id,
    compress_tool_results,
    estimate_tokens,
)
from frontier_agent.core.runtime.loop.compact import (
    SPILL_MANIFEST_HEADER as _SPILL_MANIFEST_HEADER,
)
from frontier_agent.core.runtime.loop.compact_llm import LLMSummaryCompactor

__all__ = [
    "InputTokenGauge",
    "InputTokenThresholdPolicy",
    "TieredCompactor",
    "compaction_trigger_tokens",
]

logger = logging.getLogger(__name__)

# Fraction of ``max_len`` at which tiered compaction fires. At 262,144 this
# leaves ~52k tokens before the endpoint's hard ceiling — a margin one assistant
# turn can cross on its own once reasoning is replayed into history as
# ``<think>`` (measured max 130,965 characters ≈ 33k tokens, plus a tool result).
#
# 0.8 is KEPT, and widening it was measured and REJECTED: on the long-reasoning
# subset (160 trials each) 0.65 compacted in 72.6% of trials versus 44.4% at 0.8
# and scored 49.7% versus 50.0% — no gain, just more history discarded and 4%
# fewer searches per trial. Making the trigger look at the request about to be
# sent (see :class:`InputTokenThresholdPolicy`) is what removed the failures; the
# margin was never the binding constraint. Left overridable so the question can
# be revisited without editing a profile:
#     AGENT_COMPACTION_TRIGGER_RATIO=0.65
_DEFAULT_TRIGGER_RATIO = 0.8
_TRIGGER_RATIO_ENV = "AGENT_COMPACTION_TRIGGER_RATIO"


def compaction_trigger_tokens(max_len: int) -> int:
    """The absolute token threshold tiered compaction should fire at.

    One definition for all three construction sites (agent-team coordinator,
    agent-team sub-agent, stateful-react), which previously each carried their own
    ``int(max_len * 0.8)`` literal.
    """
    ratio = _DEFAULT_TRIGGER_RATIO
    raw = os.getenv(_TRIGGER_RATIO_ENV)
    if raw:
        try:
            parsed = float(raw)
        except ValueError:
            logger.warning(
                "%s=%r is not a number — using %.2f", _TRIGGER_RATIO_ENV, raw, ratio,
            )
        else:
            # Outside (0, 1) a ratio either disables compaction or thrashes it
            # every turn. Refuse rather than silently honour a typo.
            if 0.0 < parsed < 1.0:
                ratio = parsed
            else:
                logger.warning(
                    "%s=%s is outside (0, 1) — using %.2f",
                    _TRIGGER_RATIO_ENV, raw, ratio,
                )
    return int(max_len * ratio)

_FULL_TEXT_PREFIX = "[Full text] "
_SPILL_MANIFEST_MAX_PATHS = 20
_SPILL_MANIFEST_MAX_CHARS = 3_000


class InputTokenGauge(BaseObserver):
    """Record the REAL input tokens of each LLM call for the compaction policy.

    The loop's ``CompactionPolicy.should_compact`` only receives the loop's own
    token *estimate*; this gauge exposes the actual ``prompt_tokens`` so the
    threshold can be expressed against the true context window. ``critical`` so
    the update lands INLINE (before the turn's compaction step), not as a
    fire-and-forget task that could race it. Read-only — never intervenes.
    """

    critical: bool = True

    def __init__(self) -> None:
        self.tokens = 0
        self.estimate = 0

    async def on_llm_response(self, ctx: TurnContext) -> None:
        u = ctx.usage or {}
        # Normalised usage uses ``prompt_tokens``; accept raw Anthropic
        # ``input_tokens`` as a fallback.
        self.tokens = int(u.get("prompt_tokens") or u.get("input_tokens") or 0)
        # Both sides of the scale must describe the SAME request. Estimating
        # ``ctx.messages`` here cannot: the loop appends this turn's assistant
        # reply before building the context, and the per-call system addendum
        # only ever exists inside ``messages_for_call``. The loop publishes the
        # estimate of the list it actually sent; 0 when absent, which leaves the
        # scale at 1.0 rather than silently understating it.
        self.estimate = int(ctx.metadata.get(INPUT_ESTIMATE_KEY, 0) or 0)
        return None

    async def on_loop_start(self, config: LoopConfig) -> None:
        self.tokens = 0
        self.estimate = 0

    def real_to_estimate_scale(self) -> float:
        """``real / estimate`` for the SAME request; 1.0 before the first reply.

        Both readers of this ratio want it measured on one snapshot — that is the
        whole reason :attr:`estimate` exists rather than being recomputed at the
        point of use. They clamp differently, so clamping is left to them.
        """
        if self.tokens > 0 and self.estimate > 0:
            return self.tokens / self.estimate
        return 1.0


class InputTokenThresholdPolicy:
    """Compact when the request ABOUT TO BE SENT would exceed ``limit``.

    The gauge alone is not enough, and reading only it killed sub-agents. The
    gauge describes the request already SENT, while ``should_compact`` runs at
    turn end — after this turn's assistant message and tool results were
    appended — so the next request is strictly larger. Trigger on the gauge alone
    and a single turn can step from "under the limit" straight past the
    endpoint's hard ceiling.

    Measured over 51 sub-agents killed by ``llm_error``: 51/51 had a last
    successful prompt just UNDER the 209,715 trigger (median 207,803) and then
    issued a request of 264,743 tokens median, which the endpoint rejected with
    HTTP 400 — losing the sub-agent and every report it had gathered. One
    assistant turn can add 40k+ tokens once reasoning is replayed as ``<think>``,
    so the trigger-to-ceiling margin is not something a reactive read can cover.

    So both are consulted: the gauge for what the endpoint really charged, and the
    loop's own estimate for what is about to go out. The estimate is uncalibrated
    (tiktoken cl100k_base standing in for the served tokenizer, measured to
    UNDER-state this checkpoint by ~14%), so it is scaled by the ratio the last
    call revealed, clamped to ``[1.0, 3.0]``: only under-estimation can cost a
    request here, so the floor matters and the ceiling is a sanity bound.
    """

    _MAX_SCALE = 3.0

    def __init__(self, gauge: InputTokenGauge, limit: int) -> None:
        self._gauge = gauge
        self._limit = limit

    def should_compact(
        self, turn: int, messages: list[Message], estimated_tokens: int,
    ) -> bool:
        real = self._gauge.tokens
        # Same-snapshot ratio, straight off the gauge. Deriving it here from
        # ``messages`` instead would reintroduce the defect the gauge's
        # ``estimate`` was added to fix: ``messages`` is the turn-END history, so
        # dividing by it inflates the denominator with the very tool results that
        # make this turn dangerous, understating the scale exactly when it is
        # needed most.
        scale = min(max(self._gauge.real_to_estimate_scale(), 1.0), self._MAX_SCALE)
        projected = int(estimated_tokens * scale)
        fire = max(real, projected) > self._limit
        if fire:
            logger.info(
                "compaction trigger: real=%d projected=%d (est=%d scale=%.2f) "
                "limit=%d turn=%d",
                real, projected, estimated_tokens, scale, self._limit, turn,
            )
        return fire


class TieredCompactor:
    """Compact old results, then summarize real history only when still needed."""

    def __init__(
        self,
        *,
        keep_tool_result: int,
        summary_llm: Any,
        relief_target: int,
        protect_tool_names: frozenset[str] = frozenset(),
        gauge: InputTokenGauge | None = None,
        spill: Callable[[str, str], str | None] | None = None,
        summary_retries: int = 2,
        summary_retry_timeout_s: float | None = None,
    ) -> None:
        self._tier1 = KeepLastNToolResultsCompactor(
            keep_tool_result=keep_tool_result,
            protect_tool_names=protect_tool_names,
            spill=spill,
        )
        # Tier 1 is not the only candidate that can win, and the others rewrite
        # the SAME protected results — which Tier 1 also keeps out of its spill
        # path, so nothing would be recoverable afterwards.
        self._protect = frozenset(protect_tool_names)
        self._spill = spill
        # ``emit_event`` has existed on LLMSummaryCompactor all along with no
        # caller, so the summary it produced reached nothing. Used here as the
        # internal channel that carries the summary text up to ``last_event``;
        # broadcasting to observers is the loop's job, not the summariser's.
        # Retries are on here and off for the bare callers of
        # ``LLMSummaryCompactor``: this is the only path where losing the summary
        # has a cost worth two extra attempts. Pass
        # ``summary_retry_timeout_s=llm_timeout`` and the whole sequence fits
        # inside the budget a single call was already allowed to spend, so
        # retrying adds no worst-case latency to the turn.
        #
        # No ceiling → no retries, whatever ``summary_retries`` says. A retry
        # count without a time bound is the defect this pair exists to prevent:
        # one summariser call can occupy the full LLM timeout, so "two more
        # attempts" reads as a small allowance and spends up to 30 minutes. The
        # two knobs are therefore not independent, and the safe reading of a
        # half-specified pair is the conservative one.
        retries = summary_retries if (summary_retry_timeout_s or 0) > 0 else 0
        if retries != summary_retries:
            logger.info(
                "TieredCompactor: summary_retries=%d ignored — no "
                "summary_retry_timeout_s to bound it",
                summary_retries,
            )
        self._tier2 = LLMSummaryCompactor(
            summary_llm=summary_llm,
            failure_fallback="deterministic",
            emit_event=self._capture_tier2,
            max_transient_retries=retries,
            retry_total_timeout_s=summary_retry_timeout_s,
        )
        self._tier2_payload: dict[str, Any] = {}
        #: What the most recent ``compact`` selected. Read by the agent loop to
        #: notify ``on_compaction`` observers; ``None`` until the first call.
        self.last_event: CompactionEvent | None = None
        self._relief_target = relief_target
        # Optional: the same gauge that drives the trigger. When present, the
        # relief check is expressed in REAL tokens instead of the raw estimate.
        self._gauge = gauge
        self._last_no_relief_estimate = 0

    @staticmethod
    def _spill_refs(messages: list[Message]) -> list[str]:
        """Collect unique spill paths from the messages that carry them.

        Reads ``Message.spill_refs``, which both producers set: the Tier 1 tool
        placeholder and the recovery index below. This replaced two heuristics
        that recovered the same paths from prose — one deciding whether a bullet
        looked like a spill path, one deciding whether a block of text WAS an
        index rather than a summary quoting one. Both existed only because the
        index was prose in a user message that the summarizer can echo; with the
        paths carried as data, an echoed header is just words.
        """
        refs: list[str] = []
        for message in messages:
            carried = message.get("spill_refs")
            if isinstance(carried, list):
                for path in carried:
                    if isinstance(path, str) and path and path not in refs:
                        refs.append(path)
                continue
            # Legacy: a Tier 1 placeholder from a history checkpointed before the
            # field existed. A fixed prefix on its own line, not a shape guess.
            content = text_of(message.get("content"))
            if message.get("role") != "tool" or _FULL_TEXT_PREFIX not in content:
                continue
            for line in content.splitlines():
                if line.startswith(_FULL_TEXT_PREFIX):
                    path = line[len(_FULL_TEXT_PREFIX):].strip()
                    if path and path not in refs:
                        refs.append(path)
        return refs

    def _spill_protected_results(self, messages: list[Message]) -> list[str]:
        """Persist protected fan-in bodies before Tier 2 replaces them.

        Tier 1 deliberately leaves these results inline and therefore never
        creates recovery files for them. Once a Tier 2 summary wins, however,
        the original tool messages disappear just like every other old result.
        Spill them at that transition so summary omissions remain recoverable.
        """
        if self._spill is None or not self._protect:
            return []
        id_to_name = _tool_names_by_call_id(messages)
        refs: list[str] = []
        for message in messages:
            if not is_tool_msg(message):
                continue
            name = id_to_name.get(message.get("tool_call_id", ""), "")
            if name not in self._protect:
                continue
            try:
                path = self._spill(name, text_of(message.get("content")))
            except Exception:
                path = None
            if path and path not in refs:
                refs.append(path)
        return refs

    @staticmethod
    def _latest_tool_result_ids(messages: list[Message]) -> frozenset[str]:
        """Tool-result ids produced by the latest assistant tool-call turn.

        Compaction runs at turn end, immediately after those results are
        appended and before the model has seen them.  A spill-less candidate
        must therefore keep them verbatim; shrinking them would discard unseen
        evidence rather than compacting history the model already consumed.
        """
        for message in reversed(messages):
            if message.get("role") != "assistant":
                continue
            ids = {
                str(call.get("id") or "")
                for call in message.get("tool_calls") or []
                if isinstance(call, dict) and call.get("id")
            }
            return frozenset(ids)
        return frozenset()

    def _spill_changed_tool_results(
        self,
        source: list[Message],
        transformed: list[Message],
    ) -> list[str]:
        """Persist every tool body a selected candidate shortened.

        The cheap compression candidates operate on the original history, not
        Tier 1, so they can shorten results Tier 1 deliberately kept as recent.
        Those results may never have been shown to the model.  Spill the source
        body before the transformed candidate becomes the final history.
        Content-addressed spill naming makes re-spilling an older result cheap
        and idempotent.
        """
        if self._spill is None:
            return []
        id_to_name = _tool_names_by_call_id(source)
        refs: list[str] = []
        for original, replacement in zip(source, transformed, strict=True):
            if not is_tool_msg(original) or not is_tool_msg(replacement):
                continue
            original_body = text_of(original.get("content"))
            if original_body == text_of(replacement.get("content")):
                continue
            name = id_to_name.get(original.get("tool_call_id", ""), "tool")
            try:
                path = self._spill(name, original_body)
            except Exception:
                path = None
            if path and path not in refs:
                refs.append(path)
        return refs

    @staticmethod
    def _with_spill_manifest(
        messages: list[Message], refs: list[str],
    ) -> list[Message]:
        """Keep a bounded session-local recovery index when Tier 1 is replaced.

        The index is its own message, carrying the paths in ``spill_refs`` and
        rendering them as text for the model. Keeping it separate from the
        summary is what removes a whole class of bug: it used to be appended into
        the summary message, so replacing it meant finding where the index
        started inside prose the summarizer wrote — and the summarizer sees the
        previous index and quotes its header.
        """
        if not refs:
            return messages
        selected: list[str] = []
        used = 0
        for path in reversed(refs):
            cost = len(path) + 3
            if len(selected) >= _SPILL_MANIFEST_MAX_PATHS or used + cost > _SPILL_MANIFEST_MAX_CHARS:
                break
            selected.append(path)
            used += cost
        selected.reverse()
        index: Message = user_msg(
            _SPILL_MANIFEST_HEADER + "\n" + "\n".join(f"- {path}" for path in selected),
        )
        index["spill_refs"] = selected

        out: list[Message] = []
        replaced = False
        for message in messages:
            # Exactly one message can be the index: the one that says it is.
            if not replaced and message.get("role") == "user" and message.get("spill_refs"):
                out.append(index)
                replaced = True
                continue
            out.append(message)
        if replaced:
            return out

        insert_at = 0
        while insert_at < len(out) and out[insert_at].get("role") == "system":
            insert_at += 1
        out.insert(insert_at, index)
        return out

    def _capture_tier2(self, payload: dict[str, Any]) -> None:
        """Stash the summariser's event. Called during Tier 2 regardless of
        whether Tier 2 goes on to win, so ``_selected`` reads it only for the
        label that actually used it."""
        self._tier2_payload = payload

    def _selected(
        self,
        label: str,
        messages: list[Message],
        *,
        before_tokens: int,
        scale: float,
        spill_refs: int,
    ) -> list[Message]:
        after_tokens = estimate_tokens(messages)
        # Only a tier that summarises has a summary. Tier 2 can run and lose to
        # a cheaper candidate, and its payload would still be in the stash — so
        # key off the selected label, not off the stash being populated.
        summary = ""
        rollback_reason = ""
        attempts = 0
        if label.startswith("tier2"):
            attempts = int(self._tier2_payload.get("attempts") or 0)
            if self._tier2_payload.get("rolled_back"):
                # The slice a failed summariser falls back to can still be the
                # smallest candidate and win. Recording only an empty summary
                # would read as "no summariser ran".
                rollback_reason = str(
                    self._tier2_payload.get("rollback_reason") or "unknown",
                )
            else:
                summary = str(self._tier2_payload.get("summary") or "")
        self.last_event = CompactionEvent(
            turn=0,          # stamped by the loop, which owns the turn counter
            seq=0,           # stamped by the loop, which owns the sequence
            selected=label,
            tokens_before=before_tokens,
            tokens_after=after_tokens,
            relief_met=scale * after_tokens <= self._relief_target,
            spill_refs=spill_refs,
            attempts=attempts,
            summary=summary,
            rollback_reason=rollback_reason,
        )
        # ``rollback_reason`` goes last: ``scripts/truncation_metrics.py`` reads
        # ``selected=(\S+)`` off the front of this line, so appending is safe.
        logger.info(
            "TieredCompactor selected=%s tokens=%d->%d scaled_after=%d "
            "relief_target=%d relief_met=%s spill_refs=%d attempts=%d "
            "rollback_reason=%s",
            label,
            before_tokens,
            after_tokens,
            int(scale * after_tokens),
            self._relief_target,
            scale * after_tokens <= self._relief_target,
            spill_refs,
            attempts,
            rollback_reason or "-",
        )
        return messages

    async def compact(self, messages: list[Message], keep_recent: int) -> list[Message]:
        # Calibrate the estimate to real tokens (see module docstring): scale by
        # the ratio the gauge saw on the pre-compaction messages, so the relief
        # threshold below means real ``max_len*0.6``, matching the trigger.
        scale = self._real_token_scale()
        # Per-compaction, so a Tier 2 summary from an earlier turn can never be
        # reported as this turn's.
        self._tier2_payload = {}
        source_estimate = estimate_tokens(messages)
        tier1 = self._tier1.compact(messages, keep_recent)
        spill_refs = self._spill_refs(tier1)
        selected_spill_ref_count = len(spill_refs)
        tier1_tokens = estimate_tokens(tier1)
        if scale * tier1_tokens <= self._relief_target:
            self._last_no_relief_estimate = 0
            return self._selected(
                "tier1", tier1, before_tokens=source_estimate, scale=scale,
                spill_refs=len(spill_refs),
            )

        # Cheap fallback can reach large results inside the protected recent
        # window. With a spill store, a selected candidate persists every body
        # it shortens below. Without one, keep the latest tool-call turn whole:
        # compaction runs before the model has seen those results even once.
        latest_result_ids = (
            frozenset() if self._spill is not None
            else self._latest_tool_result_ids(messages)
        )
        candidates: list[tuple[str, list[Message], int]] = [
            ("tier1", tier1, tier1_tokens),
        ]
        for width in (1_200, 600, 300):
            # These candidates ARE the final history when they win — no summary
            # stage follows, so protected results must survive them untouched.
            candidate = compress_tool_results(
                messages,
                max_chars=width,
                protect_tool_names=self._protect,
                preserve_tool_result_ids=latest_result_ids,
            )
            candidate_tokens = estimate_tokens(candidate)
            if candidate_tokens < min(item[2] for item in candidates):
                candidates.append((f"tool_compression_{width}", candidate, candidate_tokens))
        best_label, best, best_tokens = min(candidates, key=lambda item: item[2])
        if scale * best_tokens <= self._relief_target:
            if best_label != "tier1":
                # The manifest is this compaction's own recovery index, capped at
                # ~3 KB. Charging it against the relief target would discard a
                # candidate that already freed enough and fall through to a Tier 2
                # round-trip — slower, and under no obligation to come back
                # smaller. The logged ``tokens=`` below still reports the real
                # post-manifest size.
                candidate_refs = self._spill_changed_tool_results(messages, best)
                spill_refs = list(dict.fromkeys([*spill_refs, *candidate_refs]))
                best = self._with_spill_manifest(best, spill_refs)
            self._last_no_relief_estimate = 0
            return self._selected(
                best_label, best, before_tokens=source_estimate, scale=scale,
                spill_refs=len(spill_refs),
            )

        # A failed summary can otherwise run every turn without freeing a token.
        if (
            self._last_no_relief_estimate
            and source_estimate <= int(self._last_no_relief_estimate * 1.1)
        ):
            if best_label != "tier1":
                candidate_refs = self._spill_changed_tool_results(messages, best)
                spill_refs = list(dict.fromkeys([*spill_refs, *candidate_refs]))
                best = self._with_spill_manifest(best, spill_refs)
            return self._selected(
                f"{best_label}_cached", best, before_tokens=source_estimate,
                scale=scale, spill_refs=len(spill_refs),
            )

        # Summarize the real history, not Tier 1's placeholders. Ordinary tool
        # bodies are bounded first, but protected fan-in is passed verbatim: a
        # fixed head/tail cap on one collect_reports result can erase every
        # middle sub-agent report before the summarizer ever sees it.
        tier2_input = compress_tool_results(
            messages,
            protect_tool_names=self._protect,
            preserve_tool_result_ids=latest_result_ids,
        )
        tier2 = await self._tier2.compact(
            tier2_input, keep_recent, preserve_tool_names=self._protect,
        )
        tier2_tokens = estimate_tokens(tier2)
        if tier2_tokens < best_tokens:
            best_label = "tier2"
            protected_refs = self._spill_protected_results(messages)
            compressed_refs = self._spill_changed_tool_results(
                messages, tier2_input,
            )
            all_refs = list(dict.fromkeys([
                *spill_refs, *protected_refs, *compressed_refs,
            ]))
            best = self._with_spill_manifest(tier2, all_refs)
            best_tokens = tier2_tokens
            selected_spill_ref_count = len(all_refs)
        elif best_label != "tier1":
            candidate_refs = self._spill_changed_tool_results(messages, best)
            spill_refs = list(dict.fromkeys([*spill_refs, *candidate_refs]))
            best = self._with_spill_manifest(best, spill_refs)
            selected_spill_ref_count = len(spill_refs)
        # Compare the size BEFORE the manifest was attached, for the same reason
        # the relief-target check above excludes it: the manifest is this
        # compaction's own recovery index (~3 KB, ~750 tokens), not context the
        # compactor failed to free. Charging it here recorded a candidate that
        # genuinely beat Tier 1 — by less than the index costs — as "no relief",
        # which then suppresses the Tier 2 attempt on every following pass until
        # the estimate grows 10%, penalising a compactor that actually worked.
        # ``_selected`` re-estimates the returned messages, so the real
        # post-manifest size is still what gets logged.
        if best_tokens >= tier1_tokens:
            self._last_no_relief_estimate = source_estimate
        else:
            self._last_no_relief_estimate = 0
        return self._selected(
            best_label, best, before_tokens=source_estimate, scale=scale,
            spill_refs=selected_spill_ref_count,
        )

    def _real_token_scale(self) -> float:
        """``real / estimate`` for the last request; 1.0 without a gauge or before
        the first LLM response (falls back to the raw estimate).

        Unclamped, unlike the trigger's use of the same ratio: this one also gates
        the relief target, where a scale below 1.0 is a real measurement (the
        estimator over-stating) and flooring it would escalate to Tier 2 for
        volume that is not there.
        """
        return self._gauge.real_to_estimate_scale() if self._gauge is not None else 1.0
