from __future__ import annotations

import logging
import os
from dataclasses import replace
from typing import Any

from frontier_agent.core.llm import LLMResponse

from ._bind import _ensure_bound
from ._response import _visible_response_text, extract_usage

logger = logging.getLogger(__name__)
# ── Reasoning-runaway detection (capped-empty completions) ───────────
#
# Some reasoning models behind OpenAI-compatible gateways can burn the entire
# ``max_tokens`` budget inside
# the reasoning channel and return a *successful* response with zero
# visible content and no tool calls (``finish_reason="length"``).
# ``call_llm`` treats that signature as retriable-same-key. The very first
# runaway switches the retry to a reduced ``max_tokens`` and appends a
# transient, throwaway reminder asking for concise reasoning plus visible
# output/tool use. The reminder never enters durable message history.
# If the retry budget is exhausted the response is returned as-is so the
# loop's existing no-tool nudge stays the behavioural floor — a runaway must
# never escalate into a fatal ``llm_error`` stop past turn 1.

_RUNAWAY_MIN_OUTPUT_TOKENS = 1024
# Ceiling for retry caps after a confirmed runaway. The actual bound cap
# is derived from the observed completion usage so low-cap profiles
# (e.g. CI smoke at 1024) are not accidentally raised to this value.
_RUNAWAY_RETRY_MAX_TOKENS = 8192
# The reminder rides as a ``user`` turn, NOT a trailing ``system`` one.
# Providers disagree on non-leading system messages: the Anthropic adapter
# only lifts a LEADING system message out of the array
# (``anthropic_client._split_system``) and maps anything else through the
# ``{"role": "user"}`` fallthrough, and reasoning-model chat templates on
# SGLang/vLLM commonly render only the first system block. A user turn means
# every provider sees the same instruction in the same position.
_RUNAWAY_RECOVERY_GUIDANCE = (
    "[system reminder] The previous attempt exhausted its completion budget "
    "in private reasoning and produced no visible answer or tool call. On "
    "this retry, keep reasoning concise and produce either visible answer "
    "text or a tool call promptly."
)


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning("Invalid %s=%r; using default %d", name, raw, default)
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        logger.warning("Invalid %s=%r; using default %s", name, raw, default)
        return default


# Deploy-tunable: on one measured logic-puzzle workload the reduced-cap
# resample never recovered, so those deploys set
# FRONTIER_AGENT_RUNAWAY_MAX_RETRIES=1 to save 150-400s per persisted
# runaway. Default stays 2 (other workloads do recover on
# the reduced-cap retry). Read once at import; this is a deployment
# knob, not a per-call one.
_RUNAWAY_MAX_RETRIES = _env_int("FRONTIER_AGENT_RUNAWAY_MAX_RETRIES", 2)
_RUNAWAY_BACKOFF_S = 2.0
# Key under which the agent loop threads cross-turn runaway state
# (a mutable dict) through its metadata into ``call_llm``. Contents:
#   consecutive_turns                   — diagnostic streak counter (log only)
#   last_call_runaway_responses         — surfaced on ``llm_finished``
#   last_call_runaway_reasoning_chars   — surfaced on ``llm_finished``
#   last_call_recovered / last_call_reason — surfaced on ``llm_finished``
RUNAWAY_STATE_KEY = "_runaway_state"

def _is_runaway_response(response: Any) -> bool:
    """True for a successful response whose budget went entirely to
    reasoning: no visible content, no tool calls, and either
    ``finish_reason="length"`` or a completion-token count too large to
    be a plain empty reply (gateways that drop ``finish_reason``)."""
    if not isinstance(response, LLMResponse):
        return False
    if response.tool_calls:
        return False
    if _visible_response_text(response):
        return False
    if response.finish_reason == "length":
        return True
    usage = extract_usage(response) or {}
    return int(usage.get("completion_tokens") or 0) >= _RUNAWAY_MIN_OUTPUT_TOKENS

# Continuation asked of a model whose previous reply was cut off mid-sentence.
# Deliberately does NOT reduce ``max_tokens`` the way the runaway path does: this
# model was producing real output when the cap hit, so giving it less room would
# truncate it again sooner. Brevity is requested in words instead.
TRUNCATION_CONTINUATION_GUIDANCE = (
    "[system reminder] Your previous reply hit the output token limit and was "
    "cut off mid-sentence. The partial text is above. Continue from exactly "
    "where it stopped — do not repeat what you already wrote, and do not start "
    "over. Be brief and reach a tool call or a complete answer this time."
)


def is_truncated_with_text(response: Any) -> bool:
    """True for a reply the token cap cut off *after* it had produced text.

    The other half of :func:`_is_runaway_response`, which handles the same
    ``finish_reason="length"`` with the visible text *empty*. Between them they
    cover the signal, and the split matters because the two need opposite
    treatment: a runaway gets resampled at a smaller cap, while this one already
    contains work worth keeping and needs to be continued.

    Nothing detected this case before. It fell through as an ordinary turn,
    reached ``if not parsed_calls`` and — under ``no_tool_behavior="stop"`` —
    ended the run on a sentence cut mid-token.

    Unlike the runaway detector, there is **no completion-token fallback** for
    gateways that drop ``finish_reason``. That heuristic reads "a large
    completion with nothing visible cannot be a plain empty reply", which is
    sound only while the text is empty. With text present a large completion is
    what a long legitimate answer looks like, so the same heuristic would
    declare every one of them truncated. An explicit ``finish_reason`` is the
    only evidence that can carry this.
    """
    if not isinstance(response, LLMResponse):
        return False
    if response.tool_calls:
        return False
    if response.finish_reason != "length":
        return False
    return bool(_visible_response_text(response))


def _runaway_retry_max_tokens(response: Any) -> int | None:
    """Return a retry cap that cannot exceed the observed runaway cap.

    Each successive runaway inside one call halves again (the second
    reduction is derived from a completion that was ALREADY capped), so the
    squeeze is progressive. ``_RUNAWAY_MIN_OUTPUT_TOKENS`` is the floor: below
    it a capped-empty completion is no longer even detectable as a runaway
    (see :func:`_is_runaway_response`), so shrinking past it would trade a
    diagnosable failure for a silent empty reply.
    """
    usage = extract_usage(response) or {}
    try:
        completion_tokens = int(usage.get("completion_tokens") or 0)
    except (TypeError, ValueError):
        completion_tokens = 0
    if completion_tokens <= 0:
        return None
    if completion_tokens <= _RUNAWAY_MIN_OUTPUT_TOKENS:
        return completion_tokens
    return max(
        _RUNAWAY_MIN_OUTPUT_TOKENS,
        min(_RUNAWAY_RETRY_MAX_TOKENS, completion_tokens // 2),
    )


def _runaway_retry_max_tokens_from_cap(active_cap: Any) -> int | None:
    """Derive a safe retry cap when an early-cancelled stream has no usage.

    The active request cap is authoritative for the upper bound. This helper
    must never raise a low-cap profile toward the normal 8K recovery ceiling.
    """
    try:
        cap = int(active_cap)
    except (TypeError, ValueError):
        return None
    if cap <= 0:
        return None
    if cap <= _RUNAWAY_MIN_OUTPUT_TOKENS:
        return cap
    return max(
        _RUNAWAY_MIN_OUTPUT_TOKENS,
        min(_RUNAWAY_RETRY_MAX_TOKENS, cap // 2),
    )


def _bind_reduced_max_tokens(
    llm: Any,
    response: Any | None = None,
    *,
    active_cap: Any = None,
) -> Any:
    """Bind the runaway-retry ``max_tokens`` cap for follow-up attempts.

    Mirrors :func:`bind_temperature` — falls back to the original LLM
    when the wrapper doesn't support ``.bind()`` kwargs.
    """
    retry_max_tokens = (
        _runaway_retry_max_tokens(response)
        if response is not None
        else _runaway_retry_max_tokens_from_cap(active_cap)
    )
    if retry_max_tokens is None:
        logger.debug(
            "Runaway max_tokens cap could not be inferred from usage; "
            "retrying at the existing budget.",
        )
        return llm
    return replace(_ensure_bound(llm), max_tokens=retry_max_tokens)
