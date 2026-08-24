"""Error classification for the LLM-call retry / chain-escalation machinery."""

from __future__ import annotations

import re

_OVERLOAD_PATTERNS = (
    re.compile(r"overload", re.IGNORECASE),
    re.compile(r"capacity", re.IGNORECASE),
    re.compile(r"529", re.IGNORECASE),
    # Anthropic explicit error type from the SDK.
    re.compile(r"overloaded_error", re.IGNORECASE),
)

_CREDIT_PATTERNS = (
    re.compile(r"credit", re.IGNORECASE),
    re.compile(r"insufficient[_\s]*quota", re.IGNORECASE),
    re.compile(r"insufficient[_\s]*balance", re.IGNORECASE),
    re.compile(r"billing", re.IGNORECASE),
    re.compile(r"payment[_\s]*required", re.IGNORECASE),
    re.compile(r"\b402\b"),
)

_RATE_LIMIT_PATTERNS = (
    re.compile(r"rate[_\s]*limit", re.IGNORECASE),
    re.compile(r"\b429\b"),
)

_CONTEXT_LENGTH_PATTERNS = (
    re.compile(r"context[_\s]*length", re.IGNORECASE),
    re.compile(r"context_length_exceeded", re.IGNORECASE),
    re.compile(r"longer than the model", re.IGNORECASE),
    re.compile(r"maximum context", re.IGNORECASE),
)

# Transient network / proxy-wrap signatures. ``bad_response_status_code``
# + ``new_api_error`` are the new-api gateway's way of forwarding an
# upstream 5xx / timeout as a 400 envelope to the client — sleeping and
# retrying the same key is the right response, NOT raising a 400 as
# non-transient. Retrying the same key is appropriate for these gateway errors.
_TRANSIENT_NETWORK_PATTERNS = (
    re.compile(r"\btimeout\b", re.IGNORECASE),
    re.compile(r"timed[\s_]*out", re.IGNORECASE),
    re.compile(
        r"connection[\s_]*(?:reset|refused|aborted|error|closed)",
        re.IGNORECASE,
    ),
    re.compile(r"bad_response_status_code", re.IGNORECASE),
    re.compile(r"new_api_error", re.IGNORECASE),
    # Upstream gateway timeouts that get text-wrapped before our status
    # extractor sees them.
    re.compile(r"gateway[\s_]*time[\s_]*out", re.IGNORECASE),
    re.compile(r"upstream[\s_]*(?:timeout|error)", re.IGNORECASE),
)

# Runtime stream watchdog. Matched by type name / message text instead of
# importing ``LLMStreamStalled`` from core to keep infra free of core imports.
_STREAM_STALL_PATTERNS = (
    re.compile(r"\bLLMStreamStalled\b"),
    re.compile(r"stream[_\s-]*stalled", re.IGNORECASE),
    re.compile(r"no chunks for", re.IGNORECASE),
)

# Upstream content-moderation rejections. Same-key retry is hopeless —
# the filter is deterministic on the same input — so these advance the
# chain to the next provider when one is configured. Patterns cover the
# four providers we have first-hand evidence of:
#
# - Some gateways return 400 + ``code: data_inspection_failed``.
# - Anthropic returns ``input_filtered`` / ``output_filtered`` blocks on
#   policy violations (rare on Claude 4.x but documented).
# - OpenAI ``content_policy_violation`` / ``content_filter`` on Azure +
#   o1/gpt-4 deployments with strict moderation enabled.
# - GPT-5.x specifically emits ``Invalid prompt: we've limited access to
#   this content for safety reasons. This type of information may be used
#   to benefit or to harm people...`` as both a pre-flight 400 and a
#   mid-stream error event. The distinctive substrings survive minor wording
#   changes.
_SAFETY_FILTER_PATTERNS = (
    re.compile(r"data[_\s]*inspection[_\s]*failed", re.IGNORECASE),
    re.compile(r"content[_\s]*policy[_\s]*violation", re.IGNORECASE),
    re.compile(r"content[_\s]*filter(?:ed)?", re.IGNORECASE),
    re.compile(r"input[_\s]*filtered", re.IGNORECASE),
    re.compile(r"output[_\s]*filtered", re.IGNORECASE),
    re.compile(r"inappropriate[_\s]*content", re.IGNORECASE),
    re.compile(r"prompt[_\s]*blocked", re.IGNORECASE),
    # GPT-5.x "Invalid prompt: we've limited access to this content for
    # safety reasons..." family. We match short substrings so the check
    # survives wording tweaks and translated variants.
    re.compile(r"limited[_\s]*access[_\s]*to[_\s]*this[_\s]*content[_\s]*for[_\s]*safety", re.IGNORECASE),
    re.compile(r"may[_\s]*be[_\s]*used[_\s]*to[_\s]*benefit[_\s]*or[_\s]*to[_\s]*harm", re.IGNORECASE),
    re.compile(r"violates[_\s]*our[_\s]*usage[_\s]*policies", re.IGNORECASE),
    re.compile(r"your[_\s]*request[_\s]*was[_\s]*blocked", re.IGNORECASE),
    # OpenRouter and general content moderation blocks:
    re.compile(r"content[_\s]*moderation", re.IGNORECASE),
    re.compile(r"moderation[_\s]*policy", re.IGNORECASE),
    re.compile(r"request[_\s]*blocked", re.IGNORECASE),
    re.compile(r"safety[_\s]*system", re.IGNORECASE),
)

# Provider says it doesn't host this model. Same-key retry is futile;
# advance the chain to the next leg. Patterns cover:
# - Distributors may return ``code=model_not_found`` or
#   ``No available channel for model ...``.
# - OpenAI-compatible gateways that surface a 404 / 400 with
#   ``no_such_model`` or ``model_not_supported``.
# - Anthropic: structured ``not_found_error`` body (snake_case literal in
#   ``body.type``) + the ``NotFoundError`` SDK class name surfaced via
#   ``type(err).__name__`` in :func:`_stringify` (both openai-python and
#   anthropic-python raise a ``NotFoundError`` class on 404). Patterns
#   are written precisely — a permissive ``not[_\s]*found[_\s]*error``
#   would false-match Python's builtin ``FileNotFoundError`` and silently
#   advance the chain on unrelated file-IO errors.
_MODEL_UNAVAILABLE_PATTERNS = (
    re.compile(r"model[_\s]*not[_\s]*found", re.IGNORECASE),
    re.compile(r"no[_\s]*such[_\s]*model", re.IGNORECASE),
    re.compile(r"model[_\s]*not[_\s]*supported", re.IGNORECASE),
    re.compile(r"no[_\s]*available[_\s]*channel", re.IGNORECASE),
    re.compile(r"unsupported[_\s]*model", re.IGNORECASE),
    re.compile(r"not_found_error"),
    re.compile(r"\bNotFoundError\b"),
    # OpenRouter (observed in chaos scenario 02, 2026-05-21): a 400
    # BadRequest with body ``"X is not a valid model ID"``. Same root
    # cause as model_not_found — the gateway refuses to route. Same-key
    # retry can't fix a typo'd model name; the next chain leg may use a
    # different canonical model spec and succeed.
    re.compile(r"not[_\s]*a[_\s]*valid[_\s]*model[_\s]*id", re.IGNORECASE),
    re.compile(r"\binvalid[_\s]*model[_\s]*id\b", re.IGNORECASE),
    re.compile(r"\bunknown[_\s]*model\b", re.IGNORECASE),
)

# Authentication failures from upstream providers. A chaos test found this:
# a bare ``OPENROUTER_API_KEY`` clobber surfaced ``openai.AuthenticationError``
# with body ``{"error": {"message": "Missing Authentication header", "code":
# 401}}``. Without auth here, ``is_retriable_with_fallback`` returned False
# and the chain wrapper (a workflow's provider-chain wrapper) never advanced
# to the next chain leg — the run crashed with exit 1. Real users hit this
# every time a key is revoked or scoped wrong; rotating to the next chain leg
# (different key OR different provider) is the only recovery, hence
# chain-advance is correct.
#
# Patterns cover the four wire shapes we have first-hand evidence of:
# - OpenAI / OpenRouter raw 401 body shapes (``invalid_api_key`` /
#   ``invalid_authentication`` / bare ``unauthorized``).
# - OpenRouter's specific "Missing Authentication header" surface
#   (happens when the SDK suppresses an obviously-bogus Bearer value).
# - The openai-python SDK class name surfaced via ``type(err).__name__``
#   in ``_stringify`` — both ``AuthenticationError`` (openai-python) and
#   the equivalent anthropic-python shape.
# - Bare ``401`` status code (the bottom-of-the-barrel fallback when an
#   upstream wrapper strips structured fields but keeps the status).
_AUTH_FAILURE_PATTERNS = (
    re.compile(r"\bAuthenticationError\b"),
    re.compile(r"\bauthentication[_\s]*failed", re.IGNORECASE),
    re.compile(r"\binvalid[_\s]*api[_\s]*key", re.IGNORECASE),
    re.compile(r"\binvalid[_\s]*authentication", re.IGNORECASE),
    re.compile(r"\bmissing[_\s]*authentication", re.IGNORECASE),
    re.compile(r"\bunauthorized\b", re.IGNORECASE),
    re.compile(r"\bunauthenticated\b", re.IGNORECASE),
    re.compile(r"\b401\b"),
)


def _stringify(err: BaseException) -> str:
    """Concatenate every signal an LLM SDK might surface."""
    parts: list[str] = [type(err).__name__, str(err)]
    for attr in ("status_code", "response", "body", "message"):
        val = getattr(err, attr, None)
        if val is not None:
            parts.append(str(val))
    return " | ".join(parts)


def _get_status_code(err: BaseException) -> int | None:
    """Best-effort integer HTTP status extraction.

    Mirrors the heuristic in ``frontier_agent/core/runtime/loop/llm_client``
    so both call sites converge on the same status-attribute lookup
    order.
    """
    for attr in ("status_code", "status", "code"):
        val = getattr(err, attr, None)
        if isinstance(val, int):
            return val
    return None


def is_overloaded_error(err: BaseException) -> bool:
    """True if ``err`` indicates the upstream provider is at capacity.

    Triggers fallback key rotation (the next key shares the provider so
    capacity is rarely fixed by rotation alone — but it's the cheapest
    signal we have, and key-specific capacity quirks DO exist in
    practice).
    """
    blob = _stringify(err)
    return any(p.search(blob) for p in _OVERLOAD_PATTERNS)


def is_credit_exhausted(err: BaseException) -> bool:
    """True if ``err`` indicates the current API key has run out of
    credit / quota. Rotating to a different key usually fixes it."""
    blob = _stringify(err)
    return any(p.search(blob) for p in _CREDIT_PATTERNS)


def is_rate_limited(err: BaseException) -> bool:
    """True if ``err`` is a per-key rate-limit (429). Rotating keys
    likely helps; backing off also helps."""
    blob = _stringify(err)
    return any(p.search(blob) for p in _RATE_LIMIT_PATTERNS)


def is_context_length_error(err: BaseException) -> bool:
    """True if the input exceeds the model's context window.

    Callers must short-circuit straight to the loop's salvage / degraded-
    response path rather than retry or rotate keys — retrying will just hit
    the same wall.
    """
    blob = _stringify(err)
    return any(p.search(blob) for p in _CONTEXT_LENGTH_PATTERNS)


def is_transient_network(err: BaseException) -> bool:
    """True if ``err`` is a request-level transient (timeout / connection
    reset / upstream 5xx / proxy-wrapped upstream blip).

    Decision: sleep + retry on the SAME key. Does NOT escalate chain
    layers — the next provider would see the same transient at roughly
    the same rate, so burning fallback keys here is counter-productive.

    The 5xx-without-overload branch catches the common case where a
    proxy hands back ``502 / 503 / 504`` without any overload substring
    — that's a network problem, not a capacity problem. ``is_overloaded_error``
    keeps priority so a 503 with ``overloaded_error`` in the body still
    routes to rotation rather than backoff.
    """
    if is_stream_stall(err):
        return False
    if is_overloaded_error(err):
        return False
    # model_unavailable is also a 5xx (typically 503 from distributor
    # proxies) but the right response is "advance the chain", not
    # "backoff and retry same key" — same-key retries are guaranteed
    # to fail with the same model_not_found. Surrender precedence to
    # is_retriable_with_fallback here.
    if is_model_unavailable(err):
        return False
    status = _get_status_code(err)
    if status is not None and 500 <= status < 600:
        return True
    blob = _stringify(err)
    return any(p.search(blob) for p in _TRANSIENT_NETWORK_PATTERNS)


def is_stream_stall(err: BaseException) -> bool:
    """True when ``call_llm`` has surfaced a repeated stream watchdog stall.

    The watchdog has already spent the configured same-endpoint stall budget
    before this exception reaches an outer chain runner, so the correct chain
    decision is immediate key/provider advance rather than same-key backoff.
    """
    blob = _stringify(err)
    return any(p.search(blob) for p in _STREAM_STALL_PATTERNS)


def is_safety_filter(err: BaseException) -> bool:
    """True if ``err`` is an upstream content-moderation rejection.

    Retrying the same key is hopeless (filter is deterministic on the
    input). Caller should advance the chain to a different provider if
    one is configured; if not, the error surfaces to the user.
    """
    blob = _stringify(err)
    return any(p.search(blob) for p in _SAFETY_FILTER_PATTERNS)


def is_model_unavailable(err: BaseException) -> bool:
    """True if the provider doesn't host the requested model.

    Distributor proxies (an OpenRouter-style aggregator, a new-api
    gateway) return ``code=model_not_found`` (often with a 503 status
    when the upstream channel pool is empty) when the model name they
    received isn't routable to any backend. Same-key retry is pointless —
    the next provider leg in the chain may have a different upstream
    that DOES host the model, so advance instead.
    """
    blob = _stringify(err)
    return any(p.search(blob) for p in _MODEL_UNAVAILABLE_PATTERNS)


def is_auth_failure(err: BaseException) -> bool:
    """True if ``err`` is an upstream authentication failure (401).

    Catches the four observed shapes: ``openai.AuthenticationError`` SDK
    class, bare ``unauthorized`` text, ``invalid_api_key`` /
    ``invalid_authentication`` structured codes, and OpenRouter's
    "Missing Authentication header" wire surface. Status 403 is
    deliberately excluded — 403 means "key authenticated but not
    authorised for this resource", which often has the same root on a
    sibling provider (e.g. account scoped to specific model families).
    Surface 403 to the operator instead of silently advancing.

    Caller (chain wrapper) advances to the next leg on True. Same-key
    retry can never succeed because the rejection is deterministic on
    (current key, current model). See module docstring for the chaos
    test that discovered this gap (2026-05-21).
    """
    blob = _stringify(err)
    return any(p.search(blob) for p in _AUTH_FAILURE_PATTERNS)


# LangChain raises a bare ``ValueError("No generation chunks were
# returned")`` when an upstream stream completes but yields no usable
# *content* — the dominant failure shape for reasoning models that run
# away in the ``reasoning_content`` channel and never emit a content
# token before hitting ``max_tokens``. A single empty completion among
# the ~150 LLM calls of a long run was fatal because nothing classified it as recoverable, so
# every multi-call heavy run eventually died on one. The HTTP call
# *succeeded* — this is not a timeout/network class — so it gets its own
# detector rather than folding into ``is_transient_network``.
_EMPTY_COMPLETION_PATTERNS = (
    re.compile(
        r"no[\s_]*generation[\s_]*chunks?[\s_]*(?:were[\s_]*)?returned",
        re.IGNORECASE,
    ),
    re.compile(
        r"no[\s_]*completion[\s_]*(?:tokens?|content)[\s_]*returned",
        re.IGNORECASE,
    ),
    re.compile(r"empty[\s_]*completion", re.IGNORECASE),
)


def is_empty_completion(err: BaseException) -> bool:
    """True if the upstream returned a successful response with no content.

    Distinct from a network/timeout error: the call succeeded at the HTTP
    layer but produced zero content tokens (reasoning-runaway,
    all-tokens-in-thinking, or an empty stream). Routed through
    :func:`is_retriable_with_fallback` so the caller retries the same key
    first (a temperature>0 resample frequently recovers) and then advances
    the chain to a different provider, which always recovers.
    """
    blob = _stringify(err)
    return any(p.search(blob) for p in _EMPTY_COMPLETION_PATTERNS)


def is_retriable_with_fallback(err: BaseException) -> bool:
    """The chain-escalation trigger.

    ``overload``, ``credit_exhausted``, ``safety_filter``,
    ``model_unavailable``, AND ``auth_failure`` advance the chain layer.
    ``rate_limit`` + ``transient_network`` trigger backoff-same-key instead
    (handled by the caller, not this predicate). Each of these is
    deterministic on (current provider, current input) — only switching
    providers / keys can change the outcome.

    ``empty_completion`` also routes here: it isn't deterministic on the
    input (a temp>0 resample may recover), but the caller's same-key
    retry budget runs first, and advancing the chain afterwards is the
    guaranteed recovery — so it belongs to the same predicate.
    """
    return (
        is_overloaded_error(err)
        or is_credit_exhausted(err)
        or is_safety_filter(err)
        or is_model_unavailable(err)
        or is_auth_failure(err)
        or is_empty_completion(err)
        or is_stream_stall(err)
    )


def classify_error(err: BaseException) -> str:
    """Short reason label for the ``report.fallback`` SSE payload.

    Precedence (top wins):
      ``context_length`` → ``safety_filter`` → ``model_unavailable`` →
      ``auth_failure`` → ``overloaded`` → ``credit_exhausted`` →
      ``stream_stall`` → ``rate_limited`` → ``transient_network`` → ``other``.

    Context-length wins outright because its caller behaviour differs
    (short-circuit to salvage). Safety-filter wins next because the
    operator dashboard needs to distinguish "model refused" from
    capacity issues. Model-unavailable wins over overload because the
    operator response is different — overload is "wait or fan out",
    model-unavailable is "fix the chain config". Auth-failure sits
    above overload/credit because the operator action is also a
    config fix (rotate / revoke key) — splitting it out from
    ``other`` makes dashboards immediately point at the right knob.
    """
    if is_context_length_error(err):
        return "context_length"
    if is_safety_filter(err):
        return "safety_filter"
    if is_model_unavailable(err):
        return "model_unavailable"
    if is_auth_failure(err):
        return "auth_failure"
    if is_empty_completion(err):
        return "empty_completion"
    if is_overloaded_error(err):
        return "overloaded"
    if is_credit_exhausted(err):
        return "credit_exhausted"
    if is_stream_stall(err):
        return "stream_stall"
    if is_rate_limited(err):
        return "rate_limited"
    if is_transient_network(err):
        return "transient_network"
    return "other"


__all__ = [
    "classify_error",
    "is_auth_failure",
    "is_context_length_error",
    "is_credit_exhausted",
    "is_empty_completion",
    "is_model_unavailable",
    "is_overloaded_error",
    "is_rate_limited",
    "is_retriable_with_fallback",
    "is_safety_filter",
    "is_stream_stall",
    "is_transient_network",
]
