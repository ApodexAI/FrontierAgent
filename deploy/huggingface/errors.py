"""Upstream failure → one actionable sentence for the browser.

A public demo has two audiences for the same exception: the visitor, who needs
to know whether to retry or give up, and the operator, who needs to know which
Space Variable or Secret is wrong. Neither is served by a traceback, and a
traceback is also the most likely place for a credential to leak.
"""

from __future__ import annotations

import asyncio
import re

#: HTTP status → (short reason slug, message). Ordered by how often a
#: misconfigured Space hits them.
_STATUS_MESSAGES: dict[int, tuple[str, str]] = {
    400: (
        "upstream_bad_request",
        "The model endpoint rejected the request (400). If the model does not "
        "support tool/function calling, this demo cannot run on it.",
    ),
    401: (
        "upstream_unauthorized",
        "The model endpoint rejected the credentials (401). Check the "
        "OPENAI_API_KEY Secret.",
    ),
    403: (
        "upstream_forbidden",
        "The model endpoint refused access (403). Check that the API key may "
        "use this model.",
    ),
    404: (
        "upstream_not_found",
        "The model endpoint returned 404. Check OPENAI_BASE_URL (it must be "
        "the API base ending in '/v1', not a model web page) and that "
        "OPENAI_MODEL names a model the endpoint serves.",
    ),
    408: ("upstream_timeout", "The model endpoint timed out (408). Try again."),
    413: (
        "upstream_payload_too_large",
        "The request was too large for the model endpoint (413). Try a shorter "
        "task.",
    ),
    422: (
        "upstream_unprocessable",
        "The model endpoint could not process the request (422). It may not "
        "support tool/function calling.",
    ),
    429: (
        "upstream_rate_limited",
        "The model endpoint is rate limited (429). Please retry shortly.",
    ),
    500: ("upstream_server_error", "The model endpoint failed (500). Please retry."),
    502: ("upstream_bad_gateway", "The model endpoint is unreachable (502). Please retry."),
    503: (
        "upstream_unavailable",
        "The model endpoint is unavailable (503) — it may still be starting up.",
    ),
    504: ("upstream_gateway_timeout", "The model endpoint timed out (504). Please retry."),
}

_GENERIC_UPSTREAM = (
    "upstream_error",
    "The model endpoint could not complete the request.",
)

_STATUS_RE = re.compile(r"\b([45]\d{2})\b")

#: Provider exception *class names* → the status they correspond to. The agent
#: loop reports a failed attempt by class name only (``LLMAttemptContext.
#: error_type``), so this is the sole way to classify a failure that the
#: workflow already absorbed into a best-effort answer.
_ERROR_NAME_STATUS: dict[str, int] = {
    "BadRequestError": 400,
    "AuthenticationError": 401,
    "PermissionDeniedError": 403,
    "NotFoundError": 404,
    "ConflictError": 409,
    "UnprocessableEntityError": 422,
    "RateLimitError": 429,
    "InternalServerError": 500,
    "APIStatusError": 500,
}

_ERROR_NAME_SLUGS: dict[str, tuple[str, str]] = {
    "APITimeoutError": (
        "upstream_timeout",
        "The model endpoint did not respond in time. Please retry.",
    ),
    "APIConnectionError": (
        "upstream_unreachable",
        "The model endpoint could not be reached. Check OPENAI_BASE_URL and "
        "that the endpoint is running.",
    ),
    "APIConnectionTimeoutError": (
        "upstream_unreachable",
        "The model endpoint could not be reached in time. Check OPENAI_BASE_URL.",
    ),
    # The next two are *not* endpoint faults, and saying "the endpoint could
    # not complete the request" for them sends the operator to inspect a
    # healthy service. Both are raised by the loop's own stream watchdogs
    # (``frontier_agent.core.errors``).
    "LLMReasoningRunaway": (
        "model_reasoning_runaway",
        "The model spent its whole output budget on internal reasoning without "
        "answering, so the reply was stopped and resampled. The endpoint is "
        "healthy; retry, or give the task a larger output budget.",
    ),
    "LLMStreamStalled": (
        "upstream_stream_stalled",
        "The model endpoint accepted the request and then went silent "
        "mid-response. This usually means a saturated or dropped connection "
        "upstream, not a bad request.",
    ),
}

#: The subset of :data:`_ERROR_NAME_SLUGS` raised by the loop rather than by the
#: provider SDK. :func:`classify_error` has to consult these before its generic
#: status/timeout branches; the SDK names are left to those branches so the two
#: functions keep agreeing on them.
_WATCHDOG_ERROR_NAMES: frozenset[str] = frozenset({
    "LLMReasoningRunaway", "LLMStreamStalled",
})


def classify_error_name(name: str, detail: str = "") -> tuple[str, str]:
    """Classify a failure known only by its exception class name.

    Used when the workflow has already swallowed the provider error and
    returned a placeholder answer: the demo still has to tell the operator that
    the endpoint — not the model — is the problem.
    """
    name = str(name or "").strip()
    if name in _ERROR_NAME_SLUGS:
        return _ERROR_NAME_SLUGS[name]
    status = _ERROR_NAME_STATUS.get(name)
    if status is None:
        match = _STATUS_RE.search(detail)
        if match:
            status = int(match.group(1))
    if status is not None:
        return _STATUS_MESSAGES.get(status, _GENERIC_UPSTREAM)
    return _GENERIC_UPSTREAM


def classify_error(exc: BaseException) -> tuple[str, str]:
    """Return ``(reason_slug, user_message)`` for ``exc``."""
    from frontier_agent.core.errors import LLMError

    if isinstance(exc, asyncio.CancelledError):
        return "cancelled", "The run was cancelled."

    # ``LLMCallExhausted`` wraps the provider error it gave up on; the wrapped
    # one carries the status code worth reporting.
    root = getattr(exc, "last_exc", None) or exc

    # The loop's own watchdogs, before the generic branches below: neither is a
    # provider status, and ``LLMStreamStalled`` *is* a ``TimeoutError``, so the
    # timeout branch would otherwise absorb it and blame a slow endpoint for a
    # mid-stream black-hole.
    for candidate in (exc, root):
        slug = _ERROR_NAME_SLUGS.get(type(candidate).__name__)
        if slug and type(candidate).__name__ in _WATCHDOG_ERROR_NAMES:
            return slug

    status = _status_of(root)
    if status is not None:
        return _STATUS_MESSAGES.get(status, _GENERIC_UPSTREAM)

    if isinstance(root, (asyncio.TimeoutError, TimeoutError)):
        return (
            "upstream_timeout",
            "The model endpoint did not respond in time. Please retry.",
        )

    if _is_connection_error(root):
        return (
            "upstream_unreachable",
            "The model endpoint could not be reached. Check OPENAI_BASE_URL and "
            "that the endpoint is running.",
        )

    if isinstance(exc, LLMError) or isinstance(root, LLMError):
        reason = str(getattr(exc, "reason", "") or "").strip()
        return (
            f"llm_error{f':{reason}' if reason else ''}",
            "The model call failed" + (f" ({reason})." if reason else "."),
        )

    name = type(exc).__name__
    if name == "TaskWallTimeExceeded":
        return (
            "timeout",
            "The run exceeded this demo's time budget and was stopped.",
        )
    if name == "SandboxUnavailableError":
        return (
            "sandbox_unavailable",
            "The agent's filesystem sandbox is not available in this "
            "deployment; SANDBOX_BACKEND must be 'container' inside the Space "
            "image.",
        )

    return name, "The run failed before producing an answer."


def describe_error(exc: BaseException) -> str:
    """The user-facing sentence only."""
    return classify_error(exc)[1]


def error_reason(exc: BaseException) -> str:
    """The short machine-readable slug only."""
    return classify_error(exc)[0]


def _status_of(exc: BaseException) -> int | None:
    """Best-effort HTTP status for a provider exception.

    ``openai`` and ``httpx`` expose it differently, and some gateways only put
    it in the message text, so all three are checked.
    """
    for attribute in ("status_code", "status", "code"):
        value = getattr(exc, attribute, None)
        if isinstance(value, int) and 400 <= value <= 599:
            return value
    response = getattr(exc, "response", None)
    status = getattr(response, "status_code", None)
    if isinstance(status, int) and 400 <= status <= 599:
        return status
    # Only trust the message when the exception looks like an HTTP failure —
    # a bare "500" inside arbitrary text is not a status code.
    text = str(exc)
    if any(token in type(exc).__name__ for token in ("Status", "HTTP", "APIError")):
        match = _STATUS_RE.search(text)
        if match:
            return int(match.group(1))
    return None


def _is_connection_error(exc: BaseException) -> bool:
    if isinstance(exc, (ConnectionError, OSError)):
        return True
    name = type(exc).__name__
    return name in {
        "APIConnectionError", "ConnectError", "ConnectTimeout",
        "ReadError", "RemoteProtocolError", "APIConnectionTimeoutError",
    }


__all__ = [
    "classify_error",
    "classify_error_name",
    "describe_error",
    "error_reason",
]
