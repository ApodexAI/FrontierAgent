"""Per-task session context — task-scoped values LLM clients read at
construction time, carried on a ContextVar rather than threaded through
every call site."""

from __future__ import annotations

import os
from contextvars import ContextVar

_TASK_SESSION_ID: ContextVar[str] = ContextVar(
    "frontier_agent_task_session_id", default="",
)

# Providers whose upstream gateway keys session affinity off
# ``x-upstream-session-id``. Add new EAS-backed provider names here — call
# sites just consult ``provider_needs_session_header``.
_EAS_SESSION_PROVIDERS: frozenset[str] = frozenset({
    "apodex",
    "aliyun_pai_eas",
    "aliyun_pai_eas_35b",
})


def set_task_session_id(task_id: str) -> None:
    """Pin the current task's session id. Called once per task at the
    serve / pipeline entry; subsequent LLM constructions read it."""
    _TASK_SESSION_ID.set(str(task_id or ""))


def get_task_session_id() -> str:
    """Return the current task's session id, or ``""`` if unset."""
    return _TASK_SESSION_ID.get()


def provider_needs_session_header(provider: str) -> bool:
    """True when ``provider`` points at an Aliyun EAS-backed upstream
    that wants ``x-upstream-session-id`` for sticky routing."""
    return (provider or "") in _EAS_SESSION_PROVIDERS


def coerce_extra_headers(value: object) -> dict[str, str]:
    """Return a string→string header map, ignoring malformed values.

    Single home for the ``extra_headers`` normalisation every consumer
    needs (provider registry, reporter client kwargs, summary_llm
    candidates): non-dicts become ``{}`` and each surviving key/value is
    coerced to ``str``; ``None`` keys/values are dropped.
    """
    if not isinstance(value, dict):
        return {}
    return {
        str(k): str(v)
        for k, v in value.items()
        if k is not None and v is not None
    }


def fold_extra_headers(
    base: dict[str, str] | None, extra: object,
) -> dict[str, str]:
    """Merge a provider's ``extra_headers`` map onto ``base`` (extra wins
    per key), returning a new dict. ``extra`` that isn't a non-empty dict
    is ignored — the provider declared none. Single home for the
    registry-header fold every OpenAI-compat builder does (e.g. aliyun's
    ``X-DashScope-DataInspection``) into the client's ``default_headers``.
    """
    merged = dict(base or {})
    if isinstance(extra, dict) and extra:
        merged.update(extra)
    return merged


SESSION_AFFINITY_KEY = "x-upstream-session-id"


def mirror_session_query(headers: dict[str, str] | None) -> dict[str, str]:
    """Return the URL-query mirror of the session-affinity header.

    The Aliyun EAS UCH gateway hashes the URL query parameter named
    ``x-upstream-session-id``, not the identically named HTTP header. Clients
    therefore keep the header for log correlation and mirror only this value
    into OpenAI SDK ``default_query`` / ``extra_query``. Providers without the
    header retain their original wire shape.
    """
    if not headers:
        return {}
    for key, value in headers.items():
        if str(key).lower() == SESSION_AFFINITY_KEY and value:
            return {SESSION_AFFINITY_KEY: str(value)}
    return {}


def sticky_session_enabled() -> bool:
    """Whether session affinity is switched on at all (default: yes).

    ``FRONTIER_AGENT_LLM_STICKY_SESSION=0`` (or false/no/off) is a code-free
    kill switch for capacity-bound endpoints where pinning a task to one
    worker is undesirable. Read per call so tests / ops can toggle it.

    This is the single gate for *both* affinity carriers — the per-call
    binding in ``core/runtime/loop/_bind.py`` and the construction-time
    headers below. It has to cover the construction-time path too: the id
    is mirrored into the request URL query (see :func:`mirror_session_query`),
    which the EAS gateway actually honours, so a header injected past a
    disabled switch would still pin traffic.
    """
    return os.getenv("FRONTIER_AGENT_LLM_STICKY_SESSION", "1").strip().lower() not in (
        "0", "false", "no", "off",
    )


def eas_session_headers() -> dict[str, str]:
    """Headers to attach to every EAS-backed-provider request in the
    current task. Empty dict when no task id is set, or when the sticky
    kill switch is off — caller should pass through whatever default
    headers it already had.

    The gate is the provider check at the call site
    (:func:`provider_needs_session_header`), not the model series — pair
    this with that check.
    """
    if not sticky_session_enabled():
        return {}
    sid = _TASK_SESSION_ID.get()
    if not sid:
        return {}
    return {"x-upstream-session-id": sid}


__all__ = [
    "SESSION_AFFINITY_KEY",
    "coerce_extra_headers",
    "eas_session_headers",
    "fold_extra_headers",
    "get_task_session_id",
    "mirror_session_query",
    "provider_needs_session_header",
    "set_task_session_id",
    "sticky_session_enabled",
]
