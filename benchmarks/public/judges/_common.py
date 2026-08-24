"""Shared judge infrastructure — LLM client, retry helpers, type aliases."""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import Awaitable, Callable
from typing import Literal

from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

Verdict = Literal["CORRECT", "INCORRECT", "NOT_ATTEMPTED"]
JudgeFn = Callable[[str, str, str], Awaitable[Verdict]]
# Rubric judges additionally surface a raw score (0-10 for FS, F1 for
# WideSearch). Distinct from JudgeFn: same arguments, different return.
ScoreFn = Callable[[str, str, str], Awaitable["tuple[Verdict, float | None]"]]

_DEFAULT_JUDGE_MODEL = "gpt-4.1-2025-04-14"

# ``JUDGE_MODEL`` overrides every per-benchmark pin.
#
# Each judge pins the model its benchmark's official grader used, which is what
# makes a score comparable — but those pins are gateway-specific. Several are
# OpenRouter-style names (``openai/gpt-5``, ``google/gemini-2.5-flash``) that
# neither a plain OpenAI key nor a differently-routed gateway can serve, and an
# unreachable judge does not fail loudly: it returns NOT_ATTEMPTED for every
# question, so the run reports 0% accuracy and looks like a model failure.
#
# So the override exists, and using it is logged once — a score graded by a
# substitute model is no longer officially comparable, and that has to be
# visible in the run's own output rather than remembered.
_JUDGE_MODEL_ENV = "JUDGE_MODEL"
_override_logged = False


def _resolve_judge_model(
    model_override: str | None, default_model: str | None,
) -> str:
    """Judge model, honouring ``JUDGE_MODEL`` above every per-benchmark pin."""
    global _override_logged
    pinned = model_override or default_model or _DEFAULT_JUDGE_MODEL
    env = (os.environ.get(_JUDGE_MODEL_ENV) or "").strip()
    if not env or env == pinned:
        return pinned
    if not _override_logged:
        _override_logged = True
        logger.warning(
            "%s=%s overrides the benchmark's pinned judge model (%s). Scores "
            "graded this way are not comparable with results using the "
            "official grader.",
            _JUDGE_MODEL_ENV, env, pinned,
        )
    return env


def _make_client() -> AsyncOpenAI:
    api_key = os.environ.get("JUDGE_API_KEY") or os.environ.get("OPENAI_API_KEY")
    base_url = os.environ.get("JUDGE_BASE_URL") or os.environ.get("OPENAI_BASE_URL")
    if base_url:
        base_url = base_url.rstrip("/").removesuffix("/chat/completions")

    # Every judge call in a run sends the same long grading prompt, so a
    # session-affinity header is what makes the upstream prompt cache hit.
    # ``JUDGE_SESSION`` is set once per run by the runner and inherited by every
    # worker subprocess, so the whole batch shares one session. Gateways that
    # don't know the header ignore it.
    headers = {}
    if session := os.environ.get("JUDGE_SESSION"):
        headers["X-Llmhub-Session"] = session
    return AsyncOpenAI(api_key=api_key, base_url=base_url,
                       default_headers=headers or None)


def _build_judge_kwargs(
    prompt: str, *, max_completion_tokens: int,
    model_override: str | None = None,
    default_model: str | None = None,
    default_reasoning_effort: str | None = None,
    system: str | None = None,
    temperature: float | None = None,
) -> dict:
    """Common kwargs for ``chat.completions.create`` on judge calls.

    Model resolution: ``model_override`` (call-site pin) > ``default_model``
    (per-benchmark default) > ``_DEFAULT_JUDGE_MODEL`` (global fallback).
    Reasoning effort: ``default_reasoning_effort`` if supplied, else unset.
    """
    model = _resolve_judge_model(model_override, default_model)
    kwargs: dict = {
        "model": model,
        "messages": (
            ([{"role": "system", "content": system}] if system else [])
            + [{"role": "user", "content": prompt}]
        ),
        "max_completion_tokens": max_completion_tokens,
    }
    if default_reasoning_effort:
        kwargs["reasoning_effort"] = default_reasoning_effort
    if temperature is not None:
        kwargs["temperature"] = temperature
    return kwargs


def _model_supports_reasoning_effort(model: str) -> bool:
    """Whether the model accepts the ``reasoning_effort`` kwarg.

    OpenAI o-series (o1 / o3 / o4) and gpt-5 family accept it; classic
    chat models (gpt-4.1 / gpt-4o / gpt-4) reject it with a 400.
    """
    m = (model or "").lower()
    if m.startswith("openai/"):
        m = m[len("openai/"):]
    if m.startswith(("o1", "o3", "o4")):
        return True
    return m.startswith("gpt-5")


def _effort_chain(model: str, initial: str) -> list[str]:
    """Cascade of ``reasoning_effort`` levels to try (``initial`` → medium → low).

    For non-reasoning models returns ``[""]`` so the kwarg is never sent.
    """
    if not _model_supports_reasoning_effort(model):
        return [""]
    chain: list[str] = []
    for e in [(initial or "").lower(), "medium", "low"]:
        if e and e not in chain:
            chain.append(e)
    return chain or [""]


async def _judge_call_with_effort_fallback(kwargs: dict, *, label: str) -> str:
    """Run ``chat.completions.create`` with ``reasoning_effort`` cascade.

    Cascades the initial effort (from ``kwargs["reasoning_effort"]`` if set)
    through medium → low, retrying once at each level. Returns the completion
    string, or ``""`` if every attempt yielded empty content or errored.
    """
    initial = (kwargs.get("reasoning_effort") or "").lower()
    chain = _effort_chain(kwargs.get("model", ""), initial)
    content = ""
    for attempt_effort in chain:
        if attempt_effort:
            kwargs["reasoning_effort"] = attempt_effort
        else:
            kwargs.pop("reasoning_effort", None)
        for attempt_idx in range(2):
            try:
                resp = await _make_client().chat.completions.create(**kwargs)
            except Exception as exc:
                # API error (rate limit, 5xx, network). Sleep + continue so
                # the surrounding effort×attempt loop actually retries; the
                # earlier ``return ""`` here defeated the retry budget by
                # giving up after a single transient failure.
                logger.warning("%s judge call error (%s, effort=%s, attempt=%d): %s",
                               label, kwargs.get("model"), attempt_effort, attempt_idx, exc)
                await asyncio.sleep(1.0)
                continue
            content = (resp.choices[0].message.content or "").strip()
            if content:
                if attempt_effort != initial or attempt_idx > 0:
                    logger.info(
                        "%s judge: empty at effort=%s, recovered at effort=%s (retry=%d)",
                        label, initial, attempt_effort, attempt_idx,
                    )
                return content
            await asyncio.sleep(1.0)
    return content
