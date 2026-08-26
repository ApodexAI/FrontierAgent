"""Secondary-LLM content summarization for long web pages."""
from __future__ import annotations

import asyncio
import hashlib
import hmac
import logging
import secrets
from contextvars import ContextVar, Token
from typing import Any

import httpx

from frontier_agent.infra.config import get_config
from frontier_agent.infra.session_context import (
    coerce_extra_headers,
    fold_extra_headers,
)
from frontier_agent.infra.usage_meter import record_llm_usage

logger = logging.getLogger(__name__)

FALLBACK_TRUNCATE = 20_000
_MAX_RETRIES = 4
_TRUNCATE_STEP = 40_960
_REQUEST_TIMEOUT = 300
_KEY_FINGERPRINT_SECRET = secrets.token_bytes(32)


# ── Profile-driven override ───────────────────────────────────────────
# ``summarize()`` reads ``base_url`` / ``model`` / ``api_key`` from this
# ContextVar first, then falls back to ``get_config()`` env vars. Workflows
# install the override at task start using the profile's ``summary_llm:`` block
# resolved by ``_inject_provider_creds`` so ``api_key`` / ``base_url``
# are filled in by provider registry lookup).
#
# ContextVar — not module global — so concurrent tasks on different
# profiles don't trample each other; each asyncio task's copy of the
# context inherits independently.
_summary_llm_override: ContextVar[dict[str, Any] | None] = ContextVar(
    "_summary_llm_override", default=None,
)


def set_summary_llm_override(cfg: dict[str, Any] | None) -> Token:
    """Install a profile-derived summary-LLM block; returns a reset token.

    Pass ``cfg = None`` or an empty dict to clear (forces fallback to
    env-driven ``get_config()`` for the remainder of the context).
    Use :func:`reset_summary_llm_override` in a ``finally`` block to
    restore the prior context.
    """
    return _summary_llm_override.set(cfg or None)


def reset_summary_llm_override(token: Token) -> None:
    """Restore the previous override (paired with ``set_summary_llm_override``)."""
    _summary_llm_override.reset(token)


def _normalise_endpoint(base_url: str) -> str:
    """Tolerate either a base URL (…/v1) or the full endpoint
    (…/v1/chat/completions) — users routinely copy the base URL and
    forget the trailing path."""
    endpoint = (base_url or "").rstrip("/")
    if endpoint and not endpoint.endswith("/chat/completions"):
        endpoint = endpoint + "/chat/completions"
    return endpoint


def summary_llm_candidates() -> list[dict[str, Any]]:
    """Ordered summary-LLM attempts: primary first, then fallbacks.

    Each entry is ``{"endpoint", "model", "api_key", "provider"}`` with
    ``endpoint`` already normalised to the full chat-completions path.
    Sources, in order:

    1. Primary — profile override ContextVar (``summary_llm:`` block,
       installed by swarm's ``main_agent_node``) or, absent that, env
       ``SUMMARY_LLM_BASE_URL`` + ``SUMMARY_LLM_MODEL`` /
       ``SUMMARY_LLM_MODEL_NAME`` (both names accepted — the
       aligned tools historically read the ``_NAME`` variant;
       accepting either here kills that gotcha).
    2. Profile override's ``fallback:`` sub-block (provider creds
       already injected by ``_inject_provider_creds``).
    3. Env ``SUMMARY_LLM_FALLBACK_BASE_URL`` / ``_MODEL`` / ``_API_KEY``.
    4. Last resort, only when 1–3 yielded nothing — the primary
       OpenAI-compatible model (``OPENAI_BASE_URL`` / ``OPENAI_MODEL``),
       so an ``OPENAI_*``-only .env still gets working extraction.

    Callers walk the list on retriable failure so a saturated /
    runaway-reasoning primary (e.g. the self-hosted 397B) degrades to
    the fallback gateway instead of dropping to raw-content truncation.

    Every entry carries a live ``api_key``. Never print or log the return
    value directly — use :func:`describe_candidates`.
    """
    import os

    candidates: list[dict[str, Any]] = []

    def _push(
        base_url: str,
        model: str,
        api_key: str,
        provider: str,
        extra_headers: object = None,
    ) -> None:
        endpoint = _normalise_endpoint(base_url)
        if endpoint and model:
            candidates.append({
                "endpoint": endpoint,
                "model": model,
                "api_key": api_key or "",
                "provider": provider,
                # Provider-declared headers (e.g. aliyun
                # X-DashScope-DataInspection) injected onto the
                # ``summary_llm:`` block by ``_inject_provider_creds`` —
                # threaded so they ride this raw-httpx call too. Coerced
                # here so call sites pass the raw value.
                "extra_headers": coerce_extra_headers(extra_headers),
            })

    override = _summary_llm_override.get()
    if override and override.get("base_url") and override.get("model"):
        _push(
            str(override.get("base_url") or ""),
            str(override.get("model") or ""),
            str(override.get("api_key") or ""),
            str(override.get("_provider_label") or override.get("provider") or ""),
            override.get("extra_headers"),
        )
    else:
        config = get_config()
        _push(
            config.summary_llm_base_url,
            config.summary_llm_model
            or os.environ.get("SUMMARY_LLM_MODEL_NAME", ""),
            config.summary_llm_api_key,
            "summary_llm",
        )

    fb = (override or {}).get("fallback")
    if isinstance(fb, dict):
        _push(
            str(fb.get("base_url") or ""),
            str(fb.get("model") or ""),
            str(fb.get("api_key") or ""),
            str(fb.get("_provider_label") or fb.get("provider") or ""),
            fb.get("extra_headers"),
        )
    _push(
        os.environ.get("SUMMARY_LLM_FALLBACK_BASE_URL", ""),
        os.environ.get("SUMMARY_LLM_FALLBACK_MODEL", "")
        or os.environ.get("SUMMARY_LLM_FALLBACK_MODEL_NAME", ""),
        os.environ.get("SUMMARY_LLM_FALLBACK_API_KEY", ""),
        "summary_llm_fallback",
    )

    # Last resort — nothing configured anywhere: borrow the primary
    # OpenAI-compatible model (``OPENAI_BASE_URL`` / ``OPENAI_MODEL``). A
    # clone that fills in only ``OPENAI_*`` then gets working extraction
    # instead of raw truncation (``web_fetch``) or a hard per-fetch error
    # (``web_fetch_aligned``).
    #
    # Gated on the list being *otherwise empty* on purpose: an explicitly
    # configured summary LLM that fails should stay on the cheap truncation
    # path rather than dumping whole pages into the expensive primary model.
    if not candidates:
        config = get_config()
        _push(
            config.openai_base_url,
            config.openai_model,
            config.openai_api_key,
            "summary_llm_primary_fallback",
        )
        if candidates:
            logger.info(
                "[Summary LLM] Not configured — falling back to the primary "
                "model (%s). Set SUMMARY_LLM_BASE_URL to use a cheaper one.",
                candidates[0]["model"],
            )
    return candidates


def describe_candidates(candidates: list[dict[str, Any]]) -> str:
    """Printable candidate list with credentials reduced to a fingerprint.

    :func:`summary_llm_candidates` returns live API keys, so printing its result
    verbatim writes them into terminal scrollback, CI logs and pasted bug
    reports. Debug through this instead: it keeps what identifies a candidate
    (provider, model, endpoint) and reduces the key to its length plus a keyed,
    process-local 12-hex-digit HMAC prefix — enough to tell two keys apart in
    one run without enabling offline guesses from retained logs.
    """
    if not candidates:
        return "(no summary LLM candidates)"
    lines: list[str] = []
    for index, cand in enumerate(candidates, 1):
        key = str(cand.get("api_key") or "")
        fingerprint = (
            f"len={len(key)} #"
            f"{hmac.digest(_KEY_FINGERPRINT_SECRET, key.encode(), hashlib.sha256).hex()[:12]}"
            if key else "unset"
        )
        lines.append(
            f"{index}. provider={cand.get('provider') or '?'}"
            f" model={cand.get('model') or '?'}"
            f" endpoint={cand.get('endpoint') or '?'}"
            f" api_key={fingerprint}"
        )
    return "\n".join(lines)


_EXTRACT_INFO_PROMPT = """You are given a piece of content and the requirement of information to extract. Your task is to extract the information specifically requested. Be precise and focus exclusively on the requested information.

INFORMATION TO EXTRACT:
{focus}

INSTRUCTIONS:
1. Extract the information relevant to the focus above.
2. If the exact information is not found, extract the most closely related details.
3. Be specific and include exact details when available.
4. Clearly organize the extracted information for easy understanding.
5. Do not include general summaries or unrelated content.

CONTENT TO ANALYZE:
{content}

EXTRACTED INFORMATION:"""


def _truncate_fallback(content: str) -> str:
    if len(content) > FALLBACK_TRUNCATE:
        return content[:FALLBACK_TRUNCATE] + "\n\n[Content truncated...]"
    return content


def _build_payload(model: str, prompt: str) -> dict:
    """Build OpenAI-compatible chat-completions payload.

    GPT-5 family uses ``max_completion_tokens`` + reasoning_effort knobs;
    everything else uses the standard ``max_tokens`` + ``temperature``.

    Self-hosted reasoning models behind SGLang get
    ``chat_template_kwargs.enable_thinking=false`` — extraction is an
    auxiliary call; with thinking ON (their default) the model burns the
    whole completion budget on hidden reasoning and returns
    ``content=None``, which degrades web_fetch to raw truncation.

    The keys below are matched against the *model name*, so they are part
    of the contract with the serving stack — do not rename them for
    cosmetic reasons or the branch silently stops firing.
    """
    lowered = model.lower()
    if "gpt-5" in lowered or "gpt5" in lowered:
        return {
            "model": model,
            "max_completion_tokens": 8192,
            "messages": [{"role": "user", "content": prompt}],
            "reasoning_effort": "minimal",
            "service_tier": "flex",
        }
    payload: dict = {
        "model": model,
        "max_tokens": 8192,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 1.0,
    }
    if any(k in lowered for k in ("qwen", "apodex", "sglang", "397b")):
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    return payload


async def summarize(content: str, focus: str) -> str:
    """Extract info matching ``focus`` from ``content`` via the summary LLM.

    Walks :func:`summary_llm_candidates` in order — primary (profile
    override / env) first, then the profile ``fallback:`` block / env
    ``SUMMARY_LLM_FALLBACK_*`` — retrying each candidate up to
    ``_MAX_RETRIES`` before moving on. Returns truncated raw content if
    the summary LLM is not configured or every candidate fails — the
    caller never needs to handle the unconfigured case explicitly.
    """
    candidates = summary_llm_candidates()
    if not candidates:
        logger.debug("Summary LLM not configured — returning truncated raw content")
        return _truncate_fallback(content)

    for ci, cand in enumerate(candidates):
        summary = await _summarize_one(
            cand, content, focus, candidate_index=ci,
        )
        if summary:
            return summary
        if ci + 1 < len(candidates):
            logger.warning(
                "[Summary LLM] Candidate %d (%s) exhausted — falling back "
                "to %s",
                ci + 1, cand["model"], candidates[ci + 1]["model"],
            )

    return _truncate_fallback(content)


async def _summarize_one(
    cand: dict[str, Any],
    content: str,
    focus: str,
    *,
    candidate_index: int = 0,
) -> str:
    """One candidate's retry loop. Returns ``""`` when exhausted."""
    endpoint, model, api_key = cand["endpoint"], cand["model"], cand["api_key"]
    prompt = _EXTRACT_INFO_PROMPT.format(focus=focus, content=content)
    payload = _build_payload(model, prompt)

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    # Provider-declared headers (e.g. aliyun X-DashScope-DataInspection)
    # threaded from the resolved candidate — without the inspection-disable
    # header DashScope intermittently 400s on benign scraped content
    # (``code=data_inspection_failed``).
    headers = fold_extra_headers(headers, cand.get("extra_headers"))

    current_content = content
    for attempt in range(_MAX_RETRIES):
        try:
            async with httpx.AsyncClient(timeout=_REQUEST_TIMEOUT) as client:
                resp = await client.post(endpoint, headers=headers, json=payload)

            # Context-too-long: truncate and retry in place.
            body_text = resp.text
            if (
                resp.status_code >= 400
                and ("maximum context length" in body_text
                     or "longer than the model's context length" in body_text)
            ):
                chars_to_remove = _TRUNCATE_STEP * (attempt + 1)
                if chars_to_remove < len(current_content):
                    current_content = content[:-chars_to_remove] + "[...truncated]"
                    payload["messages"][0]["content"] = _EXTRACT_INFO_PROMPT.format(
                        focus=focus, content=current_content
                    )
                    logger.info(
                        "[Summary LLM] Context too long, dropping %d chars, retrying (attempt %d)",
                        chars_to_remove, attempt + 1,
                    )
                    continue
                return ""

            resp.raise_for_status()
            data = resp.json()
            # Raw-httpx LLM call — forward usage to the bound
            # meter so the tokens land in the top-level ``usage_summary
            # .llm["{model}@{provider}"]`` instead of vanishing.
            usage = data.get("usage") or {}
            if usage:
                record_llm_usage(
                    model=model,
                    provider=cand.get("provider") or "summary_llm",
                    prompt_tokens=int(usage.get("prompt_tokens", 0) or 0),
                    completion_tokens=int(
                        usage.get("completion_tokens", 0) or 0,
                    ),
                    cache_read_tokens=int(
                        (usage.get("prompt_tokens_details") or {}).get(
                            "cached_tokens", 0,
                        ) or 0,
                    ),
                )
            summary = (
                data.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
            )
            if summary:
                logger.info(
                    "[Summary LLM] Extracted %d chars from %d chars raw content",
                    len(summary), len(content),
                )
                return summary
            logger.warning(
                "[Summary LLM] Empty response on attempt %d (candidate %d)",
                attempt + 1, candidate_index + 1,
            )
            return ""

        except Exception as exc:
            logger.warning("[Summary LLM] Error on attempt %d: %s", attempt + 1, exc)
            if attempt < _MAX_RETRIES - 1:
                await asyncio.sleep(1 * (attempt + 1))
                continue
            return ""

    return ""
