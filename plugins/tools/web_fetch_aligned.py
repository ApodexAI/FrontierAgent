"""Jina-backed web fetch, byte-compatible with the reference agent's tool.

Extraction prompt, retry schedule and output formatting are reproduced
exactly: they shape what the agent sees, and the model was trained on this
form.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any

import httpx

from frontier_agent.core.tool import tool
from frontier_agent.infra.summary_llm import (
    FALLBACK_TRUNCATE,
    summary_llm_candidates,
)
from frontier_agent.infra.usage_meter import record_api_request, record_llm_usage
from plugins.tools._bounded_fetch import (
    MAX_REDIRECT_HOPS,
    RedirectRefused,
    binary_content_type,
    blocked_download_url,
    decode_body,
    next_hop,
    non_public_url_error,
    pin_to_address,
    read_bounded,
    strip_cross_origin_credentials,
    vet_public_url,
)
from plugins.tools._render_check import (
    MIN_RENDERED_BODY_CHARS,
    reader_body,
    unrendered_kind,
)
from plugins.tools._scrape_cache import ScrapeUnavailable, scrape_result_cache

logger = logging.getLogger(__name__)


# ── Env (read at call time, not import time — see web_search_aligned.py) ──


def _jina_api_key() -> str:
    return os.getenv("JINA_API_KEY", "")


def _jina_base_url() -> str:
    # Reference default: ``https://r.jina.ai`` (direct). The .env can
    # override to a proxy without changing this fallback.
    return os.getenv("JINA_BASE_URL", "https://r.jina.ai")


def _summary_llm_base_url() -> str | None:
    return os.environ.get("SUMMARY_LLM_BASE_URL")


def _summary_llm_model_name() -> str | None:
    return os.environ.get("SUMMARY_LLM_MODEL_NAME")


def _summary_llm_api_key() -> str | None:
    return os.environ.get("SUMMARY_LLM_API_KEY")


# Matches the reference tool's list. Just two patterns —
# Twitter/Reddit/etc are intentionally fetchable.
_BANNED_URL_PATTERNS: tuple[str, ...] = (
    "huggingface.co/datasets",
    "huggingface.co/spaces",
)


def _ensure_list(val: Any) -> Any:
    """Unwrap doubly-serialised JSON list payloads (mirrors web_search)."""
    if isinstance(val, str) and val.startswith("["):
        try:
            parsed = json.loads(val)
            if isinstance(parsed, list):
                return parsed
        except (json.JSONDecodeError, ValueError):
            pass
    return val


# ── Un-rendered ("app shell") handling ────────────────────────────────
#
# Detection lives in ``_render_check`` (shared with ``web_fetch.py``). Aliased
# to the module-private names this file uses elsewhere.
_reader_body = reader_body
_unrendered_kind = unrendered_kind
_MIN_RENDERED_BODY_CHARS = MIN_RENDERED_BODY_CHARS

# Render budget handed to Jina. The reader's own default gives an SPA too
# little time and returns the pre-hydration DOM; ``web_fetch.py`` (the
# non-aligned implementation) has always sent 20-30s. Verified on
# ``icmconjectures.com/1983-prob-8`` (2026-07-29): without this header the URL
# yields a 285-byte shell, with it 4945+ bytes of content.
_JINA_TIMEOUT_S = 30
_JINA_RETRY_TIMEOUT_S = 40


# ── Jina scraping ─────────────────────────────────────────────────────


async def _scrape_url_with_jina(
    url: str,
    custom_headers: dict[str, str] | None = None,
    max_chars: int = 102400 * 4,
    *,
    engine: str | None = None,
    no_cache: bool = False,
    render_timeout_s: int = _JINA_TIMEOUT_S,
) -> dict[str, Any]:
    """Scrape via Jina reader API. Mirrors the reference implementation exactly.

    Retries on connect/read timeouts and 5xx/408/409/425/429 with the
    fixed schedule ``[1, 2, 4, 8]`` seconds. Detects Jina's
    ``InsufficientBalanceError`` JSON body and returns it as a structured
    failure rather than treating the body as content.

    ``engine`` / ``no_cache`` / ``render_timeout_s`` drive the reader's
    rendering: the escalation retry in :func:`_fetch_single` re-requests a
    known-empty page with the browser engine, a longer render budget and
    Jina's own response cache bypassed.
    """
    api_key = _jina_api_key()
    if not api_key:
        return {"success": False, "content": "", "error": "JINA_API_KEY not set"}

    # Avoid duplicate Jina URL prefix — if the user already passed a
    # ``https://r.jina.ai/<inner>`` URL, strip the outer prefix so we
    # don't double-wrap.
    if url.startswith("https://r.jina.ai/") and url.count("http") >= 2:
        url = url[len("https://r.jina.ai/") :]

    jina_url = f"{_jina_base_url()}/{url}"
    headers = {
        "Authorization": f"Bearer {api_key}",
        # Render budget (see _JINA_TIMEOUT_S). Sent on every call, not just the
        # retry — the default budget is what loses the race in the first place.
        "x-timeout": str(render_timeout_s),
    }
    if engine:
        headers["x-engine"] = engine
    if no_cache:
        # Jina caches its own responses, so a shell it captured once is served
        # back for every later attempt. Bypass it when re-trying.
        headers["x-no-cache"] = "true"
    if custom_headers:
        headers.update(custom_headers)

    retry_delays = [1, 2, 4, 8]
    response: httpx.Response | None = None

    body = b""
    for attempt, delay in enumerate(retry_delays, 1):
        try:
            async with httpx.AsyncClient() as client, client.stream(
                "GET",
                jina_url,
                headers=headers,
                timeout=httpx.Timeout(None, connect=20, read=60),
                follow_redirects=True,
            ) as response:
                response.raise_for_status()
                # Bounded read: stop at the byte cap instead of
                # buffering a whole (possibly multi-GB) body that
                # ``[:max_chars]`` would throw away anyway.
                body, _ = await read_bounded(response)
            # Count one answered, billable Jina reader request.
            # The direct-httpx fallback (``_scrape_url_with_python``)
            # intentionally records nothing — it doesn't bill Jina.
            record_api_request("jina")
            break
        except (httpx.ConnectTimeout, httpx.ConnectError, httpx.ReadTimeout) as e:
            record_api_request("jina", requests=0, errors=1)
            if attempt < len(retry_delays):
                await asyncio.sleep(delay)
                continue
            return {"success": False, "content": "", "error": str(e)}
        except httpx.HTTPStatusError as e:
            sc = e.response.status_code
            record_api_request("jina", requests=0, errors=1)
            if (sc >= 500 or sc in [408, 409, 425, 429]) and attempt < len(
                retry_delays,
            ):
                await asyncio.sleep(delay)
                continue
            return {"success": False, "content": "", "error": str(e)}
        except Exception as e:
            record_api_request("jina", requests=0, errors=1)
            return {"success": False, "content": "", "error": str(e)}

    if response is None:
        return {"success": False, "content": "", "error": "No response received"}

    content = decode_body(response, body)
    if not content:
        return {"success": False, "content": "", "error": "Empty response from Jina"}

    # Detect Jina balance exhaustion — the body is JSON like
    # ``{"name": "InsufficientBalanceError", ...}`` rather than the page.
    try:
        maybe_err = json.loads(content)
        if (
            isinstance(maybe_err, dict)
            and maybe_err.get("name") == "InsufficientBalanceError"
        ):
            return {
                "success": False,
                "content": "",
                "error": "Jina insufficient balance",
            }
    except json.JSONDecodeError:
        pass

    return {"success": True, "content": content[:max_chars], "error": ""}


async def _scrape_url_with_python(
    url: str,
    custom_headers: dict[str, str] | None = None,
    max_chars: int = 102400 * 4,
) -> dict[str, Any]:
    """Direct httpx GET fallback when Jina fails. Same retry policy."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    }
    if custom_headers:
        headers.update(custom_headers)

    retry_delays = [1, 2, 4]

    for attempt, delay in enumerate(retry_delays, 1):
        try:
            async with httpx.AsyncClient() as client:
                # Redirects are walked by hand so every hop is vetted the way
                # the initial URL was; following them automatically lets a
                # vetted public URL answer 302 → localhost (see ``next_hop``).
                # Each hop is also pinned to the address that passed the check,
                # so the name cannot resolve to something else at connect time,
                # and credentials are dropped when a hop changes origin.
                hop_url, hop_headers = url, headers
                for _ in range(MAX_REDIRECT_HOPS):
                    refusal, addresses = await vet_public_url(hop_url)
                    if refusal:
                        raise RedirectRefused(refusal)
                    dial_url, dial_headers, extensions = pin_to_address(
                        hop_url, addresses, hop_headers,
                    )
                    async with client.stream(
                        "GET",
                        dial_url,
                        headers=dial_headers,
                        timeout=httpx.Timeout(None, connect=20, read=60),
                        follow_redirects=False,
                        extensions=extensions,
                    ) as response:
                        target = await next_hop(response, hop_url)
                        if target is not None:
                            hop_headers = strip_cross_origin_credentials(
                                hop_headers, hop_url, target,
                            )
                            hop_url = target
                            continue
                        response.raise_for_status()
                        # Direct fetch sees raw origin bytes (no Jina text
                        # conversion), so a data-blob content-type means a
                        # dataset/archive download — tell the agent plainly
                        # instead of feeding mojibake to the SUMMARY_LLM.
                        blob_type = binary_content_type(
                            response.headers.get("content-type"),
                        )
                        if blob_type:
                            declared = response.headers.get("content-length", "?")
                            return {
                                "success": False,
                                "content": "",
                                "error": (
                                    f"binary content ({blob_type}, "
                                    f"{declared} bytes) — a data file, not a "
                                    "web page; do not fetch it as text"
                                ),
                            }
                        body, _ = await read_bounded(response)
                        break
                else:
                    return {
                        "success": False, "content": "",
                        "error": "too many redirects",
                    }
            content = decode_body(response, body)
            if not content:
                return {"success": False, "content": "", "error": "Empty response"}
            return {"success": True, "content": content[:max_chars], "error": ""}
        except RedirectRefused as e:
            # A refused hop is a policy decision, not a transport fault: do not
            # burn the remaining retries re-requesting the same chain.
            return {"success": False, "content": "", "error": str(e)}
        except (httpx.ConnectTimeout, httpx.ConnectError, httpx.ReadTimeout) as e:
            if attempt < len(retry_delays):
                await asyncio.sleep(delay)
                continue
            return {"success": False, "content": "", "error": str(e)}
        except httpx.HTTPStatusError as e:
            sc = e.response.status_code
            if (sc >= 500 or sc in [408, 409, 425, 429]) and attempt < len(
                retry_delays,
            ):
                await asyncio.sleep(delay)
                continue
            return {"success": False, "content": "", "error": str(e)}
        except Exception as e:
            return {"success": False, "content": "", "error": str(e)}

    return {"success": False, "content": "", "error": "All retries exhausted"}


# ── LLM extraction ───────────────────────────────────────────────────

# Copied from the reference implementation. The wording matters: it shapes
# the LLM's extraction style and ultimately the content the agent sees.
_EXTRACT_INFO_PROMPT = """You are given a piece of content and the requirement of information to extract. Your task is to extract the information specifically requested. Be precise and focus exclusively on the requested information.

INFORMATION TO EXTRACT:
{}

INSTRUCTIONS:
1. Extract the information relevant to the focus above.
2. If the exact information is not found, extract the most closely related details.
3. Be specific and include exact details when available.
4. Clearly organize the extracted information for easy understanding.
5. Do not include general summaries or unrelated content.

CONTENT TO ANALYZE:
{}

EXTRACTED INFORMATION:"""


def _truncate_fallback(content: str) -> str:
    """Raw-content path used when no summary LLM is reachable."""
    if len(content) > FALLBACK_TRUNCATE:
        return content[:FALLBACK_TRUNCATE] + "\n\n[Content truncated...]"
    return content


async def _extract_info_with_llm(
    content: str,
    info_to_extract: str,
    truncate_last_num_chars: int = -1,
) -> dict[str, Any]:
    """Call the cheap SUMMARY_LLM to focus-extract from the scraped page.

    Resolution + fallback (2026-06-03): candidates come from
    ``frontier_agent.infra.summary_llm.summary_llm_candidates()`` — profile
    ``summary_llm:`` block (contextvar override, installed by the agent_team
    ``main_agent_node``) first, then its ``fallback:`` sub-block, then
    env ``SUMMARY_LLM_*`` / ``SUMMARY_LLM_FALLBACK_*``. Each candidate
    keeps the reference retry policy; a candidate that
    exhausts retries (or returns empty content — runaway reasoning)
    falls through to the next.
    """
    if not content or not content.strip():
        return {"success": False, "extracted_info": "", "error": "Empty content"}

    # No focus — nothing to extract. Hand back the raw page, matching
    # ``web_fetch``, instead of prompting the LLM with an empty focus.
    if not info_to_extract or not info_to_extract.strip():
        return {
            "success": True,
            "extracted_info": _truncate_fallback(content),
            "error": "",
        }

    candidates = summary_llm_candidates()
    if not candidates:
        # Legacy quirk preserved: BASE_URL set without MODEL_NAME used to
        # run with model="default" — keep that working.
        legacy_base = _summary_llm_base_url()
        if legacy_base:
            candidates = [{
                "endpoint": legacy_base,
                "model": _summary_llm_model_name() or "default",
                "api_key": _summary_llm_api_key() or "",
                "provider": "summary_llm",
            }]
        else:
            # No summary LLM anywhere — not even the primary-model last
            # resort ``summary_llm_candidates()`` appends. Degrade to
            # truncated raw content like ``web_fetch`` does instead of
            # failing every fetch: a page the agent can read beats an
            # error string.
            logger.warning(
                "Summary LLM not configured — returning truncated raw content",
            )
            return {
                "success": True,
                "extracted_info": _truncate_fallback(content),
                "error": "",
            }

    text = content
    if truncate_last_num_chars > 0:
        text = content[:-truncate_last_num_chars] + "[...truncated]"

    last: dict[str, Any] = {
        "success": False, "extracted_info": "", "error": "No response",
    }
    for cand in candidates:
        last = await _extract_with_candidate(cand, content, text, info_to_extract)
        if last["success"] and last["extracted_info"]:
            return last
    return last


async def _extract_with_candidate(
    cand: dict[str, str],
    content: str,
    text: str,
    info_to_extract: str,
) -> dict[str, Any]:
    """One candidate's extraction attempt (reference retry policy)."""
    endpoint = cand["endpoint"]
    model = cand["model"]
    prompt = _EXTRACT_INFO_PROMPT.format(info_to_extract, text)

    payload: dict[str, Any] = {
        "model": model,
        "max_tokens": 8192,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 1.0,
    }
    # GPT-5/4-style models reject ``max_tokens`` and need the new key.
    if "gpt" in model:
        payload["max_completion_tokens"] = payload.pop("max_tokens")
        if "gpt-5" in model.lower() or "gpt5" in model.lower():
            payload["service_tier"] = "flex"
            payload["reasoning_effort"] = "minimal"
    # Self-hosted reasoning models behind SGLang: extraction is an
    # auxiliary call — disable thinking so the reasoning prefix doesn't
    # eat the whole budget and return content=None. These keys match the
    # *model name*, so they are a contract with the serving stack — do
    # not rename them cosmetically or the branch stops firing.
    if any(k in model.lower() for k in ("qwen", "apodex", "sglang", "397b")):
        payload["chat_template_kwargs"] = {"enable_thinking": False}

    headers: dict[str, str] = {"Content-Type": "application/json"}
    api_key = cand.get("api_key") or ""
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    retry_delays = [1, 2, 4, 8]
    response: httpx.Response | None = None

    for attempt, delay in enumerate(retry_delays, 1):
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    endpoint,
                    headers=headers,
                    json=payload,
                    timeout=httpx.Timeout(None, connect=30, read=300),
                )

            # Context-overflow recovery: chop a chunk off the tail and
            # retry. Each attempt cuts a larger slice (40K, 80K, 120K…).
            if response and (
                "exceeds the model's maximum context length" in response.text
                or "longer than the model's context length" in response.text
            ):
                payload["messages"][0]["content"] = _EXTRACT_INFO_PROMPT.format(
                    info_to_extract,
                    content[: -(40960 * attempt)] + "[...truncated]",
                )
                continue

            response.raise_for_status()
            break
        except httpx.HTTPError as e:
            # GPT-5 sometimes rejects ``service_tier`` — drop and retry.
            if (
                "gpt-5" in model.lower() or "gpt5" in model.lower()
            ) and "service_tier" in payload:
                payload.pop("service_tier", None)
            # Retry every transient HTTP/network error within the budget.
            # The SUMMARY_LLM proxy returns 401 under bursty parallel load
            # even with a valid key — a follow-up request almost always
            # succeeds.
            if attempt < len(retry_delays):
                await asyncio.sleep(delay)
                continue
            return {"success": False, "extracted_info": "", "error": str(e)}
        except Exception as e:
            return {"success": False, "extracted_info": "", "error": str(e)}

    if response is None:
        return {"success": False, "extracted_info": "", "error": "No response"}

    try:
        data = response.json()
    except json.JSONDecodeError as e:
        return {
            "success": False,
            "extracted_info": "",
            "error": f"JSON parse error: {e}",
        }

    if data.get("choices"):
        try:
            extracted = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as e:
            return {"success": False, "extracted_info": "", "error": str(e)}
        # This raw-httpx LLM call bypasses the middleware
        # chain entirely — without this forward its tokens appear
        # nowhere (not even per-run). Lands in the top-level
        # ``usage_summary.llm["{model}@summary_llm"]`` slot.
        usage = data.get("usage") or {}
        if usage:
            record_llm_usage(
                model=model,
                provider=cand.get("provider") or "summary_llm",
                prompt_tokens=int(usage.get("prompt_tokens", 0) or 0),
                completion_tokens=int(usage.get("completion_tokens", 0) or 0),
                cache_read_tokens=int(
                    (usage.get("prompt_tokens_details") or {}).get(
                        "cached_tokens", 0,
                    ) or 0,
                ),
            )
        return {"success": True, "extracted_info": extracted, "error": ""}

    return {
        "success": False,
        "extracted_info": "",
        "error": f"Unexpected response: {data}",
    }


# ── Single-URL fetch + extract ────────────────────────────────────────


async def _fetch_single(
    url: str,
    info_to_extract: str,
    custom_headers: dict[str, str] | None = None,
) -> str:
    """Scrape one URL (Jina → fallback to direct) then run LLM extraction.

    The scrape is served through ``scrape_result_cache`` (single-flight,
    cross-run) so sibling agents fetching the same URL share one Jina
    round-trip. Only the SCRAPE is cached — the SUMMARY_LLM extraction below
    still runs per call so each run keeps its own ``info_to_extract`` focus.
    """
    # Vet the target before ANYTHING leaves the process — before the scrape
    # cache and before Jina, which would otherwise be handed an internal URL.
    # This implementation is the one the shipped react profile selects
    # (``web_fetch_impl: aligned``), so the guard has to live here too and not
    # only in ``plugins/tools/web_fetch.py``.
    non_public = await non_public_url_error(url)
    if non_public:
        return (
            f"Blocked: {non_public}. Only public http(s) endpoints may be "
            "fetched. Use a public source."
        )

    if any(pat in url for pat in _BANNED_URL_PATTERNS):
        return "Blocked: scraping Hugging Face datasets/spaces is not allowed."

    blocked_ext = blocked_download_url(url)
    if blocked_ext:
        return (
            f"Blocked: this URL is a dataset/archive download ({blocked_ext}), "
            "not a web page. Do not download data files — read the dataset's "
            "documentation/landing page instead, or use aggregate API queries."
        )

    # The initial body and browser retry each flow through several decisions
    # (retry, selection, cache admission, and the final warning). Reuse each
    # verdict within this fetch instead of reparsing the same HTML every time.
    render_verdicts: list[tuple[str, str | None]] = []

    def _render_kind(content: str) -> str | None:
        for seen, verdict in render_verdicts:
            if content is seen or content == seen:
                return verdict
        verdict = _unrendered_kind(content)
        render_verdicts.append((content, verdict))
        return verdict

    async def _scrape() -> str:
        scrape = await _scrape_url_with_jina(url, custom_headers)
        if scrape["success"]:
            kind = _render_kind(scrape["content"])
            if kind is not None:
                # HTTP 200 with a navigation-only body: the page had not
                # hydrated when the reader captured it. Escalate ONCE — browser
                # engine, longer render budget, Jina's cache bypassed — before
                # believing the emptiness.
                logger.info(
                    "Jina returned an un-rendered page for %s (%s, %d chars) "
                    "— retrying with the browser engine",
                    url, kind, len(scrape["content"]),
                )
                retry = await _scrape_url_with_jina(
                    url,
                    custom_headers,
                    engine="browser",
                    no_cache=True,
                    render_timeout_s=_JINA_RETRY_TIMEOUT_S,
                )
                if retry["success"] and _render_kind(retry["content"]) is None:
                    return retry["content"]
                # Still nothing. Keep whichever attempt carried more body — a
                # legitimately tiny page reads as "empty" too, and turning that
                # into an error would lose real content.
                if retry["success"] and len(
                    _reader_body(retry["content"]),
                ) > len(_reader_body(scrape["content"])):
                    scrape = retry
                if _render_kind(scrape["content"]) == "shell":
                    raise ScrapeUnavailable(
                        "the page is a JavaScript app shell — it returned no "
                        "content even with browser rendering and no cache. "
                        "Fetching it again the same way will not help: look for "
                        "the same material at another source (the site's own "
                        "API/JSON endpoint, a PDF or an archive copy), or search "
                        "for the page title instead"
                    )
            return scrape["content"]

        logger.warning(
            "Jina failed for %s: %s, trying direct", url, scrape["error"],
        )
        # Plain httpx executes no JavaScript, so for a shell page this fallback
        # can only ever confirm the emptiness — say so instead of handing the
        # extractor a navigation-only DOM to describe.
        scrape = await _scrape_url_with_python(url, custom_headers)
        if not scrape["success"]:
            raise ScrapeUnavailable(scrape["error"])
        if _render_kind(scrape["content"]) == "shell":
            raise ScrapeUnavailable(
                "the reader was unavailable and the raw HTML is a JavaScript "
                "app shell with no content in it (a direct fetch runs no "
                "JavaScript). Try another source for the same material rather "
                "than re-fetching this URL"
            )
        return scrape["content"]

    try:
        # Custom headers can change what the origin returns, so a header-bearing
        # request bypasses the URL-keyed shared cache to avoid serving content
        # fetched under different headers.
        if custom_headers:
            content = await _scrape()
        else:
            content = await scrape_result_cache.get_or_scrape(
                url,
                _scrape,
                # Return a low-confidence short page to this caller, but do not
                # publish it as the process-wide answer. A later fetch must be
                # free to win the render race.
                should_cache=lambda scraped: _render_kind(scraped) is None,
            )
    except ScrapeUnavailable as exc:
        return f"[ERROR]: Scraping failed: {exc}"

    result = await _extract_info_with_llm(content, info_to_extract)
    if not result["success"]:
        return f"[ERROR]: Extraction failed: {result['error']}"
    extracted = result["extracted_info"]
    if _render_kind(content) == "empty":
        return (
            f"[POSSIBLY NOT RENDERED] The page at {url} remained suspiciously "
            "short after the browser-render retry. This result was deliberately "
            "not cached. Use it if it answers the question; otherwise switch to "
            "another source instead of repeatedly fetching this URL.\n\n"
            f"{extracted}"
        )
    return extracted


# ── Tool ──────────────────────────────────────────────────────────────


@tool(name="web_fetch")
async def web_fetch_aligned(
    url: str | list[str],
    info_to_extract: str | list[str] = "",
    custom_headers: dict[str, str] | None = None,
) -> str:
    """Fetch content from a URL and extract specific types of information.

    Args:
        url: The URL to fetch, or a list of URLs to fetch in parallel
        info_to_extract: The specific types of information to extract (usually a question), or a list of extraction prompts (one per URL). Omit to get the raw page back
        custom_headers (Dict[str, str]): Additional headers to include in the request

    Returns:
        Extracted information as plain text. For multiple URLs, results are numbered
    """
    # The reference tool accepts JSON-encoded lists for both fields — mirror that.
    url = _ensure_list(url)
    info_to_extract = _ensure_list(info_to_extract)

    urls = url if isinstance(url, list) else [url]
    urls = [u for u in urls if u and u.strip()]
    if not urls:
        return "[ERROR]: url is required and cannot be empty."

    # ``info_to_extract`` broadcast rules, matching the reference tool:
    #   - len matches urls       → pair up 1:1
    #   - exactly one entry      → broadcast to every URL
    #   - mismatched (>=2 != N)  → join into one prompt and broadcast
    #   - plain string           → broadcast
    if isinstance(info_to_extract, list):
        if len(info_to_extract) == len(urls):
            infos = info_to_extract
        elif len(info_to_extract) == 1:
            infos = info_to_extract * len(urls)
        else:
            infos = [" ".join(info_to_extract)] * len(urls)
    else:
        infos = [info_to_extract] * len(urls)

    # Dedup identical (url, info) pairs so a multi-URL call doesn't burn
    # SUMMARY_LLM tokens on duplicates.
    seen: set[tuple[str, str]] = set()
    deduped_urls: list[str] = []
    deduped_infos: list[str] = []
    for u, info in zip(urls, infos, strict=False):
        key = (u, info)
        if key in seen:
            continue
        seen.add(key)
        deduped_urls.append(u)
        deduped_infos.append(info)
    urls = deduped_urls
    infos = deduped_infos

    try:
        results = await asyncio.gather(
            *[
                _fetch_single(u, info, custom_headers)
                for u, info in zip(urls, infos, strict=False)
            ],
        )

        # The reference tool formats single and multi-URL identically:
        # ``[N] URL: <u>\n    Info: <text>``.
        lines: list[str] = []
        for i, (u, text) in enumerate(zip(urls, results, strict=False), 1):
            lines.append(f"[{i}] URL: {u}")
            lines.append(f"    Info: {text}")
        return "\n".join(lines)

    except Exception as e:
        return f"[ERROR]: Unexpected error: {e!s}"


__all__ = ["web_fetch_aligned"]
