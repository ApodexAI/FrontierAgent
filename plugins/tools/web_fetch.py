"""Web scraping tool with academic URL routing."""

from __future__ import annotations

import asyncio
import logging

import httpx

from frontier_agent.core.tool import tool
from frontier_agent.infra.config import FrontierAgentConfig, get_config
from frontier_agent.infra.summary_llm import summarize as _summary_llm_summarize
from frontier_agent.infra.usage_meter import record_api_request
from plugins.tools._academic_fetch import (
    biorxiv_to_pdf,
    extract_pmcid,
    fetch_pmc_fulltext,
    fetch_unpaywall_oa_url,
    is_garbage_content,
    pubmed_to_pmc,
    resolve_doi,
    route_url,
)
from plugins.tools._bounded_fetch import (
    MAX_REDIRECT_HOPS,
    RedirectRefused,
    binary_content_type,
    blocked_download_url,
    next_hop,
    non_public_url_error,
    pin_to_address,
    read_bounded,
    strip_cross_origin_credentials,
    vet_public_url,
)
from plugins.tools._render_check import unrendered_kind
from plugins.tools._scrape_cache import (
    ScrapeUnavailable,
    format_skip_message,
    scrape_result_cache,
)
from plugins.tools._scrape_cache import cache as _scrape_cache

logger = logging.getLogger(__name__)

_MAX_RETRIES = 3
_SHORT_CONTENT_THRESHOLD = 500       # re-try via Unpaywall when content is this short
_PAYWALL_SHORT_THRESHOLD = 500       # tighter check used for known paywall domains

# Below this many chars the raw page is returned verbatim instead of being
# routed through the summary LLM, even when ``info_to_extract`` is set. The
# summary LLM exists purely to keep 50KB+ pages from blowing the agent's
# context; a short page (a policy paragraph, a financial snippet, a small
# table) costs nothing to carry whole, and paraphrasing it through a
# temperature=1.0 extractor risks drift.
#
# Threshold picked from an offline A/B on real pages: long policy/legal
# text (13-24K chars) summarised faithfully, but a short, number-dense
# page fabricated a derived percentage and back-filled figures that were
# never on the page. Drift tracks number density, not length, so the line
# sits high enough to route short number-dense pages to raw (a few K
# tokens whole — no real context cost) while long-form still gets
# compressed. Tune upward if headroom allows.
_SUMMARY_MIN_CHARS = 12_000


@tool
async def web_fetch(
    url: str | list[str],
    info_to_extract: str | list[str] = "",
) -> str:
    """Scrape and extract content from one or more web pages.

    Automatic backend selection based on URL domain: PMC / PubMed / bioRxiv /
    medRxiv URLs go to the corresponding OA API; known paywall domains are
    routed via Unpaywall; everything else goes through Jina Reader. Retry,
    negative cache, and arXiv PDF→HTML redirect are applied automatically.

    A non-empty ``info_to_extract`` routes the raw content through a cheap
    summary LLM that extracts only the information requested. With it
    omitted the raw extracted text is returned (subject to overflow trim).

    Args:
        url: A URL string, or a list of URLs for parallel fetch.
        info_to_extract: Optional focus for the summary LLM. A single
            string applies to every URL; a list pairs with ``url`` 1:1.

    Returns:
        For a single URL, the extracted content directly. For a list, a
        numbered block per URL: ``[i] URL: …\\n    Info: …``.
    """
    urls, focuses = _normalise_inputs(url, info_to_extract)
    if not urls:
        return "Error: URL is required."

    if len(urls) == 1:
        return await _fetch_one(urls[0], focuses[0])

    results = await asyncio.gather(
        *(_fetch_one(u, f) for u, f in zip(urls, focuses, strict=False))
    )
    return "\n\n".join(
        f"[{i}] URL: {u}\n    Info: {r}"
        for i, (u, r) in enumerate(zip(urls, results, strict=False), 1)
    )


def _normalise_inputs(
    url: str | list[str], info_to_extract: str | list[str],
) -> tuple[list[str], list[str]]:
    """Coerce the LangChain payload into paired URL + focus lists."""
    from plugins.tools._coerce import coerce_json_list
    urls = coerce_json_list(url) if isinstance(url, str) else url
    if isinstance(urls, str):
        urls = [urls]
    elif not isinstance(urls, list):
        urls = []
    urls = [u.strip() for u in urls if isinstance(u, str) and u.strip()]

    focuses = (
        coerce_json_list(info_to_extract) if isinstance(info_to_extract, str)
        else info_to_extract
    )
    if isinstance(focuses, str):
        focuses = [focuses] * len(urls)
    elif isinstance(focuses, list):
        focuses = [str(f) for f in focuses]
        if len(focuses) < len(urls):
            focuses = focuses + [""] * (len(urls) - len(focuses))
    else:
        focuses = [""] * len(urls)
    return urls, focuses


async def _fetch_one(url: str, info_to_extract: str) -> str:
    """Run the full fetch pipeline for a single URL."""
    if not url or not url.strip():
        return "Error: URL is required."

    url = url.strip()
    if not url.startswith(("http://", "https://")):
        url = f"https://{url}"

    # Vet the target before ANYTHING leaves the process — including the scrape
    # provider's request, which would otherwise be handed an internal URL.
    non_public = await non_public_url_error(url)
    if non_public:
        return (
            f"[BLOCKED] {non_public}. Only public http(s) endpoints may be "
            "fetched. Use a public source."
        )

    # Operator hard-block (WEB_DOMAIN_BLACKLIST_EXTRA) — same list that
    # filters web_search results, enforced here so a direct URL from page
    # content cannot bypass it.
    from plugins.tools.web_search import is_domain_blocked
    if is_domain_blocked(url):
        return (
            f"[BLOCKED] The domain of {url} is on the operator blocklist "
            f"and must not be accessed. Use a different source."
        )

    blocked_ext = blocked_download_url(url)
    if blocked_ext:
        return (
            f"[BLOCKED] This URL is a dataset/archive download ({blocked_ext}), "
            "not a web page. Do not download data files — read the dataset's "
            "documentation/landing page instead, or use aggregate API queries."
        )

    # arXiv PDFs consistently fail extraction — redirect to HTML abstract page.
    if "arxiv.org/pdf/" in url:
        html_url = url.replace("/pdf/", "/abs/").split(".pdf")[0]
        logger.info("Redirecting arxiv PDF → HTML: %s", html_url)
        url = html_url
    elif "arxiv.org/pdf" in url:
        html_url = url.replace("/pdf", "/abs")
        logger.info("Redirecting arxiv PDF → HTML: %s", html_url)
        url = html_url

    # Negative-cache check: skip URLs that recently returned 403/422/429.
    cached = _scrape_cache.check(url)
    if cached is not None:
        msg = format_skip_message(url, cached)
        logger.info("web_fetch skipped (cached %d): %s", cached.status, url[:60])
        return msg

    config = get_config()
    route = route_url(url)

    # A fresh scrape is classified in the scrape function, the cache predicate,
    # and the caller-facing warning path. Keep the verdicts for this one fetch
    # so HTML parsing happens once per distinct response body.
    render_verdicts: list[tuple[str, str | None]] = []

    def _render_kind(content: str) -> str | None:
        for seen, verdict in render_verdicts:
            if content is seen or content == seen:
                return verdict
        verdict = unrendered_kind(content)
        render_verdicts.append((content, verdict))
        return verdict

    # Single-flight cross-run cache: sibling agents fetching the same URL share
    # one round-trip. Only validated, non-garbage content is cached; the empty
    # / garbage branches raise so the failure is neither stored nor shared.
    async def _scrape() -> str:
        content = await _fetch_via_route(url, route, config)
        # Post-fetch quality check: Jina can return a CAPTCHA/login page as a
        # 200, and some un-listed paywalls slip past the route table. Try one
        # Unpaywall-driven retry (cheap — fails fast when no DOI resolves).
        content = await _maybe_recover_via_unpaywall(url, route, content, config)
        if not content:
            raise ScrapeUnavailable("empty")
        # Final garbage check — Jina can return a CAPTCHA/login page as success.
        if is_garbage_content(content):
            raise ScrapeUnavailable("garbage")
        # A pre-hydration DOM survived even the browser-engine escalation
        # above. Only the high-confidence ``shell`` verdict fails here: a
        # merely short body ("empty") is legitimate content often enough that
        # erroring on it would lose real pages.
        if _render_kind(content) == "shell":
            raise ScrapeUnavailable("unrendered")
        return content

    try:
        content = await scrape_result_cache.get_or_scrape(
            url,
            _scrape,
            # A low-confidence short page is still useful enough to return,
            # but it may be a pre-hydration race. Never let it become the
            # process-wide answer for every later fetch of this URL.
            should_cache=lambda scraped: _render_kind(scraped) is None,
        )
    except ScrapeUnavailable as exc:
        if str(exc) == "unrendered":
            return (
                f"[NOT RENDERED] The page at {url} is a JavaScript app that "
                f"served no content even with browser rendering. Fetching it "
                f"again the same way will not help: look for the same material "
                f"at another source (the site's own API/JSON endpoint, a PDF, "
                f"or an archive copy), or search for the page title instead."
            )
        if str(exc) == "garbage":
            return (
                f"[ACCESS BLOCKED] The page at {url} is behind a paywall or "
                f"anti-bot protection. Please try searching for an open-access "
                f"version (arxiv.org, PMC, institutional repositories)."
            )
        return f"Could not extract content from {url}"

    _scrape_cache.record_success(url)

    # LLM extraction if requested — uses dedicated SUMMARY_LLM_* config and
    # gracefully returns truncated raw content when unconfigured. Skipped for
    # short pages (< _SUMMARY_MIN_CHARS): they carry whole at no context cost,
    # and returning them verbatim avoids paraphrase drift on the exact numbers
    # / scope qualifiers the agent asked for. The focus is still served — the
    # agent reads ``info_to_extract`` straight from the raw page.
    if (
        info_to_extract
        and info_to_extract.strip()
        and len(content) >= _SUMMARY_MIN_CHARS
    ):
        output = await _summary_llm_summarize(content, info_to_extract)
    else:
        from plugins.tools._overflow import maybe_overflow
        output = maybe_overflow("web_fetch", content)

    if _render_kind(content) == "empty":
        return (
            f"[POSSIBLY NOT RENDERED] The page at {url} remained suspiciously "
            "short after the browser-render retry. This result was deliberately "
            "not cached. Use it if it answers the question; otherwise switch to "
            "another source instead of repeatedly fetching this URL.\n\n"
            f"{output}"
        )
    return output


# ── Routing ───────────────────────────────────────────────────────────────

async def _fetch_via_route(url: str, route: str, config: FrontierAgentConfig) -> str:
    """Dispatch to the domain-specific backend; always falls back to Jina."""
    if route == "pmc":
        return await _fetch_pmc(url, config)
    if route == "pubmed":
        return await _fetch_pubmed(url, config)
    if route == "biorxiv":
        return await _fetch_biorxiv(url, config)
    if route == "paywall":
        return await _fetch_paywall(url, config)

    # Generic route — Jina first, then trafilatura fallback.
    content = ""
    if config.jina_api_key:
        content = await _jina_scrape(url, config.jina_api_key, config.jina_base_url)
    if not content:
        content = await _direct_scrape(url)
    return content


async def _fetch_pmc(url: str, config: FrontierAgentConfig) -> str:
    pmcid = extract_pmcid(url)
    if pmcid:
        text = await fetch_pmc_fulltext(pmcid)
        if text:
            return text
    return await _jina_or_empty(url, config)


async def _fetch_pubmed(url: str, config: FrontierAgentConfig) -> str:
    pmcid = await pubmed_to_pmc(url)
    if pmcid:
        text = await fetch_pmc_fulltext(pmcid)
        if text:
            return text
    return await _jina_or_empty(url, config)


async def _fetch_biorxiv(url: str, config: FrontierAgentConfig) -> str:
    pdf_url = biorxiv_to_pdf(url)
    text = ""
    if pdf_url:
        logger.info("[bioRxiv] Auto PDF: %s", pdf_url)
        text = await _jina_or_empty(pdf_url, config)
    if not text or len(text) < _SHORT_CONTENT_THRESHOLD:
        fallback = await _jina_or_empty(url, config)
        if fallback and len(fallback) > len(text):
            text = fallback
    return text


async def _fetch_paywall(url: str, config: FrontierAgentConfig) -> str:
    doi = await resolve_doi(url)
    text = ""
    if doi:
        oa_url = await fetch_unpaywall_oa_url(doi)
        if oa_url:
            logger.info("[Paywall bypass] %s → OA PDF: %s", url[:60], oa_url)
            text = await _jina_or_empty(oa_url, config)
    if not text or len(text) < _PAYWALL_SHORT_THRESHOLD:
        fallback = await _jina_or_empty(url, config)
        if fallback and len(fallback) > len(text):
            text = fallback
    return text


async def _jina_or_empty(url: str, config: FrontierAgentConfig) -> str:
    """Run Jina with the existing retry/cache logic; empty string on failure."""
    if not config.jina_api_key:
        return ""
    try:
        return await _jina_scrape(url, config.jina_api_key, config.jina_base_url)
    except Exception as exc:
        logger.warning("Jina fetch failed for %s: %s", url[:60], exc)
        return ""


async def _maybe_recover_via_unpaywall(
    url: str, route: str, content: str, config: FrontierAgentConfig,
) -> str:
    """If the routed fetch returned garbage or was suspiciously short on a
    paywall domain, attempt one Unpaywall-driven retry.

    Skipped for ``pmc``/``pubmed`` — if the BioC API couldn't resolve the
    article there's no DOI detour worth trying that Jina wouldn't already hit.
    """
    if route in ("pmc", "pubmed"):
        return content

    garbage = is_garbage_content(content)
    short_on_paywall = (
        route == "paywall"
        and content
        and len(content) < _PAYWALL_SHORT_THRESHOLD
    )
    if not garbage and not short_on_paywall:
        return content

    reason = "garbage" if garbage else "short"
    logger.warning(
        "[Quality] %s content detected for %s — trying Unpaywall fallback",
        reason, url[:60],
    )
    doi = await resolve_doi(url)
    if not doi:
        return content
    oa_url = await fetch_unpaywall_oa_url(doi)
    if not oa_url:
        return content

    alt = await _jina_or_empty(oa_url, config)
    if alt and not is_garbage_content(alt) and len(alt) > len(content):
        logger.info("[Quality] Unpaywall fallback succeeded: %d chars", len(alt))
        return alt
    return content


def _extract_jina_body(raw: str) -> str:
    """Return the usable content from a Jina response body.

    Jina's POST endpoint returns JSON wrapped as ``{"code":200,"status":...,
    "data":{"title":...,"content":"markdown..."}}`` when ``Accept: application/json``
    is set. Some proxies may return raw markdown instead. Handle both shapes
    and also detect balance errors.
    """
    if not raw:
        return ""
    stripped = raw.lstrip()
    if not stripped.startswith("{"):
        return raw
    try:
        import json as _json
        data = _json.loads(raw)
    except (ValueError, TypeError):
        return raw

    if isinstance(data, dict):
        # Balance-error sentinel.
        if data.get("name") == "InsufficientBalanceError":
            logger.error("Jina API: Insufficient balance")
            return ""
        # Wrapped success payload.
        inner = data.get("data")
        if isinstance(inner, dict) and isinstance(inner.get("content"), str):
            return inner["content"]
    return raw


async def _jina_request(
    url: str,
    api_key: str,
    base_url: str,
    *,
    browser: bool,
) -> tuple[int, str]:
    """Single Jina call. Returns (status_code, body_text). Status -1 on transport error.

    ``browser=True`` switches the Jina extraction engine to ``browser`` (real
    page render), which recovers many origins that block the direct fetch
    engine with 403/422. Uses a longer ``X-Timeout`` so dynamic content has
    time to load.
    """
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
        "X-Timeout": "30" if browser else "20",
    }
    if browser:
        headers["X-Engine"] = "browser"
        # The escalation exists because the first attempt produced nothing
        # usable; Jina caches its own responses, so without this the retry can
        # be answered from that same result.
        headers["X-No-Cache"] = "true"

    try:
        async with httpx.AsyncClient(timeout=60) as client:
            async with client.stream(
                "POST",
                base_url,
                headers=headers,
                json={"url": url},
            ) as resp:
                # Bounded read — a Jina conversion of a huge document
                # must not buffer past the byte cap.
                body, _ = await read_bounded(resp)
            # One answered Jina request is billable; non-2xx
            # statuses still consumed a reader call upstream, so count
            # them as requests too and tag the error separately.
            record_api_request(
                "jina", errors=0 if resp.status_code < 400 else 1,
            )
            # Jina returns UTF-8 JSON but omits ``charset`` in Content-Type,
            # which makes httpx fall back to chardet sniffing — chardet
            # mis-classifies CJK-heavy bodies with mostly-ASCII headers as
            # Windows-1252, producing mojibake like ``千と千尋`` →
            # ``åã¨åå°``. Force UTF-8 to skip the sniff entirely.
            return resp.status_code, body.decode("utf-8", errors="replace")
    except httpx.TimeoutException:
        record_api_request("jina", requests=0, errors=1)
        return -1, "timeout"
    except Exception as e:
        logger.error("Jina transport error for %s: %s", url[:60], e)
        record_api_request("jina", requests=0, errors=1)
        return -1, str(e)


async def _jina_scrape(url: str, api_key: str, base_url: str) -> str:
    """Scrape via Jina Reader with escalation retry.

    Strategy:
    1. Default (fast) engine first.
    2. On 403/422, escalate once to ``X-Engine: browser`` — covers SPAs and
       origins that block the direct fetch engine. Only caches the URL as
       banned if the browser-engine attempt also fails.
    3. Also escalate on a 200 whose body came back un-rendered: an SPA that had
       not hydrated when the reader captured it answers 200 with a
       navigation-only DOM, so a status-only rule never noticed (see
       ``_render_check``). The retry additionally bypasses Jina's own response
       cache, which would otherwise hand back the same shell.
    4. Retry 429/5xx with exponential backoff (unchanged).
    """
    escalated = False

    for attempt in range(_MAX_RETRIES):
        status, body = await _jina_request(url, api_key, base_url, browser=escalated)

        if status == 200:
            content = _extract_jina_body(body)
            kind = unrendered_kind(content)
            if kind is not None and not escalated:
                logger.info(
                    "Jina returned an un-rendered page for %s (%s, %d chars) "
                    "— escalating to browser engine", url[:60], kind, len(content),
                )
                escalated = True
                continue
            return content

        if status == 403:
            if not escalated:
                logger.info("Jina 403 for %s — escalating to browser engine", url[:60])
                escalated = True
                continue
            _scrape_cache.record_failure(url, 403)
            logger.warning("Jina 403 for %s after browser retry — cached (1h)", url[:60])
            return ""

        if status == 422:
            if not escalated:
                logger.info("Jina 422 for %s — escalating to browser engine", url[:60])
                escalated = True
                continue
            _scrape_cache.record_failure(url, 422)
            logger.info("Jina 422 for %s after browser retry — recorded", url[:60])
            return ""

        if status == 429:
            wait = 2 ** attempt
            logger.warning("Jina 429 rate limited for %s, retrying in %ds", url[:60], wait)
            await asyncio.sleep(wait)
            continue

        if status >= 500:
            wait = 2 ** attempt
            logger.warning("Jina %d server error for %s, retrying in %ds", status, url[:60], wait)
            await asyncio.sleep(wait)
            continue

        if status == -1:
            # Transport error / timeout — treat like a soft retry.
            logger.warning("Jina transport failure for %s: %s (attempt %d)", url[:60], body[:60], attempt + 1)
            if attempt < _MAX_RETRIES - 1:
                await asyncio.sleep(1)
                continue
            return ""

        # Any other status — log and abort without caching.
        logger.error("Jina HTTP %d for %s (body: %s)", status, url[:60], body[:200])
        return ""

    # Retries exhausted on 429/5xx — record as rate-limited so we back off.
    _scrape_cache.record_failure(url, 429)
    logger.error("Jina scrape exhausted retries for %s — cached 429 (5min)", url[:60])
    return ""


async def _direct_scrape(url: str) -> str:
    """Direct scrape with httpx + trafilatura extraction."""
    try:
        import trafilatura
    except ImportError:
        return ""

    try:
        headers = {
            # Sent to every site this scrapes. Identify your deployment
            # here if you want operators to be able to contact you — the
            # default names the project, not any account.
            "User-Agent": (
                "Mozilla/5.0 (compatible; FrontierAgent/1.0; "
                "+https://github.com/ApodexAI/FrontierAgent)"
            ),
        }
        # Redirects are walked by hand so every hop is vetted: following them
        # automatically would let a vetted public URL hand us a 302 to
        # localhost or a metadata endpoint (see ``next_hop``).
        async with httpx.AsyncClient(timeout=60, follow_redirects=False) as client:
            hop_url, hop_headers = url, headers
            for _ in range(MAX_REDIRECT_HOPS):
                refusal, addresses = await vet_public_url(hop_url)
                if refusal:
                    raise RedirectRefused(refusal)
                dial_url, dial_headers, extensions = pin_to_address(
                    hop_url, addresses, hop_headers,
                )
                async with client.stream(
                    "GET", dial_url, headers=dial_headers, extensions=extensions,
                ) as resp:
                    target = await next_hop(resp, hop_url)
                    if target is not None:
                        hop_headers = strip_cross_origin_credentials(
                            hop_headers, hop_url, target,
                        )
                        hop_url = target
                        continue
                    resp.raise_for_status()
                    # A data blob (dataset zip, media) has no page text for
                    # trafilatura; skip the download instead of buffering it.
                    if binary_content_type(resp.headers.get("content-type")):
                        logger.warning(
                            "Direct scrape skipped binary content for %s", url[:60],
                        )
                        return ""
                    body, _ = await read_bounded(resp)
                    break
            else:
                logger.warning("Direct scrape exceeded redirect limit for %s", url[:60])
                return ""
    except RedirectRefused as exc:
        logger.warning("Direct scrape refused a redirect for %s: %s", url[:60], exc)
        return ""
    except httpx.TimeoutException:
        logger.warning("Direct scrape timeout for %s", url[:60])
        return ""
    except Exception as e:
        logger.warning("Direct scrape failed for %s: %s", url[:60], e)
        return ""

    # Pass raw bytes so trafilatura reads the HTML ``<meta charset>``
    # itself instead of trusting httpx's chardet sniff — pages that mix
    # heavy ASCII boilerplate with a small CJK payload get mis-classified
    # as Windows-1252 by chardet (UTF-8 → mojibake).
    return trafilatura.extract(body) or ""
