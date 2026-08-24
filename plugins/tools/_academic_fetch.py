"""Academic content fetchers with smart URL routing."""
from __future__ import annotations

import logging
import re
from typing import Literal
from urllib.parse import urlparse

import httpx

from frontier_agent.infra.config import get_config

logger = logging.getLogger(__name__)

Route = Literal["pmc", "pubmed", "biorxiv", "paywall", "jina"]

# Paywall / anti-bot domains that need Unpaywall detour.
PAYWALL_DOMAINS: frozenset[str] = frozenset({
    "www.sciencedirect.com", "linkinghub.elsevier.com",
    "onlinelibrary.wiley.com", "chemistry-europe.onlinelibrary.wiley.com",
    "advanced.onlinelibrary.wiley.com", "analyticalsciencejournals.onlinelibrary.wiley.com",
    "nph.onlinelibrary.wiley.com", "faseb.onlinelibrary.wiley.com",
    "link.aps.org", "journals.aps.org",
    "www.cell.com",
    "academic.oup.com",
    "www.tandfonline.com",
    "pubs.rsc.org", "www.rsc.org",
    "www.science.org",
    "www.jneurosci.org",
    "ashpublications.org",
    "pubs.aip.org",
    "journals.sagepub.com",
    "pubs.acs.org",
})

_GARBAGE_SIGNALS: tuple[str, ...] = (
    "security verification", "captcha", "cloudflare",
    "access denied", "please verify", "robot",
    "enable javascript", "browser check",
    "cookies are required", "please enable cookies",
    "sign in to access", "institutional login",
    "subscribe to read", "purchase this article",
)

# Unpaywall requires a contact address and Crossref asks for one in the
# User-Agent; both throttle or block anonymous traffic. There is deliberately
# no default: a shared fallback address would collect every deployment's
# traffic, and the polite-pool courtesy only works if the address is yours.
_CONTACT_UA_TEMPLATE = "FrontierAgent/1.0 (mailto:{email})"
_ANON_CROSSREF_UA = "FrontierAgent/1.0"


# ── URL extraction helpers ────────────────────────────────────────────────

def extract_pmcid(url: str) -> str:
    """Extract PMC id from URL. e.g. ``pmc.ncbi.nlm.nih.gov/articles/PMC1234567/``."""
    match = re.search(r"(PMC\d+)", url)
    return match.group(1) if match else ""


def extract_doi(url: str) -> str:
    """Try to extract DOI from URL. Returns empty string if not found."""
    match = re.search(r"doi\.org/(10\.\d{4,}/[^\s]+)", url)
    if match:
        return match.group(1).rstrip("/")
    match = re.search(r"/(10\.\d{4,}/[^\s?#]+)", url)
    if match:
        return match.group(1).rstrip("/")
    return ""


def route_url(url: str) -> Route:
    """Classify URL for backend selection. Pure — no I/O."""
    domain = urlparse(url).netloc
    if "pmc.ncbi.nlm.nih.gov" in domain:
        return "pmc"
    if "pubmed.ncbi.nlm.nih.gov" in domain:
        return "pubmed"
    if "biorxiv.org" in domain or "medrxiv.org" in domain:
        return "biorxiv"
    if domain in PAYWALL_DOMAINS:
        return "paywall"
    return "jina"


def is_garbage_content(text: str) -> bool:
    """Detect anti-bot pages, login walls, CAPTCHAs, empty pages."""
    if not text:
        return True
    low = text[:2000].lower()
    return any(sig in low for sig in _GARBAGE_SIGNALS)


def biorxiv_to_pdf(url: str) -> str:
    """Convert bioRxiv/medRxiv URL to full PDF URL. Returns empty if not applicable."""
    if not ("biorxiv.org" in url or "medrxiv.org" in url):
        return ""
    clean = url.split("?")[0].split("#")[0].rstrip("/")
    if clean.endswith(".pdf"):
        return url
    if "/content/" in clean:
        return clean + ".full.pdf"
    return ""


# ── Async fetch helpers ───────────────────────────────────────────────────

def _unpaywall_email() -> str:
    """Contact address for Unpaywall, or "" when unconfigured.

    Callers treat "" as "skip Unpaywall": querying it without an address is a
    terms violation, and a placeholder address just gets the deployment
    rate-limited.
    """
    return get_config().unpaywall_email or ""


def _crossref_ua() -> str:
    """Crossref User-Agent, with the contact address when one is configured."""
    email = get_config().unpaywall_email
    return _CONTACT_UA_TEMPLATE.format(email=email) if email else _ANON_CROSSREF_UA


def _ncbi_params(base: dict | None = None) -> dict:
    """Attach ``api_key`` if configured — raises PubMed/E-utils rate limit."""
    params = dict(base or {})
    config = get_config()
    if config.ncbi_api_key:
        params["api_key"] = config.ncbi_api_key
    return params


async def fetch_pmc_fulltext(pmcid: str, *, timeout: int = 30) -> str:
    """Fetch full text from PMC Open Access API (BioC JSON format)."""
    if not pmcid:
        return ""
    url = (
        "https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful/pmcoa.cgi/BioC_json/"
        f"{pmcid}/unicode"
    )
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.get(url, params=_ncbi_params())
    except Exception as exc:
        logger.warning("[PMC API] Failed for %s: %s", pmcid, exc)
        return ""

    if resp.status_code != 200 or len(resp.text) < 500:
        return ""
    # PMC returns "[Error]..." or HTML for non-OA content.
    preview = resp.text[:100].lower()
    if resp.text.strip().startswith("[Error]") or "<html" in preview:
        logger.info("[PMC API] %s not in OA subset", pmcid)
        return ""
    try:
        data = resp.json()
    except Exception as exc:
        logger.warning("[PMC API] JSON parse failed for %s: %s", pmcid, exc)
        return ""
    items = data if isinstance(data, list) else [data]
    parts: list[str] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        for doc in item.get("documents", []):
            for passage in doc.get("passages", []):
                text = passage.get("text", "")
                if text:
                    parts.append(text)
    fulltext = "\n\n".join(parts)
    if fulltext:
        logger.info("[PMC API] Fetched %d chars for %s", len(fulltext), pmcid)
    return fulltext


async def pubmed_to_pmc(url: str, *, timeout: int = 15) -> str:
    """Convert PubMed URL → PMC id via E-utilities elink. Empty on failure."""
    match = re.search(r"pubmed\.ncbi\.nlm\.nih\.gov/(\d+)", url)
    if not match:
        return ""
    pmid = match.group(1)
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.get(
                "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/elink.fcgi",
                params=_ncbi_params({
                    "dbfrom": "pubmed",
                    "db": "pmc",
                    "id": pmid,
                    "retmode": "json",
                }),
            )
    except Exception as exc:
        logger.warning("[PubMed->PMC] Failed for %s: %s", pmid, exc)
        return ""

    if resp.status_code != 200:
        return ""
    try:
        data = resp.json()
    except Exception:
        return ""
    for ls in data.get("linksets", []):
        for ldb in ls.get("linksetdbs", []):
            if ldb.get("dbto") == "pmc":
                links = ldb.get("links", [])
                if links:
                    return f"PMC{links[0]}"
    return ""


async def fetch_unpaywall_oa_url(doi: str, *, timeout: int = 15) -> str:
    """Query Unpaywall API for an OA PDF URL. Empty string on any failure."""
    if not doi:
        return ""
    if not (email := _unpaywall_email()):
        # Unpaywall rejects requests without a contact address. Degrade to
        # "no OA copy found" rather than spending a request that cannot succeed.
        logger.debug("[Unpaywall] Skipped for DOI %s: UNPAYWALL_EMAIL unset", doi)
        return ""
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.get(
                f"https://api.unpaywall.org/v2/{doi}",
                params={"email": email},
            )
    except Exception as exc:
        logger.warning("[Unpaywall] Failed for DOI %s: %s", doi, exc)
        return ""

    if resp.status_code != 200:
        return ""
    try:
        data = resp.json()
    except Exception:
        return ""

    best = data.get("best_oa_location") or {}
    pdf_url = best.get("url_for_pdf") or best.get("url") or ""
    if pdf_url:
        logger.info("[Unpaywall] Found OA for DOI %s: %s", doi, pdf_url)
        return pdf_url
    for loc in data.get("oa_locations", []) or []:
        pdf_url = loc.get("url_for_pdf") or loc.get("url") or ""
        if pdf_url:
            logger.info("[Unpaywall] Found OA for DOI %s: %s", doi, pdf_url)
            return pdf_url
    return ""


async def crossref_doi_lookup(
    url: str, title_hint: str = "", *, timeout: int = 10
) -> str:
    """Try to find DOI via CrossRef when URL doesn't expose one."""
    query = title_hint if title_hint else url
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.get(
                "https://api.crossref.org/works",
                params={"query": query, "rows": 1},
                headers={"User-Agent": _crossref_ua()},
            )
    except Exception as exc:
        logger.warning("[CrossRef] Lookup failed: %s", exc)
        return ""

    if resp.status_code != 200:
        return ""
    try:
        items = resp.json().get("message", {}).get("items", [])
    except Exception:
        return ""
    if items:
        doi = items[0].get("DOI", "")
        if doi:
            logger.info("[CrossRef] Found DOI %s for query: %.60s", doi, query)
            return doi
    return ""


async def resolve_doi(url: str, *, title_hint: str = "") -> str:
    """Convenience: extract DOI from URL, fall back to CrossRef lookup."""
    doi = extract_doi(url)
    if doi:
        return doi
    return await crossref_doi_lookup(url, title_hint)
