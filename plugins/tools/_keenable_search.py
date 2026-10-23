"""Keyless Keenable search transport for the native web-search tools."""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

import httpx

from frontier_agent.infra.usage_meter import record_api_request

logger = logging.getLogger(__name__)

_PUBLIC_URL = "https://api.keenable.ai/v1/search/public"
_KEYED_URL = "https://api.keenable.ai/v1/search"
_APP_TITLE = "FrontierAgent"
_MIN_RESULTS = 10
_MAX_RESULTS = 50
_MAX_RETRIES = 3
# Keenable returns page text rather than a short search-engine snippet; cap it
# so ten results stay a compact tool observation.
_SNIPPET_CHARS = 1000


def _api_key() -> str:
    """Optional key, read at call time because ``.env`` loads after import."""
    return (os.getenv("KEENABLE_API_KEY") or "").strip()


def _headers(api_key: str) -> dict[str, str]:
    headers = {"Content-Type": "application/json", "X-Keenable-Title": _APP_TITLE}
    if api_key:
        headers["X-API-Key"] = api_key
    return headers


def _snippet(value: Any) -> str:
    """Collapse page text into one line and cap its length."""
    text = " ".join(str(value or "").split())
    if len(text) > _SNIPPET_CHARS:
        text = text[:_SNIPPET_CHARS].rstrip() + "..."
    return text


def _normalise_result(value: Any) -> dict[str, Any]:
    """Convert a Keenable response into the native Serper-shaped result."""
    rows = value.get("results") if isinstance(value, dict) else None
    organic = [
        {
            "title": str(item.get("title") or ""),
            "link": str(item["url"]),
            "snippet": _snippet(item.get("snippet") or item.get("description")),
        }
        for item in rows or []
        if isinstance(item, dict) and item.get("url")
    ]
    return {"organic": organic} if organic else {}


async def _search(
    client: httpx.AsyncClient,
    url: str,
    headers: dict[str, str],
    query: str,
    max_results: int,
) -> dict[str, Any]:
    """One Keenable query, retrying rate limits and server errors."""
    payload = {"query": query, "max_results": max_results}
    for attempt in range(_MAX_RETRIES):
        response = await client.post(url, json=payload, headers=headers)
        retryable = response.status_code == 429 or response.status_code >= 500
        if retryable and attempt < _MAX_RETRIES - 1:
            record_api_request("keenable", requests=0, retries=1)
            await asyncio.sleep(2 ** attempt)
            continue
        response.raise_for_status()
        record_api_request("keenable")
        return _normalise_result(response.json())
    return {}


async def keenable_search_batch(
    queries: list[str],
    *,
    num_results: int,
    gl: str = "us",
    hl: str = "en",
    tbs: str = "",
) -> list[dict[str, Any]] | str:
    """Search Keenable for each query concurrently.

    Works without a key; ``KEENABLE_API_KEY`` is optional and only raises the
    rate limits. Locale and time controls are rejected when requested because
    this backend does not expose them. Each query asks for at least ten and at
    most fifty results, and all candidates are returned so callers can limit
    results after display filtering and URL deduplication. The selected
    provider never falls back to Serper on a transport or rate-limit error.
    """
    if gl != "us" or hl != "en" or tbs:
        return (
            "Keenable search does not support custom region, language, or "
            "time filters. Select Serper to use those search options."
        )
    if not queries:
        return []

    api_key = _api_key()
    url = _KEYED_URL if api_key else _PUBLIC_URL
    headers = _headers(api_key)
    max_results = max(_MIN_RESULTS, min(int(num_results), _MAX_RESULTS))
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=10.0)) as client:
            tasks = [
                asyncio.create_task(_search(client, url, headers, query, max_results))
                for query in queries
            ]
            try:
                return list(await asyncio.gather(*tasks))
            except BaseException:
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                raise
    except httpx.HTTPStatusError as error:
        status = error.response.status_code
        logger.warning("Keenable search request failed with HTTP %d", status)
        record_api_request("keenable", requests=0, errors=1)
        return f"Keenable search returned HTTP {status}."
    except (httpx.HTTPError, ValueError, TypeError, KeyError):
        logger.warning("Keenable search request failed")
        record_api_request("keenable", requests=0, errors=1)
        return "Keenable search failed. Check network access and try again."


__all__ = ["keenable_search_batch"]
