"""Opt-in Firecrawl research through Monid for the native web-search tools."""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any

import httpx

from frontier_agent.infra.usage_meter import record_api_request

logger = logging.getLogger(__name__)

_API_URL = "https://api.monid.ai"
_RUN_TIMEOUT = 300.0
_RESULT_SCHEMA = {
    "type": "object",
    "properties": {"organic": {"type": "array", "items": {
        "type": "object",
        "properties": {key: {"type": "string"} for key in ("title", "link", "snippet")},
        "required": ["title", "link", "snippet"],
    }}},
    "required": ["organic"],
}


async def _search(
    client: httpx.AsyncClient, headers: dict[str, str], query: str, num_results: int,
) -> dict[str, Any] | str:
    payload = {
        "provider": "firecrawl", "endpoint": "/agent",
        "input": {"queryParams": {}, "pathParams": {}, "body": {
            "prompt": f"Research this query and return up to {num_results} web results "
                      f"with source links, titles and snippets: {query}",
            "schema": _RESULT_SCHEMA,
        }},
    }
    response = await client.post(
        f"{_API_URL}/v1/run", headers=headers,
        content=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
    )
    response.raise_for_status()
    run = response.json()
    if response.status_code == 202:
        run_id = run["runId"]
        deadline = asyncio.get_running_loop().time() + _RUN_TIMEOUT
        while asyncio.get_running_loop().time() < deadline:
            await asyncio.sleep(2)
            response = await client.get(f"{_API_URL}/v1/runs/{run_id}", headers=headers)
            response.raise_for_status()
            run = response.json()
            if run.get("status") in ("COMPLETED", "FAILED", "TIMED_OUT", "STOPPED", "BLOCKED"):
                break
        else:
            return "Monid research timed out."
    if run.get("status") not in (None, "COMPLETED"):
        return f"Monid run {run['status']}."
    provider_response = run.get("providerResponse") or {}
    status = provider_response.get("httpStatus", 200)
    if status >= 400:
        logger.warning("Firecrawl research returned HTTP %d", status)
        return f"Firecrawl research returned HTTP {status}."
    output = run["output"]
    # Firecrawl wraps the schema-shaped extraction in its data field.
    return output.get("data", output)


async def monid_search_batch(
    queries: list[str], *, num_results: int, gl: str = "us", hl: str = "en", tbs: str = "",
) -> list[dict[str, Any]] | str:
    """Research queries through Firecrawl, returning native Serper-shaped results."""
    if gl != "us" or hl != "en" or tbs:
        return (
            "Monid research does not support custom region, language, or "
            "time filters. Select Serper to use those search options."
        )
    if not queries:
        return []
    api_key = (os.getenv("MONID_API_KEY") or "").strip()
    if not api_key:
        return "Monid research requires MONID_API_KEY."
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json; charset=utf-8",
    }
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=10.0)) as client:
            results = []
            for query in queries:
                result = await _search(client, headers, query, num_results)
                record_api_request("monid", errors=int(isinstance(result, str)))
                if isinstance(result, str):
                    return result
                results.append(result)
            return results
    except httpx.HTTPStatusError as error:
        logger.warning("Monid research request failed with HTTP %d", error.response.status_code)
        return f"Monid research returned HTTP {error.response.status_code}."
    except (httpx.HTTPError, ValueError, TypeError, KeyError):
        logger.warning("Monid research request failed")
        return "Monid research failed. Check network access and try again."


__all__ = ["monid_search_batch"]
