"""Keyless Parallel Search MCP transport for the native web-search tools."""

from __future__ import annotations

import json
import logging
import os
from contextlib import suppress
from importlib.metadata import PackageNotFoundError, version
from typing import Any
from uuid import uuid4

import httpx

from frontier_agent.infra.session_context import get_task_session_id
from frontier_agent.infra.usage_meter import record_api_request

logger = logging.getLogger(__name__)

_MCP_URL = "https://search.parallel.ai/mcp"
_INITIAL_PROTOCOL_VERSION = "2025-03-26"
_PROVIDER_NAMES = frozenset({"serper", "parallel"})


def selected_search_provider() -> str:
    """Return the explicit search backend, keeping Serper as the default."""
    return (os.getenv("WEB_SEARCH_PROVIDER") or "serper").strip().lower()


def valid_search_provider(value: str | None = None) -> bool:
    """Whether the selected backend is supported."""
    provider = selected_search_provider() if value is None else value.strip().lower()
    return provider in _PROVIDER_NAMES


def _user_agent() -> str:
    """Identify the calling project and its installed version truthfully."""
    try:
        project_version = version("frontier-agent")
    except PackageNotFoundError:
        project_version = "dev"
    return f"FrontierAgent/{project_version} (+https://github.com/ApodexAI/FrontierAgent)"


def _session_id() -> str:
    """Use the current task id when available, or generate a request id."""
    task_id = get_task_session_id().strip()
    if task_id and len(task_id) <= 100:
        return task_id
    return uuid4().hex


def _headers(
    session_id: str | None = None,
    protocol_version: str | None = None,
) -> dict[str, str]:
    headers = {
        "Accept": "application/json, text/event-stream",
        "Content-Type": "application/json",
        "User-Agent": _user_agent(),
    }
    if session_id:
        headers["Mcp-Session-Id"] = session_id
    if protocol_version:
        headers["MCP-Protocol-Version"] = protocol_version
    return headers


def _event_response(body: str, request_id: int | None) -> dict[str, Any] | None:
    """Read the JSON-RPC response from a Streamable HTTP SSE body."""
    for line in body.splitlines():
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if not payload or payload == "[DONE]":
            continue
        try:
            value = json.loads(payload)
        except json.JSONDecodeError:
            continue
        if not isinstance(value, dict):
            continue
        if request_id is None or value.get("id") == request_id:
            return value
    return None


async def _post(
    client: httpx.AsyncClient,
    payload: dict[str, Any],
    *,
    request_id: int | None,
    session_id: str | None = None,
    protocol_version: str | None = None,
) -> tuple[dict[str, Any] | None, httpx.Response]:
    response = await client.post(
        _MCP_URL,
        json=payload,
        headers=_headers(session_id, protocol_version),
    )
    if not 200 <= response.status_code < 300:
        raise httpx.HTTPStatusError(
            f"Parallel Search MCP returned HTTP {response.status_code}",
            request=response.request,
            response=response,
        )
    content_type = response.headers.get("content-type", "").lower()
    if "text/event-stream" in content_type:
        result = _event_response(response.text, request_id)
    elif response.content:
        value = response.json()
        result = value if isinstance(value, dict) else None
    else:
        result = None
    return result, response


def _records(value: Any, depth: int = 0) -> list[dict[str, Any]]:
    """Find result rows in either MCP structured content or JSON text."""
    if depth > 16:
        return []
    if isinstance(value, dict):
        if any(key in value for key in ("url", "link")):
            return [value]
        for key in ("results", "organic", "search_results", "web_results", "data"):
            rows = value.get(key)
            if isinstance(rows, list):
                records = [item for item in rows if isinstance(item, dict)]
                if records:
                    return records
            if isinstance(rows, dict):
                found = _records(rows, depth + 1)
                if found:
                    return found
        for nested in value.values():
            found = _records(nested, depth + 1)
            if found:
                return found
    elif isinstance(value, list):
        records = [item for item in value if isinstance(item, dict)]
        if records and any(any(key in item for key in ("url", "link")) for item in records):
            return records
        for item in value:
            found = _records(item, depth + 1)
            if found:
                return found
    return []


def _normalise_result(result: dict[str, Any]) -> dict[str, Any]:
    """Convert a Parallel MCP result into the native Serper-shaped result."""
    structured = result.get("structuredContent")
    text_parts = [
        item.get("text", "")
        for item in result.get("content", [])
        if isinstance(item, dict) and item.get("type") == "text"
    ]
    text_value = "\n\n".join(item for item in text_parts if item)
    parsed_text: Any = None
    if text_value:
        with suppress(json.JSONDecodeError):
            parsed_text = json.loads(text_value)
    source = structured if structured is not None else parsed_text
    records = _records(source)
    if not records and isinstance(source, dict):
        nested = source.get("data")
        records = _records(nested)

    organic: list[dict[str, Any]] = []
    for item in records:
        link = item.get("link") or item.get("url") or ""
        excerpts = item.get("excerpts")
        if isinstance(excerpts, list):
            snippet = "\n".join(str(part) for part in excerpts if part)
        else:
            snippet = item.get("snippet") or item.get("description") or item.get("content") or ""
        organic.append({
            "title": str(item.get("title") or ""),
            "link": str(link),
            "snippet": str(snippet),
            "date": str(item.get("published_date") or item.get("date") or ""),
        })
    if organic:
        return {"organic": organic}

    return {}


async def parallel_search_batch(
    queries: list[str],
    *,
    num_results: int,
    gl: str = "us",
    hl: str = "en",
    tbs: str = "",
) -> list[dict[str, Any]] | str:
    """Discover and call Parallel's native MCP search tool for each query.

    Locale and time controls are rejected when requested because the anonymous
    MCP tool does not expose them. The selected provider never falls back to
    Serper on a transport, rate-limit, or tool error.
    """
    if gl != "us" or hl != "en" or tbs:
        return (
            "Parallel Search MCP does not support custom region, language, or "
            "time filters. Select Serper to use those search options."
        )
    if not queries:
        return []

    task_id = _session_id()
    protocol_version = _INITIAL_PROTOCOL_VERSION
    server_session_id: str | None = None
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(45.0, connect=10.0)) as client:
            initialized, response = await _post(
                client,
                {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "initialize",
                    "params": {
                        "protocolVersion": _INITIAL_PROTOCOL_VERSION,
                        "capabilities": {},
                        "clientInfo": {"name": "FrontierAgent", "version": _user_agent().split("/")[1].split(" ")[0]},
                    },
                },
                request_id=1,
            )
            server_session_id = response.headers.get("Mcp-Session-Id")
            if not initialized or initialized.get("error"):
                raise ValueError("initialize failed")
            protocol_version = str(
                (initialized.get("result") or {}).get("protocolVersion")
                or _INITIAL_PROTOCOL_VERSION
            )

            await _post(
                client,
                {"jsonrpc": "2.0", "method": "notifications/initialized"},
                request_id=None,
                session_id=server_session_id,
                protocol_version=protocol_version,
            )

            tools: list[dict[str, Any]] = []
            cursor: str | None = None
            seen_cursors: set[str] = set()
            request_id = 2
            for _ in range(20):
                params = {"cursor": cursor} if cursor else {}
                listed, _ = await _post(
                    client,
                    {"jsonrpc": "2.0", "id": request_id, "method": "tools/list", "params": params},
                    request_id=request_id,
                    session_id=server_session_id,
                    protocol_version=protocol_version,
                )
                request_id += 1
                if not listed or listed.get("error"):
                    raise ValueError("tool discovery failed")
                result = listed.get("result") or {}
                tools.extend(item for item in result.get("tools", []) if isinstance(item, dict))
                next_cursor = result.get("nextCursor")
                if not next_cursor:
                    break
                if next_cursor in seen_cursors:
                    raise ValueError("tool discovery did not advance")
                seen_cursors.add(next_cursor)
                cursor = str(next_cursor)
            else:
                raise ValueError("tool discovery page limit exceeded")

            if not any(item.get("name") == "web_search" for item in tools):
                raise ValueError("web_search tool unavailable")

            results: list[dict[str, Any]] = []
            for query in queries:
                called, _ = await _post(
                    client,
                    {
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "method": "tools/call",
                        "params": {
                            "name": "web_search",
                            "arguments": {
                                "objective": query,
                                "search_queries": [query],
                                "session_id": task_id,
                            },
                        },
                    },
                    request_id=request_id,
                    session_id=server_session_id,
                    protocol_version=protocol_version,
                )
                request_id += 1
                if not called or called.get("error"):
                    raise ValueError("web_search call failed")
                tool_result = called.get("result") or {}
                if tool_result.get("isError"):
                    raise ValueError("web_search returned an error")
                normalised = _normalise_result(tool_result)
                normalised["organic"] = (normalised.get("organic") or [])[:
                    max(1, min(int(num_results), 100))
                ]
                results.append(normalised)
                record_api_request("parallel")
            return results
    except httpx.HTTPStatusError as error:
        status = error.response.status_code
        logger.warning("Parallel Search MCP request failed with HTTP %d", status)
        return f"Parallel Search MCP returned HTTP {status}."
    except (httpx.HTTPError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        logger.warning("Parallel Search MCP request failed during search")
        return "Parallel Search MCP search failed. Check network access and try again."
    finally:
        if server_session_id:
            try:
                async with httpx.AsyncClient(timeout=httpx.Timeout(5.0, connect=2.0)) as client:
                    await client.delete(
                        _MCP_URL,
                        headers=_headers(server_session_id, protocol_version),
                    )
            except httpx.HTTPError:
                logger.debug("Parallel Search MCP session cleanup failed")


__all__ = ["parallel_search_batch", "selected_search_provider", "valid_search_provider"]
