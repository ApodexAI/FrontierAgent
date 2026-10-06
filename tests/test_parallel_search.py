"""Native provider selection and anonymous Parallel Search MCP transport."""

from __future__ import annotations

import asyncio
import importlib
import json
from types import SimpleNamespace
from typing import Any

import pytest

from plugins.tools import _parallel_search as parallel


class _Response:
    def __init__(
        self,
        value: dict[str, Any] | None = None,
        *,
        session_id: str | None = None,
    ) -> None:
        self.status_code = 200
        self.request = object()
        self.content = json.dumps(value).encode() if value is not None else b""
        self.text = self.content.decode()
        self.headers = {"content-type": "application/json"}
        if session_id:
            self.headers["Mcp-Session-Id"] = session_id

    def json(self) -> dict[str, Any]:
        return json.loads(self.text)


class _FakeClient:
    calls: list[tuple[str, dict[str, Any] | None, dict[str, str]]]

    def __init__(self, **_: Any) -> None:
        self.calls = _FakeClient.calls

    async def __aenter__(self) -> _FakeClient:
        return self

    async def __aexit__(self, *_: Any) -> None:
        return None

    async def post(
        self,
        url: str,
        *,
        json: dict[str, Any],
        headers: dict[str, str],
    ) -> _Response:
        self.calls.append((url, json, headers))
        method = json.get("method")
        if method == "initialize":
            return _Response(
                {"jsonrpc": "2.0", "id": 1, "result": {"protocolVersion": "2025-03-26"}},
                session_id="test-session",
            )
        if method == "notifications/initialized":
            return _Response()
        if method == "tools/list":
            return _Response({
                "jsonrpc": "2.0",
                "id": json["id"],
                "result": {"tools": [{"name": "web_search"}]},
            })
        if method == "tools/call":
            return _Response({
                "jsonrpc": "2.0",
                "id": json["id"],
                "result": {
                    "content": [{"type": "text", "text": "Found a result."}],
                    "structuredContent": {
                        "results": [{
                            "title": "Example",
                            "url": "https://example.com/page",
                            "excerpts": ["A useful excerpt."],
                        }],
                    },
                },
            })
        raise AssertionError(f"unexpected MCP method: {method}")

    async def delete(self, url: str, *, headers: dict[str, str]) -> _Response:
        self.calls.append((url, None, headers))
        return _Response()


def test_serper_remains_the_implicit_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("WEB_SEARCH_PROVIDER", raising=False)
    assert parallel.selected_search_provider() == "serper"
    assert parallel.valid_search_provider()
    assert parallel.valid_search_provider("parallel")
    assert not parallel.valid_search_provider("other")


@pytest.mark.asyncio
async def test_parallel_mcp_discovers_and_calls_search_with_project_user_agent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _FakeClient.calls = []
    monkeypatch.setattr(parallel.httpx, "AsyncClient", _FakeClient)

    result = await parallel.parallel_search_batch(
        ["FrontierAgent MCP search"], num_results=5,
    )

    assert result == [{
        "organic": [{
            "title": "Example",
            "link": "https://example.com/page",
            "snippet": "A useful excerpt.",
            "date": "",
        }],
    }]
    assert [call[1]["method"] if call[1] else "delete" for call in _FakeClient.calls] == [
        "initialize", "notifications/initialized", "tools/list", "tools/call", "delete",
    ]
    for _, payload, headers in _FakeClient.calls:
        assert headers["User-Agent"].startswith("FrontierAgent/")
        assert "github.com/ApodexAI/FrontierAgent" in headers["User-Agent"]
        assert "Authorization" not in headers
        assert "X-API-KEY" not in headers
        if payload and payload.get("method") in {"tools/list", "tools/call"}:
            assert headers["MCP-Protocol-Version"] == "2025-03-26"
            assert headers["Mcp-Session-Id"] == "test-session"


@pytest.mark.asyncio
async def test_parallel_mcp_rejects_filters_it_cannot_honor() -> None:
    result = await parallel.parallel_search_batch(
        ["current information"], num_results=5, gl="gb",
    )
    assert isinstance(result, str)
    assert "custom region" in result


@pytest.mark.asyncio
async def test_parallel_mcp_rejects_counts_above_anonymous_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _FakeClient.calls = []
    monkeypatch.setattr(parallel.httpx, "AsyncClient", _FakeClient)

    result = await parallel.parallel_search_batch(
        ["current information"], num_results=11,
    )

    assert isinstance(result, str)
    assert "up to 10 results per query" in result
    assert "Select Serper" in result
    assert _FakeClient.calls == []


@pytest.mark.asyncio
async def test_parallel_mcp_dispatches_list_queries_concurrently(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    active = 0
    peak_active = 0

    class ConcurrentClient(_FakeClient):
        async def post(
            self,
            url: str,
            *,
            json: dict[str, Any],
            headers: dict[str, str],
        ) -> _Response:
            nonlocal active, peak_active
            if json.get("method") == "tools/call":
                active += 1
                peak_active = max(peak_active, active)
                await asyncio.sleep(0.02)
                active -= 1
            return await super().post(url, json=json, headers=headers)

    _FakeClient.calls = []
    monkeypatch.setattr(parallel.httpx, "AsyncClient", ConcurrentClient)

    result = await parallel.parallel_search_batch(
        ["first query", "second query"], num_results=3,
    )

    assert isinstance(result, list)
    assert len(result) == 2
    assert peak_active == 2
    calls = [
        payload for _, payload, _ in _FakeClient.calls
        if payload and payload.get("method") == "tools/call"
    ]
    assert len({payload["id"] for payload in calls}) == 2
    assert {
        payload["params"]["arguments"]["objective"] for payload in calls
    } == {"first query", "second query"}


@pytest.mark.asyncio
async def test_original_parallel_route_keeps_its_domain_exclusions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = importlib.import_module("plugins.tools.web_search")
    monkeypatch.setenv("WEB_SEARCH_PROVIDER", "parallel")
    async def search(*_args: Any, **_kwargs: Any) -> list[dict[str, Any]]:
        return [{
            "organic": [
                {"title": "Video", "link": "https://youtube.com/watch/1", "snippet": ""},
                {"title": "Useful page", "link": "https://example.com/page", "snippet": "Text."},
            ],
        }]

    monkeypatch.setattr(module, "parallel_search_batch", search)
    output = await module.web_search.ainvoke({"q": "sample topic"})

    assert "Useful page" in output
    assert "youtube.com" not in output


@pytest.mark.asyncio
async def test_aligned_parallel_route_keeps_results_from_each_query(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = importlib.import_module("plugins.tools.web_search_aligned")
    monkeypatch.setenv("WEB_SEARCH_PROVIDER", "parallel")
    requested_limits: list[int] = []

    async def search(
        queries: list[str], *, num_results: int, **_: Any,
    ) -> list[dict[str, Any]]:
        requested_limits.append(num_results)
        return [
            {"organic": [
                {"title": f"{query} result {index}",
                 "link": f"https://{query}{index}.example.com/page",
                 "snippet": "Useful result."}
                for index in range(1, 3)
            ]}
            for query in queries
        ]

    monkeypatch.setattr(module, "parallel_search_batch", search)
    output = await module.web_search_aligned.ainvoke({
        "q": ["first", "second"],
        "num": 2,
    })

    assert requested_limits == [2]
    assert output.count("URL: https://") == 4
    assert "first result 1" in output
    assert "second result 1" in output


@pytest.mark.asyncio
async def test_parallel_errors_do_not_fall_back_to_serper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = importlib.import_module("plugins.tools.web_search")
    monkeypatch.setenv("WEB_SEARCH_PROVIDER", "parallel")
    monkeypatch.delenv("SERPER_API_KEY", raising=False)

    async def failed(*_args: Any, **_kwargs: Any) -> str:
        return "Parallel Search MCP returned HTTP 429."

    async def no_serper(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        raise AssertionError("explicit Parallel selection must not call Serper")

    monkeypatch.setattr(module, "parallel_search_batch", failed)
    monkeypatch.setattr(module, "raw_web_search", no_serper)
    output = await module.web_search.ainvoke({"q": "sample topic"})

    assert "Parallel Search MCP returned HTTP 429" in output


def test_react_profile_loader_keeps_both_search_implementations_on_parallel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from apodex.agent_tools import terminal_tool_registry
    from plugins.tools.web_search import web_search
    from plugins.tools.web_search_aligned import web_search_aligned
    from workflows.stateful_react_agent.nodes.main_agent import (
        _replace_tool_impls,
        _tools_for_stateful_react,
    )
    from workflows.stateful_react_agent.profile import load_react_profile

    monkeypatch.setenv("WEB_SEARCH_PROVIDER", "parallel")
    monkeypatch.delenv("REACT_NO_WEB", raising=False)
    resource_mgr = SimpleNamespace(
        all_tools=terminal_tool_registry(),
        global_tool_policy=None,
    )

    for profile_name, expected in (
        ("simple", web_search),
        ("tui", web_search_aligned),
    ):
        agent = load_react_profile(profile_name)["agent"]
        tools = _replace_tool_impls(
            _tools_for_stateful_react(resource_mgr, agent), agent,
        )
        selected = next(tool for tool in tools if tool.name == "web_search")
        assert selected is expected
        assert parallel.selected_search_provider() == "parallel"


@pytest.mark.parametrize("implementation", ["original", "aligned"])
@pytest.mark.parametrize("scenario", ["filtered", "duplicates", "multiple_queries"])
@pytest.mark.asyncio
async def test_parallel_limits_displayed_results_after_filtering_and_deduplication(
    monkeypatch: pytest.MonkeyPatch, implementation: str, scenario: str,
) -> None:
    def row(title: str, url: str) -> dict[str, Any]:
        return {"title": title, "url": url, "excerpts": ["Useful text."],
                "publish_date": "2026-09-29"}

    first = row("First useful", "https://example.com/first")
    second = row("Second useful", "https://example.com/second")
    extra = row("Over limit", "https://example.com/extra")
    blocked_url = (
        "https://youtube.com/watch/1" if implementation == "original"
        else "https://huggingface.co/datasets/example"
    )
    blocked = row("Blocked result", blocked_url)
    if scenario == "filtered":
        queries, limit = ["first"], 1
        responses = {"first": [blocked, first, extra]}
        expected = ["First useful"]
    elif scenario == "duplicates":
        queries, limit = ["first"], 2
        responses = {"first": [first, first, second, extra]}
        expected = ["First useful", "Second useful"]
    else:
        queries, limit = ["first", "second"], 1
        responses = {"first": [first, second], "second": [first, blocked, second, extra]}
        expected = ["First useful", "Second useful"]

    class ResultsClient(_FakeClient):
        async def post(self, url, *, json, headers):
            if json.get("method") == "tools/call":
                query = json["params"]["arguments"]["objective"]
                return _Response({
                    "jsonrpc": "2.0", "id": json["id"],
                    "result": {"structuredContent": {"results": responses[query]}},
                })
            return await super().post(url, json=json, headers=headers)

    _FakeClient.calls = []
    monkeypatch.setattr(parallel.httpx, "AsyncClient", ResultsClient)
    monkeypatch.setenv("WEB_SEARCH_PROVIDER", "parallel")
    if implementation == "original":
        module = importlib.import_module("plugins.tools.web_search")
        output = await module.web_search.ainvoke({"q": queries, "num_results": limit})
    else:
        module = importlib.import_module("plugins.tools.web_search_aligned")
        output = await module.web_search_aligned.ainvoke({"q": queries, "num": limit})
        assert output.count("Date: 2026-09-29") == len(expected)

    assert output.count("URL: https://") == len(expected)
    for title in expected:
        assert output.count(title) == 1
    assert "Blocked result" not in output
    assert "Over limit" not in output


@pytest.mark.parametrize("field", ["publish_date", "published_date", "date"])
@pytest.mark.parametrize("structured", [True, False])
def test_parallel_normalises_publication_dates(field: str, structured: bool) -> None:
    payload = {"results": [{"url": "https://example.com", field: "2026-09-29"}]}
    result = (
        {"structuredContent": payload} if structured
        else {"content": [{"type": "text", "text": json.dumps(payload)}]}
    )
    assert parallel._normalise_result(result)["organic"][0]["date"] == "2026-09-29"
