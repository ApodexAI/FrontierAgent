"""Opt-in Keenable search transport for the native web-search tools."""

from __future__ import annotations

import asyncio
import importlib
import json
from collections.abc import Callable
from typing import Any

import httpx
import pytest

from plugins.tools import _keenable_search as keenable
from plugins.tools import _parallel_search as parallel


def _row(title: str, url: str, snippet: str = "Useful text.") -> dict[str, Any]:
    return {
        "title": title,
        "url": url,
        "description": "",
        "snippet": snippet,
        "acquired_at": "2026-10-01T00:00:00Z",
    }


def _install(
    monkeypatch: pytest.MonkeyPatch,
    handler: Callable[[httpx.Request], Any],
) -> list[httpx.Request]:
    """Route the module's HTTP client through ``handler`` and record requests."""
    requests: list[httpx.Request] = []
    real_client = httpx.AsyncClient

    async def recording(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        response = handler(request)
        if not isinstance(response, httpx.Response):
            response = await response
        return response

    def client(**kwargs: Any) -> httpx.AsyncClient:
        return real_client(transport=httpx.MockTransport(recording), **kwargs)

    monkeypatch.setattr(keenable.httpx, "AsyncClient", client)
    return requests


def _results(*rows: dict[str, Any]) -> Callable[[httpx.Request], httpx.Response]:
    return lambda _request: httpx.Response(200, json={"results": list(rows)})


def test_keenable_is_an_opt_in_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("WEB_SEARCH_PROVIDER", raising=False)
    assert parallel.selected_search_provider() == "serper"
    assert parallel.valid_search_provider("keenable")
    monkeypatch.setenv("WEB_SEARCH_PROVIDER", "Keenable")
    assert parallel.selected_search_provider() == "keenable"
    assert parallel.valid_search_provider()


@pytest.mark.asyncio
async def test_keyless_search_uses_the_public_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("KEENABLE_API_KEY", raising=False)
    requests = _install(monkeypatch, _results(
        _row("Example", "https://example.com/page", "First line.\n\n## Heading\nSecond."),
    ))

    result = await keenable.keenable_search_batch(["FrontierAgent search"], num_results=5)

    assert result == [{
        "organic": [{
            "title": "Example",
            "link": "https://example.com/page",
            "snippet": "First line. ## Heading Second.",
        }],
    }]
    assert len(requests) == 1
    request = requests[0]
    assert str(request.url) == "https://api.keenable.ai/v1/search/public"
    assert request.headers["X-Keenable-Title"] == "FrontierAgent"
    assert "X-API-Key" not in request.headers
    assert "Authorization" not in request.headers
    assert json.loads(request.content) == {"query": "FrontierAgent search", "max_results": 10}


@pytest.mark.asyncio
async def test_optional_api_key_selects_the_keyed_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("KEENABLE_API_KEY", "test-key")
    requests = _install(monkeypatch, _results(_row("Example", "https://example.com")))

    result = await keenable.keenable_search_batch(["query"], num_results=10)

    assert isinstance(result, list)
    assert str(requests[0].url) == "https://api.keenable.ai/v1/search"
    assert requests[0].headers["X-API-Key"] == "test-key"
    assert requests[0].headers["X-Keenable-Title"] == "FrontierAgent"


@pytest.mark.parametrize(("requested", "sent"), [(1, 10), (25, 25), (100, 50)])
@pytest.mark.asyncio
async def test_result_count_stays_within_the_supported_range(
    monkeypatch: pytest.MonkeyPatch, requested: int, sent: int,
) -> None:
    monkeypatch.delenv("KEENABLE_API_KEY", raising=False)
    requests = _install(monkeypatch, _results(_row("Example", "https://example.com")))

    await keenable.keenable_search_batch(["query"], num_results=requested)

    assert json.loads(requests[0].content)["max_results"] == sent


@pytest.mark.parametrize("options", [{"gl": "gb"}, {"hl": "de"}, {"tbs": "qdr:d"}])
@pytest.mark.asyncio
async def test_rejects_filters_it_cannot_honor(
    monkeypatch: pytest.MonkeyPatch, options: dict[str, str],
) -> None:
    requests = _install(monkeypatch, _results())

    result = await keenable.keenable_search_batch(["query"], num_results=5, **options)

    assert isinstance(result, str)
    assert "custom region, language, or time filters" in result
    assert "Select Serper" in result
    assert requests == []


def test_long_page_text_is_capped() -> None:
    text = "word " * 1000
    result = keenable._normalise_result({"results": [_row("Long", "https://example.com", text)]})
    snippet = result["organic"][0]["snippet"]
    assert len(snippet) <= keenable._SNIPPET_CHARS + 3
    assert snippet.endswith("...")
    assert keenable._normalise_result({"results": [{"title": "No URL"}]}) == {}


@pytest.mark.asyncio
async def test_rate_limits_are_retried_then_reported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    delays: list[float] = []

    async def no_wait(delay: float) -> None:
        delays.append(delay)

    monkeypatch.setattr(keenable.asyncio, "sleep", no_wait)
    requests = _install(monkeypatch, lambda _request: httpx.Response(429))

    result = await keenable.keenable_search_batch(["query"], num_results=5)

    assert result == "Keenable search returned HTTP 429."
    assert len(requests) == 3
    assert delays == [1, 2]


@pytest.mark.asyncio
async def test_server_error_is_retried_before_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def no_wait(_delay: float) -> None:
        return None

    statuses = iter([503, 200])

    def handler(_request: httpx.Request) -> httpx.Response:
        status = next(statuses)
        if status != 200:
            return httpx.Response(status)
        return httpx.Response(200, json={"results": [_row("Example", "https://example.com")]})

    monkeypatch.setattr(keenable.asyncio, "sleep", no_wait)
    requests = _install(monkeypatch, handler)

    result = await keenable.keenable_search_batch(["query"], num_results=5)

    assert isinstance(result, list)
    assert result[0]["organic"][0]["link"] == "https://example.com"
    assert len(requests) == 2


@pytest.mark.asyncio
async def test_list_queries_are_dispatched_concurrently(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    active = 0
    peak_active = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal active, peak_active
        active += 1
        peak_active = max(peak_active, active)
        await asyncio.sleep(0.02)
        active -= 1
        query = json.loads(request.content)["query"]
        return httpx.Response(200, json={
            "results": [_row(f"{query} result", f"https://{query}.example.com")],
        })

    requests = _install(monkeypatch, handler)

    result = await keenable.keenable_search_batch(["first", "second"], num_results=3)

    assert isinstance(result, list)
    assert [data["organic"][0]["title"] for data in result] == [
        "first result", "second result",
    ]
    assert len(requests) == 2
    assert peak_active == 2


@pytest.mark.asyncio
async def test_keenable_errors_do_not_fall_back_to_serper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = importlib.import_module("plugins.tools.web_search")
    monkeypatch.setenv("WEB_SEARCH_PROVIDER", "keenable")
    monkeypatch.delenv("SERPER_API_KEY", raising=False)

    async def failed(*_args: Any, **_kwargs: Any) -> str:
        return "Keenable search returned HTTP 429."

    async def no_serper(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        raise AssertionError("explicit Keenable selection must not call Serper")

    monkeypatch.setattr(module, "keenable_search_batch", failed)
    monkeypatch.setattr(module, "raw_web_search", no_serper)
    output = await module.web_search.ainvoke({"q": "sample topic"})

    assert "Keenable search returned HTTP 429" in output


@pytest.mark.asyncio
async def test_aligned_route_rejects_options_keenable_cannot_honor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = importlib.import_module("plugins.tools.web_search_aligned")
    monkeypatch.setenv("WEB_SEARCH_PROVIDER", "keenable")

    output = await module.web_search_aligned.ainvoke({"q": "sample", "page": 2})

    assert output.startswith("[ERROR]: Keenable search does not support location, page")


@pytest.mark.parametrize("implementation", ["original", "aligned"])
@pytest.mark.parametrize("scenario", ["filtered", "duplicates", "multiple_queries"])
@pytest.mark.asyncio
async def test_keenable_limits_displayed_results_after_filtering_and_deduplication(
    monkeypatch: pytest.MonkeyPatch, implementation: str, scenario: str,
) -> None:
    first = _row("First useful", "https://example.com/first")
    second = _row("Second useful", "https://example.com/second")
    extra = _row("Over limit", "https://example.com/extra")
    blocked_url = (
        "https://youtube.com/watch/1" if implementation == "original"
        else "https://huggingface.co/datasets/example"
    )
    blocked = _row("Blocked result", blocked_url)
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

    def handler(request: httpx.Request) -> httpx.Response:
        query = json.loads(request.content)["query"]
        return httpx.Response(200, json={"results": responses[query]})

    monkeypatch.delenv("KEENABLE_API_KEY", raising=False)
    monkeypatch.setenv("WEB_SEARCH_PROVIDER", "keenable")
    _install(monkeypatch, handler)
    if implementation == "original":
        module = importlib.import_module("plugins.tools.web_search")
        output = await module.web_search.ainvoke({"q": queries, "num_results": limit})
    else:
        module = importlib.import_module("plugins.tools.web_search_aligned")
        output = await module.web_search_aligned.ainvoke({"q": queries, "num": limit})

    assert output.count("URL: https://") == len(expected)
    for title in expected:
        assert output.count(title) == 1
    assert "Blocked result" not in output
    assert "Over limit" not in output
