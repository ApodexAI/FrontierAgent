"""Monid transport and opt-in native search routing."""

from __future__ import annotations

import importlib
import json

import httpx
import pytest

from plugins.tools import _monid_search as monid
from plugins.tools._parallel_search import selected_search_provider, valid_search_provider


@pytest.mark.asyncio
@pytest.mark.parametrize("module_name", ["web_search", "web_search_aligned"])
async def test_monid_request_and_native_result(monkeypatch, module_name):
    monkeypatch.setenv("MONID_API_KEY", "test-key")
    monkeypatch.setenv("WEB_SEARCH_PROVIDER", "monid")
    monkeypatch.delenv("SERPER_API_KEY", raising=False)
    monkeypatch.delenv("FIRECRAWL_API_KEY", raising=False)
    calls = []

    def handle(request):
        calls.append(request)
        assert str(request.url) == "https://api.monid.ai/v1/run"
        assert request.headers["Authorization"] == "Bearer test-key"
        assert request.headers["Content-Type"] == "application/json; charset=utf-8"
        payload = json.loads(request.content)
        assert payload["provider"] == "firecrawl"
        assert payload["endpoint"] == "/agent"
        assert payload["input"] == {
            "queryParams": {}, "pathParams": {}, "body": {
                "prompt": "Research this query and return up to 10 web results "
                          "with source links, titles and snippets: 中文",
                "schema": monid._RESULT_SCHEMA,
            },
        }
        assert "中文".encode() in request.content
        return httpx.Response(200, json={"output": {"data": {"organic": [{
            "title": "Example", "link": "https://example.com", "snippet": "Research result.",
        }]}}})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handle))
    monkeypatch.setattr(monid.httpx, "AsyncClient", lambda **kwargs: client)
    module = importlib.import_module(f"plugins.tools.{module_name}")
    tool = getattr(module, module_name)
    output = await tool.ainvoke({"q": "中文"})
    assert "Example" in output
    assert "https://example.com" in output
    assert "Research result." in output
    assert len(calls) == 1
    assert valid_search_provider("monid")


@pytest.mark.asyncio
@pytest.mark.parametrize("scenario", ["success", "provider", "http", "poll_http", "failed"])
async def test_monid_polling_and_errors(monkeypatch, scenario):
    monkeypatch.setenv("MONID_API_KEY", "test-key")
    calls = []

    async def no_sleep(seconds):
        pass

    def handle(request):
        calls.append(request)
        if request.method == "POST":
            assert request.url.path == "/v1/run"
            if scenario == "http":
                return httpx.Response(401, json={"code": 401, "message": "Bad key"})
            return httpx.Response(202, json={"runId": "research-1"})
        assert str(request.url) == "https://api.monid.ai/v1/runs/research-1"
        if scenario == "poll_http":
            return httpx.Response(503, json={"code": 503, "message": "Unavailable"})
        if scenario == "failed":
            return httpx.Response(200, json={"status": "FAILED"})
        if scenario == "provider":
            return httpx.Response(200, json={"status": "COMPLETED", "providerResponse": {
                "httpStatus": 429, "error": {"message": "Rate limited"},
            }})
        return httpx.Response(200, json={"status": "COMPLETED", "output": {"organic": []}})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handle))
    monkeypatch.setattr(monid.httpx, "AsyncClient", lambda **kwargs: client)
    monkeypatch.setattr(monid.asyncio, "sleep", no_sleep)
    result = await monid.monid_search_batch(["query"], num_results=1)
    assert result == {
        "success": [{"organic": []}], "provider": "Firecrawl research returned HTTP 429.",
        "http": "Monid research returned HTTP 401.",
        "poll_http": "Monid research returned HTTP 503.", "failed": "Monid run FAILED.",
    }[scenario]
    assert len(calls) == (1 if scenario == "http" else 2)


def test_default_provider_unchanged_without_monid_key(monkeypatch):
    monkeypatch.delenv("MONID_API_KEY", raising=False)
    monkeypatch.delenv("WEB_SEARCH_PROVIDER", raising=False)
    assert selected_search_provider() == "serper"
