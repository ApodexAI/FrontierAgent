from __future__ import annotations

from frontier_agent.infra.summary_llm import describe_candidates
from plugins.tools._academic_fetch import biorxiv_to_pdf, route_url
from plugins.tools._render_check import _looks_like_html_document, unrendered_kind


def test_academic_routes_validate_the_parsed_hostname() -> None:
    assert route_url("https://pmc.ncbi.nlm.nih.gov/articles/PMC1/") == "pmc"
    assert route_url("https://subdomain.biorxiv.org/content/1") == "biorxiv"
    assert route_url("https://pmc.ncbi.nlm.nih.gov.evil.test/PMC1") == "jina"
    assert route_url("https://biorxiv.org@evil.test/content/1") == "jina"
    assert route_url("https://[") == "jina"


def test_biorxiv_pdf_conversion_rejects_hostname_confusion() -> None:
    expected = "https://www.biorxiv.org/content/10.1/123.full.pdf"
    assert biorxiv_to_pdf("https://www.biorxiv.org/content/10.1/123") == expected
    assert biorxiv_to_pdf("https://biorxiv.org.evil.test/content/10.1/123") == ""
    assert biorxiv_to_pdf("https://biorxiv.org@evil.test/content/10.1/123") == ""
    assert biorxiv_to_pdf("https://[") == ""


def test_render_check_handles_many_leading_comments_without_redos() -> None:
    comments = "<!-- --><!-- -->" * 10_000
    assert _looks_like_html_document(f"{comments}<html><body></body></html>")


def test_render_check_still_identifies_an_empty_app_shell() -> None:
    comments = "<!-- --><!-- -->" * 100
    content = f"{comments}<html><body><div id='root'><!-- --></div><script></script></body></html>"
    assert unrendered_kind(content) == "shell"


def test_api_key_fingerprint_is_keyed_and_does_not_expose_the_key() -> None:
    candidate = {
        "provider": "test",
        "model": "model",
        "endpoint": "https://example.test/v1",
        "api_key": "password-like-low-entropy-value",
    }
    first = describe_candidates([candidate])
    second = describe_candidates([candidate])

    assert first == second
    assert candidate["api_key"] not in first
    assert "len=31 #" in first
