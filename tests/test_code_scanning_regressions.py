from __future__ import annotations

from frontier_agent.infra.summary_llm import describe_candidates
from plugins.tools._academic_fetch import biorxiv_to_pdf, route_url
from plugins.tools._render_check import (
    _looks_like_html_document,
    _parse_visible_body,
    unrendered_kind,
)


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


def test_render_check_void_elements_do_not_desync_the_mount_stack() -> None:
    # HTMLParser emits no end tag for a bare <meta>/<link>, so each one used to
    # leak a stack frame and make every later pop close the wrong element.
    content = (
        "<html><head><meta charset='utf-8'><link rel='x'></head>"
        "<body><div id='__next'></div><script></script></body></html>"
    )
    parser = _parse_visible_body(content)
    assert parser._mount_stack == []
    assert parser.has_empty_app_mount is True
    assert unrendered_kind(content) == "shell"

    # A self-closing void tag must stay balanced too.
    assert _parse_visible_body("<div id='root'><img src='s.gif'/></div>")._mount_stack == []


def test_render_check_accepts_a_bom_anywhere_in_the_leading_run() -> None:
    assert _looks_like_html_document("\ufeff\ufeff<html><body></body></html>")
    assert _looks_like_html_document("\n\ufeff<html><body></body></html>")
    assert _looks_like_html_document("\ufeff <!-- c --><html>")
    assert not _looks_like_html_document("\ufeffplain text")


def test_api_key_status_does_not_derive_or_expose_a_fingerprint() -> None:
    candidate = {
        "provider": "test",
        "model": "model",
        "endpoint": "https://example.test/v1",
        "api_key": "password-like-low-entropy-value",
    }
    output = describe_candidates([candidate])

    assert candidate["api_key"] not in output
    assert "api_key=set" in output
    assert "len=" not in output
    assert "#" not in output
    assert "api_key=unset" in describe_candidates([{**candidate, "api_key": ""}])
