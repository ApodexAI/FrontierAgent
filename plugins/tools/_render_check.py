"""Detect a scrape that came back un-rendered (a JavaScript "app shell")."""

from __future__ import annotations

import html
import re
from html.parser import HTMLParser

# A reader prefixes its markdown with a ``Key: value`` block (Jina emits
# Title / URL Source / Published Time / …). Only the body below it is content.
_READER_HEADER_RE = re.compile(
    r"\A(?:(?:Title|URL Source|Published Time|Content Length|Images|Links|"
    r"Warning|Markdown Content):[^\n]*\n+)+",
    re.IGNORECASE,
)
_HTML_DOC_RE = re.compile(
    r"\A(?:\ufeff|\s|<!--.*?-->|<\?xml[^>]*>)*"
    r"(?:<!doctype\s+html\b|<html\b|<head\b|<body\b)",
    re.IGNORECASE | re.DOTALL,
)
_SCRIPT_TAG_RE = re.compile(r"<script\b", re.IGNORECASE)
_NOSCRIPT_JS_RE = re.compile(
    r"<noscript\b[^>]*>[\s\S]*?"
    r"(?:enable|requires?|turn on|activate)\s+(?:your\s+)?javascript",
    re.IGNORECASE,
)
_JS_INTERSTITIAL_RE = re.compile(
    r"\A(?:"
    r"(?:please\s+)?(?:enable|turn on|activate)\s+(?:your\s+)?javascript"
    r"|javascript\s+(?:is\s+)?(?:required|disabled)"
    r"|(?:this\s+)?(?:site|page|app|application)\s+requires?\s+javascript"
    r")\b",
    re.IGNORECASE,
)
_EMPTY_APP_MOUNT_RE = re.compile(
    r"<(?:div|main|section)\b[^>]*(?:id|class)\s*=\s*['\"][^'\"]*"
    r"(?:app|root|__next|__nuxt)[^'\"]*['\"][^>]*>"
    r"(?:\s|<!--[\s\S]*?-->)*</(?:div|main|section)\s*>",
    re.IGNORECASE,
)

# Body shorter than this counts as "suspiciously empty — worth one retry".
# Measured: shell stubs carry 150-170 chars of body, the real page ~4800.
MIN_RENDERED_BODY_CHARS = 400
_MAX_SHELL_VISIBLE_BODY_CHARS = 80
_MAX_JS_INTERSTITIAL_CHARS = 200


class _VisibleBodyParser(HTMLParser):
    """Collect visible body text while ignoring script/style/template payloads."""

    _HIDDEN = frozenset({"script", "style", "template", "noscript", "svg"})

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.seen_body = False
        self.in_body = False
        self.hidden_depth = 0
        self.all_text: list[str] = []
        self.body_text: list[str] = []

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        name = tag.lower()
        if name == "body":
            self.seen_body = True
            self.in_body = True
        if name in self._HIDDEN:
            self.hidden_depth += 1

    def handle_endtag(self, tag: str) -> None:
        name = tag.lower()
        if name in self._HIDDEN and self.hidden_depth:
            self.hidden_depth -= 1
        if name == "body":
            self.in_body = False

    def handle_data(self, data: str) -> None:
        if self.hidden_depth or not data.strip():
            return
        self.all_text.append(data)
        if self.in_body:
            self.body_text.append(data)

    def visible_text(self) -> str:
        parts = self.body_text if self.seen_body else self.all_text
        return " ".join(" ".join(parts).split())


def _visible_body_text(content: str) -> str:
    parser = _VisibleBodyParser()
    try:
        parser.feed(content)
        parser.close()
    except Exception:
        return ""
    return html.unescape(parser.visible_text())


def reader_body(content: str) -> str:
    """Strip the reader's ``Title:``/``URL Source:``… preamble, leaving content."""
    return _READER_HEADER_RE.sub("", content or "", count=1).strip()


def unrendered_kind(content: str) -> str | None:
    """Classify a scrape that carries no page content.

    Returns:
        ``"shell"``  — high confidence: a raw pre-hydration DOM (a reader
            returns markdown on success, so raw HTML means it could not convert
            the page) or an explicit "enable JavaScript" interstitial. Nothing
            is recoverable by fetching the same way again.
        ``"empty"``  — low confidence: the body is merely suspiciously short. A
            legitimately tiny page looks identical, so callers should re-try but
            must not turn this into an error.
        ``None``     — looks like real page content.
    """
    text = (content or "").strip()
    if not text:
        return "empty"
    head = text[:4000]
    is_html = bool(_HTML_DOC_RE.match(text))
    visible = _visible_body_text(text) if is_html else reader_body(text)
    if (
        is_html
        and _SCRIPT_TAG_RE.search(head)
        and not visible
    ):
        return "shell"
    if (
        is_html
        and _SCRIPT_TAG_RE.search(head)
        and len(visible) < _MAX_SHELL_VISIBLE_BODY_CHARS
        and _EMPTY_APP_MOUNT_RE.search(text)
    ):
        return "shell"
    if (
        is_html
        and _NOSCRIPT_JS_RE.search(head)
        and len(visible) < MIN_RENDERED_BODY_CHARS
    ):
        return "shell"
    if (
        not is_html
        and len(visible) < _MAX_JS_INTERSTITIAL_CHARS
        and _JS_INTERSTITIAL_RE.search(visible)
    ):
        return "shell"
    if len(reader_body(text)) < MIN_RENDERED_BODY_CHARS:
        return "empty"
    return None


__all__ = ["MIN_RENDERED_BODY_CHARS", "reader_body", "unrendered_kind"]
