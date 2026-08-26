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
_HTML_START_RE = re.compile(
    r"(?:<!doctype\s+html\b|<html\b|<head\b|<body\b)", re.IGNORECASE,
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
# Body shorter than this counts as "suspiciously empty — worth one retry".
# Measured: shell stubs carry 150-170 chars of body, the real page ~4800.
MIN_RENDERED_BODY_CHARS = 400
_MAX_SHELL_VISIBLE_BODY_CHARS = 80
_MAX_JS_INTERSTITIAL_CHARS = 200


class _VisibleBodyParser(HTMLParser):
    """Collect visible body text while ignoring script/style/template payloads."""

    _HIDDEN = frozenset({"script", "style", "template", "noscript", "svg"})
    # HTMLParser never emits handle_endtag for a bare void element, so pushing
    # one onto the mount stack would leak a frame and desync every later pop.
    _VOID = frozenset({
        "area", "base", "br", "col", "embed", "hr", "img", "input",
        "link", "meta", "param", "source", "track", "wbr",
    })

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.seen_body = False
        self.in_body = False
        self.hidden_depth = 0
        self.all_text: list[str] = []
        self.body_text: list[str] = []
        self._mount_stack: list[tuple[str, bool, bool]] = []
        self.has_empty_app_mount = False

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        name = tag.lower()
        if self._mount_stack:
            mount_tag, candidate, _ = self._mount_stack[-1]
            self._mount_stack[-1] = (mount_tag, candidate, True)
        if name in self._VOID:
            # Already recorded as content for the enclosing mount above; a void
            # element has no children and never closes, so it gets no frame.
            return
        attr_map = {key.lower(): value or "" for key, value in attrs}
        marker = f"{attr_map.get('id', '')} {attr_map.get('class', '')}".lower()
        is_mount = name in {"div", "main", "section"} and any(
            token in marker for token in ("app", "root", "__next", "__nuxt")
        )
        self._mount_stack.append((name, is_mount, False))
        if name == "body":
            self.seen_body = True
            self.in_body = True
        if name in self._HIDDEN:
            self.hidden_depth += 1

    def handle_endtag(self, tag: str) -> None:
        name = tag.lower()
        if name in self._VOID:
            # Balances handle_starttag: no frame was pushed, so pop nothing.
            # Covers both a stray ``</br>`` and the endtag half that
            # handle_startendtag synthesizes for ``<img/>``.
            return
        if self._mount_stack:
            mount_tag, candidate, has_content = self._mount_stack.pop()
            if mount_tag == name and candidate and not has_content:
                self.has_empty_app_mount = True
        if name in self._HIDDEN and self.hidden_depth:
            self.hidden_depth -= 1
        if name == "body":
            self.in_body = False

    def handle_data(self, data: str) -> None:
        if data.strip() and self._mount_stack:
            mount_tag, candidate, _ = self._mount_stack[-1]
            self._mount_stack[-1] = (mount_tag, candidate, True)
        if self.hidden_depth or not data.strip():
            return
        self.all_text.append(data)
        if self.in_body:
            self.body_text.append(data)

    def visible_text(self) -> str:
        parts = self.body_text if self.seen_body else self.all_text
        return " ".join(" ".join(parts).split())


def _parse_visible_body(content: str) -> _VisibleBodyParser:
    parser = _VisibleBodyParser()
    try:
        parser.feed(content)
        parser.close()
    except Exception:
        pass
    return parser


def _visible_body_text(content: str) -> str:
    return html.unescape(_parse_visible_body(content).visible_text())


def _looks_like_html_document(content: str) -> bool:
    """Recognize an HTML document prefix without backtracking over comments."""
    position = 0
    length = len(content)
    while position < length:
        # A BOM may repeat or trail whitespace when documents are concatenated,
        # so skip it wherever it appears in the leading run, not just at index 0.
        while position < length and (
            content[position].isspace() or content[position] == "\ufeff"
        ):
            position += 1
        if content.startswith("<!--", position):
            end = content.find("-->", position + 4)
            if end < 0:
                return False
            position = end + 3
            continue
        if content[position:position + 5].lower() == "<?xml":
            end = content.find(">", position + 5)
            if end < 0:
                return False
            position = end + 1
            continue
        break
    return bool(_HTML_START_RE.match(content, position))


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
    is_html = _looks_like_html_document(text)
    parsed_body = _parse_visible_body(text) if is_html else None
    visible = (
        html.unescape(parsed_body.visible_text())
        if parsed_body is not None else reader_body(text)
    )
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
        and parsed_body is not None
        and parsed_body.has_empty_app_mount
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
