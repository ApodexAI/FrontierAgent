"""Small deterministic citation helpers shared by reporter implementations."""

from __future__ import annotations

import json
import re
from typing import Any

_CITE_GROUP_RE = re.compile(r"\[(\s*\d[\d,;\s]*)\]")
_TRAILING_REFS_RE = re.compile(
    r"(?im)^(?:#{2,4}\s*)?(References|参考文献|参考资料|Sources|来源)\s*$[\s\S]*\Z",
)
_INFO_TITLE_PREFIX_RE = re.compile(r"^Info:\s*#?\s*", re.IGNORECASE)
_URL_SCHEMES = ("http://", "https://")


def coerce_url(value: Any) -> str:
    """Return the first valid HTTP(S) URL represented by ``value``."""
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return next(
            (item.strip() for item in value if isinstance(item, str) and item.strip().startswith(_URL_SCHEMES)),
            "",
        )
    if not isinstance(value, str):
        return ""
    text = value.strip()
    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = json.loads(text)
        except (json.JSONDecodeError, ValueError):
            parsed = []
        if isinstance(parsed, list):
            url = coerce_url(parsed)
            if url:
                return url
        match = re.search(r"https?://[^\s\"',\]]+", text)
        return match.group(0) if match else ""
    return text if text.startswith(_URL_SCHEMES) else ""


def coerce_title(value: Any, fallback_url: str) -> str:
    """Return a single-line, bounded title or ``fallback_url``."""
    if isinstance(value, (list, tuple)):
        value = next((item for item in value if isinstance(item, str) and item.strip()), "")
    text = _INFO_TITLE_PREFIX_RE.sub("", str(value or "")).strip()
    json_shaped = (text.startswith("[") and text.endswith("]")) or (
        text.startswith("{") and text.endswith("}")
    )
    if not text or len(text) > 200 or "\n" in text or "\r" in text or json_shaped:
        return fallback_url
    return text


def citation_numbers(text: str) -> list[int]:
    """Extract numeric markers from ``[N]`` and ``[1, 2]`` forms."""
    numbers: list[int] = []
    for match in _CITE_GROUP_RE.finditer(text or ""):
        numbers.extend(int(number) for number in re.findall(r"\d+", match.group(1)))
    return numbers


def strip_trailing_references(text: str) -> str:
    """Remove a plausible trailing References/Sources footer."""
    if not text:
        return ""
    match = _TRAILING_REFS_RE.search(text)
    if match is None or match.start() < len(text) * 0.3:
        return text
    return text[: match.start()].rstrip()
