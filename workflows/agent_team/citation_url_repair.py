"""Repair citation URLs in the coordinator's answer against real evidence."""

from __future__ import annotations

import logging
import re
from typing import Any
from urllib.parse import urlsplit

logger = logging.getLogger(__name__)

# A URL token as it appears in prose or as a markdown link target.
# ``()[]<>`` are excluded so ``[text](url)`` and ``<url>`` yield the target
# alone; CJK punctuation is excluded because Chinese answers run it flush
# against the URL with no space.
_URL_RE = re.compile(r"https?://[^\s<>\[\]()\"'`，、；：。！？]+")

# Trailing characters that are punctuation of the sentence, not of the URL.
# ``/`` and alphanumerics are never stripped. A closing bracket cannot
# appear here (the pattern above already stops before one).
_TRAILING_PUNCT = ".,;:!?'\"*_"

# Page-extension suffixes a model appends when "completing" a URL to match
# the shape of its neighbours. Stripped for comparison only — never from
# the string that gets written back.
_PAGE_SUFFIXES = (".html", ".htm", ".shtml", ".shtm")

# Distinguishes "key absent" from "key present but ambiguous (None)" in the
# shape index — the two need different handling and ``None`` is a real value
# there.
_MISSING = object()


def _split_trailing_punct(token: str) -> tuple[str, str]:
    """Split a matched token into (url, trailing sentence punctuation)."""
    end = len(token)
    while end > 0 and token[end - 1] in _TRAILING_PUNCT:
        end -= 1
    return token[:end], token[end:]


def _normalize(url: str) -> str:
    """Key that collapses the rewrites a model performs on a URL.

    Drops scheme, ``www.``, query, fragment, a trailing slash and one page
    extension. Host variants that serve genuinely different pages
    (``m.``, ``amp.``) are deliberately KEPT distinct: collapsing them
    would let one page's URL be rewritten into another's.
    """
    try:
        parts = urlsplit(url)
    except ValueError:
        return ""
    host = parts.netloc.lower()
    if host.startswith("www."):
        host = host[4:]
    path = parts.path.rstrip("/")
    for suffix in _PAGE_SUFFIXES:
        if path.endswith(suffix):
            path = path[: -len(suffix)]
            break
    if not host:
        return ""
    return f"{host}{path}"


def _card_url(card: Any) -> str:
    if not isinstance(card, dict):
        return ""
    source = card.get("source")
    if not isinstance(source, dict):
        return ""
    url = source.get("url")
    return url.strip() if isinstance(url, str) else ""


def build_url_whitelist(
    cards: list[dict[str, Any]],
) -> tuple[set[str], dict[str, str | None], set[str]]:
    """Index evidence-card URLs for lookup.

    Returns ``(search_exact, search_by_shape, all_exact)`` where
    ``search_by_shape`` maps a :func:`_normalize` key to the single search
    URL carrying it, or ``None`` when two different search URLs share the
    key — an ambiguous key is a key we must not rewrite through, since
    picking either spelling could send the reader to the wrong page.
    """
    search_exact: set[str] = set()
    all_exact: set[str] = set()
    by_shape: dict[str, str | None] = {}

    for card in cards if isinstance(cards, list) else []:
        url = _card_url(card)
        if not url:
            continue
        all_exact.add(url)
        if str(card.get("source_type") or "search") != "search":
            continue
        search_exact.add(url)
        key = _normalize(url)
        if not key:
            continue
        existing = by_shape.get(key, _MISSING)
        if existing is _MISSING:
            by_shape[key] = url
        elif existing != url:
            by_shape[key] = None

    return search_exact, by_shape, all_exact


def repair_citation_urls(
    answer: str,
    cards: list[dict[str, Any]],
) -> tuple[str, dict[str, int]]:
    """Rewrite URLs in ``answer`` to the spelling the tools actually returned.

    Never raises, never deletes, and returns ``answer`` unchanged when
    there are no evidence cards (a code/math run cites nothing, and an
    empty whitelist must not be read as "every URL is wrong").
    """
    if not isinstance(answer, str) or not answer:
        return answer if isinstance(answer, str) else "", _empty_stats()

    search_exact, by_shape, all_exact = build_url_whitelist(cards)
    if not all_exact:
        return answer, _empty_stats()

    stats = _empty_stats()

    def _replace(match: re.Match[str]) -> str:
        token = match.group(0)
        url, trailing = _split_trailing_punct(token)
        if not url:
            return token
        stats["checked"] += 1

        if url in search_exact:
            return token

        key = _normalize(url)
        replacement = by_shape.get(key, _MISSING) if key else _MISSING
        if replacement is _MISSING:
            # No search card has this shape at all.
            if url in all_exact:
                stats["visit_only"] += 1
            else:
                stats["unmatched"] += 1
            return token
        if replacement is None:
            stats["ambiguous"] += 1
            return token
        if replacement == url:
            return token

        stats["repaired"] += 1
        logger.info(
            "citation_url_repair: rewrote %s -> %s", url, replacement,
        )
        return f"{replacement}{trailing}"

    try:
        repaired = _URL_RE.sub(_replace, answer)
    except Exception as exc:
        logger.warning("citation_url_repair failed, keeping answer: %s", exc)
        return answer, _empty_stats()

    return repaired, stats


def _empty_stats() -> dict[str, int]:
    return {
        "checked": 0,
        "repaired": 0,
        "ambiguous": 0,
        "unmatched": 0,
        "visit_only": 0,
    }


__all__ = ["build_url_whitelist", "repair_citation_urls"]
