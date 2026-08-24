"""Candidate preparation for fast_reporter_v1's evidence-review call."""

from __future__ import annotations

from typing import Any

from workflows._shared.cited_report_finalizer import (
    coerce_title,
    coerce_url,
)

# Bound the candidate set so the review call retains room for its response.
CANDIDATE_TOKEN_BUDGET = 80_000


def card_snippet(card: dict[str, Any]) -> str:
    """Best available verbatim text for a card.

    Prefers ``supporting_quotes[0]`` (the search snippet as the source wrote
    it) over ``claim``, because the prompt tells the writer to match claims
    against snippets, and a quote is closer to what the URL actually says.
    """
    quotes = card.get("supporting_quotes")
    if isinstance(quotes, list):
        for quote in quotes:
            text = str(quote or "").strip()
            if text:
                return text
    return str(card.get("claim") or "").strip()


def _dedup_by_url_keep_longer_snippet(
    cards: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Keep one card per distinct URL: whichever has the LONGER snippet.

    Tiebreaks on "more informative text" rather than arrival order, since the
    review call is shown only the snippet text. Preserves first-seen order
    among the survivors.
    """
    by_url: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for card in cards:
        if not isinstance(card, dict):
            continue
        raw_source = card.get("source")
        source = raw_source if isinstance(raw_source, dict) else {}
        url = coerce_url(source.get("url"))
        if not url:
            continue
        snippet = card_snippet(card)
        if not snippet:
            continue
        existing = by_url.get(url)
        if existing is None:
            by_url[url] = card
            order.append(url)
        elif len(snippet) > len(card_snippet(existing)):
            by_url[url] = card
    return [by_url[url] for url in order]


def prepare_candidates(
    cards: list[dict[str, Any]],
    *,
    budget: int = CANDIDATE_TOKEN_BUDGET,
) -> list[dict]:
    """Gate -> dedup(longer snippet) -> token-budget-truncate. No scoring.

    Admission requires a URL and non-empty :func:`card_snippet` text.
    Candidates are kept in evidence-arrival order and accumulated until the
    estimated token cost (``len(text) // 3``) of the text
    :func:`render_candidates` will actually emit for them so far —
    ``"[i] {title} | {url}\\n{snippet}"`` blocks joined by ``"\\n\\n"`` —
    would exceed ``budget``, at which point the remainder is truncated.

    Each returned candidate carries ``card`` — a reference to the original
    evidence-card object, not a copy — so a caller needing the original
    ``claim``/nested ``source`` can read it back off ``card`` directly.
    """
    if not isinstance(cards, list):
        return []

    admitted = _dedup_by_url_keep_longer_snippet(cards)

    candidates: list[dict] = []
    used_chars = 0
    for card in admitted:
        raw_source = card.get("source")
        source = raw_source if isinstance(raw_source, dict) else {}
        url = coerce_url(source.get("url"))
        snippet = card_snippet(card)
        title = coerce_title(source.get("title"), url)
        source_type = str(card.get("source_type") or "search").strip() or "search"

        index = len(candidates) + 1
        block = f"[{index}] {title} | {url}\n{snippet}"
        # "\n\n" separator only applies between blocks, i.e. once per
        # candidate after the first.
        added_chars = len(block) + (2 if candidates else 0)
        if (used_chars + added_chars) // 3 > budget:
            break
        used_chars += added_chars

        candidates.append({
            "url": url,
            "title": title,
            "snippet": snippet,
            "source_type": source_type,
            "card": card,
        })
    return candidates


def render_candidates(cands: list[dict]) -> str:
    """Render the numbered candidate block the review prompt shows."""
    blocks = []
    for i, cand in enumerate(cands, 1):
        blocks.append(
            f"[{i}] {cand.get('title', '')} | {cand.get('url', '')}\n"
            f"{cand.get('snippet', '')}",
        )
    return "\n\n".join(blocks)


__all__ = [
    "CANDIDATE_TOKEN_BUDGET",
    "card_snippet",
    "prepare_candidates",
    "render_candidates",
]
