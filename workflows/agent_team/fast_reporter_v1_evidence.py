"""Review fast-reporter evidence and project canonical citation inputs.

The model selects mechanically gathered candidates by index, so it cannot add
invented URLs; both citation projections share the reviewed title.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from workflows._shared.citation_contract import coerce_citation_url
from workflows.agent_team.prompts_fast_reporter_v1 import (
    REVIEW_SYSTEM_PROMPT,
    build_review_prompt,
)
from workflows.agent_team.report_fast_select import render_candidates

logger = logging.getLogger(__name__)

CallLLM = Callable[[str, str], Awaitable[str]]

QUALITY_VALUES = ("high", "medium", "low")
DEFAULT_QUALITY = "medium"

# One synthetic run: fast_reporter_v1 has no per-sub-agent DAG to preserve,
# and Gate 1 walks every run's nodes into one URL-keyed dict regardless, so
# splitting them would change nothing but the node id prefix.
RUN_ID = "mode_c"

_MAX_TITLE_CHARS = 200

# Trailing whitespace, ASCII truncation dots (".", "..", "..."), the Unicode
# ellipsis ("…"), or a CJK full stop ("。") — in any mix, e.g. "NF- ...".
# Matched only at the end of the string, so an internal ellipsis
# ("Part 1 ... Part 2") is untouched.
_TRAILING_TRUNCATION_RE = re.compile(r"[\s.…。]+$")


def _clean_title(value: str) -> str:
    """Strip a trailing truncation marker a search engine's title may carry.

    Search engines truncate long page titles with an ellipsis before this
    code ever sees them (observed:
    ``'NANOFLARE 1000Z 疾光1000Z NF1000Z NF- ...'``). The renderer then
    appends ``". {url}"`` right after the title
    (``citation_utils.py:125``), so a trailing marker turns into a
    ``".. http"``-shaped run that the verifier's ``reference_titles_sane``
    check flags as a degraded reference. A single trailing period is
    stripped too, for the same reason — appending ``". "`` after it would
    otherwise read as a doubled period.
    """
    return _TRAILING_TRUNCATION_RE.sub("", value)


@dataclass(frozen=True)
class ReviewedNode:
    """One candidate the reviewer kept, ready to project into both shapes.

    ``title`` is what Gate 1 will render (the reviewer's, or the candidate's
    own when the reviewer supplied none). ``page_title`` is always the
    candidate's original, kept so the verifier can check that a real page
    title was never discarded in favour of a claim sentence.
    """

    url: str
    title: str
    page_title: str
    snippet: str
    source_type: str
    quality: str


def _coerce_quality(raw: Any) -> str:
    value = str(raw or "").strip().lower()
    return value if value in QUALITY_VALUES else DEFAULT_QUALITY


def _node_from_candidate(
    candidate: dict[str, Any], *, title: str, quality: str,
) -> ReviewedNode | None:
    url = coerce_citation_url(candidate.get("url"))
    if not url:
        return None
    # page_title is kept verbatim on the dataclass (see class docstring) —
    # only the resolved, rendered title is cleaned of truncation markers.
    page_title = str(candidate.get("title") or "").strip()
    reviewer_title = _clean_title((title or "").strip()[:_MAX_TITLE_CHARS])
    resolved = reviewer_title or _clean_title(page_title)
    source_type = str(candidate.get("source_type") or "search").strip() or "search"
    return ReviewedNode(
        url=url,
        title=resolved,
        page_title=page_title,
        snippet=str(candidate.get("snippet") or "").strip(),
        source_type=source_type,
        quality=quality,
    )


def parse_review_reply(
    raw: str, candidates: list[dict[str, Any]],
) -> tuple[list[ReviewedNode], str]:
    """Parse call 1's reply into nodes. Returns ``(nodes, fallback_reason)``.

    ``fallback_reason`` is ``""`` when at least one node survived. The three
    distinct failure strings let the caller record *why* it degraded rather
    than only *that* it did.

    Never raises. ``AttributeError`` / ``TypeError`` are caught alongside
    ``JSONDecodeError`` because a model that returns ``{"nodes": ["a", "b"]}``
    — a string array where objects were asked for — fails inside the loop,
    not at ``json.loads``.
    """
    text = raw if isinstance(raw, str) else str(raw or "")
    try:
        start = text.find("{")
        end = text.rfind("}") + 1
        if start < 0 or end <= start:
            return [], "unparseable review reply"
        parsed = json.loads(text[start:end])
    except (json.JSONDecodeError, AttributeError, TypeError):
        return [], "unparseable review reply"

    if not isinstance(parsed, dict):
        return [], "unparseable review reply"

    raw_nodes = parsed.get("nodes")
    if not isinstance(raw_nodes, list) or not raw_nodes:
        return [], "empty node list"

    seen: set[int] = set()
    nodes: list[ReviewedNode] = []
    for item in raw_nodes:
        if not isinstance(item, dict):
            continue
        try:
            index = int(item.get("i") or 0)
        except (TypeError, ValueError):
            continue
        if index in seen or not (1 <= index <= len(candidates)):
            continue
        seen.add(index)
        node = _node_from_candidate(
            candidates[index - 1],
            title=str(item.get("title") or ""),
            quality=_coerce_quality(item.get("quality")),
        )
        if node is not None:
            nodes.append(node)

    if not nodes:
        return [], "no valid node indices"
    return nodes, ""


async def review_candidates(
    *,
    question: str,
    native_draft: str,
    candidates: list[dict[str, Any]],
    call_llm: CallLLM,
    cap: int,
) -> tuple[list[ReviewedNode], dict[str, Any]]:
    """Call 1: one LLM pass labelling candidates with a title and a tier.

    Fail-open on every failure mode — LLM error, unparseable reply, empty or
    all-invalid selection — by keeping every candidate at
    ``quality="medium"`` and recording the reason. Never raises: with all
    candidates at one tier, Gate 1 degrades to arrival order plus its cap,
    which is exactly the mechanical adapter this call exists to improve on.
    """
    stats: dict[str, Any] = {
        "candidates_considered": len(candidates),
        "selection_fallback": "",
    }

    def _fallback(reason: str) -> tuple[list[ReviewedNode], dict[str, Any]]:
        stats["selection_fallback"] = reason
        nodes = [
            node
            for node in (
                _node_from_candidate(c, title="", quality=DEFAULT_QUALITY)
                for c in candidates
            )
            if node is not None
        ]
        return nodes, stats

    if not candidates:
        return _fallback("no candidates to review")

    try:
        raw = await call_llm(
            REVIEW_SYSTEM_PROMPT,
            build_review_prompt(
                question=question,
                native_draft=native_draft,
                candidates_block=render_candidates(candidates),
                cap=cap,
            ),
        )
    except Exception as exc:
        logger.warning("review_candidates: call_llm raised: %s", exc)
        return _fallback(f"llm_error: {exc}")

    nodes, reason = parse_review_reply(raw, candidates)
    if reason:
        logger.warning("review_candidates: %s — keeping all candidates", reason)
        return _fallback(reason)
    return nodes, stats


def project_evidence(
    nodes: list[ReviewedNode],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Project reviewed nodes into ``(methods_info, run_snapshots)``.

    Field choices are dictated by ``_citation_grade_core``
    (``citation_contract.py``), which admits a node only when its
    ``source_type`` is ``search`` or ``visit``, its URL survives
    ``coerce_citation_url``, and it has a snippet. Under
    ``provenance_strict`` it additionally requires ``snippet_verbatim`` and a
    ``fetch_status`` other than ``"failed"``: ``fetch_status`` is set here
    truthfully to ``"ok"`` (the candidate gate already required a tool
    result that returned). ``snippet_verbatim`` is NOT uniformly true —
    it is derived from ``source_type``, because only ``search`` cards carry
    a snippet lifted verbatim out of the raw search result
    (``evidence_observer.py``); ``visit`` (``web_fetch``) cards carry
    a line pulled from an LLM-generated summary, which is a paraphrase, not
    a quote. This chain never passes ``provenance_strict`` today, so this
    has no behavioural effect yet, but the field must not lie — otherwise
    turning strict mode on later would silently admit ``visit`` paraphrases
    as if they were verbatim quotes.

    ``claim`` mirrors the snippet in the snapshot shape. That is the field
    Gate 1b's builder falls back to when ``metadata.title`` is absent, and
    it never is here — but leaving it empty would make the fallback path
    render an empty label if it ever fired.
    """
    dag_nodes: dict[str, Any] = {}
    evidence: dict[str, Any] = {}
    for i, node in enumerate(nodes, start=1):
        node_id = f"{RUN_ID}::E{i}"
        common = {
            "id": node_id,
            "snippet": node.snippet,
            "source_type": node.source_type,
            "fetch_status": "ok",
            "snippet_verbatim": node.source_type == "search",
            "source_quality": node.quality,
            "metadata": {"title": node.title},
        }
        dag_nodes[node_id] = {**common, "type": "evidence", "url": node.url}
        evidence[node_id] = {**common, "source_url": node.url, "claim": node.snippet}

    return (
        {RUN_ID: {"dag": {"nodes": dag_nodes}}},
        {RUN_ID: {"snapshot": {"evidence": evidence}}},
    )


def join_reference_rows(
    references: list[dict[str, str]],
    nodes: list[ReviewedNode],
    *,
    tag_citations: bool,
) -> list[dict[str, str]]:
    """Re-attach snippets to the Gate-1 whitelist, in whitelist order.

    Gate 1 returns ``{title, url}`` only, but two consumers need more: the
    writer prompt needs each source's snippet to match claims against, and
    offline tooling needs it for scoring. This joins the whitelist back onto
    the nodes it came from, by URL.

    ``tag_citations=True`` appends ``→ cite as [N]`` to each snippet, where
    ``N`` is the row's 1-based whitelist position. That makes Gate 2's
    contract text literally true — it promises the writer that every
    evidence node carries its index — so the writer copies the number
    instead of guessing it from titles.

    Row order is the whitelist's (quality-ranked), never the nodes'.
    """
    by_url = {}
    for node in nodes:
        by_url.setdefault(node.url, node)

    rows: list[dict[str, str]] = []
    for index, ref in enumerate(references, start=1):
        url = coerce_citation_url(ref.get("url", ""))
        node = by_url.get(url)
        snippet = node.snippet if node else ""
        if tag_citations:
            snippet = f"{snippet}\n→ cite as [{index}]".strip()
        rows.append({
            "title": str(ref.get("title") or ""),
            "url": url,
            "snippet": snippet,
            "page_title": node.page_title if node else "",
            "quality": node.quality if node else DEFAULT_QUALITY,
            "source_type": node.source_type if node else "search",
        })
    return rows


__all__ = [
    "DEFAULT_QUALITY",
    "QUALITY_VALUES",
    "RUN_ID",
    "ReviewedNode",
    "join_reference_rows",
    "parse_review_reply",
    "project_evidence",
    "review_candidates",
]
