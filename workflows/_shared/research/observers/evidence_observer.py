"""EvidenceObserver — passive observer that extracts EvidenceCards from tool results.

Handles web_search, web_fetch, bash, and write_file results. Extracted cards are
accumulated in TurnContext.metadata["evidence_cards"] across turns.
"""
from __future__ import annotations

import logging
import re

from frontier_agent.core.loop_types import (
    AgentLoopResult,
    Intervention,
    LoopConfig,
    ToolResult,
    TurnContext,
)
from workflows._shared.research.evidence import new_evidence_id

logger = logging.getLogger(__name__)

# Shared regex for extracting generated image paths from tool output.
# Used by agent.py for output artifact detection.
IMAGE_PATH_RE = re.compile(
    r'((?:/(?:tmp|private/tmp|mnt/agent|mnt/frontier_agent)/)'
    r'[^\s"\']+\.(?:png|jpg|jpeg|webp|gif|svg))',
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Domain authority scoring table
# ---------------------------------------------------------------------------

_AUTHORITY_DOMAINS: dict[str, float] = {
    ".gov": 0.12,
    ".edu": 0.10,
    ".org": 0.06,
    "nature.com": 0.10,
    "science.org": 0.10,
    "arxiv.org": 0.08,
    "pubmed.ncbi": 0.08,
    "ieee.org": 0.08,
    "acm.org": 0.08,
    "reuters.com": 0.07,
    "bbc.com": 0.06,
    "nytimes.com": 0.06,
    "github.com": 0.05,
    "stackoverflow.com": 0.05,
    "wikipedia.org": 0.04,
}


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def score_relevance(
    query: str,
    claim: str,
    url: str,
    position: int,
    total_results: int,
) -> float:
    """Score evidence relevance in [0.0, 1.0].

    Components:
      - position (0.35): linear decay 1.0 → 0.3 across results
      - query-claim keyword overlap (0.35)
      - domain authority (0.15)
      - snippet quality (0.15)
    """
    # Coerce text inputs — LLMs sometimes pass list/dict args (e.g. web_fetch
    # called with ``url=["a", "b"]``). re.findall would TypeError on non-str.
    if not isinstance(query, str):
        query = str(query) if query is not None else ""
    if not isinstance(claim, str):
        claim = str(claim) if claim is not None else ""
    if not isinstance(url, str):
        url = str(url) if url is not None else ""

    # Position score: linear decay 1.0 to 0.3
    pos_score = 1.0 - 0.7 * (position / (total_results - 1)) if total_results > 1 else 1.0

    # Query-claim keyword overlap (supports CJK + Latin words ≥ 2 chars)
    q_words = set(w.lower() for w in re.findall(r"[\u4e00-\u9fff]+|[a-zA-Z]{2,}", query))
    c_words = set(w.lower() for w in re.findall(r"[\u4e00-\u9fff]+|[a-zA-Z]{2,}", claim))
    overlap_score = len(q_words & c_words) / len(q_words) if q_words else 0.5

    # Domain authority
    domain_score = 0.0
    url_lower = url.lower()
    for domain, boost in _AUTHORITY_DOMAINS.items():
        if domain in url_lower:
            domain_score = boost
            break
    domain_score = min(1.0, domain_score / 0.12)

    # Snippet quality based on claim length
    claim_len = len(claim.strip())
    if claim_len < 20:
        quality_score = 0.2
    elif claim_len < 50:
        quality_score = 0.5
    elif claim_len <= 300:
        quality_score = 0.8 + 0.2 * min(1.0, claim_len / 300)
    else:
        quality_score = 0.9

    score = (
        0.35 * pos_score
        + 0.35 * overlap_score
        + 0.15 * domain_score
        + 0.15 * quality_score
    )
    return round(max(0.1, min(1.0, score)), 3)


# ---------------------------------------------------------------------------
# URL / title structural helpers
# ---------------------------------------------------------------------------


def _coerce_url_arg(raw: object) -> str:
    """Coerce a ``tool_args["url"]`` value into one canonical URL string.

    Handles every shape the LLM has been observed to emit:

    * native ``list`` / ``tuple`` of URLs
    * a single URL string
    * a JSON-array literal *string* like ``'["https://a", "https://b"]'``
      (the LLM JSON-encodes the list before placing it in the slot)
    * ``None`` or other types

    Returns the first non-empty ``http(s)://`` URL it finds, or ``""``.
    Always returns a plain string — never a list literal, never a JSON
    blob — so downstream consumers (dedup keys, ``source.url``) can rely
    on the field being a single URL.
    """
    if raw is None:
        return ""
    if isinstance(raw, (list, tuple)):
        for item in raw:
            if isinstance(item, str) and item.strip().startswith(("http://", "https://")):
                return item.strip()
        return ""
    s = str(raw).strip()
    if s.startswith("[") and s.endswith("]"):
        # JSON-array-literal-as-string. Parse if possible; otherwise
        # regex-extract the first http(s) URL — many LLMs miss a quote
        # or comma but the URLs themselves are still visible.
        import json
        try:
            parsed = json.loads(s)
        except (json.JSONDecodeError, ValueError):
            parsed = None
        if isinstance(parsed, list):
            for item in parsed:
                if isinstance(item, str) and item.strip().startswith(("http://", "https://")):
                    return item.strip()
        m = re.search(r"https?://[^\s\"',\]]+", s)
        if m:
            return m.group(0)
        return ""
    if s.startswith(("http://", "https://")):
        return s
    return ""


def _title_from_url(url: str) -> str:
    """Derive a human-readable title from a URL alone.

    For web_fetch evidence cards the raw page title is no longer
    available by the time the tool result reaches us (summary_llm
    replaced the Jina markdown with a free-form summary), so scavenging
    the result body for a title is a guaranteed leak surface — that is
    how LLM ``<think>`` content ended up as ``[27] Hmm, the user wants
    specific financial information extracted from the prov.`` in the
    References block.

    This helper sidesteps the leak entirely: title comes from the URL,
    nothing else. Strategy: pick the last meaningful path segment, swap
    dashes / underscores for spaces, drop trailing extensions, and fall
    back to the hostname when the path is empty / numeric / opaque.
    """
    if not url:
        return ""
    try:
        from urllib.parse import urlparse
        parsed = urlparse(url)
    except (ValueError, TypeError):
        return url

    host = (parsed.netloc or "").lstrip(".").removeprefix("www.")
    segments = [s for s in (parsed.path or "").split("/") if s]
    # Walk segments from the end, pick the first that looks "slug-shaped"
    # (letters present, longer than a short numeric id). Skips trailing
    # ``index.html`` / ``amp`` / pure numeric ids / single tokens.
    for seg in reversed(segments):
        clean = re.sub(r"\.(?:html?|php|aspx?|md|pdf|amp)$", "", seg, flags=re.IGNORECASE)
        if len(clean) < 6:
            continue
        if not re.search(r"[A-Za-z]", clean):
            continue
        words = re.split(r"[-_]+", clean)
        if len(words) < 2 and len(clean) < 12:
            continue
        title_words = [w[:1].upper() + w[1:] for w in words if w]
        slug_title = " ".join(title_words)[:120]
        if host:
            return f"{slug_title} — {host}"
        return slug_title
    return host or url


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------


def extract_evidence_cards(
    tool_name: str,
    tool_args: dict,
    result: str,
    turn: int,
    agent: str = "",
) -> list[dict]:
    """Extract evidence cards from a tool result.

    Returns a list of evidence card dicts (not yet wrapped in EvidenceCard
    dataclass so callers can store them as plain JSON-serialisable dicts).
    """
    cards: list[dict] = []

    # Tool results occasionally arrive as non-strings (LangChain ToolMessage
    # content can be a list of structured blocks for vision / Anthropic
    # tool_use roundtrips). Downstream regex / .startswith / .split call
    # sites would TypeError; coerce once here.
    if not isinstance(result, str):
        result = str(result) if result is not None else ""
    if not isinstance(tool_args, dict):
        tool_args = {}

    if tool_name == "web_search":
        query = tool_args.get("query", "")

        # Parse format: [N] **title**\nsnippet\nURL: link
        result_pattern = re.compile(r"\[(\d+)\]\s+\*\*(.+?)\*\*\n(.*?)\nURL:\s*(\S+)")
        matches = list(result_pattern.finditer(result))
        total = len(matches)

        for i, m in enumerate(matches):
            title = m.group(2).strip()
            snippet = m.group(3).strip()
            url = m.group(4).strip()
            if title or url:
                cards.append(
                    {
                        "id": new_evidence_id(),
                        "claim": snippet or title,
                        "source": {"title": title, "url": url},
                        "relevance_score": score_relevance(
                            query, snippet or title, url, i, total
                        ),
                        "query": query,
                        "supporting_quotes": [snippet[:300]] if snippet else [],
                        # Search snippets are verbatim excerpts from the tool's
                        # result block. Downstream citation projection uses
                        # this provenance to distinguish them from web_fetch's
                        # LLM-generated visit summaries.
                        "source_type": "search",
                    }
                )

        # Direct Answer section (prepend so it appears first)
        answer_match = re.search(
            r"## Direct Answer\n\*\*(.+?)\*\*\n(.+?)(?=\n\n---|\Z)",
            result,
            re.DOTALL,
        )
        if answer_match:
            cards.insert(
                0,
                {
                    "id": new_evidence_id(),
                    "claim": answer_match.group(2).strip()[:300],
                    "source": {"title": answer_match.group(1).strip(), "url": ""},
                    "relevance_score": 0.9,
                    "query": query,
                    "supporting_quotes": [answer_match.group(2).strip()[:300]],
                    "source_type": "search",
                },
            )

    elif tool_name == "web_fetch":
        # ``tool_args["url"]`` is observed in the wild as: a single URL
        # string, a Python list of URLs, *or* a JSON-array literal string
        # the LLM produced when it tried to encode a list. ``_coerce_url_arg``
        # collapses all three shapes into one canonical URL string so we
        # never store a list / JSON blob in ``source.url``.
        url = _coerce_url_arg(tool_args.get("url"))

        if result.startswith("Could not extract") or result.startswith("Error"):
            return cards
        if not url:
            return cards

        # ``web_fetch`` results are ``[N] URL: u\n    Info: <summary_llm
        # output>`` — there is no raw page title to scrape from the body
        # anymore, so we derive the title purely from the URL. This kills
        # the entire "body line → title" leak surface (LLM reasoning,
        # error preambles, summary_llm self-talk) at the source instead
        # of trying to blacklist every possible reasoning shape after
        # the fact.
        title = _title_from_url(url)

        # First substantial paragraph as claim
        claim = ""
        for line in result.split("\n"):
            line = line.strip()
            if (
                len(line) > 50
                and not line.startswith("#")
                and not line.startswith("[")
                and not line.startswith("-")
            ):
                claim = line[:300]
                break

        cards.append(
            {
                "id": new_evidence_id(),
                "claim": claim or result[:200],
                "source": {"title": title, "url": url},
                "relevance_score": score_relevance(url, claim or result[:200], url, 0, 1),
                "query": url,
                "supporting_quotes": [claim[:500]] if claim else [],
                # ``web_fetch`` returns a summary produced by the fetch
                # pipeline, not a verbatim page quote.
                "source_type": "visit",
            }
        )

    # bash / write_file: surface generated file paths (charts, images, HTML)
    if tool_name in ("bash", "write_file"):
        file_paths = re.findall(
            r"/tmp/[\w/.\-]+\.(?:png|jpg|jpeg|svg|gif|html)", result
        )
        for path in file_paths:
            ext = path.rsplit(".", 1)[-1].lower()
            cards.append(
                {
                    "id": new_evidence_id(),
                    "claim": f"Generated file: {path}",
                    "source": {"title": f"{tool_name}: {str(tool_args)[:80]}", "url": ""},
                    "relevance_score": 0.9,
                    "query": str(tool_args)[:100],
                    "supporting_quotes": [result[:300]],
                    "file_path": path,
                    "is_chart": ext in ("png", "jpg", "jpeg", "svg", "gif"),
                }
            )

    # finance sandbox: parse CITATIONS blocks the LLM is instructed to emit
    # (see plugins/tools/finance_tools.py::_FINANCE_QUICK_REF) and surface
    # generated dataset/chart files. Without this, fiscal.ai / EODHD fetches
    # never reach the notebook → assertion → report pipeline.
    if tool_name in (
        "run_python_code_in_finance_sandbox",
        "run_command_in_finance_sandbox",
    ):
        cards.extend(_extract_finance_cards(tool_name, tool_args, result))

    for c in cards:
        c.setdefault("turn", turn)
        c.setdefault("agent", agent)

    return cards


# ---------------------------------------------------------------------------
# Finance sandbox extractor
# ---------------------------------------------------------------------------


# Match a "CITATIONS:" header followed by one or more "- ..." bullet lines.
# The block ends at the first blank line or non-bullet line. The literal
# "\n" sequence shows up in some sandbox stringified outputs (Execution(...)
# repr) so we tolerate both real newlines and the escape pair.
_CITATIONS_BLOCK_RE = re.compile(
    r"CITATIONS:\s*(?:\\n|\n)((?:\s*(?:\\n|\n)?\s*-\s+[^\n\\]+(?:\\n|\n)?)+)",
    re.IGNORECASE,
)

# Citation bullet, optionally prefixed with "[N]". The trailing
# "(provider hint)" part is captured separately so we can split it out
# of the claim and use it as the evidence card's source title.
_CITATION_BULLET_RE = re.compile(
    r"^\s*(?:\[\d+\]\s*)?(?P<claim>.+?)\.\s+"
    r"(?P<provider>[A-Za-z][\w. -]*?)"
    r"\s*\((?P<endpoint>[^)]+)\)\s*\.?\s*$"
)

# Files the sandbox generates land here (only allowed write-path under
# the resource sandbox). Match data + chart extensions.
_FINANCE_FILE_RE = re.compile(
        r"/tmp/(?:agent|frontier_agent)-outputs/[\w/.\-]+"
    r"\.(?:csv|parquet|json|jsonl|tsv|xlsx|png|jpg|jpeg|svg|gif|html)"
)


def _extract_finance_cards(
    tool_name: str, tool_args: dict, result: str,
) -> list[dict]:
    cards: list[dict] = []
    if not result or result.startswith("[ERROR]"):
        return cards

    query = str(
        tool_args.get("code_block")
        or tool_args.get("command")
        or ""
    )[:200]

    # 1. CITATIONS bullets — one evidence card per bullet, with the
    #    "(endpoint, retrieved date)" hint preserved as the source title.
    for block in _CITATIONS_BLOCK_RE.finditer(result):
        body = block.group(1)
        # Split on either real or literal-escape newlines, trim "- " prefix.
        bullets = re.split(r"(?:\\n|\n)", body)
        for raw in bullets:
            text = raw.strip()
            if not text.startswith("-"):
                continue
            text = text.lstrip("-").strip()
            if not text:
                continue
            m = _CITATION_BULLET_RE.match(text)
            if m:
                claim = m.group("claim").strip()
                provider = m.group("provider").strip()
                endpoint = m.group("endpoint").strip()
                title = f"{provider} ({endpoint})"
            else:
                claim = text
                title = "Finance sandbox"
            cards.append(
                {
                    "id": new_evidence_id(),
                    "claim": claim[:300],
                    "source": {"title": title, "url": ""},
                    "relevance_score": 0.85,
                    "query": query,
                    "supporting_quotes": [text[:500]],
                }
            )

    # 2. Generated dataset / chart files — surface the path so the chart
    #    materializer (report.py::_materialize_charts) can pick them up
    #    and so the notebook has a concrete artifact to anchor facts to.
    seen_paths: set[str] = set()
    for path in _FINANCE_FILE_RE.findall(result):
        if path in seen_paths:
            continue
        seen_paths.add(path)
        ext = path.rsplit(".", 1)[-1].lower()
        cards.append(
            {
                "id": new_evidence_id(),
                "claim": f"Generated file: {path}",
                "source": {"title": f"{tool_name}: {query[:80]}", "url": ""},
                "relevance_score": 0.9,
                "query": query[:100],
                "supporting_quotes": [],
                "file_path": path,
                "is_chart": ext in ("png", "jpg", "jpeg", "svg", "gif"),
            }
        )

    return cards


# ---------------------------------------------------------------------------
# Observer
# ---------------------------------------------------------------------------


class EvidenceObserver:
    """Observer that accumulates evidence cards in TurnContext.metadata.

    Evidence cards are stored under the key ``"evidence_cards"`` as a list of
    plain dicts (JSON-serialisable).

    critical=True because callers read metadata["evidence_cards"] from the
    result — writes must complete before run_agent_loop returns.
    """

    critical: bool = True

    async def on_tool_result(self, ctx: TurnContext, result: ToolResult) -> None:
        if result.is_error:
            return
        cards = extract_evidence_cards(
            result.name, result.args, result.result, ctx.turn, ctx.role_id,
        )
        if cards:
            existing = ctx.metadata.setdefault("evidence_cards", [])
            existing.extend(cards)
        elif result.name in ("web_search", "web_fetch"):
            # Silent zero-card extraction on a search/scrape tool almost
            # always means the result format drifted (Serper added a new
            # section, Jina returned HTML, etc.) and the regex missed.
            # Log the first 200 chars so operators can spot format drift
            # and fix the regex before accuracy degrades.
            raw = result.result or ""
            if not isinstance(raw, str):
                raw = str(raw)
            preview = raw[:200].replace("\n", " ")
            logger.warning(
                "[EvidenceObserver] %s returned 0 cards — preview: %r",
                result.name, preview,
            )

    # No-op hooks required by the observer base. Previously carried
    # type: ignore[override] because the signatures did not actually match
    # BaseObserver; they now do, so no suppression is needed.
    async def on_loop_start(self, config: LoopConfig) -> None:
        pass

    async def on_llm_response(self, ctx: TurnContext) -> Intervention | None:
        return None

    async def on_turn_end(self, ctx: TurnContext) -> Intervention | None:
        return None

    async def on_loop_end(self, result: AgentLoopResult) -> None:
        pass
