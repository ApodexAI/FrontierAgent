"""Shared helpers for deterministic inline-citation report finalization."""

from __future__ import annotations

import asyncio
import logging
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

from frontier_agent.core.messages import system_msg, text_of, user_msg
from workflows._shared.citation_utils import (
    _format_references_section,
    finalize_report,
)

logger = logging.getLogger(__name__)


Reference = Mapping[str, str]


class CitationValidationError(ValueError):
    """Raised when a report body cannot be finalized with valid citations."""

    def __init__(self, issues: Sequence[str]) -> None:
        self.issues = list(issues)
        super().__init__("; ".join(self.issues))


_CITE_GROUP_RE = re.compile(r"\[(\s*\d[\d,;\s]*)\]")

# A citation index written with trailing descriptive junk inside the
# bracket — ``[2的相关背景]`` / ``[2 - see above]`` — that ``_CITE_GROUP_RE``
# does NOT recognize (so it survives as literal text and leaks into the
# rendered report). The leading 1-3 digits are a plausible citation index;
# the char right after must be a NON-digit, NON-separator, NON-bracket so
# that multi-cites (``[2, 3]`` / ``[2; 3]``) and 4-digit years
# (``[2024年]``) are left untouched. Everything up to the close bracket is
# then dropped, normalizing to ``[N]``.
_MALFORMED_CITE_RE = re.compile(r"\[\s*(\d{1,3})\s*[^\d,;\s\]][^\]]*\]")


def normalize_malformed_citations(text: str) -> tuple[str, int]:
    """Rewrite ``[<idx><trailing text>]`` → ``[<idx>]``. Returns ``(text, n)``.

    Belt-and-suspenders for the citation contract: the LLM occasionally
    writes a descriptive citation (``[2的相关背景]``) instead of the bare
    ``[2]``. Left as-is it is neither renumbered nor validated — it just
    leaks. This collapses it to the canonical form so the downstream
    renumber/validate sees a real ``[N]``.
    """
    if not text:
        return text, 0
    count = 0

    def _sub(m: re.Match) -> str:
        nonlocal count
        count += 1
        return f"[{m.group(1)}]"

    return _MALFORMED_CITE_RE.sub(_sub, text), count


# A markdown footnote reference (``[^idc_infra]`` / ``[^gs_dc]``) the LLM
# minted for a claim whose source isn't in the citation whitelist. The
# Deterministic Citation Contract mandates numeric ``[N]`` markers only and
# tells the LLM to keep an un-sourced claim as analysis *without* a citation;
# a footnote-style marker is off-contract. Worse, the LLM never writes the
# matching ``[^id]: …`` definition, so the marker has nothing to resolve to
# and leaks into the rendered report as literal ``[^id]`` text. The label is
# ``^`` + the usual footnote id charset (letters, digits, ``_``, ``-``); the
# definition form is the same token followed by ``:``.
# The optional single leading space is consumed with the marker so a
# space-separated ``word [^x] more`` collapses cleanly to ``word more``
# instead of leaving a double space; CJK text (``亿[^x]，``) has no such
# space, so nothing is over-consumed there. Only ONE space is eaten, so
# line-leading indentation is never swallowed.
_FOOTNOTE_REF_RE = re.compile(r" ?\[\^([A-Za-z0-9_-]+)\]")
_FOOTNOTE_DEF_LINE_RE = re.compile(r"(?m)^[ \t]*\[\^([A-Za-z0-9_-]+)\]:.*(?:\n|$)")


def strip_orphan_footnote_citations(text: str) -> tuple[str, int]:
    """Strip off-contract markdown footnote citations. Returns ``(text, n)``.

    Belt-and-suspenders for the Deterministic Citation Contract, which pins
    the reporter to numeric ``[N]`` markers and a system-appended References
    block. The LLM occasionally mints a markdown footnote marker
    (``$7,580亿[^idc_infra]``) for a claim whose source isn't in the
    whitelist, but never writes the ``[^idc_infra]: …`` definition. With no
    definition the marker can't resolve, the contract's ``[N]`` renumber
    ignores it, and it leaks into the final report as literal ``[^idc_infra]``
    text — a broken reference.

    This removes every footnote reference whose definition is absent, plus
    any orphan definition lines, leaving the surrounding claim intact as
    analysis (exactly what the contract prescribes for un-sourced claims). A
    footnote that *does* have a matching definition is left untouched — that
    is a well-formed footnote the renderer can resolve, so removing it would
    be lossy; the contract simply shouldn't produce one.
    """
    if not text:
        return text, 0
    defined = {m.group(1) for m in _FOOTNOTE_DEF_LINE_RE.finditer(text)}
    count = 0

    def _sub(m: re.Match) -> str:
        nonlocal count
        if m.group(1) in defined:
            return m.group(0)
        count += 1
        return ""

    return _FOOTNOTE_REF_RE.sub(_sub, text), count


# A bullet list whose items got flattened onto one line — the LLM emits
# ``...句子。 - **下一条**：...`` with no newline before the second/third
# ``- ``, so only the first renders as a list item and the rest collapse
# into the preceding item's text. We re-break a ``- **`` bullet that
# directly follows sentence-ending punctuation onto its own line. The
# trailing ``**`` requirement (a bold lead-in, which every flattened
# bullet here uses) keeps this off ordinary in-prose hyphens / em-dashes.
_INLINE_BULLET_RE = re.compile(r"(?<=[。！？.!?])[ \t]+-[ \t]+(?=\*\*)")


def normalize_inline_bullets(text: str) -> tuple[str, int]:
    """Re-break a ``- **`` bullet flattened onto the prior line. ``(text, n)``.

    Markdown only renders ``- `` as a list item at line start. When the
    LLM drops the newline (``...这一时间框架。 - **每次最多3人共享**``) the
    second and later items collapse into the first item's body. This puts
    each such bullet back on its own line so the list renders.
    """
    if not text:
        return text, 0
    n = len(_INLINE_BULLET_RE.findall(text))
    return _INLINE_BULLET_RE.sub("\n- ", text), n
_TRAILING_REFS_RE = re.compile(
    r"(?im)^(?:#{2,4}\s*)?(References|参考文献|参考资料|Sources|来源)\s*$[\s\S]*\Z",
)
_URL_SCHEMES = ("http://", "https://")

# web_fetch's tool output uses ``[N] URL: …\n    Info: …`` — when the
# regex-driven title fallback in ``extract_evidence_cards`` picks up the
# ``Info:`` line we'd otherwise emit ``[N] Info: # …`` in References.
_INFO_TITLE_PREFIX_RE = re.compile(r"^Info:\s*#?\s*", re.IGNORECASE)

# Structural defence-in-depth for titles. ``EvidenceObserver`` now
# derives web_fetch titles from the URL alone (no body scavenging), but
# third-party tool plugins or future tools may still hand us free-form
# strings. Rather than blacklist every reasoning-preamble shape we've
# seen (``Hmm,`` / ``Let me`` / ``我需要`` / future-LLM variants), we
# reject titles whose *shape* says "this is not a title":
#
# * longer than 200 chars (real titles are short; reasoning is paragraphs)
# * contains a newline (titles are single-line)
# * looks like a JSON literal (``[...]`` or ``{...}`` at both ends)
#
# Triggering any of these falls back to the URL — which is always safe.
_TITLE_MAX_LEN = 200


def _looks_like_non_title(title: str) -> bool:
    if not title:
        return True
    if len(title) > _TITLE_MAX_LEN:
        return True
    if "\n" in title or "\r" in title:
        return True
    stripped = title.strip()
    return bool((stripped.startswith("[") and stripped.endswith("]")) or (stripped.startswith("{") and stripped.endswith("}")))


def coerce_url(value: Any) -> str:
    """Coerce ``source.url`` to a single canonical URL string.

    Inputs seen in the wild (in priority order):

    * plain string ``"https://..."``
    * Python list / tuple of URLs (LLM batched ``web_fetch``)
    * a JSON-array literal *string* like ``'["https://a", "https://b"]'``
      (the LLM JSON-encoded the list before placing it in the slot — the
      tool extractor stored the raw string verbatim)

    Returns the first non-empty ``http(s)://`` URL, or ``""``. Never
    returns a list literal or JSON blob — the rendered References block
    can rely on getting a real URL string.
    """
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        for item in value:
            if isinstance(item, str):
                stripped = item.strip()
                if stripped.startswith(_URL_SCHEMES):
                    return stripped
        return ""
    if not isinstance(value, str):
        return ""
    s = value.strip()
    if s.startswith("[") and s.endswith("]"):
        # JSON-array-literal-as-string. Try JSON first; fall back to
        # regex extraction (LLMs sometimes miss a quote / comma).
        import json
        try:
            parsed = json.loads(s)
        except (json.JSONDecodeError, ValueError):
            parsed = None
        if isinstance(parsed, list):
            for item in parsed:
                if isinstance(item, str) and item.strip().startswith(_URL_SCHEMES):
                    return item.strip()
        m = re.search(r"https?://[^\s\"',\]]+", s)
        if m:
            return m.group(0)
        return ""
    # Plaintext path: only accept real ``http(s)://`` URLs. Upstream
    # extractors occasionally place a human label ("Oracle earnings
    # release") in ``source_url`` when the LLM hallucinates a citation
    # without a real source; returning the label here makes the rendered
    # References row "[N] Title. Title" because the renderer falls back
    # to URL for missing title. Reject non-URL strings so the upstream
    # ``if not url: continue`` guard drops the entry instead.
    if s.startswith(_URL_SCHEMES):
        return s
    return ""


def coerce_title(value: Any, fallback_url: str) -> str:
    """Clean a ``source.title`` field for the References block.

    Strips the ``Info:`` prefix that legacy extractors used to leave on
    web_fetch titles, then applies a structural sanity check: anything
    that does not look title-shaped (too long, multi-line, JSON-shaped)
    is replaced with ``fallback_url``. This protects against any
    upstream source — third-party tool plugins, custom observers, future
    extractors — without depending on a blacklist of "known bad" phrases.
    """
    if isinstance(value, (list, tuple)):
        value = next(
            (v for v in value if isinstance(v, str) and v.strip()),
            "",
        )
    if not isinstance(value, str):
        value = str(value) if value is not None else ""
    title = _INFO_TITLE_PREFIX_RE.sub("", value).strip()
    if _looks_like_non_title(title):
        return fallback_url
    return title


def citation_numbers(text: str) -> list[int]:
    """Extract numeric citation markers from ``[N]`` and ``[1, 2]`` forms."""
    nums: list[int] = []
    for match in _CITE_GROUP_RE.finditer(text or ""):
        nums.extend(int(n) for n in re.findall(r"\d+", match.group(1)))
    return nums


def has_any_citation(text: str) -> bool:
    return bool(citation_numbers(text))


def strip_trailing_references(text: str) -> str:
    """Drop a trailing References/Sources block, if it is plausibly a footer.

    The match must begin after the first 30% of the body. This avoids deleting
    an unusual but legitimate mid-body section named "References".
    """
    if not text:
        return ""
    match = _TRAILING_REFS_RE.search(text)
    if match is None:
        return text
    if match.start() < len(text) * 0.3:
        return text
    return text[: match.start()].rstrip()


def build_citation_maps(
    references: Sequence[Reference],
) -> tuple[dict[str, str], dict[str, str]]:
    """Build ``{N -> url}`` / ``{N -> title}`` from references.

    URL-less entries are silently skipped so the visible ``[N]`` numbering
    matches the *position* in ``references``, not the position after
    compaction. Heavy_reporter / swarm_reporter already pre-filter empty
    URLs upstream, so in practice ``citation_map`` keys are dense
    ``"1".."N"``; the gap-tolerant skip is defence-in-depth — callers
    that hand us a raw list with holes won't silently produce a body
    citing ``[2]`` when reference 2 has no URL (the orphan-citation check
    in :func:`validate_citation_body` catches it instead).
    """
    citation_map: dict[str, str] = {}
    citation_titles: dict[str, str] = {}
    for i, ref in enumerate(references, 1):
        key = str(i)
        url = (ref.get("url") or "").strip()
        title = (ref.get("title") or "").strip()
        if not url:
            continue
        citation_map[key] = url
        if title:
            citation_titles[key] = title
    return citation_map, citation_titles


def render_references_block(references: Sequence[Reference]) -> str:
    lines: list[str] = []
    for i, ref in enumerate(references, 1):
        title = (ref.get("title") or ref.get("url") or "").strip()
        url = (ref.get("url") or "").strip()
        if not url:
            continue
        lines.append(f"[{i}] {title} — {url}")
    return "\n".join(lines)


def format_references_block_plain(
    references: Sequence[Reference],
    *,
    language: str = "auto",
) -> str:
    """Render the canonical ``## References`` block in plain ``[N] title. url``
    style without going through :func:`finalize_report`'s renumber-and-validate
    pipeline.

    Used by the heavy_reporter degraded-finalize fallback paths so users see
    the system's canonical URL list even when validation/repair couldn't make
    the body's ``[N]`` markers consistent. The body's markers may stay
    dangling, but at least the source list is visible — strictly better UX
    than no References at all.
    """
    citation_map, citation_titles = build_citation_maps(references)
    if not citation_map:
        return ""
    return _format_references_section(
        citation_map, citation_titles, language=language, style="plain",
    )


def validate_citation_body(
    body: str,
    *,
    max_ref: int,
    valid_indices: Sequence[int] | None = None,
) -> list[str]:
    """Validate that every inline ``[N]`` marker in the body resolves.

    Two modes:

    * **Range mode** (default): ``[N]`` is valid iff ``1 <= N <= max_ref``.
      Used by callers that hand us a contiguous references list and
      cannot easily compute the membership set.
    * **Membership mode** (``valid_indices`` provided): ``[N]`` is valid
      iff ``N in valid_indices``. Catches "orphan" citations — indices
      that fall in the range but point to a URL-less / dropped reference,
      where range mode would let them silently retarget at renumber.
    """
    nums = citation_numbers(body)
    if not nums:
        return ["no inline [N] citations were emitted"]
    if valid_indices is not None:
        valid = set(valid_indices)
        invalid = sorted({n for n in nums if n not in valid})
        if invalid:
            valid_list = ", ".join(str(n) for n in valid_indices) or "(none)"
            return [
                "orphan citation indices: "
                + ", ".join(str(n) for n in invalid)
                + f" (valid indices are: {valid_list})",
            ]
        return []
    invalid = sorted({n for n in nums if n < 1 or n > max_ref})
    if invalid:
        return [
            "out-of-range citation indices: "
            + ", ".join(str(n) for n in invalid)
            + f" (valid range is 1..{max_ref})",
        ]
    return []


# Parse rows like ``[3] Title — https://example.com``, ``3. Title - URL``,
# ``[3] Title. URL``, ``- [3] Title (URL)`` — the formats reporter LLMs
# tend to emit when they ignore the contract and write their own
# References block. The capture is permissive on the title/URL separator
# (em-dash, ASCII dash, period, colon, "—", "–") and tolerates surrounding
# whitespace / markdown link wrapping.
_REF_ROW_RE = re.compile(
    r"""(?mx)
    ^\s*[-*]?\s*                            # optional bullet
    (?:\[\s*(?P<n1>\d+)\s*\]|(?P<n2>\d+)\.) # [3] or 3.
    \s+
    (?P<rest>.+?)
    \s*$
    """,
)

# Pull URL out of the row body — accepts bare URLs, ``[title](url)`` md
# links, and parenthesised URLs.
_URL_IN_ROW_RE = re.compile(
    r"\((?P<paren>https?://[^\s)]+)\)|(?P<bare>https?://[^\s)\]]+)",
)


def parse_trailing_references(text: str) -> dict[str, dict[str, str]]:
    """Extract ``{N: {title, url}}`` from a trailing References footer.

    Returns ``{}`` when there is no plausible References footer or no rows
    parse cleanly. Used to detect Mode-B silent corruption: when the
    reporter LLM ignores the citation contract and writes its own
    ``## References`` block, body ``[N]`` markers and the LLM's own
    References numbering may bind to URLs different from the system's
    canonical ``report_references[N-1]``. Detection lets the repair pass
    surface the mismatch instead of producing silently-wrong citations.
    """
    if not text:
        return {}
    match = _TRAILING_REFS_RE.search(text)
    if match is None or match.start() < len(text) * 0.3:
        return {}
    footer = text[match.start():]
    out: dict[str, dict[str, str]] = {}
    for row in _REF_ROW_RE.finditer(footer):
        idx = row.group("n1") or row.group("n2")
        if not idx:
            continue
        rest = row.group("rest").strip()
        url_match = _URL_IN_ROW_RE.search(rest)
        url = ""
        title = rest
        if url_match:
            url = url_match.group("paren") or url_match.group("bare") or ""
            # Strip the URL substring + any trailing punctuation/whitespace
            # so the remaining text is the title.
            title = (rest[:url_match.start()] + rest[url_match.end():]).strip()
            title = title.rstrip(".,;:—–-([) ").strip()
            # Markdown-link form ``[Title](url)`` — pull the title out of
            # the brackets if the residue still has them.
            md_link = re.match(r"^\[(?P<t>.+?)\]\s*$", title)
            if md_link:
                title = md_link.group("t").strip()
        out[str(int(idx))] = {"title": title, "url": url}
    return out


def _normalize_url(url: str) -> str:
    """Loose URL canonicalisation for cross-source comparison.

    Strips scheme, ``www.``, trailing slash, and trailing punctuation.
    Used by :func:`detect_url_misalignment` so that ``http://www.x.com/`` ==
    ``https://x.com`` for misalignment detection — we are looking for *clearly
    different* URLs, not bit-for-bit equality.
    """
    if not url:
        return ""
    u = url.strip().rstrip(".,;:)]")
    u = re.sub(r"^https?://", "", u, flags=re.IGNORECASE)
    u = re.sub(r"^www\.", "", u, flags=re.IGNORECASE)
    u = u.rstrip("/").lower()
    return u


def detect_url_misalignment(
    body: str,
    references: Sequence[Reference],
) -> list[str]:
    """Return issues when the body's own References URLs disagree with the
    system's canonical ``report_references`` at the same index.

    No-op when the body has no trailing References footer or when none of
    its parsed rows include a URL. Conservative: only flags indices that
    are present in BOTH lists with a clearly different URL — partial /
    missing rows are tolerated.
    """
    llm_refs = parse_trailing_references(body)
    if not llm_refs:
        return []
    issues: list[str] = []
    for i, ref in enumerate(references, 1):
        sys_url = _normalize_url(ref.get("url") or "")
        if not sys_url:
            continue
        llm_row = llm_refs.get(str(i))
        if not llm_row:
            continue
        llm_url = _normalize_url(llm_row.get("url") or "")
        if not llm_url or llm_url == sys_url:
            continue
        # Substring match in either direction (handles ``en.wikipedia.org/wiki/Foo``
        # vs ``en.wikipedia.org/wiki/Foo?lang=en`` style minor drifts).
        if llm_url in sys_url or sys_url in llm_url:
            continue
        issues.append(
            f"citation [{i}] is misaligned: body's own References lists "
            f"{llm_row.get('url')!r} at index {i}, but the system reference "
            f"[{i}] is {ref.get('url')!r} — body's [{i}] markers will "
            f"silently retarget when the canonical References block is "
            f"appended. Rewrite citations using the indices from "
            f"``get_report_references()``."
        )
    return issues


# Programmatic numeric grounding audit.
#
# Picks up where validate_citation_body / detect_url_misalignment stop:
# those check that ``[N]`` markers resolve and URLs align, but say
# nothing about whether the specific NUMBERS the body cites against
# ``[N]`` actually appear in reference [N]'s snippet. The 2026-05-20
# fabrication-prone smoke (``temp/2026-05-20_phase3_fabrication_smoke/``)
# showed 23% of numeric tokens in the final report were cited but absent
# from any cited snippet — fabrication, citation drift, or unlabelled
# derivations like ``-2.5% from $220.61 → $215.10``.

# Tighter regex than the smoke-analysis script's: matches financial /
# scientific number tokens most prone to fabrication. Excludes bare
# years (``2026``) and section labels (``Q4 FY2026``) because flagging
# those produces noise without catching real fabrications.
#
# Magnitude → multiplier maps for cross-notation form generation.
# When adding a new entry, you do NOT need to change the regex — bare
# branches build their alternation dynamically from this dict, and
# the parser uses longest-prefix matching for ``$``-prefixed tokens.
# Missing a magnitude here just means the token falls back to raw-
# string substring matching instead of getting cross-notation
# equivalents. That's a graceful degradation, not a bug.
#
# ``兆`` is ambiguous (mainland: 10^12 = trillion; Taiwan/HK Cantonese:
# 10^6 = million). We use the mainland convention.
_CN_MAGNITUDE: dict[str, float] = {
    "万亿": 1e12,
    "兆": 1e12,
    "千亿": 1e11,
    "百亿": 1e10,
    "亿": 1e8,
    "千万": 1e7,
    "百万": 1e6,
    "万": 1e4,
    "千": 1e3,
}
# Pre-computed longest-first ordering for greedy prefix matching.
_CN_MAGNITUDE_LONGEST_FIRST: tuple[str, ...] = tuple(
    sorted(_CN_MAGNITUDE.keys(), key=len, reverse=True),
)
_EN_MAGNITUDE_FULL: dict[str, float] = {
    "billion": 1e9, "million": 1e6, "trillion": 1e12,
}
_EN_MAGNITUDE_SHORT: dict[str, float] = {"b": 1e9, "m": 1e6, "t": 1e12}

# Build a magnitude alternation from the dict (longest-first so the
# regex tries ``万亿`` before ``万`` / ``亿`` and doesn't truncate).
_CN_MAGNITUDE_ALT: str = "|".join(_CN_MAGNITUDE_LONGEST_FIRST)

# Numeric-token capture for the audit. Design (Plan B — broad capture
# for ``$``-prefixed tokens + dictionary-anchored capture for bare):
#
# Branch order matters. The specific shapes (bps / %) come BEFORE the
# generic Chinese-suffix branch, otherwise greedy CJK eating would
# swallow ``个基`` from ``88个基点`` and break the bps↔% equivalence.
#
# ``$``-prefixed Chinese-suffix branch (3) is tried BEFORE the English
# branch (4) — otherwise the English branch greedily matches ``$1.04``,
# backtracks to ``$1`` (since ``.`` isn't a word boundary), and loses
# the magnitude. The ``$`` branch deliberately uses a broad ``[一-鿿]
# {1,2}`` capture (graceful fallback in the parser): an unknown
# magnitude still produces a token, just without cross-notation
# equivalents.
#
# Bare branches (5-6) use the **known-magnitude allowlist** instead of
# broad CJK capture. Without ``$`` to anchor the intent, broad CJK
# would catch ``2 年期`` / ``5 月`` / ``Q1 季度`` as numeric tokens —
# they're labels, not value claims. Restricting to dictionary entries
# avoids the false positives at the cost of not catching bare
# unknown-magnitude tokens (users should write the ``$`` prefix
# anyway for currency claims).
_NUMERIC_GROUNDING_RX = re.compile(
    # 1. Basis points (specific — must precede generic CJK).
    r"(?<!\w)\d+(?:\.\d+)?\s*(?:bps|basis\s*points|个基点|基点)"
    # 2. Percentage (English % + Chinese 个百分点).
    r"|(?<!\w)\d+(?:\.\d+)?\s*(?:%|个百分点)"
    # 3. $-prefixed Chinese suffix. Tries the known-magnitude allowlist
    #    first (longest-first, so ``万亿`` beats ``万``); falls back to a
    #    **single** CJK character so unknown magnitudes (``$5京``) still
    #    capture for the parser's graceful-degradation path. Single-char
    #    fallback avoids the previous ``{1,2}`` over-capture that ate
    #    trailing context (``$250 亿下`` → ``$250 亿``).
    r"|\$\s*[\d,]+(?:\.\d+)?\s*(?:(?:" + _CN_MAGNITUDE_ALT + r")|[一-鿿])"
    # 4. $-prefixed with optional English magnitude or none. The
    #    ``(?!\w|\.\d)`` lookahead rejects partial matches like ``$1``
    #    when ``$1.04万亿`` follows (``.0`` blocks the lookahead so
    #    branch 3 catches the full token) while still allowing
    #    sentence-ending ``$100.`` (period not followed by digit).
    r"|\$\s*[\d,]+(?:\.\d+)?\s*(?:billion|million|trillion|B|M|T)?(?!\w|\.\d)"
    # 5. Bare with English magnitude (``5 billion``).
    r"|(?<!\w)[\d,]+(?:\.\d+)?\s*(?:billion|million|trillion)\b"
    # 6. Bare with **known** Chinese magnitude — allowlist from the
    #    ``_CN_MAGNITUDE`` dict, longest-first. New magnitudes added to
    #    the dict become matchable here without any regex edit.
    r"|(?<![\d.])[\d,]+(?:\.\d+)?\s*(?:" + _CN_MAGNITUDE_ALT + r")",
    re.IGNORECASE,
)

# Citation tail must follow within this many chars of the number's end
# for the audit to bind them together. 80 chars covers ``[5]`` / ``[5][7]``
# / ``[5, 7]`` / ``(see [5])`` / a short parenthetical between number and
# cite. Wider windows pick up unrelated citations later in the sentence.
_NUMERIC_CITE_WINDOW = 80

# Derivation markers — when these appear within a small window around
# a numeric token, the value is an explicitly labelled derivation per
# the numeric-grounding prompt's bucket 2 ("Explicitly derived from
# cited inputs, show the math") rather than a fabrication. Common
# shapes seen in reporter output:
#
#   ~$215.10 (2.5% below the current $220.61 [3])
#   $214.38 (2× ATR, ~2.8% below close) [8]
#   ~$207.00 (below 200-day SMA, ~6.0% below close) [9]
#   $215.10 (= $220.61 × 0.975 [3])
#
# Without this rescue, the audit flags every derived figure as suspect
# even though the prompt explicitly allows them — pushing the
# suspect-ratio metric well above the real fabrication rate.
_DERIVATION_SIGNAL_RX = re.compile(
    r"%\s+(?:below|above|from|of)\b"
    r"|×\s*ATR\b"
    r"|=\s*\$\d"
    r"|\b(?:below|above)\s+"
    r"(?:the\s+(?:current\s+)?\$?\d|recent|"
    r"\d+-day|S\d\b|R\d\b|the\s+\d+|\$\d)",
    re.IGNORECASE,
)


@dataclass
class NumericGroundingReport:
    """Per-token verdict from :func:`validate_numeric_grounding`.

    ``grounded`` carries verbatim-matched tokens; ``grounded_derived``
    carries explicitly-labelled derivations the prompt allows
    (``~1.4% below close``, ``2× ATR``, ``= $215.10``); ``suspect``
    carries cited tokens that match neither — the real fabrication
    risk class. Each pair is ``(token, citation_index)``.

    ``uncited`` is just the token text — no citation to point at.
    """

    grounded: list[tuple[str, int]] = field(default_factory=list)
    grounded_derived: list[tuple[str, int]] = field(default_factory=list)
    suspect: list[tuple[str, int]] = field(default_factory=list)
    uncited: list[str] = field(default_factory=list)

    @property
    def total(self) -> int:
        return (
            len(self.grounded) + len(self.grounded_derived)
            + len(self.suspect) + len(self.uncited)
        )

    @property
    def suspect_ratio(self) -> float:
        return len(self.suspect) / max(self.total, 1)


# (Magnitude dicts moved up — they're referenced by the regex above.)


def _fmt_value_strings(v: float) -> set[str]:
    """Common decimal renderings: '5', '5.0', '5.00'-trimmed, '%g',
    and comma-grouped ``5,000`` when integer. Comma form matters for
    grounding thousand-scale claims (``$5千`` ↔ ``$5,000``)."""
    out: set[str] = set()
    if v == int(v):
        out.add(str(int(v)))
        out.add(f"{int(v):,}")
    out.add(f"{v:g}")
    out.add(f"{v:.1f}")
    out.add(f"{v:.2f}".rstrip("0").rstrip("."))
    return out


def _cross_notation_forms(value: float) -> set[str]:
    """Generate string equivalents for a USD value across en/zh notations.

    For a value like ``5_000_000_000`` (5 billion), emits forms like
    ``$5 billion``, ``5 billion``, ``$5B``, ``$5b``, ``$50亿``, ``50亿``,
    ``$50亿美元``, etc. Also emits thousand-scale forms (``$5千`` ↔
    ``$5,000``) so a Chinese token like ``$5千`` grounds against an
    English snippet writing ``$5,000`` and vice versa.

    Filters out absurd magnitudes (a value of 5e9 won't try ``0.005
    trillion`` because the ratio is too unfamiliar in finance prose).
    """
    out: set[str] = set()

    # Raw-integer / comma-grouped form (covers $5,000 ↔ $5千 axis where
    # English notation doesn't use a magnitude word at all).
    if 0 < value < 1e7:
        for s in _fmt_value_strings(value):
            out.add(f"${s}")
            out.add(s)

    # English magnitudes — only emit ranges typical in finance text.
    for divisor, unit, short in (
        (1e12, "trillion", "t"),
        (1e9, "billion", "b"),
        (1e6, "million", "m"),
    ):
        v = value / divisor
        if v < 0.01 or v >= 10_000:
            continue
        for s in _fmt_value_strings(v):
            out.add(f"${s} {unit}")
            out.add(f"{s} {unit}")
            out.add(f"${s}{short}")
            out.add(f"{s}{short}")

    # Chinese magnitudes — covers the full ladder, not just 万亿/亿/万,
    # so a Chinese report writing ``$5千`` (= $5,000) generates the
    # English ``$5,000`` form for grounding.
    for divisor, unit in (
        (1e12, "万亿"),
        (1e11, "千亿"),
        (1e10, "百亿"),
        (1e8, "亿"),
        (1e7, "千万"),
        (1e6, "百万"),
        (1e4, "万"),
        (1e3, "千"),
    ):
        v = value / divisor
        if v < 0.01 or v >= 10_000:
            continue
        for s in _fmt_value_strings(v):
            out.add(f"${s}{unit}")
            out.add(f"{s}{unit}")
            out.add(f"${s} {unit}")
            out.add(f"{s} {unit}")
            out.add(f"{s}{unit}美元")
            out.add(f"${s}{unit}美元")

    return out


@lru_cache(maxsize=256)
def _numeric_equivalent_forms(tok: str) -> frozenset[str]:
    """Equivalent lower-cased string forms a snippet might use for ``tok``.

    Handles three notation axes:
    1. Punctuation: ``$1,234`` ↔ ``$1234`` (thousands-separator)
    2. English magnitude: ``$30.8B`` ↔ ``$30.8 billion`` ↔ ``30.8 billion``
    3. Cross-language magnitude: ``$50亿`` ↔ ``$5 billion`` ↔ ``5,000 million``
       (because the report may be Chinese while the source snippet is
       English, or vice versa)

    Also handles basis points (English ``88 bps`` ↔ Chinese ``88 基点``
    / ``88 个基点`` ↔ ``0.88%``).

    Doesn't try to unify across genuinely different values (``$30.8
    billion`` vs ``30,800 million`` are equivalent — yes; but ``$30
    million`` vs ``$30 billion`` are not).
    """
    raw = tok.lower().strip()
    forms = {raw, raw.replace(",", ""), raw.replace(" ", "")}
    forms.add(raw.replace(",", "").replace(" ", ""))

    # Parser branch order: try the *specific* units (bps, %, English
    # magnitudes) BEFORE the generic Chinese-suffix fallback. Otherwise
    # tokens like ``88 个基点`` get caught by the broad CJK branch with
    # an unrecognised suffix ``个基点`` and lose their bps↔% equivalence.

    # 1. Basis points: 88 bps / 88 个基点 / 88 基点
    m = re.match(
        r"([\d,]+(?:\.\d+)?)\s*(?:bps|basis\s*points|个基点|基点)\s*$",
        raw, re.IGNORECASE,
    )
    if m:
        try:
            n = float(m.group(1).replace(",", ""))
        except ValueError:
            return frozenset(forms)
        for s in _fmt_value_strings(n):
            forms.add(f"{s} bps")
            forms.add(f"{s}bps")
            forms.add(f"{s} basis points")
            forms.add(f"{s} 个基点")
            forms.add(f"{s}个基点")
            forms.add(f"{s} 基点")
            forms.add(f"{s}基点")
        # 88 bps ↔ 0.88%
        pct = n / 100
        for s in _fmt_value_strings(pct):
            forms.add(f"{s}%")
            forms.add(f"{s} %")
        return frozenset(forms)

    # 2. Percentage (English % / Chinese 个百分点): 1% ↔ 100 bps.
    m = re.match(
        r"([\d,]+(?:\.\d+)?)\s*(?:%|个百分点)\s*$",
        raw,
    )
    if m:
        try:
            n = float(m.group(1).replace(",", ""))
        except ValueError:
            return frozenset(forms)
        for s in _fmt_value_strings(n):
            forms.add(f"{s}%")
            forms.add(f"{s} %")
            forms.add(f"{s}个百分点")
            forms.add(f"{s} 个百分点")
        bps = n * 100
        for s in _fmt_value_strings(bps):
            forms.add(f"{s}bps")
            forms.add(f"{s} bps")
            forms.add(f"{s} 基点")
            forms.add(f"{s}基点")
            forms.add(f"{s} 个基点")
            forms.add(f"{s}个基点")
        return frozenset(forms)

    # 3. English full magnitude: $5 billion / 5 billion / $5.0 trillion
    m = re.match(
        r"\$?\s*([\d,]+(?:\.\d+)?)\s*(billion|million|trillion)\s*$",
        raw, re.IGNORECASE,
    )
    if m:
        num_str, unit = m.group(1), m.group(2).lower()
        try:
            n = float(num_str.replace(",", ""))
        except ValueError:
            return frozenset(forms)
        value = n * _EN_MAGNITUDE_FULL[unit]
        forms.update(_cross_notation_forms(value))
        return frozenset(forms)

    # 4. English short magnitude: $5B / 5.0b / 5T
    m = re.match(
        r"\$?\s*([\d,]+(?:\.\d+)?)\s*([bmt])\s*$",
        raw.replace(" ", ""),
    )
    if m:
        num_str, unit = m.group(1), m.group(2)
        try:
            n = float(num_str.replace(",", ""))
        except ValueError:
            return frozenset(forms)
        value = n * _EN_MAGNITUDE_SHORT[unit]
        forms.update(_cross_notation_forms(value))
        return frozenset(forms)

    # 5. Chinese-suffix branch — broad capture, longest-prefix match
    # against the magnitude dict. Unknown suffix → keep raw forms only
    # (graceful fallback — substring match still works if both body
    # and snippet use the same notation; missing magnitudes only lose
    # cross-notation equivalence, not detection).
    m = re.match(
        r"\$?\s*([\d,]+(?:\.\d+)?)\s*([一-鿿]+)\s*$",
        raw,
    )
    if m:
        num_str, suffix = m.group(1), m.group(2)
        for magnitude_word in _CN_MAGNITUDE_LONGEST_FIRST:
            if suffix.startswith(magnitude_word):
                try:
                    n = float(num_str.replace(",", ""))
                except ValueError:
                    return frozenset(forms)
                value = n * _CN_MAGNITUDE[magnitude_word]
                forms.update(_cross_notation_forms(value))
                # Canonical Chinese form without currency identifier
                # (so ``$50亿美`` matches ``$50亿`` in a snippet).
                forms.add(f"${n:g}{magnitude_word}")
                forms.add(f"{n:g}{magnitude_word}")
                return frozenset(forms)
        # Unknown magnitude → graceful fallback (raw forms only).
        return frozenset(forms)

    return frozenset(forms)


_SNIPPET_DECIMAL_DOLLAR_RX = re.compile(
    r"\$?\s*([\d,]+)\.(\d+)\s*([bmt])\b",
    re.IGNORECASE,
)


def _decimal_precision_match(tok: str, snippet: str) -> bool:
    """True when ``tok`` is a body number that rounds back from a
    higher-precision value present in ``snippet``.

    Catches the common case where the writer drops a decimal place for
    readability (``$903.4B``) while the source carries the full precision
    (``$903.394B``). Equivalence holds iff ``round(snippet_value,
    body_precision) == body_value`` — strictly tighter than a percentage
    tolerance, so it cannot accidentally equate ``$903.4B`` with
    ``$900.0B`` or ``$50B`` with ``$5B``.

    Only handles ``$NNN.DDD[BMT]`` (dollar + decimal + single-letter
    magnitude) — the shape that triggered the regression. Other tokens
    drop through to ``suspect``.
    """
    m = re.match(r"\$?\s*([\d,]+)\.(\d+)\s*([bmt])\s*$", tok.strip(), re.IGNORECASE)
    if not m:
        return False
    body_int = m.group(1).replace(",", "")
    body_frac = m.group(2)
    unit = m.group(3).lower()
    precision = len(body_frac)
    try:
        body_value = float(f"{body_int}.{body_frac}")
    except ValueError:
        return False
    for sm in _SNIPPET_DECIMAL_DOLLAR_RX.finditer(snippet):
        if sm.group(3).lower() != unit:
            continue
        try:
            snip_value = float(
                f"{sm.group(1).replace(',', '')}.{sm.group(2)}"
            )
        except ValueError:
            continue
        if round(snip_value, precision) == body_value:
            return True
    return False


def validate_numeric_grounding(
    body: str,
    *,
    snippet_lookup: Callable[[int], str] | None = None,
) -> NumericGroundingReport:
    """Classify each specific number in ``body`` as grounded / suspect / uncited.

    Args:
        body: The report markdown (post strip-trailing-references).
        snippet_lookup: ``index -> snippet_text``. When None, the audit
            degrades to citation-presence only — every cited number is
            grounded (no snippet-content check). A heavy-backend caller with
            its own evidence store would wire this to that store's snippet
            lookup; the default-research path with no such store passes None.

    Returns:
        :class:`NumericGroundingReport`. Callers decide what to do with
        suspects: log + continue (warn mode) or raise + repair (strict).
        This function never raises on suspects — its job is to classify,
        not enforce.
    """
    if not body:
        return NumericGroundingReport()

    snippet_cache: dict[int, str] = {}

    def snippet_for(idx: int) -> str:
        if snippet_lookup is None:
            return ""
        if idx not in snippet_cache:
            try:
                text = snippet_lookup(idx) or ""
            except Exception:
                # A misbehaving lookup must not crash the audit — treat as
                # "no snippet available", which conservatively marks the
                # token suspect rather than silently passing it.
                text = ""
            snippet_cache[idx] = text.lower()
        return snippet_cache[idx]

    report = NumericGroundingReport()
    for m in _NUMERIC_GROUNDING_RX.finditer(body):
        tok = m.group(0).strip()
        tail = body[m.end(): m.end() + _NUMERIC_CITE_WINDOW]
        # ``[3][8]`` / ``[3, 8]`` / ``[3] (see also [8])`` — collect all
        # indices from every ``[…]`` bracket pair in the window. The
        # first bracket counts as long as it sits in the 80-char window
        # (already enforced by the tail slice); subsequent brackets only
        # count if they tail-follow the previous one within ≤4 chars,
        # since once non-cite text intervenes they belong to a different
        # claim. Reuses ``_CITE_GROUP_RE`` from the existing citation
        # parsing — keeps the bracket regex single-sourced.
        ids: list[int] = []
        prev_end: int | None = None
        for cm in _CITE_GROUP_RE.finditer(tail):
            if prev_end is not None and cm.start() - prev_end > 4:
                break
            ids.extend(int(n) for n in re.findall(r"\d+", cm.group(1)))
            prev_end = cm.end()
        if not ids:
            report.uncited.append(tok)
            continue
        if snippet_lookup is None:
            report.grounded.append((tok, ids[0]))
            continue
        forms = _numeric_equivalent_forms(tok)
        matched = any(
            any(f in snippet_for(idx) for f in forms)
            for idx in ids
        )
        if matched:
            report.grounded.append((tok, ids[0]))
            continue
        # Decimal-precision rescue: body rounded ``$903.4B`` against a
        # snippet carrying the higher-precision ``$903.394B``. The check
        # is mathematically strict (rounds back at body's precision), so
        # ``$903.4B`` vs ``$900.0B`` still classifies suspect.
        if any(
            _decimal_precision_match(tok, snippet_for(idx)) for idx in ids
        ):
            report.grounded.append((tok, ids[0]))
            continue
        # Rescue check: if the token's immediate surroundings carry a
        # derivation marker (``~1.4% below close``, ``2× ATR``,
        # ``= $215.10``), the value is the prompt's bucket-2 form —
        # explicitly derived from cited inputs with shown math — not a
        # fabrication. The window straddles the token (20 chars before,
        # 80 chars after) because some markers cross the token boundary
        # (e.g. ``= $215.10`` has the ``=`` before the dollar).
        context = body[max(0, m.start() - 20): m.end() + 80]
        if _DERIVATION_SIGNAL_RX.search(context):
            report.grounded_derived.append((tok, ids[0]))
        else:
            report.suspect.append((tok, ids[0]))
    return report


def finalize_cited_report(
    body: str,
    references: Sequence[Reference],
    *,
    language: str = "auto",
    style: str = "markdown",
    snippet_lookup: Callable[[int], str] | None = None,
) -> str:
    """Validate citations, then append canonical References (+ citation-map
    in ``markdown`` style).

    Uses membership mode so a body citing an index that was skipped by
    :func:`build_citation_maps` (URL-less reference) fails closed rather
    than silently retargeting to a different URL after renumber.

    When ``snippet_lookup`` is provided, also runs the numeric-grounding
    audit and logs a warning summary if any tokens classify as suspect.
    The audit never raises — callers decide whether a suspect ratio
    above some threshold warrants a repair pass. ``snippet_lookup`` is
    opt-in because not every caller has snippet access (a caller without
    an evidence store passes None; a heavy-backend reporter would wire it to
    that store's snippet lookup).
    """
    clean_body = strip_trailing_references(body).strip()
    citation_map, citation_titles = build_citation_maps(references)
    valid_indices = [int(k) for k in citation_map]
    issues = validate_citation_body(
        clean_body,
        max_ref=len(citation_map),
        valid_indices=valid_indices,
    )
    if issues:
        raise CitationValidationError(issues)
    if snippet_lookup is not None:
        audit = validate_numeric_grounding(
            clean_body, snippet_lookup=snippet_lookup,
        )
        if audit.suspect:
            logger.warning(
                "numeric grounding audit: %d suspect of %d tokens (%.1f%%)"
                "; %d grounded-derived, %d uncited",
                len(audit.suspect), audit.total, audit.suspect_ratio * 100,
                len(audit.grounded_derived), len(audit.uncited),
            )
            for tok, idx in audit.suspect[:5]:
                logger.warning("  suspect %r cited to [%d]", tok, idx)
    return finalize_report(
        clean_body, citation_map, citation_titles, language, style=style,
    )


async def repair_citations_once(
    *,
    llm: Any,
    question: str,
    draft_answer: str,
    body: str,
    references_block: str,
    issues: Sequence[str],
    timeout_s: float,
) -> str:
    """Ask the reporter LLM to repair citation markers without rewriting facts."""
    system = """You repair citation markers in a Markdown report body.

Rules:
- Preserve the report's meaning, structure, claims, numbers, dates, and language.
- Use only citation indices from the provided Available references list.
- Add [N] citations to supported factual claims.
- Remove or replace invalid/out-of-range citation markers.
- Do NOT add a References/Sources section. Output only the repaired body.
- Do NOT add preamble, analysis, tool markup, or code fences."""

    user = f"""# Original question

{question or "(unspecified)"}

# Validation errors to fix

{chr(10).join(f"- {issue}" for issue in issues)}

# Available references

{references_block}

# Original coordinator/report draft

{draft_answer}

# Body to repair

{body}

# Task

Return the same Markdown body with valid inline [N] citations. Output only
the body. Do not append References."""

    response = await asyncio.wait_for(
        llm.chat([
            system_msg(system),
            user_msg(user),
        ]),
        timeout=timeout_s,
    )
    content = response.content if hasattr(response, "content") else str(response)
    return content if isinstance(content, str) else text_of(content)


__all__ = [
    "CitationValidationError",
    "NumericGroundingReport",
    "build_citation_maps",
    "citation_numbers",
    "coerce_title",
    "coerce_url",
    "detect_url_misalignment",
    "finalize_cited_report",
    "format_references_block_plain",
    "has_any_citation",
    "normalize_inline_bullets",
    "normalize_malformed_citations",
    "parse_trailing_references",
    "render_references_block",
    "repair_citations_once",
    "strip_orphan_footnote_citations",
    "strip_trailing_references",
    "validate_citation_body",
    "validate_numeric_grounding",
]
