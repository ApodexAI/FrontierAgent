"""Citation-contract gates for cited report generation."""

from __future__ import annotations

import re
from typing import Any

from workflows._shared.citation_utils import (
    _normalize_multi_citations,
)
from workflows._shared.cited_report_finalizer import (
    coerce_title,
    coerce_url,
    format_references_block_plain,
    has_any_citation,
    normalize_inline_bullets,
    normalize_malformed_citations,
    render_references_block,
    strip_orphan_footnote_citations,
    strip_trailing_references,
)

# ── Gate 0: URL verifiability (shared by every citation-grade gate) ───

# Markers that make a syntactically-valid http(s) URL un-fetchable: the LLM
# minted it by collapsing several real filings into a glob, or left a
# placeholder. ``coerce_url`` accepts these (the scheme is valid), so without
# this check they flow straight into the canonical References block —
# laundering a fabricated citation as a system-generated source. Observed in
# the q3 eval: ``gild-202*.htm`` / ``msft-10k_202*.htm`` / ``tsla-10k_202*.htm``.
# Kept to high-confidence markers only (a bare space, a glob ``*``, an ellipsis,
# or bracket/angle placeholders) so a real URL — which percent-encodes these —
# is never rejected.
_FABRICATED_URL_MARKERS: tuple[str, ...] = (
    " ", "*", "...", "…", "{", "}", "<", ">",
)


def is_fabricated_url(url: str) -> bool:
    """True when a syntactically-valid URL carries an un-fetchable marker.

    Detects globbed / placeholder URLs (``gild-202*.htm``) the LLM fabricates
    by collapsing multiple real sources into one wildcard. This is a shape
    check, not a liveness probe — it never makes a network call; it only
    rejects URLs that cannot possibly resolve.
    """
    if not url:
        return False
    return any(marker in url for marker in _FABRICATED_URL_MARKERS)


def coerce_citation_url(raw: Any) -> str:
    """``coerce_url`` + fabricated-URL rejection, for citation whitelisting.

    Returns the canonical URL string, or ``""`` when it is empty / malformed
    (``coerce_url``) OR carries a fabrication marker (:func:`is_fabricated_url`).
    Every citation-grade gate coerces through this instead of
    bare ``coerce_url`` so a globbed / placeholder URL can never anchor a
    ``[N]`` or reach the References block.
    """
    url = coerce_url(raw)
    if not url or is_fabricated_url(url):
        return ""
    return url


# ── Gate 1: build references from raw snapshots ───────────────────────


_QUALITY_WEIGHT: dict[str, int] = {"high": 3, "medium": 2, "low": 1}
_TRAILING_REFS_FOOTER_RE = re.compile(
    r"(?im)^(?:#{2,4}\s*)?(References|参考文献|参考资料|Sources|来源)\s*$[\s\S]*\Z",
)


def _citation_grade_core(
    *,
    source_type: str,
    url: str,
    has_snippet: bool,
    snippet_verbatim: bool,
    fetch_status: str,
    strict: bool,
) -> bool:
    """Single source of truth for "can this evidence anchor a ``[N]``".

    The two shape-specific wrappers (:func:`_is_citation_grade_evidence` for
    the snapshot evidence dump, :func:`_is_citation_grade_dag_node`
    for the methods_info DAG dict) normalise their fields and delegate here,
    so the snapshot path and the canonical path can never drift apart.

    Base gate (always on):
    - ``source_type`` whitelisted to ``search`` / ``visit`` (empty/unset OK for
      legacy nodes); a blacklist that only dropped ``inference`` let mislabelled
      fabricated nodes through.
    - URL survives :func:`coerce_citation_url` (real + non-fabricated).
    - has a snippet to cite a substring of.

    ``strict`` (Evidence Provenance Contract, P3) additionally requires:
    - ``snippet_verbatim`` — an LLM-paraphrased snippet can't anchor a quote.
    - ``fetch_status != "failed"`` — a source the tool never fetched isn't real.

    ``strict`` is gated behind the ``provenance_strict`` config flag (default
    off) so it only bites once P2/P2b have populated those fields; with the
    flag off this is byte-identical to the pre-contract gate + URL check.
    """
    st = (source_type or "").lower()
    if st and st not in ("search", "visit"):
        return False
    if not coerce_citation_url(url):
        return False
    if not has_snippet:
        return False
    if strict:
        if not snippet_verbatim:
            return False
        if (fetch_status or "").lower() == "failed":
            return False
    return True


def _is_citation_grade_evidence(
    ev: dict[str, Any], *, strict: bool = False,
) -> bool:
    """True when a snapshot evidence dict can support report citations.

    Shape wrapper over :func:`_citation_grade_core` for a serialised run
    snapshot's ``evidence`` dump shape (``source_url`` / ``snippet`` /
    ``metadata.snippet``).
    """
    snippet = str(ev.get("snippet") or "").strip()
    has_snippet = bool(snippet)
    if not has_snippet:
        metadata = ev.get("metadata") or {}
        if isinstance(metadata, dict):
            has_snippet = bool(str(metadata.get("snippet") or "").strip())
    return _citation_grade_core(
        source_type=str(ev.get("source_type") or ""),
        url=ev.get("source_url", ""),
        has_snippet=has_snippet,
        snippet_verbatim=bool(ev.get("snippet_verbatim", False)),
        fetch_status=str(ev.get("fetch_status") or "unknown"),
        strict=strict,
    )


def build_report_references_from_snapshots(
    run_snapshots: dict[str, dict[str, Any]],
    *,
    cap: int = 30,
    provenance_strict: bool = False,
) -> list[dict[str, str]]:
    """Build the numbered ``[N]`` whitelist directly from run-phase output.

    Walks every run's serialised evidence snapshot (the dict-of-dict shape
    ``project_evidence`` produces as its ``run_snapshots`` return value; see
    ``fast_reporter_v1_evidence.py``) and surfaces one entry per URL, ranking
    by ``source_quality`` (high > medium > low) and breaking ties on
    first-seen order so the resulting indices are stable across reruns of
    the same evidence pool.

    Title resolution order matches V1:

    1. ``metadata.title`` — the DAG analyzer is asked to copy the page
       title from the tool response into this slot.
    2. ``claim`` — the human-readable atomic claim, used when the LLM
       omits ``metadata.title``.
    3. URL (via :func:`coerce_title`'s structural sanity fallback) when
       neither candidate looks title-shaped.

    Returns at most ``cap`` entries. The :func:`compose_citation_contract`
    fragment then pins the LLM to exactly these indices.
    """
    by_url: dict[str, tuple[int, int, dict[str, str]]] = {}
    first_seen = 0
    for snap_dict in (run_snapshots or {}).values():
        if not isinstance(snap_dict, dict):
            continue
        snapshot = snap_dict.get("snapshot") or {}
        evidence_map = snapshot.get("evidence") or {}
        if not isinstance(evidence_map, dict):
            continue
        for ev in evidence_map.values():
            if not isinstance(ev, dict):
                continue
            if not _is_citation_grade_evidence(ev, strict=provenance_strict):
                continue
            url = coerce_citation_url(ev.get("source_url", ""))
            metadata = ev.get("metadata") or {}
            raw_title = (
                (metadata.get("title") if isinstance(metadata, dict) else "")
                or ev.get("claim")
                or ""
            )
            title = coerce_title(raw_title, url)
            quality = _QUALITY_WEIGHT.get(str(ev.get("source_quality") or ""), 0)
            first_seen += 1
            existing = by_url.get(url)
            # Keep the strictly-higher-quality entry for a given URL; tie
            # breaks on first-seen so reruns over the same evidence pool
            # produce stable indices.
            if existing is not None and existing[0] >= quality:
                continue
            by_url[url] = (
                quality,
                first_seen,
                {"title": title, "url": url},
            )

    ordered = sorted(
        by_url.values(),
        key=lambda item: (-item[0], item[1]),
    )
    return [entry for _quality, _idx, entry in ordered[:cap]]


# ── Gate 1b: build references from the reporter's canonical DAG ───────


def _is_citation_grade_dag_node(
    node: dict[str, Any], *, strict: bool = False,
) -> bool:
    """True when a methods_info DAG node can support citations.

    Shape wrapper over :func:`_citation_grade_core` for the evidence-DAG dict
    shape (``type`` / ``url``|``source_url`` / ``snippet``|``content``).
    """
    if str(node.get("type") or "").lower() != "evidence":
        return False
    has_snippet = bool(
        str(node.get("snippet") or "").strip()
        or str(node.get("content") or "").strip()
    )
    return _citation_grade_core(
        source_type=str(node.get("source_type") or ""),
        url=node.get("url") or node.get("source_url") or "",
        has_snippet=has_snippet,
        snippet_verbatim=bool(node.get("snippet_verbatim", False)),
        fetch_status=str(node.get("fetch_status") or "unknown"),
        strict=strict,
    )


def build_report_references_from_methods_info(
    methods_info: dict[str, dict[str, Any]],
    *,
    cap: int = 30,
    provenance_strict: bool = False,
) -> list[dict[str, str]]:
    """Build the numbered ``[N]`` whitelist from ``methods_info``.

    This is the canonical path because ``methods_info`` (see
    ``project_evidence`` in ``fast_reporter_v1_evidence.py``) is the richer
    evidence projection: it can carry source URLs that are absent from the
    sanitized ``run_snapshots`` fallback.
    """
    by_url: dict[str, tuple[int, int, dict[str, str]]] = {}
    first_seen = 0
    for record in (methods_info or {}).values():
        if not isinstance(record, dict):
            continue
        dag = record.get("dag") or {}
        if not isinstance(dag, dict):
            continue
        nodes = dag.get("nodes") or {}
        if not isinstance(nodes, dict):
            continue
        for node in nodes.values():
            if not isinstance(node, dict):
                continue
            if not _is_citation_grade_dag_node(node, strict=provenance_strict):
                continue
            url = coerce_citation_url(node.get("url") or node.get("source_url") or "")
            metadata = node.get("metadata") or {}
            raw_title = (
                (metadata.get("title") if isinstance(metadata, dict) else "")
                or node.get("title")
                or node.get("claim")
                or node.get("content")
                or ""
            )
            title = coerce_title(raw_title, url)
            quality = _QUALITY_WEIGHT.get(str(node.get("source_quality") or ""), 0)
            first_seen += 1
            existing = by_url.get(url)
            if existing is not None and existing[0] >= quality:
                continue
            by_url[url] = (
                quality,
                first_seen,
                {"title": title, "url": url},
            )

    ordered = sorted(
        by_url.values(),
        key=lambda item: (-item[0], item[1]),
    )
    return [entry for _quality, _idx, entry in ordered[:cap]]


def backfill_reference_titles_from_snapshots(
    references: list[dict[str, str]],
    run_snapshots: dict[str, dict[str, Any]] | None,
) -> list[dict[str, str]]:
    """Re-attach real page titles to a methods_info-derived whitelist.

    The canonical whitelist is built from ``methods_info`` (see
    :func:`build_report_references_from_methods_info`), whose DAG some
    extractors rebuild from an evidence schema that only carries ``snippet``
    / ``url`` — never the page title — so title resolution there falls
    through to the snippet ``content``, which
    :func:`coerce_title` rejects to the bare URL (rendering as ``[N] url.
    url``) or, when the snippet is a short single line, keeps as a
    claim-shaped pseudo-title.

    The *original* tool execution did capture page titles into the run
    snapshots' ``evidence[*].metadata.title``. This walks those snapshots,
    builds a ``url -> real_title`` map (real titles only — entries that are
    empty or that ``coerce_title`` collapsed to the URL are excluded), and
    prefers that authoritative title over whatever the methods_info path
    produced for the same URL.

    The snapshot ``metadata.title`` is the most authoritative label we have
    (captured at tool-execution time), so it wins over both the URL
    fallback (the ``url. url`` bug) AND a claim-sentence pseudo-title that
    the methods_info resolver produced when it fell through to ``content``.
    Claim-as-label survives only when the snapshot ALSO has no real title
    for that URL — i.e. genuinely "no real title available" — which is the
    intended last-resort behaviour. No-op when there are no references or
    snapshots, so the methods_info path stays the source of truth for
    *which* URLs make the whitelist; this only improves their labels.
    """
    if not references or not run_snapshots:
        return references
    snap_refs = build_report_references_from_snapshots(
        run_snapshots, cap=10_000,
    )
    title_by_url: dict[str, str] = {}
    for ref in snap_refs:
        url = coerce_url(ref.get("url", ""))
        title = (ref.get("title") or "").strip()
        if url and title and title != url:
            title_by_url.setdefault(url, title)
    if not title_by_url:
        return references
    out: list[dict[str, str]] = []
    for ref in references:
        url = coerce_url(ref.get("url", ""))
        title = (ref.get("title") or "").strip()
        better = title_by_url.get(url)
        if url and better and better != title:
            ref = {**ref, "title": better}
        out.append(ref)
    return out


def _body_without_references_footer(text: str) -> str:
    """Return report body text with a final References/Sources footer ignored."""
    match = _TRAILING_REFS_FOOTER_RE.search(text or "")
    if match is None:
        return text
    return text[: match.start()].rstrip()


# Heading line that opens a "write your own References" instruction inside a
# report prompt — matches ``### ## References`` / ``## References`` /
# ``#### References`` etc., but only when the whole line is just hashes +
# "References" (so inline mentions like "References at end with URL" are
# untouched). Used to excise that instruction when the citation contract is
# active, so the prompt no longer fights the contract (the prompt says
# "write a [N] Title. URL list", the contract says "don't" — see the
# duplicate-references post-mortem). Matched section runs up to the next
# top-level ``## `` heading.
_PROMPT_REFS_SECTION_RE = re.compile(
    r"(?ms)^#{1,4}[ \t]*#*[ \t]*References[ \t]*$.*?(?=^## )",
)


def strip_references_section_from_prompt(prompt: str) -> str:
    """Remove a "write your own ## References section" instruction block.

    Only needed when the deterministic citation contract is active. The base
    report prompts (``soft3_table`` / ``soft3_table_clean``) carry a
    ``### ## References`` block telling the
    LLM to author a ``[N] Title. URL`` list; the contract tells it NOT to
    and the system appends the canonical list instead. Leaving both in the
    system prompt is what made the reporter emit a second, header-less
    reference list that slipped past the footer strip and produced two
    reference blocks. Excising the instruction here removes the conflict at
    its source rather than papering over it downstream.

    No-op when the prompt has no such heading (e.g. ``short_answer``, which
    mentions references only inline).
    """
    if not prompt:
        return prompt
    return _PROMPT_REFS_SECTION_RE.sub("", prompt)


_INLINE_CITE_RE = re.compile(r"\[(\d+)\]")


def _renumber_by_first_appearance(
    body: str,
    references: list[dict[str, str]],
) -> tuple[str, list[dict[str, str]]]:
    """Renumber inline ``[N]`` markers by order of first appearance.

    The whitelist (``references``) is quality-ranked, so ``[1]`` is the
    highest-quality source — not the first one cited. Readers expect a
    report's citation numbers to climb as they read (``[1]`` then ``[2]``
    …), so this remaps each cited index to its first-appearance rank and
    returns the cited references in that same order. Uncited whitelist
    entries are dropped (a References list of sources the body never cites
    is noise).

    ``references`` is 1-indexed against the body (``references[0]`` is
    ``[1]``). Markers outside ``1..len(references)`` are left untouched —
    they're orphan citations the upstream whitelist can't resolve, and
    silently renumbering them would retarget them onto a real source.

    Returns ``(new_body, ordered_references)``. When the body cites no
    in-range index, returns ``(body, [])``.
    """
    normalized = _normalize_multi_citations(body, max_ref=len(references))
    order: list[int] = []
    seen: set[int] = set()
    for match in _INLINE_CITE_RE.finditer(normalized):
        num = int(match.group(1))
        if num in seen or not (1 <= num <= len(references)):
            continue
        seen.add(num)
        order.append(num)
    if not order:
        return body, []

    old_to_new = {old: new for new, old in enumerate(order, start=1)}
    # Two-phase replace via a NUL sentinel so renumbering never collides
    # with an index that is itself a replacement target (e.g. [2]→[1]
    # while [1]→[3] is still pending).
    new_body = normalized
    for old in sorted(old_to_new, reverse=True):
        new_body = new_body.replace(f"[{old}]", f"[\x00{old_to_new[old]}\x00]")
    for new in old_to_new.values():
        new_body = new_body.replace(f"[\x00{new}\x00]", f"[{new}]")

    ordered_references = [references[old - 1] for old in order]
    return new_body, ordered_references


# ── Gate 2: compose contract for system prompt ────────────────────────


def build_citation_index_map(
    references: list[dict[str, str]],
) -> dict[str, int]:
    """Map each whitelist URL → its 1-based citation index.

    The returned index matches both the order
    :func:`compose_citation_contract` renders (``[1]`` is ``references[0]``)
    and the order :func:`finalize_report_with_canonical_references`
    renumbers against, so a tag stamped onto an evidence node from this map
    points at the same source the final References block resolves it to.

    This is the binding that closes the citation-mismatch hole: the
    reporter LLM reads evidence via its evidence-lookup tools (node IDs +
    URLs), not the reference list, so without an inline index next to each
    node it has to *guess* which ``[N]`` a claim maps to and picks topically-
    plausible-but-wrong numbers. Stamping the index onto the evidence lets
    it copy the number instead of guessing.

    First URL wins on collision (the whitelist already deduped by URL, so
    collisions only arise from malformed input). Entries without a usable
    URL are skipped.
    """
    out: dict[str, int] = {}
    for i, ref in enumerate(references or [], start=1):
        if not isinstance(ref, dict):
            continue
        url = coerce_url(ref.get("url", ""))
        if url and url not in out:
            out[url] = i
    return out


def compose_citation_contract(references: list[dict[str, str]]) -> str:
    """Render the citation contract block for injection into the reporter's
    system prompt.

    Returns an empty string when ``references`` is empty — the caller
    should NOT append the contract in that case, otherwise the LLM is
    told "use one of these indices" with an empty index list and either
    refuses to cite anything or hallucinates indices.
    """
    if not references:
        return ""
    references_block = render_references_block(references)
    return f"""

# Deterministic Citation Contract

The system appends the final References section after you submit, using the
numbered list below, and renumbers your inline markers. This contract
OVERRIDES every earlier instruction about writing references — including any
``## References`` section described in the report structure above.

- Write ONLY the report body. Do NOT emit any reference list of your own:
  no ``## References`` / ``## Sources`` / ``## 参考文献`` heading, and no
  trailing run of bare ``[N] URL`` / ``[N] Title. URL`` lines (with or
  without a heading). The system builds that section for you.
- Use ONLY numeric ``[N]`` markers for citations. Do NOT use markdown
  footnote syntax (``[^id]`` / ``[^idc_infra]``) — those are not supported,
  never get a definition, and leak into the report as broken text. If a
  claim's source isn't in the list below, keep it as analysis WITHOUT any
  marker; do not invent a ``[^…]`` footnote for it.
- Use inline ``[N]`` citations for specific sourced claims. Every sourced
  evidence node in the briefing above is tagged with its citation index as
  ``→ cite as [N]``. When a claim comes from that node, use exactly that
  tagged ``[N]`` — do NOT pick an index yourself by matching titles in the
  list below, and never invent another index. ``N`` must be one of the
  indices in the list below; never reuse the user's numbering of items they
  mention in the question. If an evidence node carries no ``→ cite as [N]``
  tag, it has no citable source — keep its claim as analysis without a
  citation rather than attaching an unrelated index.
- Cite in whatever order the narrative needs — do NOT try to keep the
  numbers ascending. The system renumbers them by order of first
  appearance, so ``[1]`` ends up being the first source you cite.
- If a factual claim cannot be tied to one of these references, either keep
  it as analysis without a citation or qualify it clearly.
- Tables: every sourced numeric cell should carry its ``[N]`` citation in
  the cell.

Available references:
{references_block}
"""


# ── Gate 3: strip LLM refs + append canonical block ───────────────────


def finalize_report_with_canonical_references(
    body: str,
    *,
    references: list[dict[str, str]],
    language: str = "auto",
) -> str:
    """Strip any LLM-written References footer and append the canonical block.

    Idempotent:

    - Empty body → unchanged.
    - No references whitelist → unchanged (caller's gate-1 produced
      nothing, so there's no canonical block to append; passing the body
      through avoids stripping a legitimate trailing References section
      written by the LLM when no whitelist was built).
    - Body with no ``[N]`` markers → unchanged. Appending a References
      block to a non-cited report would be jarring and matches V1's
      ``_emit_degraded_and_append_references`` ``citation_count == 0``
      guard. The strip+append is meant for reports that *do* cite.
    """
    if not body:
        return body
    if not references:
        return body
    if not has_any_citation(_body_without_references_footer(body)):
        return body
    # Drop any heading-led References footer the LLM still wrote (defence
    # in depth — the contract now also tells it not to). A header-less
    # bare ``[N] URL`` list is intentionally NOT stripped here: the fix for
    # that is the contract + prompt excision (so the LLM never writes it),
    # not a downstream strip.
    stripped = strip_trailing_references(body).rstrip()
    # Collapse malformed citations (``[2的相关背景]`` → ``[2]``) the LLM
    # occasionally writes, so the renumber below sees a real ``[N]`` marker
    # instead of leaking the descriptive bracket into the final report.
    stripped, _n_malformed = normalize_malformed_citations(stripped)
    # Strip off-contract markdown footnote markers (``[^idc_infra]``) the LLM
    # minted for un-whitelisted claims but never defined — left in, they leak
    # into the rendered report as literal broken-reference text because the
    # ``[N]`` renumber below only sees numeric markers.
    stripped, _n_footnote = strip_orphan_footnote_citations(stripped)
    # Re-break any bullet the LLM flattened onto the prior line
    # (``...句子。 - **下一条**``) so the list renders instead of collapsing
    # into one run-on item.
    stripped, _n_bullets = normalize_inline_bullets(stripped)
    # Renumber the surviving inline markers by first appearance so the
    # reader sees [1], [2], [3]… in reading order, and emit the canonical
    # block in that same order (cited sources only).
    renumbered, ordered_references = _renumber_by_first_appearance(
        stripped, references,
    )
    refs_block = format_references_block_plain(
        ordered_references, language=language or "auto",
    )
    if not refs_block:
        return body
    return f"{renumbered.rstrip()}\n\n{refs_block}".rstrip() + "\n"


__all__ = [
    "backfill_reference_titles_from_snapshots",
    "build_citation_index_map",
    "build_report_references_from_methods_info",
    "build_report_references_from_snapshots",
    "coerce_citation_url",
    "compose_citation_contract",
    "finalize_report_with_canonical_references",
    "is_fabricated_url",
    "strip_references_section_from_prompt",
]
