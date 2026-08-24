"""Citation renumbering and deterministic report finalization helpers."""

from __future__ import annotations

import json
import re
from urllib.parse import urlparse

from frontier_agent.utils.language import is_chinese_label

# Match bracket groups whose body is *only* digits + commas / semicolons /
# whitespace, with at least one digit somewhere. ``[1]`` / ``[1, 13, 14]`` /
# ``[1; 2]`` / ``[ 5 , 6 ]`` all match; ``[citation needed]`` and
# ``[1, foo]`` do not. The captured group is the inner body.
_CITE_GROUP_RE = re.compile(r"\[(\s*\d[\d,;\s]*)\]")


def _normalize_multi_citations(report: str, *, max_ref: int | None = None) -> str:
    """Rewrite ``[1, 13, 14]`` → ``[1][13][14]``.

    The LLM frequently emits comma-separated multi-citations (a more readable
    form), but downstream renumber + reference rendering both expect the
    single-number form ``[N]``. Without this pass, ``re.findall(r"\\[(\\d+)\\]",
    ...)`` skips every multi-citation block, so the renumbered citation_map
    ends up with only the handful of citations that happened to appear as
    ``[N]`` standalone — see the bug post-mortem in
    ``data/reports/70feda8cbb39.md`` (Strategy A smoke test).

    ``max_ref``, when given, gates the rewrite on plausibility: a group is
    only split when every number in it could be a real 1-indexed citation
    (``1 <= n <= max_ref``). Real reports also carry bracketed numeric
    ranges/id-pairs copied verbatim from source data (e.g. ``[684, 821]``
    from a revenue-projection table, far outside any real reference list) —
    without this gate those get corrupted into ``[684][821]``, silently
    dropping the comma that made the range legible. ``max_ref=None`` keeps
    the old unconditional behavior, for callers with no reference list to
    check against (:func:`renumber_citations`'s citation map isn't bounded
    to a contiguous ``1..N`` range).
    """
    def _expand(match: re.Match) -> str:
        body = match.group(1)
        nums = [int(n) for n in re.findall(r"\d+", body)]
        if len(nums) <= 1:
            # Single-number bracket — leave it alone (avoid touching
            # mathematical notation, code references, etc.).
            return match.group(0)
        if max_ref is not None and not all(1 <= n <= max_ref for n in nums):
            return match.group(0)
        return "".join(f"[{n}]" for n in nums)

    return _CITE_GROUP_RE.sub(_expand, report)


def renumber_citations(
    report: str, citation_map: dict[str, str]
) -> tuple[str, dict[str, str]]:
    """Renumber citations in an LLM report so they are sequential (1, 2, 3, ...).

    The LLM sees all evidence numbered 1..N but often only cites a subset,
    producing gaps like [1][2][4][10][15]. This remaps them to [1][2][3][4][5].

    Multi-citation forms like ``[1, 13, 14]`` are normalized to
    ``[1][13][14]`` first so they participate in the renumber pass too.

    Returns (new_report, new_citation_map).
    """
    report = _normalize_multi_citations(report)
    used_nums = sorted(set(int(m) for m in re.findall(r"\[(\d+)\]", report)))
    if not used_nums:
        return report, citation_map

    old_to_new: dict[int, int] = {}
    for new_num, old_num in enumerate(used_nums, start=1):
        old_to_new[old_num] = new_num

    # Replace largest numbers first to avoid partial replacement
    new_report = report
    for old_num in sorted(old_to_new.keys(), reverse=True):
        new_num = old_to_new[old_num]
        if old_num != new_num:
            new_report = new_report.replace(
                f"[{old_num}]", f"[__CITE_{new_num}__]"
            )

    for new_num in old_to_new.values():
        new_report = new_report.replace(f"[__CITE_{new_num}__]", f"[{new_num}]")

    new_citation_map: dict[str, str] = {}
    for old_num, new_num in old_to_new.items():
        old_key = str(old_num)
        if old_key in citation_map:
            new_citation_map[str(new_num)] = citation_map[old_key]

    return new_report, new_citation_map


def _format_references_section(
    citation_map: dict[str, str],
    citation_titles: dict[str, str] | None = None,
    language: str = "auto",
    style: str = "markdown",
) -> str:
    """Build a visible References section for downloaded reports.

    ``style="markdown"`` (default) — ``N. [title](url)`` numbered list,
    rendered as clickable links in the frontend. Used by the HTTP API path
    (the research workflows).

    ``style="plain"`` — ``[N] title. url`` plain text, no markdown links.
    Used by SDK-CLI workflows where the output
    is consumed as raw text rather than rendered as HTML.

    Frontend's `MarkdownRenderer` strips any trailing References/参考文献
    section before rendering (it emits its own), so adding this block is safe
    for in-app display and strictly improves offline / downloaded markdown.
    """
    if not citation_map:
        return ""

    heading = "## 参考文献" if is_chinese_label(language) else "## References"
    titles = citation_titles or {}

    lines: list[str] = [heading, ""]
    for key in sorted(citation_map, key=lambda k: int(k) if k.isdigit() else 0):
        url = (citation_map.get(key) or "").strip()
        title = (titles.get(key) or "").strip()
        if not title and url:
            host = urlparse(url).netloc or url
            title = host
        # When the upstream whitelist couldn't recover a real page title it
        # leaves ``title`` empty or equal to the URL; printing both then
        # reads as a duplicated URL (``[N] url. url`` / ``[N] [url](url)``).
        # Collapse to a single URL in that case.
        has_title = bool(title) and title != url
        if style == "plain":
            if url:
                lines.append(f"[{key}] {title}. {url}" if has_title else f"[{key}] {url}")
            else:
                lines.append(f"[{key}] {title or '(no source)'}")
        else:
            if url:
                lines.append(
                    f"{key}. [{title}]({url})" if has_title else f"{key}. [{url}]({url})"
                )
            else:
                lines.append(f"{key}. {title or '(no source)'}")
    lines.append("")
    return "\n".join(lines)


def finalize_report(
    report: str,
    citation_map: dict[str, str],
    citation_titles: dict[str, str] | None = None,
    language: str = "auto",
    *,
    style: str = "markdown",
) -> str:
    """Renumber citations, append a visible References section, and (in
    ``markdown`` style) a machine-readable citation-map HTML comment for
    frontend parsing.

    ``style="plain"`` produces plain ``[N] title. url`` references with no
    citation-map comment — used by SDK-CLI workflows whose consumer reads
    the markdown as text.
    """
    # Build url→title using the pre-renumber keys, so we can re-key titles
    # against the post-renumber citation_map (which uses new keys but same
    # URLs).
    title_by_url: dict[str, str] = {}
    for key, url in citation_map.items():
        title = (citation_titles or {}).get(key, "").strip()
        if url and title:
            title_by_url.setdefault(url, title)

    report, citation_map = renumber_citations(report, citation_map)

    renumbered_titles: dict[str, str] = {}
    for new_key, url in citation_map.items():
        if url in title_by_url:
            renumbered_titles[new_key] = title_by_url[url]

    parts = [report.rstrip()]
    refs = _format_references_section(
        citation_map, renumbered_titles, language, style=style,
    )
    if refs:
        parts.append(refs)
    if style != "plain":
        map_json = json.dumps(citation_map, ensure_ascii=False)
        parts.append(f"<!-- citation-map: {map_json} -->")
    return "\n\n".join(parts) + "\n"
