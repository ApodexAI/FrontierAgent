"""Post-generation report body cleaners."""

from __future__ import annotations

import logging
import re
from collections.abc import Callable, Iterator
from typing import Any

logger = logging.getLogger(__name__)

# Sentinels for :func:`with_code_protected`. Unicode private-use characters, so
# no regex in the post-processing gates (brackets, digits-in-brackets, runs of
# spaces, ASCII word boundaries) can match a masked line — and no real report
# text can collide with them.
_MASK_OPEN, _MASK_CLOSE = "\ue000", "\ue001"
_MASK_RE = re.compile(f"{_MASK_OPEN}(\\d+){_MASK_CLOSE}")

__all__ = [
    "clean_report_1_0",
    "clean_report_v3",
    "extract_cited_urls",
    "segment_lines",
    "strip_document_fence_wrapper",
    "unwrap_inline_image_fences",
    "with_code_protected",
]

# ── 1.0 (frozen) ──────────────────────────────────────────────────────


def clean_report_1_0(report: str) -> str:
    """Remove internal identifiers and artifacts.

    Frozen: the shipped report path is defined by this exact behaviour, bugs
    included, so any change here is a silent production change for every
    reporting run.
    """
    report = re.sub(r'\(信息来自\s*(?:pro_r|stt_r|mm_run|run)\S+\)', '', report)
    report = re.sub(r'\[(?:pro_r|stt_r|mm_run|run)\d+:[A-Z]\d+\]', '', report)
    report = re.sub(r'(?:pro_r|stt_r|mm_run|run)\d+:[A-Z]\d+', '', report)
    report = re.sub(r'\((?:pro_r|stt_r|mm_run|run)\d+\)', '', report)
    report = re.sub(r'(?:pro_r|stt_r|mm_run|run)\d+', '', report)
    report = re.sub(r'(?<![A-Za-z0-9%/\-])[ESF]\d{1,3}(?=[,.\s;:)\]|}]|$)', '', report)
    report = re.sub(r'\(\s*,?\s*\)', '', report)
    report = re.sub(r',\s*,', ',', report)
    # Collapse runs of spaces in prose only. Fenced code blocks live at
    # odd indices after split on ``` … ```; leaving them untouched
    # preserves PEP-8 4-space Python indentation that the prose-side
    # regex would otherwise destroy.
    #
    # Robustness: when the fence count is odd (unclosed trailing
    # fence — truncated LLM output) the split's last "prose" segment
    # actually contains code. Detect that case and treat the tail
    # from the dangling opener onward as code too, so we don't
    # collapse indentation inside it.
    fence_count = len(re.findall(r'```', report))
    if fence_count % 2 == 1:
        # Find the LAST opening fence and keep everything after it
        # verbatim. Splitting up to that point is still safe because
        # the prefix has an even number of fences.
        last_open = report.rfind('```')
        head, tail = report[:last_open], report[last_open:]
        head_parts = re.split(r'(```[\s\S]*?```)', head)
        for i in range(0, len(head_parts), 2):
            head_parts[i] = re.sub(r'  +', ' ', head_parts[i])
        report = ''.join(head_parts) + tail
    else:
        parts = re.split(r'(```[\s\S]*?```)', report)
        for i in range(0, len(parts), 2):
            parts[i] = re.sub(r'  +', ' ', parts[i])
        report = ''.join(parts)
    report = re.sub(r'\n{3,}', '\n\n', report)
    heading_match = re.search(r'^#\s', report, re.MULTILINE)
    if heading_match and heading_match.start() > 0:
        report = report[heading_match.start():]
    return report.strip()


# ── v3 building blocks ────────────────────────────────────────────────

_FENCE_RE = re.compile(r"^\s{0,3}(?P<char>`|~)(?P<run>(?P=char){2,})")

# ``` / ```markdown / ```md — a whole-document wrapper the writer sometimes
# opens (and often forgets to close) around the entire report.
_WRAPPER_INFO_RE = re.compile(r"^\s{0,3}`{3,}\s*(?:markdown|md)?\s*$", re.IGNORECASE)

# The same opener with NO info string — ambiguous, so it needs corroboration
# before being treated as a whole-document wrapper (see
# :func:`strip_document_fence_wrapper`).
_BARE_FENCE_RE = re.compile(r"^\s{0,3}`{3,}\s*$")

# Inline code spans, so prose scrubs skip ``search()`` / ``[E12]`` written as code.
_INLINE_CODE_RE = re.compile(r"`[^`\n]+`")

# The 1.0 run-id families. Kept (scoped to prose) as a cheap safety net for
# legacy traces and offline replays; no current pipeline mints these names.
_LEGACY_ID_SUBS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r'\(信息来自\s*(?:pro_r|stt_r|mm_run|run)\S+\)'), ''),
    (re.compile(r'\[(?:pro_r|stt_r|mm_run|run)\d+:[A-Z]\d+\]'), ''),
    (re.compile(r'(?:pro_r|stt_r|mm_run|run)\d+:[A-Z]\d+'), ''),
    (re.compile(r'\((?:pro_r|stt_r|mm_run|run)\d+\)'), ''),
    (re.compile(r'(?:pro_r|stt_r|mm_run|run)\d+'), ''),
)

_DOUBLE_COMMA_RE = re.compile(r',\s*,')

# Collapse only runs of spaces that FOLLOW a non-space character, so leading
# indentation survives everywhere — the second line of defence for code the
# fence scanner could not identify (writer omitted the opening fence).
_MID_LINE_SPACES_RE = re.compile(r'(?<=\S) {2,}')

# Deliberately the same trigger as 1.0 (``^#\s`` — an H1 only, not ``##``).
# Broadening it to ``#{1,6}`` made the trim fire on reports whose first heading
# is an H2, which cost real content: logs/leclere-image.json opens with the
# answer sentence ("找到了。…") plus a spelling note before its first ``##``, and
# 1.0 never touched it. v3's trim is therefore a strict subset of 1.0's.
_ATX_HEADING_RE = re.compile(r'^#\s')

# A preamble is short by nature; never let the trim eat a real document.
_PREAMBLE_MIN_BUDGET = 400
_PREAMBLE_BUDGET_RATIO = 10


def _fence_marker(line: str) -> tuple[str, int] | None:
    """Return ``(char, length)`` when ``line`` is a fence marker, else ``None``."""
    match = _FENCE_RE.match(line)
    if not match:
        return None
    return match.group("char"), len(match.group("run")) + 1


def segment_lines(report: str) -> list[tuple[str, bool]]:
    """Split ``report`` into ``(line, in_code)`` pairs by scanning fences.

    A fence marker line is itself reported as code, so markers are never
    rewritten. Unlike the 1.0 ``re.split`` pairing this cannot mis-align: a
    stray fence shifts nothing, and an unclosed fence simply leaves the tail
    marked as code (the safe direction — code is preserved, not scrubbed).

    Closing follows CommonMark: only a fence of the SAME character and at least
    the opener's length closes a block. That matters for reports that document
    markdown itself, where a ```` ````markdown ```` block legitimately contains
    ```` ``` ```` lines — a naive "any fence toggles" scan flips state on the
    inner markers and hands the enclosed code to the prose scrubs.
    """
    out: list[tuple[str, bool]] = []
    open_fence: tuple[str, int] | None = None
    for line in report.split("\n"):
        marker = _fence_marker(line)
        if marker is None:
            out.append((line, open_fence is not None))
            continue
        char, length = marker
        if open_fence is None:
            open_fence = (char, length)
        elif char == open_fence[0] and length >= open_fence[1]:
            open_fence = None
        else:
            # An inner marker (different char, or shorter run): part of the
            # block's content, not a delimiter.
            out.append((line, True))
            continue
        out.append((line, True))
    return out


def with_code_protected(body: str, transform: Callable[[str], str]) -> str:
    """Run ``transform`` over ``body`` with all code hidden from it.

    Two kinds of code are masked: every fenced-code line (fence markers
    included), and every inline code span (`` `…` ``) on an otherwise-prose
    line. Each is swapped for a sentinel that contains no brackets, no
    spaces and no ASCII word characters, so the document-wide regexes in the
    report post-processing gates cannot match inside either. The sentinels
    are swapped back afterwards, giving code that is byte-identical to the
    input.

    This is how the citation gates are made code-safe without touching their
    logic: they keep running exactly as they always have — only the
    text they are shown changes. Rewriting each of their regexes instead would
    mean editing eight-plus patterns across two shared modules, with a new
    boundary case behind each one.

    It also fixes a quieter problem: a JS/JSON literal like ``[1091, 1915]``
    otherwise reads as a grouped citation, so its numbers enter the renumber map
    and the citation validator. Hidden code can't pollute either — fenced or
    inline (``` `value = data[2]` ``` is exactly as exposed as the fenced
    form, since Gate 3's renumber is a plain document-wide string replace
    with no notion of backticks).
    """
    if _MASK_OPEN in body or _MASK_CLOSE in body:
        # Astronomically unlikely (private-use codepoints), but if the body
        # already contains a sentinel the restore pass could splice a code line
        # into the wrong place. Corrupting the report is far worse than leaving
        # the gates unprotected for this one run, so bail out loudly instead.
        logger.warning(
            "with_code_protected: body already contains the mask sentinel — "
            "running the transform unprotected to avoid mis-restoring content",
        )
        return transform(body)
    lines = body.split("\n")
    states = [in_code for _, in_code in segment_lines(body)]
    tokens: list[str] = []
    masked: list[str] = []

    def _mask_inline_span(match: re.Match[str]) -> str:
        tokens.append(match.group(0))
        return f"{_MASK_OPEN}{len(tokens) - 1}{_MASK_CLOSE}"

    for line, in_code in zip(lines, states, strict=False):
        if in_code:
            masked.append(f"{_MASK_OPEN}{len(tokens)}{_MASK_CLOSE}")
            tokens.append(line)
        else:
            # Fence markers are already claimed by the branch above, so any
            # backtick span reaching here is genuine inline code, never a
            # fence delimiter.
            masked.append(_INLINE_CODE_RE.sub(_mask_inline_span, line))
    if not tokens:
        return transform(body)

    out = transform("\n".join(masked))

    def _restore(match: re.Match[str]) -> str:
        index = int(match.group(1))
        return tokens[index] if index < len(tokens) else match.group(0)

    restored, n = _MASK_RE.subn(_restore, out)
    if n != len(tokens):
        logger.warning(
            "with_code_protected: %d/%d code spans survived the transform — "
            "a gate dropped protected content", n, len(tokens),
        )
    return restored


def strip_document_fence_wrapper(report: str) -> tuple[str, bool]:
    """Drop a fence that wraps the whole document.

    Writers routinely answer ``submit_report`` with the entire report inside a
    ```` ```markdown ```` block, and just as routinely forget the closing fence.
    That stray opener is what mis-aligns 1.0's fence pairing (every real code
    block lands in a "prose" slot and gets its indentation flattened). Removing
    it up front is deterministic and independent of the scanner.

    Only an info-string-free / ``markdown`` / ``md`` opener on the first
    non-blank line qualifies, so a report that legitimately starts with a
    ```` ```python ```` block is untouched.

    An opener with NO info string is ambiguous — plenty of reports open with a
    bare ```` ``` ```` block holding an ASCII diagram or table. Those are only
    treated as a wrapper when what follows actually looks like a document
    (a heading, or a further fence); otherwise the block is left alone. Getting
    this wrong would strip a real code block's opening fence, i.e. re-create by
    hand exactly the damage this module exists to prevent.

    The trailing fence is dropped only when it actually closes the wrapper. A
    final fence is ambiguous on its own — it may equally be the closer of the
    last inner code block, which is the common case because writers usually
    forget the wrapper's closer. Fence parity of everything between the opener
    and that final line resolves it: an even count leaves the final fence
    unmatched (so it is the wrapper's), an odd count means it closes an inner
    block and must stay.
    """
    lines = report.split("\n")
    first = next((i for i, ln in enumerate(lines) if ln.strip()), None)
    if first is None or not _WRAPPER_INFO_RE.match(lines[first]):
        return report, False
    if _BARE_FENCE_RE.match(lines[first]):
        # Corroboration must come from OUTSIDE the first block: a heading
        # anywhere, or a fence after this block's own closer. Counting that
        # closer would make every report opening with a bare ``` diagram look
        # like a wrapper.
        close_idx = next(
            (i for i in range(first + 1, len(lines)) if _FENCE_RE.match(lines[i])),
            len(lines),
        )
        looks_like_document = any(
            _ATX_HEADING_RE.match(ln) for ln in lines[first + 1:]
        ) or any(_FENCE_RE.match(ln) for ln in lines[close_idx + 1:])
        if not looks_like_document:
            return report, False
    last = next(
        (i for i in range(len(lines) - 1, first, -1) if lines[i].strip()), None,
    )
    end = len(lines)
    if last is not None and _FENCE_RE.match(lines[last]):
        inner_fences = sum(
            1 for ln in lines[first + 1:last] if _FENCE_RE.match(ln)
        )
        if inner_fences % 2 == 0:
            end = last
    return "\n".join(lines[first + 1:end]), True


# A single line that is ENTIRELY one image — no leading/trailing prose.
_IMAGE_ONLY_LINE_RE = re.compile(r"^[ \t]*!\[[^\]]*\]\(([^)\n]+)\)[ \t]*$")

# ``[N] Title. URL`` — the exact References-line shape every report_prompts
# variant mandates. Line-anchored so an in-text ``[3]`` citation mid-sentence
# never matches; only a full standalone line in this shape does.
_REFERENCE_LINE_RE = re.compile(r"^\[\d+\]\s+.+?\.\s+(\S+)\s*$", re.MULTILINE)


def extract_cited_urls(report: str) -> set[str]:
    """URLs the report itself cites in a ``[N] Title. URL`` line.

    Used by :func:`unwrap_inline_image_fences` as a fallback signal when no
    ``canonical_references`` whitelist is available (citation_contract
    disabled for this call) — same idea, weaker guarantee (it trusts the
    writer's own References text instead of the independently-built DAG
    whitelist).
    """
    return set(_REFERENCE_LINE_RE.findall(report))


def _iter_fenced_blocks(lines: list[str]) -> Iterator[tuple[int, int]]:
    """Yield ``(open_idx, close_idx)`` for each CLOSED fenced block.

    Same state machine as :func:`segment_lines` (CommonMark closing rule:
    same fence character, closer run length >= opener's), re-derived here
    instead of reverse-engineered from ``segment_lines``' flattened
    ``(line, in_code)`` output — that output can't distinguish two fenced
    blocks placed back to back (no blank line between a closer and the next
    opener) from one contiguous span, which matters because each block needs
    its OWN opener line inspected for its own info string.

    An unclosed trailing fence yields nothing for that span — the safe
    direction, matching ``segment_lines``.
    """
    # One Optional holding both halves — as two variables the checker could not
    # see that the index and its marker are always set and cleared together.
    open_block: tuple[int, tuple[str, int]] | None = None
    for i, line in enumerate(lines):
        marker = _fence_marker(line)
        if marker is None:
            continue
        if open_block is None:
            open_block = (i, marker)
            continue
        open_at, open_marker = open_block
        if marker[0] == open_marker[0] and marker[1] >= open_marker[1]:
            yield open_at, i
            open_block = None
        # else: inner marker (different char / shorter run) — content, not a
        # delimiter for THIS block.


def unwrap_inline_image_fences(
    report: str, *, known_urls: set[str] | None = None,
) -> str:
    """Strip a fence wrapping ONLY image markdown so it renders as an image.

    Distinct from :func:`strip_document_fence_wrapper` above, which only
    handles a fence around the WHOLE document. Writers sometimes wrap a
    single inline ``![alt](url)`` partway through the body in a
    ```` ```markdown ```` / ```` ```md ```` / bare ```` ``` ```` fence, which
    renders as literal text instead of an image.

    Fence detection is CommonMark-correct (built on the same
    :func:`_fence_marker` primitive as :func:`segment_lines`): backtick OR
    tilde, any run length >= 3, closer must be the same character and at
    least as long as the opener. A naive fixed-length regex mismatches a
    legitimate ```` ````markdown ```` (4-backtick) block and corrupts it
    instead of unwrapping it.

    Unwrapping is gated on citation discipline, not just the fence's content
    shape: every image URL inside a candidate block must exactly match
    either ``known_urls`` (the caller's ``canonical_references`` whitelist —
    pass this when available; it is independently built from the DAG, not
    from anything the writer typed) or a ``[N] Title. URL`` line the report
    cites elsewhere in its OWN text (:func:`extract_cited_urls` — the weaker
    fallback signal for when no whitelist was supplied). Every report_prompts
    variant already requires an embedded image's URL to be a cited source, so
    this is a self-consistency check, not a guess — and it is what stops a
    block from being unwrapped when the writer is demonstrating markdown
    syntax rather than citing a source (that example URL has no reason to
    also be a citation, so it stays fenced). A block with no matching
    citation is left untouched — the safe direction; this can only fail to
    unwrap a real image, never wrongly render a fake one.

    Only unwraps a block whose EVERY body line is, on its own, exactly one
    image — a genuine code block that happens to print ``![...]`` (e.g. in a
    comment, or mixed with other code) is left untouched, fence markers and
    all.
    """
    if not report:
        return report
    cited = extract_cited_urls(report)
    if known_urls:
        cited |= known_urls
    lines = report.split("\n")
    # Keyed by the opener's line index, so the reassembly pass below is a
    # single linear scan instead of a per-line lookup.
    by_start: dict[int, tuple[int, str]] = {}
    for start, end in _iter_fenced_blocks(lines):
        opener = lines[start]
        marker_match = _FENCE_RE.match(opener)
        assert marker_match is not None  # _iter_fenced_blocks only yields fence lines
        info = opener[marker_match.end():].strip().lower()
        if info not in ("", "markdown", "md"):
            continue
        body = lines[start + 1:end]
        # Collect first, then require a full match set: ``all`` over a list of
        # Optionals does not narrow the elements for the group() reads below.
        image_matches = [_IMAGE_ONLY_LINE_RE.match(ln) for ln in body]
        matched = [m for m in image_matches if m is not None]
        if not body or len(matched) != len(image_matches):
            continue
        if not all((m.group(1) or "").strip() in cited for m in matched):
            continue
        by_start[start] = (end, "\n".join(ln.strip() for ln in body))
    if not by_start:
        return report
    out: list[str] = []
    i = 0
    n = len(lines)
    while i < n:
        hit = by_start.get(i)
        if hit is not None:
            end, replacement = hit
            out.append(replacement)
            i = end + 1
        else:
            out.append(lines[i])
            i += 1
    return "\n".join(out)


def _node_id_label_pattern(node_ids: set[str]) -> re.Pattern[str] | None:
    """Regex matching the given node ids **only in internal-label shape**.

    ``[E12]`` / ``(E12)`` / ``E12:`` / ``src=R5/E12`` are the navigation labels
    a reporter copies out of its evidence-lookup tool output. A bare ``E12`` in a sentence
    is not matched: it is indistinguishable from ordinary vocabulary ("Euro 5
    (E5)", "F1 score") and ``internal_label_scrub`` draws the same line.

    The label shapes are deliberately narrow so they cannot bite a URL:
    ``E12:`` requires end-of-line/whitespace after the colon (so ``S3://bucket``
    keeps its scheme), and the slash form requires the literal ``src=`` hint
    (so a ``…/E5`` URL path segment is left alone).
    """
    ids = sorted((i for i in node_ids if re.fullmatch(r"[A-Za-z]+\d{1,4}", i)), key=len,
                 reverse=True)
    if not ids:
        return None
    alt = "|".join(re.escape(i) for i in ids)
    return re.compile(
        rf"\[\s*(?:{alt})\s*\]"                  # [E12]
        rf"|\(\s*(?:{alt})\s*\)"                 # (E12)
        rf"|(?<![A-Za-z0-9])(?:{alt}):(?=\s|$)"  # E12: 数据显示…
        rf"|src=\S*?/(?:{alt})\b",               # src=R5/E12
    )


def _scrub_prose_line(
    line: str, *, label_re: re.Pattern[str] | None, stats: dict[str, Any],
) -> str:
    """Apply the prose-only scrubs to one line, skipping inline code spans."""

    def _scrub(chunk: str) -> str:
        for pattern, repl in _LEGACY_ID_SUBS:
            chunk, n = pattern.subn(repl, chunk)
            stats["ids_scrubbed"] += n
        if label_re is not None:
            chunk, n = label_re.subn("", chunk)
            stats["node_ids_scrubbed"] += n
        chunk = _DOUBLE_COMMA_RE.sub(",", chunk)
        collapsed, n = _MID_LINE_SPACES_RE.subn(" ", chunk)
        if n:
            stats["lines_space_collapsed"] += 1
        return collapsed

    out: list[str] = []
    pos = 0
    for match in _INLINE_CODE_RE.finditer(line):
        out.append(_scrub(line[pos:match.start()]))
        out.append(match.group(0))
        pos = match.end()
    out.append(_scrub(line[pos:]))
    return "".join(out)


def clean_report_v3(
    report: str,
    *,
    internal_node_ids: set[str] | None = None,
) -> tuple[str, dict[str, Any]]:
    """Code-aware report cleaner for the v3 / agent_team chain.

    Returns ``(body, stats)``. ``internal_node_ids`` is the set of DAG node ids
    this run actually minted, sourced from the caller's own evidence
    tracking; when empty the node-id scrub is a no-op rather than falling
    back to a character-class guess.
    """
    stats: dict[str, Any] = {
        "wrapper_stripped": False,
        "fence_unbalanced": False,
        "preamble_trimmed_chars": 0,
        "preamble_trim_skipped": False,
        "ids_scrubbed": 0,
        "node_ids_scrubbed": 0,
        "lines_space_collapsed": 0,
    }
    if not report:
        return report, stats

    body, wrapper = strip_document_fence_wrapper(report)
    stats["wrapper_stripped"] = wrapper

    label_re = _node_id_label_pattern(internal_node_ids or set())

    segments = segment_lines(body)
    cleaned = [
        line if in_code else _scrub_prose_line(line, label_re=label_re, stats=stats)
        for line, in_code in segments
    ]
    stats["fence_unbalanced"] = bool(segments) and _fence_state_open(segments)
    if stats["fence_unbalanced"]:
        # Never auto-repair: an odd fence count cannot tell us whether the
        # dangling marker is an unclosed opener (append) or an orphan closer
        # (drop), and guessing wrong renders an empty code block.
        logger.warning(
            "clean_report_v3: unbalanced code fence in report body "
            "(%d lines) — leaving markup as written", len(segments),
        )

    body = _collapse_prose_blank_runs(cleaned)
    body = _trim_preamble(body, stats=stats)
    return body.strip(), stats


def _fence_state_open(segments: list[tuple[str, bool]]) -> bool:
    """True when the fence scan ends inside a code block."""
    in_code = False
    for line, _ in segments:
        if _FENCE_RE.match(line):
            in_code = not in_code
    return in_code


def _collapse_prose_blank_runs(lines: list[str]) -> str:
    """``\\n{3,}`` → ``\\n\\n``, but only outside fenced code.

    Blank-line runs inside a code block can be meaningful (and are certainly
    not the LLM's spacing habit this rule targets), so the collapse is applied
    per prose region instead of over the whole document.
    """
    out: list[str] = []
    in_code = False
    blanks = 0
    for line in lines:
        if _FENCE_RE.match(line):
            in_code = not in_code
            out.append(line)
            blanks = 0
            continue
        if in_code:
            out.append(line)
            continue
        if line.strip():
            blanks = 0
            out.append(line)
            continue
        blanks += 1
        if blanks <= 1:
            out.append(line)
    return "\n".join(out)


def _trim_preamble(body: str, *, stats: dict[str, Any]) -> str:
    """Drop the writer's chatter before the first heading, when it is short.

    Same trigger as 1.0 (``^#\\s``), plus two guards:

    - the heading must be **outside** a fence, so a ``# comment`` line inside a
      shell/Python block can no longer behead the document;
    - what gets removed must be preamble-sized (``max(400, len//10)`` chars),
      so a mis-detected heading can cost at most a short prefix instead of the
      whole report.

    Both guards only ever *suppress* a trim 1.0 would have done, so this can
    never discard text 1.0 kept.
    """
    offset = 0
    for line, in_code in segment_lines(body):
        if not in_code and _ATX_HEADING_RE.match(line):
            break
        offset += len(line) + 1
    else:
        # No prose heading at all (the only ``#`` lines are inside fences, or
        # the report simply has no headings) — nothing to trim.
        return body
    if offset == 0:
        return body
    budget = max(_PREAMBLE_MIN_BUDGET, len(body) // _PREAMBLE_BUDGET_RATIO)
    if offset > budget:
        stats["preamble_trim_skipped"] = True
        logger.warning(
            "clean_report_v3: first prose heading is %d chars in (budget %d) — "
            "keeping the body intact instead of trimming", offset, budget,
        )
        return body
    stats["preamble_trimmed_chars"] = offset
    return body[offset:]
