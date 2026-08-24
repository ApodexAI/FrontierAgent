"""Deliverable-preservation + citation-bracket-safety helpers shared by."""

from __future__ import annotations

import logging
import re
from typing import Any

from workflows._shared.cited_report_finalizer import (
    strip_trailing_references,
)

logger = logging.getLogger(__name__)

FENCE = "`" * 3

_OUTPUTS_PATH_RE = re.compile(r"/outputs/[\w./\-]+")

# Sentinels for :func:`neutralize_non_citation_brackets`. Private-use
# codepoints distinct from ``report_clean``'s own ````/````
# sentinels, so the two masking passes can never collide when nested.
_MASK_OPEN2, _MASK_CLOSE2 = "\U000e0000", "\U000e0001"

# Same bracket shape ``citation_numbers``/``validate_citation_body`` parse as
# a (possibly multi-index) citation: ``[1]``, ``[1, 2]``, ``[1; 2]``. Kept
# structurally identical (not imported — that regex is private to the shared
# module) so this only ever masks what the validator would otherwise treat as
# a citation.
_BRACKET_GROUP_RE = re.compile(r"\[(\s*\d[\d,;\s]*)\]")

# A single-backtick inline code span on one line. Code never carries a
# citation — only data (``arr[0]``, line numbers, ids) — so any bracketed
# integer found in here is masked unconditionally, regardless of magnitude.
_INLINE_CODE_RE = re.compile(r"`[^`\n]+`")

# Generous floor under the plausibility ceiling (see
# :func:`_is_plausible_citation_group`), independent of any particular
# caller's ``max_ref``.
_CITATION_ABSOLUTE_CEILING = 100

# Headroom above ``max_ref`` inside which a bracketed integer is still
# treated as a possible (merely out-of-range) citation, so it reaches
# ``validate_citation_body``'s existing orphan check instead of being
# silently masked as data. See :func:`_is_plausible_citation_group`.
_CITATION_CEILING_HEADROOM = 60


class DeliverablesLostError(RuntimeError):
    """The rewrite dropped every instance of a deliverable class.

    Raised when the native answer carried ``/outputs/`` paths or fenced code
    blocks and the rewrite has none left. Citations do not compensate for
    losing the artefact the user asked for, so the caller fails open.
    """


def deliverable_metrics(report: str, *, native: str) -> dict[str, Any]:
    """Measure what the rewrite kept from the native answer. Never mutates.

    ``/outputs/`` paths are counted only in native's *body* — its own
    trailing References/Sources footer is excluded before matching. The
    writer is deliberately instructed to never reproduce that footer (Gate
    3 replaces it with the canonical, evidence-card-derived list
    unconditionally), so a path that only appears there as a self-citation
    footnote (e.g. "[4] ... full results at `/outputs/x.txt`") was never
    something the rewrite could have kept — Gate 1 has no evidence card for
    a self-generated analysis file, only for web-sourced ones. Counting it
    as "lost" measured a promise the writer contract never made. A path
    mentioned in the body (e.g. "saved to `/outputs/report.md`") still
    counts: that is real prose the fidelity rule protects.
    """
    native_body = strip_trailing_references(native or "")
    native_paths = set(_OUTPUTS_PATH_RE.findall(native_body))
    report_paths = set(_OUTPUTS_PATH_RE.findall(report or ""))
    native_blocks = (native or "").count(FENCE) // 2
    report_blocks = (report or "").count(FENCE) // 2
    return {
        "native_outputs_paths": len(native_paths),
        "report_outputs_paths": len(report_paths),
        "outputs_paths_kept": len(native_paths & report_paths),
        "native_code_blocks": native_blocks,
        "report_code_blocks": report_blocks,
        "fence_balanced": (report or "").count(FENCE) % 2 == 0,
        "chars": len(report or ""),
        "bloat_ratio": (
            round(len(report or "") / len(native), 3) if native else None
        ),
    }


def deliverables_lost(metrics: dict[str, Any]) -> bool:
    """True only on TOTAL loss of a deliverable class the native answer had.

    Partial loss is deliberately allowed: the writer may legitimately merge or
    reorder paths. Total loss is different — the artefact the user asked for
    is simply gone.

    The path test is ``outputs_paths_kept`` (the intersection with the native
    answer's paths), not ``report_outputs_paths`` (any path at all): a writer
    that drops the real ``/outputs/chart.png`` and invents a *different*
    ``/outputs/...`` path has lost the artefact just as completely.
    """
    if metrics.get("native_outputs_paths", 0) > 0 and not metrics.get(
        "outputs_paths_kept", 0,
    ):
        return True
    return bool(
        metrics.get("native_code_blocks", 0) > 0
        and not metrics.get("report_code_blocks", 0),
    )


def _bracket_numbers(group_text: str) -> list[int]:
    return [int(n) for n in re.findall(r"\d+", group_text)]


def _is_plausible_citation_group(nums: list[int], *, max_ref: int) -> bool:
    """True iff every number in a bracket group could be a real citation.

    Citations are 1-indexed against the reference list Gate 1 built
    (bounded by fast_reporter_v1's own citation cap). The ceiling is
    ``max(max_ref + headroom, 100)`` — a floor of
    ``_CITATION_ABSOLUTE_CEILING`` so a thin reference list cannot shrink
    the safety margin, plus headroom above ``max_ref`` itself so an
    ordinary off-by-a-few citation mistake still reaches
    ``validate_citation_body``'s orphan-index repair path rather than
    being silently masked here as data. ``0`` is always excluded:
    1-indexing makes it structurally impossible as a citation.
    """
    if not nums:
        return False
    ceiling = max(
        max_ref + _CITATION_CEILING_HEADROOM, _CITATION_ABSOLUTE_CEILING,
    )
    return all(1 <= n <= ceiling for n in nums)


def neutralize_non_citation_brackets(
    text: str, *, max_ref: int,
) -> tuple[str, list[str]]:
    """Mask bracketed integers that cannot be real ``[N]`` citations.

    Two independent reasons a bracketed-integer group is masked:

    * it sits inside inline code (a single-backtick span) — code never
      carries a citation, only data (``qs[0]``, indices, ids). **Note:** Fenced
      (triple-backtick) code blocks are NOT handled by this function; the
      caller must strip fenced blocks out of the body first by wrapping this
      call in :func:`workflows._shared.report_clean.with_code_protected`.
    * in prose, its value is implausible as a 1-indexed citation into a
      ``max_ref``-entry list (see :func:`_is_plausible_citation_group`) —
      catches ``[684, 821]``, a plain numeric range, not a footnote.

    Masking, not deleting: returns ``(masked_text, tokens)`` where each
    masked span's exact original text is kept in ``tokens``, indexed by the
    sentinel embedded in ``masked_text``.

    Granularity is per-number, not per-group, for a MIXED group (some
    numbers plausible, some not — e.g. ``[0, 3]`` against a 40-reference
    list): only the implausible members are masked; the plausible ones
    survive as a live, real citation.
    """
    if not text:
        return text, []
    if _MASK_OPEN2 in text or _MASK_CLOSE2 in text:
        # Astronomically unlikely (private-use codepoints), but corrupting
        # the report on a body that already contains our sentinel is far
        # worse than leaving these brackets unmasked for this one run.
        logger.warning(
            "report_fast_guard: body already contains the neutralizer's "
            "sentinel — skipping bracket neutralization for this run",
        )
        return text, []

    tokens: list[str] = []

    def _mask(literal: str) -> str:
        tokens.append(literal)
        return f"{_MASK_OPEN2}{len(tokens) - 1}{_MASK_CLOSE2}"

    def _mask_match(match: re.Match[str]) -> str:
        return _mask(match.group(0))

    def _is_plausible_number(n: int) -> bool:
        return _is_plausible_citation_group([n], max_ref=max_ref)

    def _mask_if_implausible(match: re.Match[str]) -> str:
        nums = _bracket_numbers(match.group(1))
        if _is_plausible_citation_group(nums, max_ref=max_ref):
            return match.group(0)                          # all plausible
        plausible = [n for n in nums if _is_plausible_number(n)]
        if not plausible:
            return _mask_match(match)                       # all implausible
        # Mixed: split into individual brackets, then mask only the
        # implausible ones. Order preserved.
        return "".join(
            f"[{n}]" if _is_plausible_number(n) else _mask(f"[{n}]")
            for n in nums
        )

    def _mask_inline_code_span(match: re.Match[str]) -> str:
        return _BRACKET_GROUP_RE.sub(_mask_match, match.group(0))

    masked = _INLINE_CODE_RE.sub(_mask_inline_code_span, text)
    masked = _BRACKET_GROUP_RE.sub(_mask_if_implausible, masked)
    return masked, tokens


__all__ = [
    "FENCE",
    "DeliverablesLostError",
    "deliverable_metrics",
    "deliverables_lost",
    "neutralize_non_citation_brackets",
]
