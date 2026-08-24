"""Token estimation and text truncation for context compression."""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any, Literal

# ---------------------------------------------------------------------------
# Lazy-loaded tiktoken encoder
# ---------------------------------------------------------------------------

# Broad CJK regex: Unified Ideographs, Ext-A, radicals, strokes,
# Hiragana, Katakana, CJK compatibility, fullwidth forms
_CJK_RE = re.compile(
    r"[\u2e80-\u2eff\u3000-\u303f\u3040-\u30ff\u3400-\u4dbf"
    r"\u4e00-\u9fff\uf900-\ufaff\uff00-\uffef]"
)


def _get_tokenizer() -> Any:
    """Return the cl100k_base encoder without ever blocking the loop.

    Delegates to the shared non-blocking loader: returns ``None`` while
    the encoder is still loading on its daemon thread (callers fall back
    to the CJK heuristic), never a synchronous network fetch on the loop
    thread. See ``tokenizer.py`` for the 2026-06 wedge history.
    """
    from frontier_agent.core.runtime.loop.tokenizer import get_encoding_nonblocking
    return get_encoding_nonblocking("cl100k_base")


# ---------------------------------------------------------------------------
# Token estimation
# ---------------------------------------------------------------------------


def estimate_tokens(text: str) -> int:
    """Estimate the token count for a plain text string.

    Uses tiktoken cl100k_base when available; falls back to a CJK-aware
    heuristic (each CJK character ≈ 1 token, Latin text ≈ chars / 4).
    """
    if not text:
        return 0

    enc = _get_tokenizer()
    if enc is not None:
        try:
            return len(enc.encode(text, disallowed_special=()))
        except Exception:
            pass

    # Heuristic fallback
    cjk_count = len(_CJK_RE.findall(text))
    other_count = len(text) - cjk_count
    return cjk_count + (other_count // 4)


def truncate_text_to_tokens(
    text: str,
    max_tokens: int,
    *,
    marker: str = "\n[... older context truncated to fit token budget ...]",
    estimator: Callable[[str], int] = estimate_tokens,
    keep: Literal["head", "tail"] = "head",
) -> str:
    """Keep the largest text prefix — or suffix — that fits a token budget.

    ``estimator`` and ``marker`` are injectable so specialized callers can
    preserve their existing tokenizer and user-facing truncation language.

    ``keep="head"`` (default) drops the newest text and appends the marker —
    right for summaries and tool output, where the opening lines carry the
    identity of the content. ``keep="tail"`` drops the oldest text and
    prepends the marker — right for reasoning traces, where the conclusion
    and the tool-use intent sit at the end.

    Note that ``estimator`` need not be monotonic in the slice length, so the
    binary search returns a near-maximal slice rather than a provably maximal
    one. The budget itself is always respected.
    """
    if keep not in ("head", "tail"):
        raise ValueError(f"keep must be 'head' or 'tail', got {keep!r}")
    if max_tokens <= 0:
        return ""
    if estimator(text) <= max_tokens:
        return text
    # A marker wider than the whole budget would leave room for a single
    # character of real text and still overshoot; drop it instead.
    effective_marker = "" if estimator(marker) >= max_tokens else marker
    target = max(1, max_tokens - estimator(effective_marker))
    lo, hi = 0, len(text)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        chunk = text[:mid] if keep == "head" else text[-mid:]
        if estimator(chunk) <= target:
            lo = mid
        else:
            hi = mid - 1
    if keep == "head":
        return text[:lo] + effective_marker
    return effective_marker + text[-lo:] if lo else effective_marker
