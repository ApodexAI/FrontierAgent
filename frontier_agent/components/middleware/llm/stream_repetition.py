"""Stream-level repetition detector — kills degenerate LLM output loops."""

from __future__ import annotations

import logging
from typing import Any

from frontier_agent.components.middleware.llm.base import (
    LLMCallContext,
    LLMMiddleware,
)
from frontier_agent.core.llm import StreamDelta

logger = logging.getLogger(__name__)

__all__ = ["StreamRepetitionDetectorMiddleware"]


_STATE_KEY = "_stream_repetition_state"


class StreamRepetitionDetectorMiddleware(LLMMiddleware):
    """Abort streams that fall into exact-pattern repetition loops.

    Args:
        min_pattern_len: shortest repeated substring to detect (chars).
            Below ~20 chars there are too many natural English word
            repeats; below ~10 chars the false-positive rate explodes.
        max_pattern_len: longest repeated substring to detect. Caps
            the scan cost — patterns larger than ~500 chars are rare
            in real degenerate output (the model has run out of ideas
            and is replaying short fragments, not paragraphs).
        min_repeats: how many consecutive identical copies trigger an
            abort. 6 is the tuned value; lower triggers earlier
            (false-positive risk on legitimate enumerations like "1) X
            2) X 3) X") and higher wastes tokens before catching.
        min_text_len: don't run the scan until at least this much text
            has streamed. Short outputs (e.g. a one-line answer) can't
            contain a 6-fold repeat without being legitimately short.
        check_interval: re-scan every N new characters. Smaller means
            faster detection at the cost of more CPU on the hot path.
    """

    def __init__(
        self,
        *,
        min_pattern_len: int = 30,
        max_pattern_len: int = 500,
        min_repeats: int = 6,
        min_text_len: int = 800,
        check_interval: int = 200,
    ) -> None:
        if min_pattern_len < 1:
            raise ValueError("min_pattern_len must be >= 1")
        if max_pattern_len < min_pattern_len:
            raise ValueError("max_pattern_len must be >= min_pattern_len")
        if min_repeats < 2:
            raise ValueError("min_repeats must be >= 2 (1 repeat = no repeat)")
        if min_text_len < min_pattern_len * min_repeats:
            # A repeat of length L appearing R times needs ≥ L*R chars.
            # Without this floor, the scan would run on text that's too
            # short to ever match — pure CPU waste.
            min_text_len = min_pattern_len * min_repeats
        self._min_pattern_len = min_pattern_len
        self._max_pattern_len = max_pattern_len
        self._min_repeats = min_repeats
        self._min_text_len = min_text_len
        self._check_interval = check_interval

    @property
    def name(self) -> str:
        return "stream_repetition_detector"

    async def on_chunk(
        self,
        ctx: LLMCallContext,
        delta: StreamDelta,
        full_text: str,
    ) -> bool:
        """Run the exact-pattern check against the tail of ``full_text``.

        Returns ``True`` to abort the stream. State is per-call,
        keyed in ``ctx.metadata`` so concurrent streams through one
        middleware instance don't interfere.
        """
        state = self._get_state(ctx)
        if state["detected"]:
            return True  # belt-and-suspenders: shouldn't fire post-abort

        text_len = len(full_text)
        if text_len < self._min_text_len:
            return False

        state["chars_since_check"] += len(delta.content or "")
        if state["chars_since_check"] < self._check_interval:
            return False
        state["chars_since_check"] = 0

        match = self._scan_tail(full_text)
        if match is None:
            return False

        pat_len, repeats = match
        state["detected"] = True
        pattern_preview = full_text[-pat_len:][:80]
        logger.warning(
            "StreamRepetitionDetector: aborting stream after %d chars "
            "(call_index=%d) — exact pattern len=%d repeats=%d preview=%r",
            text_len, ctx.call_index, pat_len, repeats, pattern_preview,
        )
        ctx.metadata["stream_repetition_pattern_len"] = pat_len
        ctx.metadata["stream_repetition_repeats"] = repeats
        return True

    # ── State scoping ───────────────────────────────────────────────

    def _get_state(self, ctx: LLMCallContext) -> dict[str, Any]:
        """Per-call mutable state stored in ``ctx.metadata``.

        Keyed by ``call_index`` so a single middleware instance can
        observe many concurrent streams without trampling state. The
        middleware doesn't allocate dicts globally — each call gets
        its own and is GC'd when the call completes.
        """
        bag = ctx.metadata.setdefault(_STATE_KEY, {})
        key = ctx.call_index
        if key not in bag:
            bag[key] = {
                "chars_since_check": 0,
                "detected": False,
            }
        return bag[key]

    # ── Detection algorithm ─────────────────────────────────────────

    def _scan_tail(self, text: str) -> tuple[int, int] | None:
        """Return ``(pattern_len, repeat_count)`` if a repeat is found.

        Walks ``pattern_len`` from min upward; for each
        length takes the tail substring of that length and counts how
        many consecutive copies sit at the very end of the text.
        First length with ≥ ``min_repeats`` consecutive copies wins.
        """
        text_len = len(text)
        # A repeat must fit ``min_repeats`` copies at the tail — past
        # ``text_len // min_repeats`` chars there isn't room.
        max_scan = min(self._max_pattern_len, text_len // self._min_repeats)
        if max_scan < self._min_pattern_len:
            return None

        for pat_len in range(self._min_pattern_len, max_scan + 1):
            pattern = text[text_len - pat_len:]
            repeats = 1
            pos = text_len - pat_len
            while pos >= pat_len and text[pos - pat_len: pos] == pattern:
                repeats += 1
                pos -= pat_len
            if repeats >= self._min_repeats:
                return pat_len, repeats
        return None
