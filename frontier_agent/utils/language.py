"""Generic language detection and instruction utilities."""

from __future__ import annotations

import json
import logging
import os
import re
from collections.abc import Awaitable, Callable
from typing import Any

logger = logging.getLogger(__name__)

# ISO/short code → display label. Free-form labels (e.g. "simplified Chinese")
# pass through ``normalize_language`` unchanged.
LANGUAGE_NAMES: dict[str, str] = {
    "en": "English",
    "zh": "Simplified Chinese",
    "zh-cn": "Simplified Chinese",
    "zh-hans": "Simplified Chinese",
    "zh-tw": "Traditional Chinese",
    "zh-hant": "Traditional Chinese",
    "ja": "Japanese",
    "ko": "Korean",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "ru": "Russian",
    "ar": "Arabic",
    "pt": "Portuguese",
    "it": "Italian",
}


def is_language_detect_enabled() -> bool:
    """Return whether automatic language inference should run at all.

    Toggle: ``LANGUAGE_DETECT_ENABLED`` env var (default ``true``).
    Treat ``0`` / ``false`` / ``no`` / ``off`` (case-insensitive) as off.
    Covers both the heuristic (in :func:`resolve_language`) and any LLM
    detector — when off, ``state["language"]`` flows through as the user
    supplied it (typically ``"auto"`` → normalized to ``""`` → no language
    line in any prompt).
    """
    val = os.getenv("LANGUAGE_DETECT_ENABLED", "true").strip().lower()
    return val not in {"0", "false", "no", "off"}


def is_chinese_label(language: str | None) -> bool:
    """Return True if ``language`` denotes any flavour of Chinese.

    Accepts ISO codes (``"zh"``, ``"zh-cn"``, ``"zh-tw"``) and free-form
    labels emitted by an LLM detector (``"simplified Chinese"``,
    ``"中文"``). Used by report-side helpers that swap section headings
    or wording based on the report's target language.
    """
    if not language:
        return False
    label = language.strip().lower()
    return bool(label) and (
        label.startswith("zh") or "chinese" in label or "中文" in language
    )


def detect_language(text: str) -> str:
    """Heuristic language detection from text.

    Returns a free-form display label (e.g. ``"Simplified Chinese"``).
    Biased toward CJK on Chinese-English mixed input — even a small amount
    of Chinese in an otherwise-English query is taken as a signal that the
    user wants the answer in Chinese.
    """
    if not text:
        return "English"

    total = max(len(text), 1)
    cjk = len(re.findall(r"[一-鿿]", text))
    kana = len(re.findall(r"[぀-ヿ]", text))
    hangul = len(re.findall(r"[가-힯]", text))
    cyrillic = len(re.findall(r"[Ѐ-ӿ]", text))
    arabic = len(re.findall(r"[؀-ۿ]", text))

    # Kana is unique to Japanese — Chinese never uses it, so any kana wins.
    if kana > 0:
        return "Japanese"
    if hangul >= max(2, total * 0.05):
        return "Korean"
    # Mixed Chinese-English biases to Chinese: ≥2 CJK chars (or 5% density)
    # flips the answer to Chinese. The ``>=`` matters for short prompts —
    # ``"你好"`` / ``"中文?"`` / ``"你好 GPT"`` all have exactly 2 CJK chars.
    if cjk >= max(2, total * 0.05):
        return "Simplified Chinese"
    if cyrillic > total * 0.30:
        return "Russian"
    if arabic > total * 0.30:
        return "Arabic"
    return "English"


def normalize_language(language: str | None) -> str:
    """Map any accepted language value to a display label.

    Returns ``""`` for unset / ``"auto"`` so callers can treat it as
    "no preference set". Known short codes are expanded; unknown values
    pass through verbatim (which is what we want for free-form LLM output
    like ``"simplified Chinese"``).
    """
    if not language:
        return ""
    raw = language.strip()
    if not raw or raw.lower() == "auto":
        return ""
    code = raw.lower()
    return LANGUAGE_NAMES.get(code, raw)


def resolve_language(state: dict[str, Any]) -> str:
    """Get the resolved language label from state, detecting if 'auto'.

    Used by nodes that fire before / alongside the LLM detector, where
    ``state["language"]`` may still be ``"auto"``. After the detector runs
    state already holds a resolved label, and this function is a no-op
    normalize.

    Respects :func:`is_language_detect_enabled`: when the toggle is off
    and the user didn't pin a language, returns ``""`` instead of running
    the heuristic — so operators can fully disable automatic language
    inference end-to-end.
    """
    lang = state.get("language", "auto")
    if not lang or str(lang).lower() == "auto":
        if not is_language_detect_enabled():
            return ""
        return detect_language(state.get("original_question", ""))
    return normalize_language(lang)


def language_instruction(language: str) -> str:
    """Return a prompt fragment forcing the model to answer in ``language``.

    Returns an empty string when no instruction is needed:
    * unset / empty / ``"auto"`` — caller didn't ask for a language
    * English / ``"en"`` — the model's default, no need to nudge

    Callers can therefore always append ``language_instruction(...)`` to
    their prompt without an extra ``if`` — disabled detection means an
    empty string and the prompt line just isn't there.
    """
    label = normalize_language(language)
    if not label or label == "English":
        return ""
    return (
        f"\n\nIMPORTANT: You MUST respond entirely in {label}. "
        f"All text output must be in {label}."
    )


# ── LLM-backed detector ───────────────────────────────────────────────
#
# A single LLM call that resolves the answer language for *any* language,
# covering the cases the character heuristic above collapses to English
# (Spanish, Vietnamese, Thai, …). Lives here in core so every workflow can
# share it; both entry points fall back to :func:`detect_language` on any LLM failure.

# Cap the user question fed to the detector. The heuristic only needs a few
# hundred chars to gauge CJK density, but the LLM detector also has to see an
# explicit language request (e.g. "answer in English") — those tend to sit at
# the END of a longer current-turn message (pasted content first, instruction
# last), so this is a TAIL cap (see the slice below), not a prefix cap.
# Without a cap a 100k-char question would balloon every detector call; 8000
# (~2000 tokens) trades a still-negligible per-call cost for covering most
# real messages. An explicit request placed earlier than this many chars from
# the end of a single message is still missed.
_MAX_DETECT_INPUT_CHARS = 8000

DETECT_PROMPT = """Determine the most appropriate language to respond in for this user question.

User question:
{question}

Rules (apply in order):
1. If the question explicitly states or requests a specific response language \
(e.g. "answer in English", "用英文回答", "please reply in Japanese", "respond in \
español"), honor that explicit request — it always wins over the language the \
question itself happens to be written in.
2. Otherwise, infer from the question's own language:
   - Chinese-English mixed input → bias to Chinese ("simplified Chinese" or "traditional Chinese").
   - Pure non-English input → that language's English name (e.g. "Japanese", "Spanish", "Korean").
   - Pure English → "English".

Output a single JSON object with exactly one key: "language".
Output ONLY valid JSON, no other text.

Example: {{"language": "simplified Chinese"}}
"""


def _parse_language(content: str) -> str:
    """Extract ``language`` from the first JSON object in ``content``."""
    start = content.find("{")
    end = content.rfind("}") + 1
    if start < 0 or end <= start:
        raise ValueError("no JSON object in response")
    data = json.loads(content[start:end])
    label = str(data.get("language", "")).strip()
    if not label:
        raise ValueError("empty 'language' field")
    return label


async def detect_language_from_prompt(
    question: str,
    ask: Callable[[str], Awaitable[str]],
) -> str:
    """Resolve the answer language for ``question`` via a single LLM call.

    ``ask`` is any async callable that sends a prompt string to an LLM and
    returns its raw text reply — so a caller with a bring-your-own model
    (``llm.chat``) can drive the detector without a :class:`NodeContext`.
    Caller is expected to gate via :func:`is_language_detect_enabled`; this
    always issues the call. On any failure (parse error, LLM error) falls back
    to the heuristic :func:`detect_language`. Returns the normalized display
    label (e.g. ``"Simplified Chinese"``).
    """
    # Tail, not prefix: an explicit language request ("answer in English")
    # is far more likely to sit at the end of a long message (after pasted
    # content) than in the first N chars.
    truncated = question[-_MAX_DETECT_INPUT_CHARS:]
    try:
        raw = await ask(DETECT_PROMPT.format(question=truncated))
        label = _parse_language(raw)
        normalized = normalize_language(label) or label
        logger.debug("LLM language detector: %r → %r", label, normalized)
        return normalized
    except Exception as exc:
        fallback = detect_language(truncated)
        logger.warning(
            "LLM language detector failed (%s); heuristic fallback → %r",
            exc,
            fallback,
        )
        return fallback
