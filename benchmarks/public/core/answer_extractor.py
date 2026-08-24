"""Extract a concise answer from a research report."""

from __future__ import annotations

import logging
import re
import string
from typing import Any

from frontier_agent.core.llm import LLMClient
from frontier_agent.core.llm_utils import extract_boxed_content
from frontier_agent.core.messages import system_msg, text_of, user_msg

logger = logging.getLogger(__name__)

_MC_DIRECT_RE = re.compile(
    r"^\s*\*?\*?\(?([A-Za-z])\)?\s*[.)]?\*?\*?\s*$",
)
_MC_EXPLICIT_PATTERNS = [
    re.compile(
        r"\bFINAL\s+ANSWER\s*:\s*\*?\*?\(?([A-Za-z])\)?\*?\*?(?![A-Za-z])",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:the\s+)?(?:correct\s+)?(?:answer|choice)\s*"
        r"(?:is|:)\s*(?:option\s*)?\*?\*?\(?([A-Za-z])\)?\*?\*?(?![A-Za-z])",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:choose|select)\s+(?:option\s*)?\*?\*?\(?([A-Za-z])\)?"
        r"\*?\*?(?![A-Za-z])",
        re.IGNORECASE,
    ),
]

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_SYSTEM_MC = """\
You are an answer extractor. Given a research report and the original question, \
output ONLY the single capital letter (A, B, C, D, or E) that best answers the question. \
Do not explain. Output nothing but the letter."""

_SYSTEM_EXACT = """\
You are an answer extractor. Given a research report and the original question, \
output ONLY the shortest possible answer.

Format rules:
- Fractions: use a/b (not decimal, not LaTeX)
- Yes/No questions: output exactly 'Yes' or 'No'
- Numbers: exact value (no approximations)
- Multiple values: comma-separated
- Math expressions: standard notation, no LaTeX
- Strip all markdown formatting

Output nothing but the answer itself."""

_MAX_EXTRACTOR_REPORT_CHARS = 6000
_ANSWER_MARKER_RE = re.compile(
    r"(?im)^\s*(?:#{1,6}\s*)?"
    r"(?:final\s+answer|answer|result)\s*[:：]\s*(.+?)\s*$",
)
_BOLD_ANSWER_RE = re.compile(r"\*\*([^*\n]{1,120})\*\*")


# ---------------------------------------------------------------------------
# LLM-based extraction (primary)
# ---------------------------------------------------------------------------

async def extract_answer_llm(
    report: dict[str, Any] | None,
    question_text: str,
    answer_type: str,
    llm: LLMClient,
) -> str:
    """Use the LLM to pull a concise answer from the report.

    Falls back to regex if the LLM call fails.
    """
    if not report:
        return ""

    # Try boxed extraction first (deterministic, no LLM cost)
    report_content = (
        report.get("content", "")
        if isinstance(report, dict)
        else str(report)
    )
    boxed = extract_boxed_content(report_content)
    if boxed:
        if answer_type == "multipleChoice":
            cleaned = boxed.strip()
            if (
                len(cleaned) == 1
                and cleaned.upper() in string.ascii_uppercase
            ):
                return cleaned.upper()
        else:
            return boxed

    content: str = report.get("content", "")
    if not content:
        return ""

    regex_answer = extract_answer_regex(report, answer_type)
    if regex_answer:
        return regex_answer

    system = _SYSTEM_MC if answer_type == "multipleChoice" else _SYSTEM_EXACT

    content = _compact_report_for_extraction(content)

    user_body = f"## Original Question\n{question_text}\n\n## Research Report\n{content}"

    try:
        resp = await llm.chat([
            system_msg(system),
            user_msg(user_body),
        ])
        raw: str = text_of(resp.content).strip()
        logger.debug("LLM extractor raw output: %r", raw)

        if answer_type == "multipleChoice":
            return _clean_mc(raw)
        return _clean_exact(raw)

    except Exception as exc:
        logger.warning("LLM extraction failed (%s), falling back to regex", exc)
        return extract_answer_regex(report, answer_type)


def _clean_mc(raw: str) -> str:
    """Normalise LLM output to a single uppercase letter.

    Preference order:
    1. Explicit "FINAL ANSWER: X" marker
    2. "the answer/choice is X" pattern
    3. Direct single letter (only if the entire response is one letter)
    4. Return empty string for ambiguous prose (do NOT guess from
       random standalone letters).
    """
    raw = raw.strip()
    # 1. Explicit FINAL ANSWER / answer / choice markers.
    for pat in _MC_EXPLICIT_PATTERNS:
        m = pat.search(raw)
        if m:
            return m.group(1).upper()
    # 3. Direct single letter (entire response is just one letter)
    m = _MC_DIRECT_RE.match(raw)
    if m:
        return m.group(1).upper()
    # 4. Reject ambiguous prose — do not guess
    return ""


def _clean_exact(raw: str) -> str:
    """Strip quotes / markdown / LaTeX noise from LLM output."""
    # Strip <think>...</think> blocks that leak from thinking models
    raw = re.sub(r'<think>[\s\S]*?</think>\s*', '', raw)
    raw = raw.strip().strip('"').strip("'").strip("`")
    # Remove leading "Answer: " prefix if LLM added one
    raw = re.sub(r"^(?:answer|the answer is)[:\s]+", "", raw, flags=re.IGNORECASE)
    # Strip markdown bold
    raw = re.sub(r"^\*\*(.+?)\*\*$", r"\1", raw)
    raw = re.sub(r"\*\*$", "", raw)
    # Strip LaTeX display delimiters: $...$, \(...\), \[...\]
    m = re.match(r"^\$(.+)\$$", raw, re.DOTALL)
    if m:
        raw = m.group(1)
    m = re.match(r"^\\\((.+)\\\)$", raw, re.DOTALL)
    if m:
        raw = m.group(1)
    m = re.match(r"^\\\[(.+)\\\]$", raw, re.DOTALL)
    if m:
        raw = m.group(1)

    cleaned = raw.strip()

    # If still too long (>100 chars), try extracting common patterns
    if len(cleaned) > 100:
        boxed = extract_boxed_content(cleaned)
        if boxed and len(boxed) <= 100:
            return boxed
        m = re.search(
            r"(?:answer|result)\s*(?:is|=)\s*[:\s]*"
            r"(.{1,80}?)(?:\.|$)",
            cleaned,
            re.IGNORECASE,
        )
        if m:
            return m.group(1).strip()
        m = re.search(
            r"^[\s\S]*?(\d+(?:/\d+)?(?:\.\d+)?)\s*$",
            cleaned,
        )
        if m:
            return m.group(1)
        m = re.search(r"\b(yes|no)\b", cleaned, re.IGNORECASE)
        if m:
            return m.group(1).capitalize()

    return cleaned


# ---------------------------------------------------------------------------
# Regex-based extraction (fallback)
# ---------------------------------------------------------------------------

_MC_PATTERNS = [
    *_MC_EXPLICIT_PATTERNS,
    _MC_DIRECT_RE,
]


def extract_answer_regex(report: dict[str, Any] | None, answer_type: str) -> str:
    """Pure-regex extraction — no LLM call."""
    if not report:
        return ""

    content: str = report.get("content", "")
    if not content:
        return ""

    if answer_type == "multipleChoice":
        return _extract_mc_regex(content)
    return _extract_exact_regex(content)


def _extract_mc_regex(content: str) -> str:
    for pat in _MC_PATTERNS:
        m = pat.search(content)
        if m:
            return m.group(1).upper()
    letters = re.findall(r"\b([A-Ea-e])\b", content)
    if letters:
        return letters[-1].upper()
    return ""


def _extract_exact_regex(content: str) -> str:
    for pat in (_ANSWER_MARKER_RE,):
        m = pat.search(content)
        if m:
            cleaned = _clean_exact(m.group(1))
            if cleaned and len(cleaned) <= 120:
                return cleaned

    for line in content.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("---") or len(line) < 3:
            continue
        marker = _ANSWER_MARKER_RE.match(line)
        if marker:
            cleaned = _clean_exact(marker.group(1))
            if cleaned and len(cleaned) <= 120:
                return cleaned
        line = re.sub(r"^\*\*(.+?)\*\*$", r"\1", line)
        bold = _BOLD_ANSWER_RE.search(line)
        if bold:
            return _clean_exact(bold.group(1))
        if len(line) <= 120:
            return _clean_exact(line)
    return ""


def _compact_report_for_extraction(content: str) -> str:
    """Return a bounded report slice that preserves likely answer regions."""
    if len(content) <= _MAX_EXTRACTOR_REPORT_CHARS:
        return content

    windows: list[str] = []
    for match in _ANSWER_MARKER_RE.finditer(content):
        start = max(0, match.start() - 1200)
        end = min(len(content), match.end() + 1200)
        windows.append(content[start:end])
        if sum(len(w) for w in windows) >= _MAX_EXTRACTOR_REPORT_CHARS // 2:
            break

    marker_budget = sum(len(w) for w in windows)
    remaining = max(1000, _MAX_EXTRACTOR_REPORT_CHARS - marker_budget)
    head_len = remaining // 2
    tail_len = remaining - head_len
    pieces = [content[:head_len]]
    pieces.extend(windows)
    pieces.append(content[-tail_len:])
    compacted = "\n\n...[middle truncated for answer extraction]...\n\n".join(pieces)
    if len(compacted) > _MAX_EXTRACTOR_REPORT_CHARS:
        if windows:
            marker_text = "\n\n".join(windows)
            remaining = max(0, _MAX_EXTRACTOR_REPORT_CHARS - len(marker_text) - 80)
            head_len = remaining // 2
            tail_len = remaining - head_len
            compacted = (
                content[:head_len]
                + "\n\n...[middle truncated for answer extraction]...\n\n"
                + marker_text
                + "\n\n...[middle truncated for answer extraction]...\n\n"
                + content[-tail_len:]
            )
        else:
            half = (_MAX_EXTRACTOR_REPORT_CHARS - 80) // 2
            compacted = (
                compacted[:half]
                + "\n\n...[extractor prompt truncated]...\n\n"
                + compacted[-half:]
            )
    return compacted
