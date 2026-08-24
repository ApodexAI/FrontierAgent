"""Keep input-method commits intact under the Kitty keyboard protocol.

Textual asks the terminal for ``KITTY_DISAMBIGUATE_ESCAPE_CODES |
KITTY_REPORT_ALL_KEYS | KITTY_REPORT_ASSOCIATED_TEXT`` (``CSI > 25 u``). The last
flag is what makes an input method usable at all: when a CJK IME commits from
its candidate window, the terminal does not send the characters as ordinary
UTF-8 bytes — it sends one key event carrying the committed text in the
sequence's third field, as decimal codepoints::

    ESC [ 32 ; ; 26377:24456:22823:24046:21035 u    ("有很大差别", committed with space)

Textual's parser understands that field. What defeats it is the guard *before*
the recognition step: while collecting an escape sequence it gives up once the
buffer passes ``_MAX_SEQUENCE_SEARCH_THRESHOLD``, which upstream sets to 32
characters, and re-issues everything collected so far as individual key presses.

Do the arithmetic for a commit of ``n`` CJK characters: ``ESC [`` (2) plus
``32;;`` (4) plus five decimal digits per codepoint with colon separators
(``6n - 1``) plus the final ``u``. The buffer therefore reaches ``6n + 5``
characters before the terminator arrives, which passes 32 at ``n = 5``. Four
characters commit cleanly; five or more are re-issued as literal text, so the
prompt fills with ``^[32;;26377:24456:22823:24046:21035u`` instead of the phrase
the user typed. It is a hard cutoff, not a race — every commit of five or more
characters is corrupted, which is most of what a candidate window is for.

Raising the ceiling is enough: the threshold only decides how long to keep
looking before concluding a sequence is unsupported, and the real safety net for
a genuinely malformed sequence is the ``ESCAPE_DELAY`` timeout (100 ms) in the
same loop, which is unaffected. A truncated sequence still ends up re-issued as
keys, just after a larger bound.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

#: Room for an 84-character commit. Phrase-level input methods commit far less
#: than that, and the cost of a generous bound is only how long a *malformed*
#: sequence takes to be recognised as malformed — which the escape-delay timeout
#: settles first anyway.
MIN_ESCAPE_SEQUENCE_LIMIT = 512


def widen_escape_sequence_limit(minimum: int = MIN_ESCAPE_SEQUENCE_LIMIT) -> int:
    """Raise Textual's escape-sequence search limit to ``minimum``.

    Returns the limit in effect afterwards, or ``0`` if this version of Textual
    no longer exposes the constant — in which case the corruption is either
    fixed upstream or has moved, and either way silently doing nothing is
    correct: this is an input-quality fix, not something to abort a session for.

    The parser reads the constant as a module global on every iteration, so
    rebinding it takes effect for parsers that already exist.
    """
    try:
        from textual import _xterm_parser
    except ImportError:  # pragma: no cover - textual is a hard dependency
        logger.debug("textual._xterm_parser is unavailable; leaving IME limit alone")
        return 0
    current = getattr(_xterm_parser, "_MAX_SEQUENCE_SEARCH_THRESHOLD", None)
    if not isinstance(current, int):
        logger.debug("textual no longer exposes the escape-sequence limit")
        return 0
    if current >= minimum:
        return current
    _xterm_parser._MAX_SEQUENCE_SEARCH_THRESHOLD = minimum
    return minimum


def ime_commit_sequence(text: str, *, key: int = 32) -> str:
    """The Kitty sequence a terminal sends when an IME commits ``text``.

    Shared with the tests so the regression guard exercises the real byte shape
    rather than a hand-copied string, and so the arithmetic above stays checkable.
    """
    codepoints = ":".join(str(ord(char)) for char in text)
    return f"\x1b[{key};;{codepoints}u"


__all__ = [
    "MIN_ESCAPE_SEQUENCE_LIMIT", "ime_commit_sequence",
    "widen_escape_sequence_limit",
]
