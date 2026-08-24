"""Unicode math-symbol sanitization for Python code blocks."""
from __future__ import annotations

_UNICODE_REPLACEMENTS: dict[str, str] = {
    "\u222B": "integral",   # Integral sign
    "\u2211": "sum",        # Summation
    "\u221E": "inf",        # Infinity
    "\u00B2": "**2",        # Superscript 2
    "\u00B3": "**3",        # Superscript 3
    "\u2074": "**4",        # Superscript 4
    "\u207B": "-",          # Superscript minus
    "\u00B9": "**1",        # Superscript 1
    "\u27E8": "<",          # Left angle bracket
    "\u27E9": ">",          # Right angle bracket
    "\u2329": "<",          # Left-pointing angle bracket
    "\u232A": ">",          # Right-pointing angle bracket
    "\u2014": "-",          # Em dash
    "\u2013": "-",          # En dash
    "\u2018": "'",          # Left single quote
    "\u2019": "'",          # Right single quote
    "\u201C": '"',          # Left double quote
    "\u201D": '"',          # Right double quote
    "\u2264": "<=",         # Less than or equal
    "\u2265": ">=",         # Greater than or equal
    "\u2260": "!=",         # Not equal
    "\u00D7": "*",          # Multiplication sign
    "\u00F7": "/",          # Division sign
    "\u2248": "==",         # Almost equal
    "\u2245": "==",         # Approximately equal
    "\u2261": "==",         # Identical to
    "\u2192": "->",         # Right arrow
    "\u2190": "<-",         # Left arrow
    "\u221A": "sqrt",       # Square root
    "\u03C0": "pi",         # Pi
}


def sanitize_code(code: str) -> str:
    """Replace Unicode math symbols with ASCII equivalents and strip non-ASCII
    bytes from comments.

    Also removes lines that start with a shell/jupyter escape (``!pip install…``)
    — those get fed to the Python interpreter, which rejects them with a
    ``SyntaxError``.

    Code-line behaviour (non-comment):
      * any char present in the replacement table is mapped to its ASCII form
      * non-ASCII chars that have no replacement are left alone — they may
        be legitimate (e.g. a unicode string literal)

    Comment-line behaviour:
      * symbol replacement runs first
      * any remaining non-ASCII bytes are replaced by ``?`` (lossy) — comments
        never affect execution, so we drop noise eagerly
    """
    if not code:
        return code

    out_lines: list[str] = []
    for line in code.split("\n"):
        # Strip jupyter/shell escapes.
        if line.lstrip().startswith("!"):
            continue

        if any(ord(ch) > 127 for ch in line):
            stripped = line.lstrip()
            if stripped.startswith("#"):
                for sym, repl in _UNICODE_REPLACEMENTS.items():
                    line = line.replace(sym, repl)
                # Lossy: comments may be in any locale, we don't need them
                # round-tripped.
                line = line.encode("ascii", "replace").decode("ascii")
            else:
                for sym, repl in _UNICODE_REPLACEMENTS.items():
                    line = line.replace(sym, repl)
        out_lines.append(line)

    return "\n".join(out_lines)
