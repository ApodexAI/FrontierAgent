"""Pure LLM utility helpers — no business logic, no runtime/components deps."""

from __future__ import annotations


def extract_boxed_content(text: str) -> str:
    r"""Extract content from the last well-formed ``\boxed{}`` in ``text``.

    Walks left-to-right, balancing braces so nested ``{...}`` inside
    the box are preserved. Returns "" if no well-formed ``\boxed{...}``
    is found. Shared by every consumer that needs to pull a boxed answer
    out of an LLM response (``benchmarks/`` answer extraction, workflow
    summary phases).
    """
    if not text:
        return ""

    matches: list[str] = []
    i = 0
    while i < len(text):
        start = text.find(r"\boxed{", i)
        if start == -1:
            break
        content_start = start + 7  # len(r'\boxed{')
        if content_start >= len(text):
            break
        brace_count = 1
        pos = content_start
        while pos < len(text) and brace_count > 0:
            if text[pos] == "{":
                brace_count += 1
            elif text[pos] == "}":
                brace_count -= 1
            pos += 1
        if brace_count == 0:
            matches.append(text[content_start:pos - 1])
            i = pos
        else:
            i = content_start

    return matches[-1] if matches else ""


__all__ = ["extract_boxed_content"]
