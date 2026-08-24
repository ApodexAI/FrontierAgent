"""``preserve_line_breaks`` — model line breaks must survive CommonMark.

``rich.markdown`` reflows soft breaks, which silently destroys any structure a
model carries in newlines alone. The references section is the visible case:
the report prompt asks for one entry per line with no bullets and no blank
lines, which is exactly the shape CommonMark merges into one paragraph.
"""
from __future__ import annotations

import pytest
from rich.console import Console
from rich.markdown import Markdown

from apodex.render import preserve_line_breaks

_REFERENCES = (
    "## 参考资料\n\n"
    "[1] https://www.channelnewsasia.com/singapore/economy-gdp-forecast-mti-growth-6310711\n"
    "[2] https://www.straitstimes.com/business/singapore-upgrades-2026-growth-forecast\n"
    "[3] https://www.zaobao.com.sg/finance/singapore/story20260811-9497810\n"
)


def _render(markdown: str, width: int = 100) -> str:
    console = Console(width=width, force_terminal=False)
    with console.capture() as capture:
        console.print(Markdown(markdown))
    return capture.get()


def test_reference_entries_each_keep_their_own_line() -> None:
    """Without this, entries run together and the ``[n]`` markers strand at
    line ends — the reported symptom."""
    before = _render(_REFERENCES)
    after = _render(preserve_line_breaks(_REFERENCES))

    assert "-6310711 [2]" in before          # two entries share a line
    assert "-6310711 [2]" not in after
    for marker in ("[1]", "[2]", "[3]"):
        assert sum(line.strip().startswith(marker) for line in after.splitlines()) == 1


def test_prose_paragraphs_get_hard_breaks() -> None:
    assert preserve_line_breaks("line one\nline two") == "line one  \nline two"


def test_code_blocks_are_left_verbatim() -> None:
    """Inside a fence the content is literal, so a trailing space is a content
    change rather than a layout one."""
    text = "prose\n\n```python\nx = 1\ny = 2   # trailing spaces matter\n```\n"

    assert preserve_line_breaks(text) == text


@pytest.mark.parametrize("text", [
    # A ``~~~`` line inside a backtick fence is CONTENT, not a closing fence.
    "```python\nprint(1)\n~~~\nprint(2)\n```\n",
    # A fence only closes on the same character at the same-or-greater length.
    "````\n```\ninner = 1\n````\n",
    # Fences nest under container prefixes.
    "> ```python\n> a = 1\n> b = 2\n> ```\n",
    "- ```python\n  a = 1\n  b = 2\n  ```\n",
    # Indented code inside a list item.
    "- item\n\n      indented = 1\n      more = 2\n",
    # An unclosed fence still protects everything after it.
    "prose\n\n```python\nx = 1\ny = 2\n",
])
def test_verbatim_blocks_survive_every_fence_shape(text: str) -> None:
    """A boolean "am I in a fence" toggle gets all of these wrong and edits the
    code it promised to preserve — hence Rich's own parser locates them."""
    assert preserve_line_breaks(text) == text


def test_indented_code_is_left_verbatim() -> None:
    text = "para\n\n    indented = 1\n    more = 2\n"

    assert preserve_line_breaks(text) == text


def test_is_idempotent() -> None:
    """The TUI can pass content through more than one render path."""
    once = preserve_line_breaks(_REFERENCES)

    assert preserve_line_breaks(once) == once


@pytest.mark.parametrize(("text", "expected"), [
    ("a\\\nb", "a\\\nb"),             # odd: existing backslash hard break
    ("a\\\\\nb", "a\\\\  \nb"),       # even: escaped literal backslash
    ("a\\ \nb", "a\\  \nb"),          # whitespace means backslash is not trailing
])
def test_backslash_hard_breaks_require_an_odd_trailing_run(
    text: str, expected: str,
) -> None:
    assert preserve_line_breaks(text) == expected


def test_no_marker_is_added_at_the_end_of_a_block() -> None:
    """A blank line already ends the block, so a marker would only leave
    trailing whitespace behind."""
    assert preserve_line_breaks("a\nb\n\nc") == "a  \nb\n\nc"


def test_tables_still_parse() -> None:
    table = "| a | b |\n|---|---|\n| 1 | 2 |"

    assert "─" in _render(preserve_line_breaks(table), width=40)


def test_lists_still_parse() -> None:
    assert "•" in _render(preserve_line_breaks("- one\n- two"), width=40)
