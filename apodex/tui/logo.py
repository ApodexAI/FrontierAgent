"""The FrontierAgent pixel-art logo, drawn in the active theme's colours.

The product is FrontierAgent, so the product is what the wordmark says; the
mountain-and-star mark is Apodex's, kept because it is the visual identity the
tool ships under (and because a frontier is a reasonable thing for a mountain to
mean). The tagline carries the company credit.

The startup block is a bitmap, not an ASCII drawing. Every pixel is one
half-cell: rows are consumed in pairs and painted with ``▀`` / ``▄`` / ``█`` so
the mark gets twice the vertical resolution a character grid would give it, and
the peaks read as diagonals instead of stair-stepped slashes.

Colour comes from :mod:`apodex.tui.themes` rather than from the artwork, so the
logo is the same object under every palette: the peaks take ``primary``, the
star takes ``accent``, the wordmark takes ``foreground`` and the tagline
``muted``. Nothing here hard-codes a hex value, which is what keeps the logo
from being the one element on screen still wearing catppuccin's mauve after the
user has switched to gruvbox.

Two pixels can share a cell only where a top row and the bottom row beneath it
carry different colours; the layout keeps the star, the mark and the wordmark on
separate row pairs so that case does not arise, and the renderer falls back to a
foreground/background split if a future edit introduces one.
"""

from __future__ import annotations

from rich.text import Text

from apodex.tui.themes import palette

# ── the mark ──────────────────────────────────────────────────────────────
# 36 columns x 20 pixel rows = 36x10 character cells.
_MARK_WIDTH = 36
_MARK_ROWS = 20

# Star: four pixel rows centred over the tall peak's apex, so it fills exactly
# two character cells and leaves the third as the gap above the mountain.
_STAR_CENTRE = 24
_STAR_ROWS: tuple[tuple[int, ...], ...] = (
    (0,), (-1, 0, 1), tuple(range(-4, 5)), (-1, 0, 1),
)

# Mountain: 14 pixel rows starting at row 6, drawn as five strokes. Each is
# ``(first row, first column, columns per row, last row)``, painted ``_STROKE``
# columns wide — laying a slope down as a run of short horizontal spans is what
# gives it the weight of the logo's rounded strokes instead of a hairline.
#
# The slopes are the artwork's, not 45 degrees, and that is what lets the shape
# survive at this size: the tall peak's flanks are steep (1.55 and 0.8 columns
# per row) while the short peak's right flank is shallow (0.6), so the two meet
# at a valley 60% of the way down instead of colliding just below the apexes.
# The short peak's left flank converges on the tall one and merges into it near
# the base, and the last stroke is the detached one inside the right flank.
_MOUNTAIN_TOP = 6
_STROKE = 3
_APEX_COL = 23
_STROKES: tuple[tuple[int, int, float, int], ...] = (
    (0, _APEX_COL, -1.55, 13),   # tall peak, left flank
    (0, _APEX_COL, 0.80, 13),    # tall peak, right flank
    (2, 7, -0.55, 13),           # short peak, left flank
    (2, 7, 0.80, 8),             # short peak, right flank — ends at the valley
    (6, 19, 0.80, 13),           # the detached inner stroke
)


def _mark_bitmap() -> list[list[str]]:
    """Return the mark as a 20-row grid of role keys (``""`` = transparent)."""
    grid = [["" for _ in range(_MARK_WIDTH)] for _ in range(_MARK_ROWS)]

    def paint(row: int, col: int, role: str, width: int = 1) -> None:
        for x in range(col, col + width):
            if 0 <= x < _MARK_WIDTH and 0 <= row < len(grid):
                grid[row][x] = role

    for row, offsets in enumerate(_STAR_ROWS):
        for offset in offsets:
            paint(row, _STAR_CENTRE + offset, "star")

    for top, col, per_row, last in _STROKES:
        for local in range(top, last + 1):
            paint(_MOUNTAIN_TOP + local, round(col + per_row * (local - top)),
                  "mark", _STROKE)

    return grid


# ── the wordmark ──────────────────────────────────────────────────────────
# A 5x6 pixel face; six rows means each wordmark line is exactly three character
# cells tall and lines up with the mark's row pairs. Only the letters the two
# lines actually use are cut.
_FONT: dict[str, tuple[str, ...]] = {
    "A": (".###.", "#...#", "#...#", "#####", "#...#", "#...#"),
    "E": ("#####", "#....", "####.", "#....", "#....", "#####"),
    "F": ("#####", "#....", "####.", "#....", "#....", "#...."),
    "G": (".###.", "#....", "#.###", "#...#", "#...#", ".###."),
    "I": ("#####", "..#..", "..#..", "..#..", "..#..", "#####"),
    "N": ("#...#", "##..#", "#.#.#", "#.#.#", "#..##", "#...#"),
    "O": (".###.", "#...#", "#...#", "#...#", "#...#", ".###."),
    "R": ("####.", "#...#", "#...#", "####.", "#..#.", "#...#"),
    "T": ("#####", "..#..", "..#..", "..#..", "..#..", "..#.."),
}
# Two lines, because FRONTIERAGENT set on one would be 77 columns of wordmark
# against a 36-column mark — a banner shaped like a ticker tape. Stacked, it
# reads as a logo and fits beside the peaks.
_WORDMARK_LINES = ("FRONTIER", "AGENT")
_LINE_PITCH = 8  # pixel rows between the two lines' tops: 6 of glyph, 2 of air
_WORDMARK_WIDTH = max(len(line) for line in _WORDMARK_LINES) * 6 - 1
_WORDMARK_ROWS = _LINE_PITCH * (len(_WORDMARK_LINES) - 1) + 6
_PRODUCT = "FrontierAgent"
_TAGLINE = "self-evolving · by Apodex"

# Beside the mark the wordmark starts at pixel row 2, which leaves the top
# character row to the star and lands the tagline on the mark's own base line.
_WORDMARK_TOP = 2   # pixel row
_TAGLINE_ROW = 9    # character row
_NAME_ROW = 4       # character row, plain-text tiers only

_GAP = 3  # character columns between the mark and the wordmark
FULL_WIDTH = _MARK_WIDTH + _GAP + _WORDMARK_WIDTH
# The tier that sets the name as plain text beside the mark; the tagline is the
# longer of the two lines it puts there.
NAMED_WIDTH = _MARK_WIDTH + _GAP + len(_TAGLINE)
# The narrowest tier: the glyph, a space and the product name, nothing else.
# The tagline is dropped rather than given its own threshold — it would need 42
# columns beside the name, and this tier only exists below 36.
ONE_LINE_WIDTH = 2 + len(_PRODUCT)


def _wordmark_bitmap(top: int) -> list[list[str]]:
    """Return both wordmark lines as a pixel grid whose first row is ``top``."""
    grid = [["" for _ in range(_WORDMARK_WIDTH)]
            for _ in range(top + _WORDMARK_ROWS)]
    for line_index, line in enumerate(_WORDMARK_LINES):
        line_top = top + line_index * _LINE_PITCH
        for index, letter in enumerate(line):
            origin = index * 6
            for row, pixels in enumerate(_FONT[letter]):
                for col, pixel in enumerate(pixels):
                    if pixel == "#":
                        grid[line_top + row][origin + col] = "word"
    return grid


# ── rendering ─────────────────────────────────────────────────────────────
# Palette attributes, not the semantic role names ``rich_style`` takes: the mark
# wants the palette's identity colours themselves, and the roles ("tool",
# "ok", "err" …) all describe transcript meaning the logo does not have.
_ROLE_COLOURS = {
    "mark": "primary",
    "star": "accent",
    "word": "foreground",
}


def _cell(top: str, bottom: str, colours: dict[str, str]) -> Text:
    """Paint one character cell from the two pixel rows it covers."""
    if not top and not bottom:
        return Text(" ")
    if top and not bottom:
        return Text("\u2580", style=colours[top])
    if bottom and not top:
        return Text("\u2584", style=colours[bottom])
    if top == bottom:
        return Text("\u2588", style=colours[top])
    # Two different colours in one cell: upper half foreground, lower half
    # background. The layout keeps the star, mark and wordmark on separate row
    # pairs so this does not arise today; it is here so an edit that moves them
    # degrades to a blended cell rather than to a crash.
    return Text("\u2580", style=f"{colours[top]} on {colours[bottom]}")


def _rows(grid: list[list[str]], colours: dict[str, str], width: int) -> list[Text]:
    """Fold a pixel grid into character rows, two pixel rows per row."""
    lines: list[Text] = []
    for index in range(0, len(grid) - 1, 2):
        line = Text()
        for col in range(width):
            line.append_text(_cell(grid[index][col], grid[index + 1][col], colours))
        lines.append(line)
    return lines


def render_logo(theme: str, width: int = 0) -> Text:
    """Return the startup logo for ``theme``, fitted to ``width`` columns.

    Four tiers, narrowing by what the pane can hold: the pixel wordmark beside
    the mark, then the name set as plain text beside it, then that name moved
    below it, then a single line. Two constraints shape the ladder — a logo that
    wraps is worse than no logo, and one that fills the transcript is worse
    still, so no tier is taller than thirteen rows even though stacking the pixel
    wordmark under the mark (nineteen) would have kept it around longer.
    ``width`` of 0 means "unconstrained" and always gets the full layout.
    """
    spec = palette(theme)
    colours = {role: getattr(spec, token)
               for role, token in _ROLE_COLOURS.items()}
    name = Text(_PRODUCT, style=f"bold {colours['word']}")
    tagline = Text(_TAGLINE, style=f"italic {spec.muted}")

    if width and width < _MARK_WIDTH:
        star = Text("\u25ed", style=colours["star"])
        return star + Text(" ") + name if width >= ONE_LINE_WIDTH else star

    mark = _rows(_mark_bitmap(), colours, _MARK_WIDTH)
    gap = Text(" " * _GAP)

    if width and width < NAMED_WIDTH:
        return Text("\n").join([*mark, Text(), Text(" ") + name, Text(" ") + tagline])

    if width and width < FULL_WIDTH:
        beside = {_NAME_ROW: name, _NAME_ROW + 2: tagline}
        return Text("\n").join(
            line + gap + beside.get(index, Text())
            for index, line in enumerate(mark)
        )

    word = _rows(_wordmark_bitmap(_WORDMARK_TOP), colours, _WORDMARK_WIDTH)
    lines = []
    for index, line in enumerate(mark):
        row = line + gap
        if index == _TAGLINE_ROW:
            row = row + Text(" ") + tagline
        elif index < len(word):
            row = row + word[index]
        lines.append(row)
    return Text("\n").join(lines)
