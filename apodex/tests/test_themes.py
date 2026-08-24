"""The theme palettes are a contract, not a mood board.

These tests re-measure the properties the palettes in :mod:`apodex.tui.themes`
were fitted for, so a future colour tweak cannot quietly reintroduce the two
failures that motivated the redesign: transcript text colliding with the
background it sat on, and a selected theme recolouring only half the UI.
"""

from __future__ import annotations

import pytest

from apodex.render import Renderer, diff_to_text
from apodex.tui.themes import (
    CLI_THEME_NAMES,
    GLYPHS,
    ROLES,
    THEME_PICKER_NAMES,
    THEME_SPECS,
    TUI_THEME_NAMES,
    ansi_fg,
    palette,
    rich_style,
    rich_styles,
    role_color,
)


# ── WCAG contrast ─────────────────────────────────────────────────────────
def _relative_luminance(hex_color: str) -> float:
    def channel(value: int) -> float:
        c = value / 255
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4

    raw = hex_color.lstrip("#")
    r, g, b = (channel(int(raw[i:i + 2], 16)) for i in (0, 2, 4))
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contrast_ratio(a: str, b: str) -> float:
    la, lb = _relative_luminance(a), _relative_luminance(b)
    return (max(la, lb) + 0.05) / (min(la, lb) + 0.05)


def test_contrast_ratio_matches_known_reference_pairs() -> None:
    """Guard the measuring stick itself before trusting it on 14 palettes."""
    assert contrast_ratio("#000000", "#ffffff") == pytest.approx(21.0, abs=0.01)
    assert contrast_ratio("#ffffff", "#ffffff") == pytest.approx(1.0, abs=0.01)
    assert contrast_ratio("#767676", "#ffffff") == pytest.approx(4.54, abs=0.05)


# Floors match what each role actually is. ``foreground`` and ``muted`` are body
# text and owe WCAG AA. ``subtle`` and the semantic roles are short bold labels,
# glyphs, borders and diff markers: 3:1 is WCAG 1.4.3 for bold text and 1.4.11
# for UI components. Holding accents to the 4.5:1 body-text floor is what turned
# Catppuccin Latte's amber into a dark brown — the metric is not the goal.
_FLOORS = {"text": 6.0, "muted": 4.5, "border": 1.6}
_UI = 3.0


@pytest.mark.parametrize("spec", THEME_SPECS, ids=lambda spec: spec.name)
@pytest.mark.parametrize("role", ROLES)
def test_every_role_clears_its_contrast_floor_on_every_bed(spec, role: str) -> None:
    """A role must be readable on all three beds, not just the screen background.

    Panels and modals are different backgrounds than the app background, so
    measuring against ``background`` alone is what let panel text drop to ~2:1.
    """
    color = rich_style(spec.name, role)
    floor = _FLOORS.get(role, _UI)
    for bed_name, bed in zip(("background", "surface", "panel"), spec.beds, strict=False):
        ratio = contrast_ratio(color, bed)
        assert ratio >= floor, (
            f"{spec.name}.{role} ({color}) is {ratio:.2f}:1 on {bed_name} "
            f"({bed}); floor is {floor}:1"
        )


@pytest.mark.parametrize("spec", THEME_SPECS, ids=lambda spec: spec.name)
def test_text_tiers_stay_visually_distinct(spec) -> None:
    """text > muted > subtle. Without real separation the tiers convey nothing."""
    worst = [
        min(contrast_ratio(getattr(spec, token), bed) for bed in spec.beds)
        for token in ("foreground", "muted", "subtle")
    ]
    text, muted, subtle = worst
    assert text > muted > subtle, f"{spec.name} tiers not ordered: {worst}"
    assert text / subtle >= 1.5, f"{spec.name} tiers too close together: {worst}"


@pytest.mark.parametrize("spec", THEME_SPECS, ids=lambda spec: spec.name)
def test_semantic_colors_are_mutually_distinguishable(spec) -> None:
    """Success/error/warning must not converge once corrected for contrast.

    Nudging colours toward a floor moves them all in one direction, so it could
    in principle collapse two states onto the same swatch.
    """
    for a, b in (("ok", "err"), ("ok", "tool"), ("err", "tool"), ("accent", "todo")):
        assert rich_style(spec.name, a) != rich_style(spec.name, b), f"{spec.name}: {a}/{b}"


# ── the palettes stay recognisably themselves ─────────────────────────────
# Upstream values for the colours that carry each theme's identity. This is the
# guarantee that was missing when the palettes were first fitted: with only
# contrast under test, "make everything AA" passed while Gruvbox's orange became
# rust and Solarized Light's teal became near-black.
_UPSTREAM = {
    "catppuccin": dict(primary="#cba6f7", secondary="#89b4fa", warning="#f9e2af",
                       error="#f38ba8", success="#a6e3a1", accent="#f5c2e7"),
    "tokyo-night": dict(primary="#7aa2f7", secondary="#bb9af7", warning="#e0af68",
                        error="#f7768e", success="#9ece6a", accent="#7dcfff"),
    "gruvbox": dict(primary="#83a598", secondary="#d3869b", warning="#fabd2f",
                    error="#fb4934", success="#b8bb26", accent="#fe8019"),
    "gruvbox-light": dict(primary="#076678", secondary="#8f3f71",
                          error="#c14a4a", success="#6c782e", accent="#c35e0a"),
    "one-dark": dict(primary="#61afef", secondary="#c678dd", warning="#e5c07b",
                     error="#e06c75", success="#98c379", accent="#56b6c2"),
    "one-light": dict(primary="#4078f2", secondary="#a626a4", warning="#986801",
                      accent="#0184bc"),
    "solarized": dict(primary="#268bd2", warning="#b58900", success="#859900",
                      accent="#2aa198"),
    "solarized-light": dict(primary="#268bd2", secondary="#6c71c4", error="#dc322f"),
    "dracula": dict(primary="#bd93f9", secondary="#8be9fd", warning="#f1fa8c",
                    success="#50fa7b", accent="#ff79c6"),
    "nord": dict(primary="#88c0d0", secondary="#81a1c1", warning="#ebcb8b",
                 success="#a3be8c", accent="#b48ead"),
    "catppuccin-latte": dict(primary="#8839ef", secondary="#1e66f5", error="#d20f39"),
    "tokyo-night-day": dict(warning="#8c6c3e", success="#587539", accent="#007197"),
}


@pytest.mark.parametrize(("name", "expected"), sorted(_UPSTREAM.items()))
def test_semantic_colors_match_upstream_exactly(name: str, expected: dict) -> None:
    """These colours ARE the theme. They must survive byte-for-byte."""
    spec = palette(name)
    for token, want in expected.items():
        assert getattr(spec, token) == want, (
            f"{name}.{token} drifted from upstream {want}"
        )


def test_signature_colors_keep_their_saturation() -> None:
    """A contrast correction must not desaturate a palette into mud.

    Gruvbox's orange and Solarized's yellow/teal are the reason someone picks
    those themes; each is checked against the hue and vividness upstream ships.
    """
    def hsv_sat(hex_color: str) -> float:
        raw = hex_color.lstrip("#")
        r, g, b = (int(raw[i:i + 2], 16) / 255 for i in (0, 2, 4))
        return 0.0 if max(r, g, b) == 0 else (max(r, g, b) - min(r, g, b)) / max(r, g, b)

    for name, token, floor in (
        ("gruvbox", "accent", 0.85),          # #fe8019 orange
        ("gruvbox", "warning", 0.75),         # #fabd2f yellow
        ("gruvbox-light", "accent", 0.85),    # #c35e0a burnt orange
        ("solarized", "warning", 0.90),       # #b58900 yellow
        ("solarized", "accent", 0.70),        # #2aa198 cyan
        ("solarized-light", "warning", 0.90),
        ("solarized-light", "accent", 0.70),
        ("catppuccin-latte", "warning", 0.80),
    ):
        value = getattr(palette(name), token)
        assert hsv_sat(value) >= floor, (
            f"{name}.{token} = {value} has saturation "
            f"{hsv_sat(value):.2f}, below {floor}"
        )


def test_quiet_text_keeps_the_palette_tint() -> None:
    """``muted`` covers a lot of surface area — notes, thinking, arg detail,
    durations, the status bar. If it goes achromatic the whole UI reads grey,
    whatever the accents do. Each tier must stay tinted like its palette."""
    def channels(hex_color: str) -> tuple[int, int, int]:
        raw = hex_color.lstrip("#")
        return tuple(int(raw[i:i + 2], 16) for i in (0, 2, 4))  # type: ignore[return-value]

    # gruvbox is warm (red > blue); Solarized is cool (blue > red). Compared as
    # a ratio, not a difference: the same cast is a smaller absolute gap on a
    # dark shade like gruvbox-light's ``#504945`` than on a pale one.
    for name, tier in (("gruvbox", "muted"), ("gruvbox", "subtle"),
                       ("gruvbox-light", "muted"), ("gruvbox-light", "subtle")):
        r, _g, b = channels(getattr(palette(name), tier))
        assert r / b >= 1.12, f"{name}.{tier} lost its warm cast"
    for name, tier in (("solarized", "muted"), ("solarized", "subtle"),
                       ("solarized-light", "muted"), ("solarized-light", "subtle")):
        r, _g, b = channels(getattr(palette(name), tier))
        assert b / r >= 1.12, f"{name}.{tier} lost its cool cast"


# ── no terminal ``dim`` ───────────────────────────────────────────────────
@pytest.mark.parametrize("name", [spec.name for spec in THEME_SPECS])
def test_no_role_uses_terminal_dim(name: str) -> None:
    """``dim`` is a terminal-side blend by an unspecified amount, so it destroys
    a measured ratio. Quiet text uses the ``muted`` / ``subtle`` tiers instead."""
    for role, style in rich_styles(name).items():
        assert "dim" not in style, f"{name}.{role} = {style!r}"
        assert style.startswith("#"), f"{name}.{role} is not an explicit colour: {style!r}"


def test_renderer_and_diffs_emit_no_dim_styles() -> None:
    """The renderer and the shared diff colouriser are dim-free too."""
    renderer = Renderer(theme="solarized-light")
    for role in ROLES:
        assert "dim" not in renderer._s(role)
    diff = diff_to_text(
        "--- a/x\n+++ b/x\n@@ -1 +1 @@\n-old\n+new\n unchanged\n",
        theme="solarized-light",
    )
    styles = {str(span.style) for span in diff.spans}
    assert styles and all("dim" not in style for style in styles), styles


# ── one palette drives both surfaces ─────────────────────────────────────
@pytest.mark.parametrize("name", TUI_THEME_NAMES)
def test_every_selectable_theme_has_a_real_palette(name: str) -> None:
    """Regression: ``dark`` and ``light`` were selectable but had no palette, so
    ``rich_style`` silently fell back to Catppuccin — a dark palette on a light
    background. Every offered name must resolve to a palette of its own."""
    spec = palette(name)
    assert spec.name in {"catppuccin", "catppuccin-latte"} or spec.name == name
    if name in ("dark", "light"):
        assert spec.name == name
        assert (spec.dark is True) == (name == "dark")


def test_light_and_dark_palettes_are_not_the_same_colours() -> None:
    light, dark = palette("light"), palette("dark")
    assert light.dark is False and dark.dark is True
    assert light.foreground != dark.foreground
    assert _relative_luminance(light.background) > _relative_luminance(dark.background)


def test_cli_and_tui_share_one_palette_source() -> None:
    """The line UI and the TUI must agree; two copies of a palette is how they
    drifted apart (the same theme name produced different transcript colours)."""
    for name in TUI_THEME_NAMES:
        renderer = Renderer(theme=name)
        for role in ROLES:
            assert renderer._s(role) == rich_style(name, role), f"{name}.{role}"


def test_mono_is_the_only_colourless_theme() -> None:
    assert set(CLI_THEME_NAMES) - set(TUI_THEME_NAMES) == {"mono"}
    assert Renderer(theme="mono")._styles == {}
    assert Renderer(theme="mono")._s("err") == ""


def test_theme_picker_offers_canonical_spellings_only() -> None:
    assert set(THEME_PICKER_NAMES) <= set(TUI_THEME_NAMES)
    assert "catppucin" not in THEME_PICKER_NAMES  # the alias, not a second row
    assert len(set(THEME_PICKER_NAMES)) == len(THEME_PICKER_NAMES)
    assert {spec.name for spec in THEME_SPECS} == set(THEME_PICKER_NAMES)


def test_catppuccin_alias_resolves_to_the_same_palette() -> None:
    assert palette("catppucin") is palette("catppuccin")
    assert palette("catppucin-latte") is palette("catppuccin-latte")


def test_unknown_theme_falls_back_without_raising() -> None:
    assert palette("no-such-theme").name == "catppuccin"


def test_ansi_escape_matches_the_role_colour() -> None:
    """The spinner and streaming thinking prefix bypass Rich, so they need the
    raw escape — it must encode the same colour the Rich style would."""
    assert rich_style("dark", "muted") == "#a9adbe"
    assert ansi_fg("dark", "muted") == "\033[38;2;169;173;190m"


# ── Agent Team worker identity ────────────────────────────────────────────
@pytest.mark.parametrize("name", TUI_THEME_NAMES)
def test_agent_identity_colours_are_distinct_in_every_theme(name: str) -> None:
    """The whole point of an identity colour is telling two workers apart.

    The roles are resolved per theme, so a palette that happens to map two of
    them to the same hex (``border`` and ``primary`` usually match) would hand
    two sub-agents the same colour on that theme alone.
    """
    from apodex.tui.themes import AGENT_IDENTITY_ROLES

    colours = [role_color(name, role) for role in AGENT_IDENTITY_ROLES]
    assert len(set(colours)) == len(colours), dict(
        zip(AGENT_IDENTITY_ROLES, colours, strict=False)
    )


@pytest.mark.parametrize("name", TUI_THEME_NAMES)
def test_agent_identity_colours_never_borrow_a_state_colour(name: str) -> None:
    """A worker's name in the success green reads as a succeeded worker."""
    from apodex.tui.themes import AGENT_IDENTITY_ROLES

    reserved = {role_color(name, role) for role in ("ok", "err")}
    for role in AGENT_IDENTITY_ROLES:
        assert role_color(name, role) not in reserved, role


def test_agent_identity_role_wraps_instead_of_failing() -> None:
    """Teams can exceed the palette; the specialty glyph carries the rest."""
    from apodex.tui.themes import AGENT_IDENTITY_ROLES, agent_identity_role

    size = len(AGENT_IDENTITY_ROLES)
    assert agent_identity_role(0) == agent_identity_role(size)
    assert agent_identity_role(-1) == agent_identity_role(0)
    assert len({agent_identity_role(i) for i in range(size)}) == size


def test_agent_kind_glyphs_are_one_cell_and_resolve_for_every_kind() -> None:
    from rich.cells import cell_len

    from apodex.tui.themes import AGENT_KINDS, agent_kind_glyph

    glyphs = [agent_kind_glyph(kind) for kind in AGENT_KINDS]
    assert len(set(glyphs)) == len(glyphs), glyphs
    for value in glyphs:
        assert cell_len(value) == 1, value
    # An unmapped kind must still render something rather than an empty column.
    assert agent_kind_glyph("nonsense") == GLYPHS["agent_generic"]


# ── glyphs follow the theme ───────────────────────────────────────────────
def test_glyphs_carry_no_colour_of_their_own() -> None:
    """Colour emoji paint themselves and ignore the surrounding foreground, so
    they cannot follow a theme. Every glyph must be a monochrome character that
    takes the theme colour like any other text."""
    emoji_ranges = ((0x1F300, 0x1FAFF), (0x1F000, 0x1F2FF), (0x23E9, 0x23FA))
    for key, value in GLYPHS.items():
        assert len(value) == 1, f"{key}: {value!r} should be a single character"
        code = ord(value)
        assert not any(lo <= code <= hi for lo, hi in emoji_ranges), (
            f"{key}: {value!r} (U+{code:04X}) has emoji presentation"
        )
        # A trailing VS16 would force emoji rendering even on a BMP character.
        assert "️" not in value, key


def test_glyphs_occupy_exactly_one_cell() -> None:
    """Status markers prefix aligned columns (the activity timeline, the plan
    pane), so a double-width glyph shifts every row it appears on. Emoji are
    always two cells — another reason they cannot serve as UI markers."""
    from rich.cells import cell_len

    for key, value in GLYPHS.items():
        assert cell_len(value) == 1, f"{key}: {value!r} is {cell_len(value)} cells wide"


def test_renderer_tool_labels_use_the_shared_glyphs() -> None:
    from apodex.render import _TOOL_GLYPH

    assert _TOOL_GLYPH["bash"].startswith(GLYPHS["bash"])
    assert _TOOL_GLYPH["read_file"].startswith(GLYPHS["read"])
    assert _TOOL_GLYPH["grep_search"].startswith(GLYPHS["search"])
    for label in _TOOL_GLYPH.values():
        assert not any(ord(ch) > 0x1F000 for ch in label), label


def test_every_served_tool_has_a_family_marker() -> None:
    """A tool missing from the label table falls back to the generic ``tool``
    marker plus its raw snake_case name. That is fine for one unknown tool and
    useless for a whole family: a research run is mostly web steps, and until
    these were mapped a page of them read as an undifferentiated column of
    identical marks. Listed explicitly rather than imported from the registry so
    this stays a cheap unit test — the cost of a new tool is one line here.
    """
    from apodex.render import _TOOL_GLYPH

    families = {
        "web": ("web_search",),
        "fetch": ("web_fetch", "download_file"),
        "code": ("run_python_code",),
        "spawn": ("create_subagent", "assign_task", "collect_reports"),
        "plan": ("todo_write", "add_task", "update_task", "finish_planning"),
        "read": ("read_file", "read_text", "file_editor_view"),
        "write": ("write_file", "create_file", "file_editor_create",
                  "file_editor_str_replace"),
        "delete": ("delete_file",),
        "search": ("grep_search", "glob_search"),
    }
    for family, names in families.items():
        for name in names:
            assert name in _TOOL_GLYPH, f"{name} would fall back to the generic marker"
            assert _TOOL_GLYPH[name].startswith(GLYPHS[family]), name
    # Nothing in the table may wear the fallback marker: that would claim a
    # family had been assigned while rendering exactly like an unmapped tool.
    assert GLYPHS["tool"] not in "".join(_TOOL_GLYPH.values())
