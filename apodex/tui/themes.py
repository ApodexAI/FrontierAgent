"""Curated colour themes shared by the Textual TUI and the line-mode CLI.

One :class:`ThemeSpec` per theme drives *both* surfaces. It carries the Textual
widget palette (background / surface / panel / semantic colours) **and** the
Rich text tiers the transcript needs, so a selected theme can never recolour the
chrome while transcript content — tool headers, thinking, plans, results — keeps
generic Rich colours that clash with it.

**Each upstream palette is the source of truth.** Gruvbox's orange is
``#fe8019`` and Solarized's yellow is ``#b58900`` because that is what makes
them those themes; a palette is an authored artifact, and repainting it to
satisfy a metric produces a generic gradient wearing the theme's name. So:

* semantic colours (``primary`` … ``error``) are used **verbatim** unless they
  fall below 3:1, and are then corrected by the smallest lightness step that
  clears it — hue and chroma are never touched. 68 of 84 are untouched;
* the three text tiers come from each designer's own ramp — gruvbox
  ``fg``/``fg3``/``fg4``, Solarized ``base1``/``base0``/``base00``, Catppuccin
  ``text``/``subtext0``/``overlay2`` — which is why quiet text keeps the
  palette's cast (gruvbox warm, Solarized teal) instead of going grey. Where a
  palette publishes no mid step, one is derived by blending the tier above
  toward that palette's *comment* colour, not toward the background;
* floors are the ones that match what each role actually is. ``foreground``
  >= 6:1 and ``muted`` >= 4.5:1 are body text and owe WCAG AA. ``subtle`` and
  the semantic colours are short bold labels, glyphs, borders and diff markers,
  so they owe 3:1 — WCAG 1.4.3 for bold text and 1.4.11 for UI components.

Measured against the worst of that theme's own ``background`` / ``surface`` /
``panel``, since panels and modals are different beds than the screen.
``tests/test_themes.py`` re-measures every floor *and* asserts the semantic
colours still match upstream, so neither property can regress alone.

Two consequences worth knowing before editing this file:

* **Nothing here uses Rich's ``dim``.** ``dim`` is a terminal-controlled blend
  by an unspecified amount, so it destroys a measured ratio — it was the direct
  cause of thinking text colliding with the background on several themes.
  Quieter text uses the ``muted`` / ``subtle`` tiers instead.
* **Glyphs are text-presentation, never colour emoji.** A colour emoji paints
  itself and ignores the surrounding foreground colour, so it cannot follow a
  theme; on light themes the old ``💭`` / ``📖`` / ``🔎`` / ``📋`` markers stayed
  dark-theme-coloured no matter what was selected. Every glyph in :data:`GLYPHS`
  is a monochrome character that takes the theme colour like any other text.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# ``catppucin`` is kept as a user-facing compatibility alias for the spelling
# used in the original theme request. ``catppuccin`` is also accepted.
_ALIASES = {
    "catppucin": "catppuccin",
    "catppucin-latte": "catppuccin-latte",
}


@dataclass(frozen=True)
class ThemeSpec:
    """A complete palette: Textual chrome tokens plus the Rich text tiers."""

    name: str
    dark: bool
    # Beds — every colour below is measured against all three.
    background: str
    surface: str
    panel: str
    # Text tiers, brightest to quietest, from the palette's own ramp.
    foreground: str
    muted: str
    subtle: str
    # Chrome.
    border: str
    # Semantic colours — upstream values, see the module docstring.
    primary: str
    secondary: str
    accent: str
    success: str
    warning: str
    error: str

    @property
    def beds(self) -> tuple[str, str, str]:
        return (self.background, self.surface, self.panel)


THEME_SPECS: tuple[ThemeSpec, ...] = (
    ThemeSpec(
        name="catppuccin", dark=True,
        background="#1e1e2e", surface="#313244", panel="#181825",
        foreground="#cdd6f4", muted="#a6adc8", subtle="#9399b2",
        border="#cba6f7",
        primary="#cba6f7", secondary="#89b4fa",
        accent="#f5c2e7", success="#a6e3a1",
        warning="#f9e2af", error="#f38ba8",
    ),
    ThemeSpec(
        # Latte's yellow/green/pink are tuned for syntax on white and sit at
        # 2.0-2.5:1 as UI labels; nudged to 3:1, still amber/green/pink.
        name="catppuccin-latte", dark=False,
        background="#eff1f5", surface="#e6e9ef", panel="#dce0e8",
        foreground="#4c4f69", muted="#606278", subtle="#7c7e92",
        border="#8839ef",
        primary="#8839ef", secondary="#1e66f5",
        accent="#c655aa", success="#319219",
        warning="#bb6c00", error="#d20f39",
    ),
    ThemeSpec(
        name="tokyo-night", dark=True,
        background="#1a1b26", surface="#24283b", panel="#16161e",
        foreground="#c0caf5", muted="#a9b1d6", subtle="#66709b",
        border="#7aa2f7",
        primary="#7aa2f7", secondary="#bb9af7",
        accent="#7dcfff", success="#9ece6a",
        warning="#e0af68", error="#f7768e",
    ),
    ThemeSpec(
        name="tokyo-night-day", dark=False,
        background="#e1e2e7", surface="#d5d6db", panel="#e9e9ed",
        foreground="#1f44a1", muted="#4b5a96", subtle="#6f779f",
        border="#2e7de9",
        primary="#2777e2", secondary="#9753f0",
        accent="#007197", success="#587539",
        warning="#8c6c3e", error="#ea195d",
    ),
    ThemeSpec(
        # Dracula publishes no mid text step, so the quiet tiers are derived by
        # blending foreground toward its comment colour (#6272a4) — hence the
        # violet cast rather than a flat grey.
        name="dracula", dark=True,
        background="#282a36", surface="#44475a", panel="#21222c",
        foreground="#f8f8f2", muted="#aeb6cc", subtle="#8193c7",
        border="#bd93f9",
        primary="#bd93f9", secondary="#8be9fd",
        accent="#ff79c6", success="#50fa7b",
        warning="#f1fa8c", error="#ff5b5a",
    ),
    ThemeSpec(
        name="nord", dark=True,
        background="#2e3440", surface="#3b4252", panel="#272c36",
        foreground="#eceff4", muted="#d8dee9", subtle="#7f8da8",
        border="#88c0d0",
        primary="#88c0d0", secondary="#81a1c1",
        accent="#b48ead", success="#a3be8c",
        warning="#ebcb8b", error="#d07078",
    ),
    ThemeSpec(
        # Fully authentic: every gruvbox colour clears 3:1 on bg0/bg1/bg0_h,
        # and the tiers are its own fg/fg3/fg4 ramp, so it stays warm.
        name="gruvbox", dark=True,
        background="#282828", surface="#3c3836", panel="#1d2021",
        foreground="#ebdbb2", muted="#bdae93", subtle="#a89984",
        border="#83a598",
        primary="#83a598", secondary="#d3869b",
        accent="#fe8019", success="#b8bb26",
        warning="#fabd2f", error="#fb4934",
    ),
    ThemeSpec(
        name="gruvbox-light", dark=False,
        background="#fbf1c7", surface="#ebdbb2", panel="#f2e5bc",
        foreground="#3c3836", muted="#504945", subtle="#7c6f64",
        border="#076678",
        primary="#076678", secondary="#8f3f71",
        accent="#c35e0a", success="#6c782e",
        warning="#ae6f04", error="#c14a4a",
    ),
    ThemeSpec(
        name="one-dark", dark=True,
        background="#282c34", surface="#353b45", panel="#21252b",
        foreground="#b6becb", muted="#9da4b2", subtle="#7d8592",
        border="#61afef",
        primary="#61afef", secondary="#c678dd",
        accent="#56b6c2", success="#98c379",
        warning="#e5c07b", error="#e06c75",
    ),
    ThemeSpec(
        name="one-light", dark=False,
        background="#fafafa", surface="#f0f0f1", panel="#e5e5e6",
        foreground="#383a42", muted="#646671", subtle="#828389",
        border="#4078f2",
        primary="#4078f2", secondary="#a626a4",
        accent="#0184bc", success="#439443",
        warning="#986801", error="#e15447",
    ),
    ThemeSpec(
        # Tiers are Solarized's own base1/base0/base00, which is why the quiet
        # text keeps the teal cast instead of going grey.
        name="solarized", dark=True,
        background="#002b36", surface="#073642", panel="#00212b",
        foreground="#a5b3b3", muted="#8a9c9e", subtle="#677d85",
        border="#268bd2",
        primary="#268bd2", secondary="#6c72c5",
        accent="#2aa198", success="#859900",
        warning="#b58900", error="#e23834",
    ),
    ThemeSpec(
        name="solarized-light", dark=False,
        background="#fdf6e3", surface="#eee8d5", panel="#f7f0dc",
        foreground="#445960", muted="#576c74", subtle="#78888a",
        border="#268bd2",
        primary="#268bd2", secondary="#6c71c4",
        accent="#15958c", success="#7b8e00",
        warning="#aa7e00", error="#dc322f",
    ),
    ThemeSpec(
        name="dark", dark=True,
        background="#12131a", surface="#1e2029", panel="#0c0d12",
        foreground="#e4e6f0", muted="#a9adbe", subtle="#868b9e",
        border="#7aa2f7",
        primary="#7aa2f7", secondary="#c4a7f7",
        accent="#5ccfe6", success="#7fd88f",
        warning="#e5b567", error="#f2788c",
    ),
    ThemeSpec(
        name="light", dark=False,
        background="#fcfcfd", surface="#f1f2f6", panel="#e7e8ee",
        foreground="#2b2d3a", muted="#585c70", subtle="#7b8098",
        border="#2563c7",
        primary="#2563c7", secondary="#8250df",
        accent="#0e7490", success="#1a7f45",
        warning="#b25f00", error="#d4283f",
    ),
)

_PALETTES: dict[str, ThemeSpec] = {spec.name: spec for spec in THEME_SPECS}
_DEFAULT = "catppuccin"

TUI_THEME_NAMES = (
    "dark",
    "light",
    "catppucin",
    "catppuccin",
    "catppucin-latte",
    "catppuccin-latte",
    "tokyo-night",
    "tokyo-night-day",
    "dracula",
    "nord",
    "gruvbox",
    "gruvbox-light",
    "one-dark",
    "one-light",
    "solarized",
    "solarized-light",
)
CLI_THEME_NAMES = (*TUI_THEME_NAMES, "mono")

# The picker intentionally shows only canonical spellings, ordered as a
# settings menu rather than exposing compatibility aliases as duplicate rows.
THEME_PICKER_NAMES = (
    "catppuccin",
    "catppuccin-latte",
    "tokyo-night",
    "tokyo-night-day",
    "dracula",
    "nord",
    "gruvbox",
    "gruvbox-light",
    "one-dark",
    "one-light",
    "solarized",
    "solarized-light",
    "dark",
    "light",
)


# ── glyph vocabulary ──────────────────────────────────────────────────────
# Text-presentation characters only — see the module docstring. Shapes are
# deliberately theme-independent (muscle memory, and greppable transcripts);
# what adapts per theme is their colour, which is exactly what colour emoji
# made impossible.
GLYPHS: dict[str, str] = {
    # Tool families. One marker per *family*, not per tool: what a scanning
    # reader wants from a column of steps is "web, web, file, shell", and a
    # unique mark per tool name would defeat that as surely as the single
    # generic ``tool`` mark the web tools used to share.
    "bash": "❯",
    "read": "≡",
    "write": "✎",
    "delete": "⌫",
    "search": "⌕",
    "web": "⊕",
    # Dashed rather than solid: fetching is a transfer in progress, and the
    # shape has to stay distinguishable from ``search`` at one cell.
    "fetch": "⇣",
    "code": "⌗",
    "image": "▨",
    "plan": "☑",
    "spawn": "⋔",
    "tool": "◆",
    # Assistant channels.
    "thinking": "✻",
    "proposal": "▤",
    # Outcomes / states.
    "ok": "✓",
    "fail": "✗",
    "pending": "○",
    "running": "◐",
    "responding": "●",
    "in_progress": "▶",
    "cancelled": "⊘",
    "stopped": "■",
    "skipped": "–",
    "error": "×",
    "approval": "!",
    "danger": "▲",
    # Conversation markers.
    "user": "›",
    "queued": "⤷",
    "pruned": "↑",
    # Agent Team worker specialties. Every sub-agent shares one ``role_id``
    # (``agent_team_sub``), so what actually distinguishes them is the
    # ``{topic}_{task_type}`` name the coordinator invents. These glyphs mark
    # the inferred task type so a row is readable without colour.
    "agent_research": "⌕",
    "agent_analysis": "∑",
    "agent_write": "✎",
    "agent_verify": "⊙",
    "agent_code": "⌗",
    "agent_data": "▦",
    "agent_plan": "☑",
    "agent_web": "⊕",
    "agent_generic": "◆",
}


def glyph(key: str) -> str:
    """Return the shared monochrome glyph for ``key`` (``''`` if unknown)."""
    return GLYPHS.get(key, "")


# ── Agent Team worker identity ────────────────────────────────────────────
# Ordered longest-idea-first: the first keyword found anywhere in the name
# wins, so ``market_research_2`` is research and ``code_review_1`` is verify
# (``review`` is checked before ``code``).
_AGENT_KIND_KEYWORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("verify", ("verif", "review", "check", "validat", "audit", "fact")),
    ("research", ("research", "search", "discover", "gather", "survey",
                  "investigat", "explor", "lit-review", "litreview")),
    ("web", ("web", "browse", "crawl", "scrape", "news")),
    ("data", ("data", "dataset", "metric", "stat", "table", "financ",
              "quant", "number")),
    ("analysis", ("analy", "compare", "comparison", "evaluat", "assess",
                  "synth", "insight")),
    ("write", ("write", "writer", "writing", "report", "draft", "summar",
               "compose", "narrativ", "edit")),
    ("code", ("code", "coding", "dev", "build", "implement", "engineer",
              "script", "debug", "test")),
    ("plan", ("plan", "roadmap", "outline", "strateg", "coordinat")),
)

AGENT_KINDS: tuple[str, ...] = (
    *(kind for kind, _ in _AGENT_KIND_KEYWORDS), "generic",
)


def agent_kind(name: str) -> str:
    """Infer a worker's specialty from the name the coordinator chose.

    Sub-agent names follow ``{topic}_{task_type}[_{N}]`` by prompt convention
    but nothing enforces it, so this is a best-effort match over the whole
    string and always resolves — unknown names get ``"generic"``.
    """
    haystack = str(name or "").lower()
    for kind, keywords in _AGENT_KIND_KEYWORDS:
        if any(keyword in haystack for keyword in keywords):
            return kind
    return "generic"


def agent_kind_glyph(kind: str) -> str:
    """Return the specialty marker for ``kind`` (falls back to generic)."""
    return GLYPHS.get(f"agent_{kind}", GLYPHS["agent_generic"])


# Identity colours deliberately exclude ``ok``/``err``: those two carry the
# run-state meaning in the very same row, and reusing them for "which worker"
# would make a green name read as a succeeded worker. ``border`` is out for a
# duller reason — most palettes set it to the same value as ``primary``, so it
# would silently hand two workers the same colour.
AGENT_IDENTITY_ROLES: tuple[str, ...] = (
    "user", "todo", "accent", "warn", "muted",
)


def agent_identity_role(index: int) -> str:
    """Map a worker's position in the team to one of the identity colours.

    Positional rather than hashed: hashing a name or session id collides
    freely across a six-colour palette, and two same-coloured rows defeat the
    whole point. The bus lists sessions in creation order and never drops
    one, so a position is stable for the life of a run; teams larger than the
    palette wrap, and the specialty glyph still separates the repeats.
    """
    return AGENT_IDENTITY_ROLES[max(0, index) % len(AGENT_IDENTITY_ROLES)]


# ── palette lookup ────────────────────────────────────────────────────────
def canonical_name(name: str) -> str:
    """Resolve aliases; unknown names fall back to the default palette."""
    resolved = _ALIASES.get(name, name)
    return resolved if resolved in _PALETTES else _DEFAULT


def palette(name: str) -> ThemeSpec:
    """Return the :class:`ThemeSpec` backing ``name``."""
    return _PALETTES[canonical_name(name)]


def blend(a: str, b: str, t: float) -> str:
    """Mix two palette colours in sRGB. ``t=0`` is ``a``, ``t=1`` is ``b``.

    Used for chrome that should read as *between* two tiers — a scrollbar thumb
    sits above its trough but must stay quieter than any text.
    """
    def channels(hex_color: str) -> tuple[int, int, int]:
        raw = hex_color.lstrip("#")
        return tuple(int(raw[i:i + 2], 16) for i in (0, 2, 4))  # type: ignore[return-value]

    ca, cb = channels(a), channels(b)
    return "#" + "".join(
        f"{round(x + (y - x) * t):02x}" for x, y in zip(ca, cb, strict=False)
    )


def active_theme(widget: object, default: str = _DEFAULT) -> str:
    """Return the theme selected on ``widget``'s app.

    Textual raises ``NoActiveAppError`` when ``.app`` is touched outside a
    running app — which is how the modals get unit-tested in isolation — so this
    degrades to the default palette instead of propagating.
    """
    try:
        app = widget.app  # type: ignore[attr-defined]
    except Exception:
        return default
    return getattr(app, "_ui_theme", default)


# Semantic role → palette attribute. Roles are what call sites ask for; the
# indirection is what lets a theme change every surface at once.
#
# ``user`` maps to ``primary`` rather than ``success`` on purpose: echoing the
# user's own input in the same green as a succeeded tool call conflated two
# unrelated meanings, which is its own kind of collision.
_ROLE_TOKENS: dict[str, str] = {
    "text": "foreground",
    "muted": "muted",
    "subtle": "subtle",
    "border": "border",
    "accent": "accent",
    "user": "primary",
    "tool": "warning",
    "warn": "warning",
    "ok": "success",
    "err": "error",
    "todo": "secondary",
    "add": "success",
    "del": "error",
    "hunk": "accent",
}

ROLES: tuple[str, ...] = tuple(_ROLE_TOKENS)


def role_color(name: str, role: str) -> str:
    """Return the hex colour a semantic ``role`` resolves to under ``name``."""
    spec = palette(name)
    return getattr(spec, _ROLE_TOKENS.get(role, "foreground"))


def rich_style(name: str, role: str, *, bold: bool = False) -> str:
    """Return a Rich style string for ``role`` in the ``name`` theme.

    Textual CSS variables theme the widgets, but transcript renderables are Rich
    objects and would otherwise keep Rich's generic ``green`` / ``yellow``.
    Resolving both from one palette is what keeps the two halves in agreement.
    """
    color = role_color(name, role)
    return f"bold {color}" if bold else color


def rich_styles(name: str) -> dict[str, str]:
    """Return every semantic role as a Rich style, for the line-mode renderer."""
    return {role: role_color(name, role) for role in _ROLE_TOKENS}


def ansi_fg(name: str, role: str) -> str:
    """Return a raw truecolor SGR escape for ``role``.

    The line renderer writes some transient lines (the spinner, the streaming
    thinking prefix) straight to stdout without going through Rich, so it needs
    the escape rather than a Rich style string.
    """
    hex_color = role_color(name, role).lstrip("#")
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
    return f"\033[38;2;{r};{g};{b}m"


# ── Textual registration ──────────────────────────────────────────────────
def _textual_theme(spec: ThemeSpec, name: str | None = None) -> Any:
    """Build a Textual ``Theme`` from ``spec``.

    ``variables`` overrides the tokens Textual would otherwise derive by
    alpha-blending: its default ``text`` is ``auto 87%`` and ``text-muted`` is
    ``auto 60%``, both of which discard the contrast we fitted. Pinning them to
    the measured tiers is what makes the CSS side agree with the Rich side.

    The scrollbar tokens are pinned for a different reason: Textual derives the
    trough by darkening the background, which clamps to pure black on our
    darkest palettes and puts a hard black gutter beside a ``#101014`` pane, and
    derives the thumb from ``primary``, which turns a saturated bar into the
    loudest thing on screen. Both are pulled back into the palette: the trough is
    ``panel``, and the thumb sits *between* trough and ``subtle`` so it reads as
    chrome rather than as content — a scrollbar at full text contrast draws the
    eye harder than anything it scrolls past.
    """
    from textual.theme import Theme

    thumb = blend(spec.panel, spec.subtle, 0.55)

    return Theme(
        name=name or spec.name,
        primary=spec.primary,
        secondary=spec.secondary,
        warning=spec.warning,
        error=spec.error,
        success=spec.success,
        accent=spec.accent,
        foreground=spec.foreground,
        background=spec.background,
        surface=spec.surface,
        panel=spec.panel,
        dark=spec.dark,
        variables={
            "text": spec.foreground,
            "text-muted": spec.muted,
            "text-disabled": spec.subtle,
            "border": spec.border,
            "border-blurred": spec.subtle,
            "scrollbar": thumb,
            "scrollbar-hover": spec.subtle,
            "scrollbar-active": spec.primary,
            "scrollbar-background": spec.panel,
            "scrollbar-background-hover": spec.panel,
            "scrollbar-background-active": spec.panel,
            "scrollbar-corner-color": spec.panel,
            # Textual derives the selected-row foreground from ``text``, whose
            # stock value is the auto-contrast ``auto 87%``. Pinning ``text`` to
            # a measured colour therefore also pinned this one, painting dark
            # body text onto the ``primary``-coloured cursor. ``auto`` restores
            # the contrast pick for the one place a palette colour is the *bed*.
            "block-cursor-foreground": "auto",
            "button-color-foreground": "auto",
        },
    )


def register_themes(app: object) -> None:
    """Register every curated theme, plus the Catppuccin spelling aliases."""
    for spec in THEME_SPECS:
        app.register_theme(_textual_theme(spec))  # type: ignore[attr-defined]
    for alias, canonical in _ALIASES.items():
        app.register_theme(  # type: ignore[attr-defined]
            _textual_theme(_PALETTES[canonical], name=alias)
        )


__all__ = [
    "CLI_THEME_NAMES", "GLYPHS", "ROLES", "THEME_PICKER_NAMES", "THEME_SPECS",
    "TUI_THEME_NAMES", "ThemeSpec", "active_theme", "ansi_fg", "canonical_name",
    "glyph", "palette", "register_themes", "rich_style", "rich_styles",
    "role_color",
]
