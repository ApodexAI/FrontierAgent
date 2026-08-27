"""Compatibility decisions for full-screen and fallback terminal modes."""

from __future__ import annotations

import pytest

from apodex.terminal import resolve_terminal_ui
from apodex.tui.themes import CLI_THEME_NAMES, TUI_THEME_NAMES


def test_curated_theme_names_are_available_to_the_cli() -> None:
    requested = {
        "catppucin", "catppucin-latte", "tokyo-night", "tokyo-night-day",
        "dracula", "nord", "gruvbox", "gruvbox-light", "one-dark",
        "one-light", "solarized", "solarized-light",
    }
    assert requested <= set(TUI_THEME_NAMES)
    assert set(TUI_THEME_NAMES) | {"mono"} == set(CLI_THEME_NAMES)


def test_cli_defaults_to_catppuccin() -> None:
    from apodex.cli import build_parser

    assert build_parser().parse_args([]).theme == "catppuccin"


def test_cli_accepts_repeatable_inputs() -> None:
    from apodex.cli import build_parser

    args = build_parser().parse_args(["--input", "a.pdf", "--input=b.png"])
    assert args.input == ["a.pdf", "b.png"]


def test_iterm_disables_textual_kitty_keyboard_by_default() -> None:
    from apodex.terminal import configure_terminal_keyboard

    environ = {"TERM_PROGRAM": "iTerm.app"}
    configure_terminal_keyboard(environ)

    assert environ["TEXTUAL_DISABLE_KITTY_KEY"] == "1"


def test_keyboard_fallback_leaves_other_terminals_unchanged() -> None:
    from apodex.terminal import configure_terminal_keyboard

    environ = {"TERM_PROGRAM": "Apple_Terminal"}
    configure_terminal_keyboard(environ)

    assert "TEXTUAL_DISABLE_KITTY_KEY" not in environ


@pytest.mark.parametrize("explicit", ["", "1", "custom"])
def test_keyboard_fallback_preserves_an_explicit_user_value(explicit) -> None:
    from apodex.terminal import configure_terminal_keyboard

    environ = {
        "TERM_PROGRAM": "iTerm.app",
        "TEXTUAL_DISABLE_KITTY_KEY": explicit,
    }
    configure_terminal_keyboard(environ)

    assert environ["TEXTUAL_DISABLE_KITTY_KEY"] == explicit


@pytest.mark.parametrize(
    ("stdin_tty", "stdout_tty", "one_shot", "no_tui", "theme", "env", "use_tui"),
    [
        (True, True, False, False, "dark", {"TERM": "xterm-256color"}, True),
        (True, True, False, False, "dark", {"TERM": "screen-256color"}, True),
        (True, True, False, False, "dark", {"TERM": "dumb"}, False),
        (False, True, False, False, "dark", {}, False),
        (True, False, False, False, "dark", {}, False),
        (True, True, True, False, "dark", {}, False),
        (True, True, False, True, "dark", {}, False),
        (True, True, False, False, "mono", {}, False),
        (True, True, False, False, "dark", {"NO_COLOR": ""}, False),
    ],
)
def test_terminal_compatibility_matrix(
    stdin_tty, stdout_tty, one_shot, no_tui, theme, env, use_tui,
):
    ui = resolve_terminal_ui(
        stdin_tty=stdin_tty,
        stdout_tty=stdout_tty,
        one_shot=one_shot,
        no_tui=no_tui,
        no_color=False,
        requested_theme=theme,
        environ=env,
    )
    assert ui.use_tui is use_tui


def test_no_color_convention_forces_mono_line_mode():
    ui = resolve_terminal_ui(
        stdin_tty=True,
        stdout_tty=True,
        one_shot=False,
        no_tui=False,
        no_color=False,
        requested_theme="light",
        environ={"NO_COLOR": "1", "TERM": "xterm-256color"},
    )
    assert ui.theme == "mono"
    assert ui.use_tui is False
    assert "colourless" in ui.reason


def test_piped_stdout_keeps_interactive_line_input():
    ui = resolve_terminal_ui(
        stdin_tty=True,
        stdout_tty=False,
        one_shot=False,
        no_tui=False,
        no_color=False,
        requested_theme="dark",
        environ={},
    )
    assert ui.interactive is True
    assert ui.use_tui is False
    assert ui.reason == "stdout is not a TTY"


# ── colour depth: a palette needs colours to exist in ─────────────────────
# Regression: the themes were correct but every value was being quantised into
# the terminal's own 8 ANSI slots, which reads as "all the colours are grey".
@pytest.mark.parametrize(
    ("environ", "expected"),
    [
        ({"COLORTERM": "truecolor", "TERM": "xterm"}, 16777216),
        ({"COLORTERM": "24bit", "TERM": "xterm"}, 16777216),
        ({"TERM": "xterm-256color"}, 256),
        ({"TERM": "xterm-truecolor"}, 16777216),
        ({"TERM": "xterm-direct"}, 16777216),
        ({"TERM": "xterm"}, 8),          # Docker's default inside `-it`
        ({"TERM": "screen"}, 8),         # tmux without 256-colour support
        ({}, 8),
        ({"TERM": "dumb"}, 8),
    ],
)
def test_detect_color_depth_mirrors_rich_and_textual(environ, expected) -> None:
    from apodex.terminal import detect_color_depth

    assert detect_color_depth(environ) == expected


def test_detect_color_depth_agrees_with_rich() -> None:
    """Pin the mirror to the real implementation rather than to our belief."""
    from rich.console import Console

    from apodex.terminal import detect_color_depth

    mapping = {8: "standard", 256: "256", 16777216: "truecolor"}
    for environ in ({"TERM": "xterm"}, {"TERM": "xterm-256color"},
                    {"TERM": "xterm-256color", "COLORTERM": "truecolor"}, {}):
        console = Console(force_terminal=True, _environ=dict(environ))
        assert mapping[detect_color_depth(environ)] == console.color_system, environ


def test_limited_color_terminal_is_reported_not_hidden() -> None:
    ui = resolve_terminal_ui(
        stdin_tty=True, stdout_tty=True, one_shot=False, no_tui=False,
        no_color=False, requested_theme="gruvbox", environ={"TERM": "xterm"},
    )
    assert ui.colors == 8
    assert "only 8 colours" in ui.color_warning
    assert "COLORTERM=truecolor" in ui.color_warning
    assert ui.use_tui is True  # degraded colour is not a reason to drop the TUI


def test_capable_terminal_warns_about_nothing() -> None:
    ui = resolve_terminal_ui(
        stdin_tty=True, stdout_tty=True, one_shot=False, no_tui=False,
        no_color=False, requested_theme="gruvbox",
        environ={"TERM": "xterm-256color", "COLORTERM": "truecolor"},
    )
    assert ui.colors == 16777216
    assert ui.color_warning == ""


def test_mono_does_not_warn_about_colour_depth() -> None:
    """``mono`` is colourless on purpose; a colour warning there is just noise."""
    ui = resolve_terminal_ui(
        stdin_tty=True, stdout_tty=True, one_shot=False, no_tui=False,
        no_color=True, requested_theme="gruvbox", environ={"TERM": "xterm"},
    )
    assert ui.theme == "mono"
    assert ui.color_warning == ""


# ── the container must inherit the host terminal's capability ─────────────
def test_container_receives_a_usable_colour_depth() -> None:
    """Regression: Docker sets ``TERM=xterm`` and forwards no ``COLORTERM``, so
    the containerised TUI — the default path on macOS — rendered every theme
    through 8 ANSI slots no matter what the host terminal could do."""
    from apodex.docker import terminal_env
    from apodex.terminal import detect_color_depth

    def env_of(args: list[str]) -> dict[str, str]:
        return dict(
            pair.split("=", 1) for flag, pair in zip(args[::2], args[1::2], strict=False)
            if flag == "-e"
        )

    # Host advertises truecolor → forwarded verbatim.
    passed = env_of(terminal_env({"TERM": "xterm-256color", "COLORTERM": "truecolor"}))
    assert passed["TERM"] == "xterm-256color"
    assert passed["COLORTERM"] == "truecolor"

    # Host says nothing (common when launched from a GUI) → 256-colour floor,
    # never the 8-colour default the container would otherwise get.
    for host in ({}, {"TERM": "xterm"}, {"TERM": "screen"}):
        depth = detect_color_depth(env_of(terminal_env(host)))
        assert depth >= 256, (host, depth)


def test_container_still_honours_an_explicitly_dumb_terminal() -> None:
    from apodex.docker import terminal_env

    assert terminal_env({"TERM": "dumb"}) == ["-e", "TERM=dumb"]
    assert "COLORTERM=truecolor" not in terminal_env({"TERM": "dumb"})


def test_container_receives_terminal_identity_and_keyboard_override() -> None:
    from apodex.docker import terminal_env
    from apodex.terminal import configure_terminal_keyboard

    host = {
        "TERM": "xterm-256color",
        "TERM_PROGRAM": "iTerm.app",
        "TERM_PROGRAM_VERSION": "3.6.11",
    }
    configure_terminal_keyboard(host)
    args = terminal_env(host)
    passed = dict(
        pair.split("=", 1)
        for flag, pair in zip(args[::2], args[1::2], strict=False)
        if flag == "-e"
    )

    assert passed["TERM_PROGRAM"] == "iTerm.app"
    assert passed["TERM_PROGRAM_VERSION"] == "3.6.11"
    assert passed["TEXTUAL_DISABLE_KITTY_KEY"] == "1"
