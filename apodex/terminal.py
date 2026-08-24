"""Terminal capability detection shared by the CLI and its smoke tests.

Keep this deliberately conservative: the full-screen UI is an enhancement,
while line mode is the compatibility baseline for pipes, simple terminals and
explicitly colourless output.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class TerminalUI:
    """Resolved presentation mode for one CLI invocation."""

    interactive: bool
    use_tui: bool
    theme: str
    reason: str
    colors: int = 16777216
    color_warning: str = ""


def detect_color_depth(environ: Mapping[str, str]) -> int:
    """Number of colours Rich and Textual will actually use.

    Both read ``COLORTERM`` then ``TERM``, in that order, and neither queries the
    terminal — so this mirrors their logic rather than guessing.
    """
    if environ.get("COLORTERM", "").strip().lower() in ("truecolor", "24bit"):
        return 16777216
    term = environ.get("TERM", "").strip().lower()
    if term == "dumb" or not term:
        return 8
    suffix = term.rpartition("-")[2]
    if suffix in ("truecolor", "direct"):
        return 16777216
    if suffix == "256color":
        return 256
    return 8


def resolve_terminal_ui(
    *,
    stdin_tty: bool,
    stdout_tty: bool,
    one_shot: bool,
    no_tui: bool,
    no_color: bool,
    requested_theme: str,
    environ: Mapping[str, str],
) -> TerminalUI:
    """Choose the richest UI the current terminal can safely support.

    ``NO_COLOR`` follows the cross-CLI convention: its presence is enough,
    even when the value is empty. ``TERM=dumb`` is treated the same way since
    cursor addressing and colour cannot be assumed there. SSH and tmux need no
    special cases; when they expose normal TTYs and a capable ``TERM`` they can
    use the same Textual path as a local terminal.
    """

    interactive = stdin_tty and not one_shot
    term_is_dumb = environ.get("TERM", "").strip().lower() == "dumb"
    mono = no_color or "NO_COLOR" in environ or requested_theme == "mono" or term_is_dumb
    theme = "mono" if mono else requested_theme

    if one_shot:
        reason = "one-shot output"
    elif not stdin_tty:
        reason = "stdin is not a TTY"
    elif not stdout_tty:
        reason = "stdout is not a TTY"
    elif no_tui:
        reason = "--no-tui requested"
    elif term_is_dumb:
        reason = "TERM=dumb"
    elif mono:
        reason = "colourless output requested"
    else:
        reason = "full-screen terminal available"

    # A palette needs colours to exist in. At 8 colours the themes are not
    # approximated but destroyed — every value snaps to one of eight ANSI slots
    # the *terminal* defines, so gruvbox's cream becomes pure white, its muted
    # tan becomes grey, and its orange and red become the same red. Say so
    # instead of letting a correct palette look broken.
    colors = 8 if mono else detect_color_depth(environ)
    color_warning = ""
    if not mono and colors < 256:
        color_warning = (
            f"terminal reports only {colors} colours, so theme colours will be "
            "replaced by your terminal's own ANSI palette. Set "
            "COLORTERM=truecolor (or TERM=xterm-256color) to see the real theme."
        )

    return TerminalUI(
        interactive=interactive,
        use_tui=(stdin_tty and stdout_tty and not one_shot and not no_tui and not mono),
        theme=theme,
        reason=reason,
        colors=colors,
        color_warning=color_warning,
    )


__all__ = ["TerminalUI", "detect_color_depth", "resolve_terminal_ui"]
