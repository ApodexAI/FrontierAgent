"""Terminal rendering — Rich when available, graceful plain-text fallback.

Streaming text (assistant content + thinking) is written incrementally to
stdout so it appears token-by-token like apodex. Discrete blocks (tool
calls, diff previews, tool results, the todo plan, the final answer) are
rendered with Rich panels when ``rich`` is installed, else plain framed
text.

Colors come from a named **theme** (including dark/light variants and popular
editor palettes) or ``mono`` for color-free output, so look-and-feel is
configurable without touching call sites. The palettes live in
:mod:`apodex.tui.themes` and are shared with the full-screen TUI — this module
holds no colours of its own, because two copies of the same palette is exactly
how the line UI and the TUI drifted apart. The renderer never raises — a
rendering failure must not abort an agent run.
"""

from __future__ import annotations

import asyncio
import sys
import time
from typing import TYPE_CHECKING, Any

from apodex.tui.themes import GLYPHS, ansi_fg, rich_styles

if TYPE_CHECKING:
    # For the checker these names are always bound: ``rich`` is a mandatory
    # dependency (see [project] dependencies in pyproject.toml), and
    # ``markdown_it`` arrives with it. The runtime try/except below stays as a
    # defensive fallback for a broken install, but a checker cannot connect the
    # names it binds to the ``_HAVE_RICH`` guard that protects every use, and
    # so reports all 19 of them as possibly-unbound.
    from markdown_it import MarkdownIt
    from rich.console import Console
    from rich.markdown import Markdown
    from rich.panel import Panel
    from rich.text import Text

try:  # Rich is optional; degrade gracefully.
    # markdown-it is Rich's own parser (and its dependency); ``preserve_line_breaks``
    # reuses it so "what counts as code" cannot drift from what Rich renders.
    from markdown_it import MarkdownIt
    from rich.console import Console
    from rich.markdown import Markdown
    from rich.panel import Panel
    from rich.text import Text
    _HAVE_RICH = True
except Exception:  # pragma: no cover - exercised only without rich
    _HAVE_RICH = False


#: Block types Rich renders verbatim — their source lines must not be touched.
_VERBATIM_BLOCKS = frozenset({"fence", "code_block", "html_block"})


def _verbatim_lines(text: str) -> set[int]:
    """0-based indices of the lines Rich will render as literal content.

    Located with Rich's own parser rather than a fence regex. Hand-rolling the
    rule means re-implementing CommonMark: a fence closes only on the same
    character at the same-or-greater length (so ``~~~`` inside a ``` block, or
    ``` inside a ```` block, is content), and fences nest under container
    prefixes such as blockquotes. A boolean "am I in a fence" toggle gets all
    three wrong and silently edits code it promised to preserve.
    """
    parser = MarkdownIt().enable("strikethrough").enable("table")
    protected: set[int] = set()
    for token in parser.parse(text):
        if token.type in _VERBATIM_BLOCKS and token.map:
            protected.update(range(*token.map))
    return protected


def _ends_with_hard_break(line: str) -> bool:
    """True when *line* already ends in a CommonMark hard line break.

    Two trailing spaces, or an ODD number of trailing backslashes: an even
    number is escaped literal backslashes (``C:\\``), which leaves the newline
    soft, so those lines still need a marker.
    """
    if line.endswith("  "):
        return True
    return (len(line) - len(line.rstrip("\\"))) % 2 == 1


def preserve_line_breaks(text: str) -> str:
    """Make a model's line breaks survive CommonMark rendering.

    ``rich.markdown`` is CommonMark, where a single newline inside a block is a
    *soft* break: the renderer reflows those lines into one paragraph. Model
    output is line-oriented, so that silently destroys any structure carried by
    newlines alone — most visibly the references section, which the report
    prompt requires as "one reference per line, plain text — no bullets, no
    blank lines between entries". Rendered as CommonMark, that becomes::

        [1] https://…-6310711 [2]
        https://…-on-stronger-ai-boom [3] https://…-9497810 [4]

    Entries run together and the markers strand at line ends. Appending the
    two-space hard-break marker keeps each line its own line while leaving the
    text valid Markdown, so headings, lists, tables, emphasis and links go on
    working exactly as before.

    Skipped where a trailing space would change meaning rather than layout:
    verbatim blocks (see :func:`_verbatim_lines`), lines that already end in a
    hard break (so this is idempotent), and lines at the end of a block, which
    need no marker because the blank line already breaks them.

    Without Rich there is nothing to repair — the plain-text path prints the
    answer as-is and every newline already survives — so the text is returned
    untouched.
    """
    if not _HAVE_RICH:
        return text
    lines = text.split("\n")
    protected = _verbatim_lines(text)
    for i, line in enumerate(lines):
        if i in protected or i + 1 in protected:
            continue
        stripped = line.rstrip()
        if not stripped or _ends_with_hard_break(line):
            continue
        if i + 1 >= len(lines) or not lines[i + 1].strip():
            continue
        lines[i] = f"{stripped}  "
    return "\n".join(lines)


# Labels pair a shared monochrome glyph with the tool's short name. The glyphs
# are text-presentation characters so they inherit the active theme's colour —
# the colour emoji these replaced painted themselves and ignored the theme.
_TOOL_GLYPH = {
    # Shell / code execution.
    "bash": f"{GLYPHS['bash']} Bash",
    "run_python_code": f"{GLYPHS['code']} Python",
    # Reading.
    "read_file": f"{GLYPHS['read']} Read",
    "read_text": f"{GLYPHS['read']} Read",
    "file_editor_view": f"{GLYPHS['read']} View",
    "view_image": f"{GLYPHS['image']} Image",
    # Writing.
    "write_file": f"{GLYPHS['write']} Write",
    "create_file": f"{GLYPHS['write']} Create",
    "file_editor_create": f"{GLYPHS['write']} Create",
    "file_editor_str_replace": f"{GLYPHS['write']} Edit",
    "delete_file": f"{GLYPHS['delete']} Delete",
    # Local search.
    "grep_search": f"{GLYPHS['search']} Grep",
    "glob_search": f"{GLYPHS['search']} Glob",
    # Web. A research run spends most of its steps here, and these three shared
    # the generic fallback marker until now — which made a page of alternating
    # search / fetch / download steps read as one undifferentiated family, the
    # opposite of what the marker column is for.
    "web_search": f"{GLYPHS['web']} Search web",
    "web_fetch": f"{GLYPHS['fetch']} Fetch",
    "download_file": f"{GLYPHS['fetch']} Download",
    # Planning / task board.
    "todo_write": f"{GLYPHS['plan']} Plan",
    "add_task": f"{GLYPHS['plan']} Plan",
    "update_task": f"{GLYPHS['plan']} Plan",
    "finish_planning": f"{GLYPHS['plan']} Planning done",
    "exit_plan_mode": f"{GLYPHS['proposal']} Plan review",
    # Agent Team orchestration.
    "create_subagent": f"{GLYPHS['spawn']} Spawn",
    "assign_task": f"{GLYPHS['spawn']} Assign",
    "collect_reports": f"{GLYPHS['spawn']} Collect",
    "stop_subagent": f"{GLYPHS['stopped']} Stop agent",
    "submit_report": f"{GLYPHS['proposal']} Report",
}

def tool_label(name: str, *, marker: bool = True) -> str:
    """The display name for ``name``: family marker plus a short human label.

    ``marker=False`` drops the family marker for rows that already carry a
    marker of their own — a result heading leads with ✓/✗, and two glyphs in one
    column stop reading as a column. The *words* still have to match the call
    row above, which is why both surfaces resolve them here rather than falling
    back to the raw ``snake_case`` tool name in one place and the label in
    another.
    """
    label = _TOOL_GLYPH.get(name, f"{GLYPHS['tool']} {name}")
    return label if marker else label.split(" ", 1)[-1]


_MAX_RESULT_CHARS = 4000
_MAX_DIFF_LINES = 200

_DEFAULT_THEME = "catppuccin"


def diff_to_text(
    diff_text: str, *, max_lines: int = _MAX_DIFF_LINES,
    theme: str = _DEFAULT_THEME,
) -> Text:
    """Colorize a unified diff into a Rich ``Text``, truncating to ``max_lines``.

    Additions, deletions, hunk headers and context lines take the ``theme``'s
    own add / del / hunk / subtle colours rather than Rich's generic
    ``green`` / ``red`` / ``cyan`` / ``dim``, so a diff shown inside a themed
    panel cannot fight the panel it sits in.

    Shared by the TUI sink and the approval modal so both surfaces color diffs
    identically. Requires rich, which the TUI always has.
    """
    styles = rich_styles(theme)
    lines = diff_text.splitlines()
    truncated = ""
    if len(lines) > max_lines:
        truncated = f"… [+{len(lines) - max_lines} more diff lines]"
        lines = lines[:max_lines]
    body = Text()
    for ln in lines:
        if ln.startswith("+") and not ln.startswith("+++"):
            body.append(ln + "\n", style=styles["add"])
        elif ln.startswith("-") and not ln.startswith("---"):
            body.append(ln + "\n", style=styles["del"])
        elif ln.startswith("@@"):
            body.append(ln + "\n", style=styles["hunk"])
        else:
            body.append(ln + "\n", style=styles["subtle"])
    if truncated:
        body.append(truncated, style=styles["subtle"])
    return body


#: Row cap for the changed-files summary. A single build step can promote
#: thousands of paths into the journal; the panel is a summary, and a terminal
#: scrollback full of ``dist/`` entries buries the answer above it.
_MAX_CHANGE_ROWS = 50


class Renderer:
    """Stateful, theme-aware console renderer shared across a session."""

    def __init__(self, *, theme: str = _DEFAULT_THEME, color: bool = True,
                 verbose: bool = True) -> None:
        # ``mono`` is the one "theme" with no palette: it means colour-free.
        self._theme = theme
        styles = None if theme == "mono" else rich_styles(theme)
        use_color = color and styles is not None and _HAVE_RICH
        self._styles: dict[str, str] = styles or {}
        self._console = Console(soft_wrap=True) if use_color else None
        self._streaming_kind: str | None = None  # "thinking" | "content" | None
        self._content_shown = False  # visible answer text already streamed/printed
        # Stream the thinking channel raw by default (users like seeing the
        # reasoning); ``/verbose`` toggles it off → a compact live indicator,
        # handy when a model's reasoning is long or rambling.
        self._verbose = verbose
        # Animated transient line ("working…" / "thinking…"). One background
        # task ticks ~10×/s so a silent model gap never *looks* frozen.
        self._spinner_task: asyncio.Task | None = None
        self._spinner_phase: str | None = None  # "working" | "thinking" | None
        self._spinner_start = 0.0
        self._think_chars = 0
        self._frame = 0
        # Optional live usage readout in the spinner / footer (set by session).
        self._usage: Any = None
        self._window = 0

    def set_usage(self, usage: Any, window: int) -> None:
        """Attach a token-usage object so the spinner can show context fill."""
        self._usage = usage
        self._window = window

    # ── low-level ────────────────────────────────────────────────────────
    def _s(self, key: str) -> str:
        """Resolve a semantic style key for the active theme (empty if mono)."""
        return self._styles.get(key, "")

    def _w(self, s: str) -> None:
        sys.stdout.write(s)
        sys.stdout.flush()

    def _print(self, *args: Any, **kw: Any) -> None:
        if self._console is not None:
            try:
                self._console.print(*args, **kw)
                return
            except Exception:
                pass
        print(*[a if isinstance(a, str) else str(a) for a in args])

    def _styled(self, text: str, key: str, *, bold: bool = False) -> str:
        st = self._bold(key) if bold else self._s(key)
        return f"[{st}]{text}[/]" if (st and self._console is not None) else text

    def _bold(self, key: str) -> str:
        """A bold variant of a role, for headline text."""
        st = self._s(key)
        return f"bold {st}" if st else "bold"

    def _sgr(self, key: str) -> str:
        """Raw truecolor escape for the lines written straight to stdout.

        The spinner and the streaming thinking prefix bypass Rich (they redraw
        in place), so they need the escape rather than a Rich style. They used
        to hard-code ``\\033[2m`` — terminal ``dim`` — which is a blend by an
        unspecified amount and is what made thinking text collide with the
        background on darker palettes.
        """
        return "" if self._console is None else ansi_fg(self._theme, key)

    def set_verbose(self, on: bool) -> None:
        """Toggle raw thinking streaming (``/verbose``)."""
        self._verbose = bool(on)

    async def _spin_loop(self) -> None:
        """Redraw the transient indicator line ~10×/s until cancelled."""
        frames = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
        try:
            while True:
                el = int(time.monotonic() - self._spinner_start)
                if self._spinner_phase == "thinking":
                    line = (f"{GLYPHS['thinking']} thinking… {el}s · "
                            f"{self._think_chars} chars  ·  /verbose to show")
                else:
                    ctx = ""
                    if self._usage is not None and self._window:
                        pct = self._usage.context_pct_left(self._window)
                        if pct is not None:
                            ctx = f" · ctx {pct}% left"
                    line = (f"{frames[self._frame % len(frames)]} working… {el}s{ctx}"
                            "  ·  Ctrl-C interrupt · type to steer")
                self._w(f"\r{self._sgr('muted')}{line}\033[0m\033[K")
                self._frame += 1
                await asyncio.sleep(0.1)
        except asyncio.CancelledError:
            pass

    def _spin_start(self, phase: str) -> None:
        """Begin (or switch) the transient indicator. No-op without color or a
        running loop. Never overlays actively-streaming content."""
        if self._console is None or self._streaming_kind == "content":
            return
        if self._spinner_phase != phase:
            self._spinner_phase = phase
            self._spinner_start = time.monotonic()
            if phase == "thinking":
                self._think_chars = 0
        if self._spinner_task is None:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                self._spinner_phase = None  # can't animate without a loop
                return
            self._spinner_task = loop.create_task(self._spin_loop())

    def _spin_stop(self) -> None:
        """Cancel the ticker and erase its line. When leaving a collapsed
        thinking span, leave a one-line ``◇ thought for Ns`` trace."""
        phase = self._spinner_phase
        if self._spinner_task is not None:
            self._spinner_task.cancel()
            self._spinner_task = None
        if phase is not None:
            self._spinner_phase = None
            self._w("\r\033[K")
            if phase == "thinking":
                el = int(time.monotonic() - self._spinner_start)
                if el >= 1:
                    self._w(f"{self._sgr('muted')}{GLYPHS['thinking']} "
                            f"thought for {el}s\033[0m\n")

    # ``working_on`` / ``working_off`` are the names the observer calls; the
    # animated ticker backs them now.
    def working_on(self) -> None:
        self._spin_start("working")

    def working_off(self) -> None:
        self._spin_stop()

    def _end_stream(self) -> None:
        """Close any open streaming line before printing a discrete block."""
        self._spin_stop()
        if self._streaming_kind is not None:
            if self._console is not None:
                self._w("\033[0m")  # drop the thinking colour before the newline
            self._w("\n")
            self._streaming_kind = None

    # ── banner / framing ────────────────────────────────────────────────
    def banner(self, *, model: str, cwd: str, auto_approve: bool, mode: str = "coding") -> None:
        title = "FrontierAgent"
        sub = (
            f"mode: {mode}   model: {model}   cwd: {cwd}   "
            f"approve: {'AUTO' if auto_approve else 'manual'}"
        )
        # Don't enumerate every command here (it drifts out of date) — point at
        # /help, which is the single source of truth, and surface the few
        # non-obvious affordances.
        hint = ("Type a task.  /help for all commands  ·  /plan plan-first  ·  "
                "type while it works to steer  ·  Ctrl-C interrupts")
        if self._console is not None:
            muted = self._s("muted")
            self._print(Panel.fit(
                f"[{self._bold('accent')}]{title}[/]\n[{muted}]{sub}[/]",
                border_style=self._s("border"),
            ))
            self._print(f"[{muted}]{hint}[/]")
        else:
            print(f"=== {title} ===\n{sub}")
            print(hint)

    def rule(self) -> None:
        self._end_stream()
        rule = "─" * 60
        self._print(f"[{self._s('subtle')}]{rule}[/]" if self._console else "-" * 60)

    def echo_user(self, text: str) -> None:
        self._end_stream()
        glyph = GLYPHS["user"]
        self._print(
            self._styled(glyph, "user", bold=True) + f" {text}"
            if self._console else f"> {text}"
        )

    # ── streaming ────────────────────────────────────────────────────────
    def thinking_delta(self, s: str) -> None:
        if not s:
            return
        if self._verbose:
            # Raw, token-by-token thinking in the muted tier — opt-in via
            # ``/verbose``. Muted is a measured colour, not terminal ``dim``.
            if self._streaming_kind != "thinking":
                self._end_stream()
                self._w(f"{self._sgr('muted')}{GLYPHS['thinking']} "
                        if self._console is not None else "[thinking] ")
                self._streaming_kind = "thinking"
            self._w(s)
            return
        # Default: collapse into the compact live ticker (count + elapsed),
        # so a verbose/rambling reasoning channel doesn't flood the screen.
        if self._streaming_kind is not None:
            self._end_stream()
        self._spin_start("thinking")
        self._think_chars += len(s)

    def content_delta(self, s: str) -> None:
        if not s:
            return
        self._content_shown = True
        if self._streaming_kind != "content":
            self._end_stream()
            if self._console is not None:
                self._w("\033[0m")  # leave the thinking colour exactly once
            self._streaming_kind = "content"
        self._w(s)

    def end_turn_text(self) -> None:
        """Close one assistant message's prose without disturbing the ticker.

        Only an open *content* stream is closed: the thinking ticker belongs to
        the phase, not the message, and stopping it here would blank the one
        indicator a user has while the next turn is being generated.
        """
        if self._streaming_kind == "content":
            self._end_stream()

    def turn_text_fallback(self, ai_text: str, thinking: str) -> None:
        """Render full turn text when no streaming deltas fired this turn."""
        self._end_stream()
        if thinking:
            t = thinking.strip()[:2000]
            self._print(
                f"[{self._s('muted')}]{GLYPHS['thinking']} {t}[/]"
                if self._console else f"[thinking] {t}"
            )
        if ai_text.strip():
            self._print(ai_text.strip())
            self._content_shown = True

    # ── tool calls / diff preview / results ───────────────────────────────
    def activity_call(self, name: str, args: dict, *, call_id: str = "") -> None:
        """Start a timeline-only row for calls with a custom main presentation."""

    def tool_call(
        self, name: str, args: dict, risk_reason: str = "", danger: bool = False,
        *, call_id: str = "",
    ) -> None:
        self._end_stream()
        label = tool_label(name)
        detail = self._summarize_args(name, args)
        mark = f"{GLYPHS['danger']} " if danger else ""
        if self._console is not None:
            line = f"{self._styled(label, 'tool', bold=True)} {detail}"
            if risk_reason:
                style = self._bold("err") if danger else self._s("muted")
                line += f"  [{style}]({mark}{risk_reason})[/]"
            self._print(line)
        else:
            print(f"{label} {detail}" + (f"  ({mark}{risk_reason})" if risk_reason else ""))
        # For bash, when the headline is a plain-English description (or a
        # leading-comment label), surface the EXACT command on a quieter sub-line
        # so the user sees both the intent and precisely what will run.
        if name == "bash":
            cmd = str(args.get("command", "")).strip()
            if cmd and cmd != detail:
                shown = cmd if len(cmd) <= 500 else cmd[:500] + " …"
                self._print(
                    f"[{self._s('muted')}]    $ {shown}[/]"
                    if self._console else f"    $ {shown}"
                )

    def diff_preview(self, diff_text: str, *, stats: tuple[int, int] | None = None) -> None:
        """Render a unified diff (proposed edit) before it is applied."""
        self._end_stream()
        lines = diff_text.splitlines()
        truncated = ""
        if len(lines) > _MAX_DIFF_LINES:
            truncated = f"… [+{len(lines) - _MAX_DIFF_LINES} more diff lines]"
            lines = lines[:_MAX_DIFF_LINES]
        title = "proposed change"
        if stats is not None:
            title += (
                f"  [{self._s('add')}]+{stats[0]}[/] [{self._s('del')}]-{stats[1]}[/]"
                if self._console else f"  +{stats[0]} -{stats[1]}"
            )
        if self._console is not None:
            body = Text()
            for ln in lines:
                if ln.startswith("+") and not ln.startswith("+++"):
                    body.append(ln + "\n", style=self._s("add"))
                elif ln.startswith("-") and not ln.startswith("---"):
                    body.append(ln + "\n", style=self._s("del"))
                elif ln.startswith("@@"):
                    body.append(ln + "\n", style=self._s("hunk"))
                else:
                    body.append(ln + "\n", style=self._s("subtle"))
            if truncated:
                body.append(truncated, style=self._s("subtle"))
            try:
                self._print(Panel(body, title=title, border_style=self._s("border"), title_align="left"))
                return
            except Exception:
                pass
        print(f"--- {title} ---\n" + "\n".join(lines) + (f"\n{truncated}" if truncated else ""))

    def activity_result(
        self, name: str, *, call_id: str = "", is_error: bool,
        ms: int = 0, outcome: str = "",
    ) -> None:
        """Complete an activity row when its result has a custom presentation."""

    def tool_result(
        self, name: str, result: str, *, is_error: bool, ms: int = 0,
        call_id: str = "",
    ) -> None:
        self._end_stream()
        body = result if isinstance(result, str) else str(result)
        truncated = ""
        if len(body) > _MAX_RESULT_CHARS:
            truncated = (f"\n… [+{len(body) - _MAX_RESULT_CHARS} chars hidden here — "
                         "full output is in the trace (/log)]")
            body = body[:_MAX_RESULT_CHARS]
        shown = tool_label(name, marker=False)
        header = (
            f"{GLYPHS['fail']} {shown} failed" if is_error else f"{GLYPHS['ok']} {shown}"
        )
        if ms:
            header += f"  ({ms} ms)"
        if self._console is not None:
            style = self._s("err") if is_error else self._s("ok")
            try:
                self._print(Panel(body + truncated, title=header, border_style=style, title_align="left"))
                return
            except Exception:
                pass
        print(f"[{header}]\n{body}{truncated}")

    # ── todo plan ──────────────────────────────────────────────────────────
    def todos(self, items: list) -> None:
        """Render the current task checklist."""
        self._end_stream()
        if not items:
            return
        done = sum(1 for i in items if getattr(i, "status", "") == "completed")
        title = f"Plan  {done}/{len(items)}"
        if self._console is not None:
            body = Text()
            for it in items:
                glyph = getattr(it, "glyph", GLYPHS["pending"])
                content = getattr(it, "content", str(it))
                status = getattr(it, "status", "pending")
                style = self._s("ok") if status == "completed" else (
                    self._s("tool") if status == "in_progress" else self._s("muted")
                )
                body.append(f"{glyph} {content}\n", style=style)
            try:
                self._print(Panel(body, title=title, border_style=self._s("todo"), title_align="left"))
                return
            except Exception:
                pass
        print(f"[{title}]")
        for it in items:
            glyph = getattr(it, "glyph", GLYPHS["pending"])
            print(f"  {glyph} {getattr(it, 'content', it)}")

    def changes(self, stats: list) -> None:
        """Render the deterministic changed-files summary (path + ±lines)."""
        if not stats:
            return
        self._end_stream()
        shown = stats[:_MAX_CHANGE_ROWS]
        extra = len(stats) - len(shown)
        title = f"Changed files ({len(stats)})  ·  /revert to undo"
        if self._console is not None:
            body = Text()
            for path, add, dele in shown:
                body.append(f"{path}  ", style=self._bold("text"))
                body.append(f"+{add}", style=self._s("add"))
                body.append(" ")
                body.append(f"-{dele}\n", style=self._s("del"))
            if extra:
                body.append(f"… and {extra} more\n", style=self._s("muted"))
            try:
                self._print(Panel(body, title=title, border_style=self._s("todo"), title_align="left"))
                return
            except Exception:
                pass
        print(f"[{title}]")
        for path, add, dele in shown:
            print(f"  {path}  +{add} -{dele}")
        if extra:
            print(f"  … and {extra} more")

    def plan_review(self, plan: str) -> None:
        """Render the agent's proposed plan (exit_plan_mode) for approval."""
        self._end_stream()
        text = plan.strip() or "_(no plan text provided)_"
        title = f"{GLYPHS['proposal']} Proposed plan  ·  approve to unlock edits"
        if self._console is not None:
            try:
                self._print(Panel(
                    Markdown(preserve_line_breaks(text)), title=title,
                    border_style=self._s("todo"), title_align="left",
                ))
                return
            except Exception:
                pass
        print(f"[{title}]\n{text}")

    # ── final / notes ──────────────────────────────────────────────────────
    def final(self, text: str, *, turns: int = 0, tool_calls: int = 0, stopped_by: str = "") -> None:
        self._end_stream()
        meta = f"turns={turns} · tools={tool_calls}" + (f" · {stopped_by}" if stopped_by else "")
        if self._usage is not None and self._usage.total:
            meta += f" · {self._usage.summary()}"
        shown = self._content_shown
        self._content_shown = False
        # The answer already streamed/printed live this turn → don't re-render
        # the whole thing (that duplicate "Result" box is what made it look
        # like the text only appeared at the end). Just show a compact footer.
        done = f"{GLYPHS['ok']} done · {meta}"
        if shown:
            self._print(f"[{self._s('muted')}]{done}[/]" if self._console else done)
            return
        # Nothing was shown live (rare) → render the answer now.
        if self._console is not None:
            try:
                self._print(Panel(
                    Markdown(preserve_line_breaks(text) or "_(no answer)_"),
                    title=f"{GLYPHS['ok']} Result", border_style=self._s("ok"),
                    subtitle=f"[{self._s('muted')}]{meta}[/]", subtitle_align="right",
                ))
                return
            except Exception:
                pass
        print(f"\n=== Result ({meta}) ===\n{text}\n")

    def note(self, msg: str) -> None:
        self._end_stream()
        self._print(f"[{self._s('muted')}]{msg}[/]" if self._console else msg)

    def queued(self, text: str) -> None:
        """A line the user typed while the agent was working (steering)."""
        self._end_stream()
        msg = f"{GLYPHS['queued']} queued — will steer at the next step: {text}"
        self._print(f"[{self._s('accent')}]{msg}[/]" if self._console else msg)

    def error(self, msg: str) -> None:
        self._end_stream()
        self._print(
            self._styled("error:", "err", bold=True) + f" {msg}"
            if self._console else f"error: {msg}"
        )

    def llm_failure(self, msg: str, *, configuration_error: bool = False) -> None:
        """Render an LLM failure without implying that a report completed."""
        self._end_stream()
        title = "LLM configuration error" if configuration_error else "LLM call failed"
        if self._console is not None:
            try:
                self._print(Panel(
                    Text(msg), title=f"{GLYPHS['fail']} {title}",
                    border_style=self._s("err"), title_align="left",
                ))
                return
            except Exception:
                pass
        print(f"\n=== {title} ===\n{msg}\n")

    def incomplete(
        self,
        text: str,
        *,
        turns: int = 0,
        tool_calls: int = 0,
        stopped_by: str = "",
    ) -> None:
        """Render partial output without implying successful delivery."""
        self._end_stream()
        meta = f"turns={turns} · tools={tool_calls}"
        if stopped_by:
            meta += f" · {stopped_by}"
        shown = self._content_shown
        self._content_shown = False
        footer = (
            f"{GLYPHS['stopped']} run incomplete · {meta} · "
            "partial output was not saved as a final report"
        )
        if shown:
            self._print(
                f"[{self._s('warn')}]{footer}[/]" if self._console else footer
            )
            return
        if self._console is not None:
            try:
                self._print(Panel(
                    Markdown(preserve_line_breaks(text) or "_(no partial output)_"),
                    title=f"{GLYPHS['stopped']} Incomplete output",
                    border_style=self._s("warn"),
                    subtitle=f"[{self._s('muted')}]{meta}[/]",
                    subtitle_align="right",
                ))
                self._print(f"[{self._s('warn')}]{footer}[/]")
                return
            except Exception:
                pass
        print(f"\n=== Incomplete output ({meta}) ===\n{text}\n\n{footer}\n")

    # ── helpers ──────────────────────────────────────────────────────────
    @staticmethod
    def _summarize_args(name: str, args: dict) -> str:
        if name == "bash":
            # Prefer the model's plain-English intent; else a leading "# comment"
            # label (cf. Claude Code's extractBashCommentLabel); else the command.
            desc = str(args.get("description", "")).strip()
            if desc:
                return desc[:200]
            cmd = str(args.get("command", "")).strip()
            first = cmd.split("\n", 1)[0].strip()
            if first.startswith("#") and not first.startswith("#!"):
                return first.lstrip("#").strip()[:200] or cmd[:200]
            return cmd[:200]
        if name == "run_python_code":
            # Same shape as bash: lead with the first line that does something,
            # and say how much else there is rather than truncating mid-snippet.
            code = str(args.get("code", "")).strip()
            body = [ln.strip() for ln in code.splitlines() if ln.strip()]
            first = next((ln for ln in body if not ln.startswith("#")), "")
            if not first:
                return "python snippet"
            return first[:140] + (f"  · {len(body)} lines" if len(body) > 1 else "")
        if name in ("todo_write", "add_task", "update_task"):
            # The task board tools carry a list of dicts. Stringified, that is a
            # row of Python source; the count plus the first entry is what the
            # reader is actually after, and the sidebar board holds the rest.
            items = (
                args.get("todos") or args.get("tasks") or args.get("updates") or []
            )
            if not isinstance(items, list):
                return "? items"
            label = f"{len(items)} item{'' if len(items) == 1 else 's'}"
            first = items[0] if items else None
            if isinstance(first, dict):
                headline = str(
                    first.get("content")
                    or first.get("description")
                    or first.get("resolution")
                    or ""
                ).strip()
                if headline:
                    return f"{label} · {headline}"[:160]
            return label
        # Most-specific argument first. ``pattern`` precedes ``path`` because a
        # search is identified by what it looks for, not where — and because
        # ``grep_search`` takes both, so leading with ``path`` printed the
        # directory twice ("src  in src") and dropped the pattern entirely.
        # ``query`` / ``url`` are here for the same reason the web tools now
        # have their own markers: they used to fall through to the ``k=v`` tail
        # below, which truncates at 40 characters and turned every search into
        # ``query=how to configure sglang for a 35B model on a sing…``.
        for key in ("pattern", "query", "url", "file_path", "path", "name"):
            if not args.get(key):
                continue
            value = args[key]
            if isinstance(value, list):
                # ``web_fetch`` accepts one URL or a batch; a stringified Python
                # list is unreadable in a one-line row.
                extra_urls = len(value) - 1
                value = str(value[0]) if value else ""
                if extra_urls > 0:
                    return f"{value[:160]}  +{extra_urls} more"
            where = ""
            if name in ("grep_search", "glob_search") and args.get("path"):
                where = f"  in {args['path']}"
            return str(value)[:160] + where
        return ", ".join(f"{k}={str(v)[:40]}" for k, v in list(args.items())[:3])
