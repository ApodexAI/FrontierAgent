"""``apodex`` command-line entry point.

Examples::

    python -m apodex                       # interactive TUI in $PWD
    python -m apodex --cwd /path/to/repo   # interactive in a repo
    python -m apodex -p "explain src/foo.py"   # one-shot, prints, exits
    python -m apodex --model qwen/qwen3.7-max --yes "add a CLI flag"
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import logging
import os
import sys
from collections.abc import MutableMapping
from typing import TYPE_CHECKING

from apodex import __version__
from apodex.profiles import get_profile, terminal_mode_names
from apodex.render import Renderer
from apodex.session import TerminalSession
from apodex.terminal import resolve_terminal_ui
from apodex.tui.themes import CLI_THEME_NAMES

if TYPE_CHECKING:
    from apodex.config import ModelConfig


def _load_env() -> None:
    """Load a ``.env`` (keys/base-url/model) from the launch directory or an
    ancestor, the way FrontierAgent's own entry points do. ``override=False`` so
    real environment variables and CLI flags always win. Must run **before**
    any ``chdir`` so it finds the repo's ``.env`` rather than the target repo.
    """
    try:
        from dotenv import find_dotenv, load_dotenv
    except Exception:
        return
    load_dotenv(".env", override=False)
    found = find_dotenv(usecwd=True)
    if found:
        load_dotenv(found, override=False)


# (substring in an engine log message) -> clean one-line note to surface
_LOG_NOTES = (
    ("LeakedToolCallRetry", "⟳ the model wrote a tool call as text — retrying in the proper format"),
    ("leaked into <think>", "⟳ recovered a tool call from the model's thinking"),
    ("LLM reasoning runaway", "⟳ the model's reasoning ran long — recovering"),
)


class _EngineLogRouter(logging.Handler):
    """Keep the terminal UI clean. Engine code logs freely (e.g. the
    ``[LeakedToolCallRetry]`` warning); without this, Python's last-resort
    handler dumps those raw to stderr mid-UI. We write every record to a
    per-session file and surface only *recovery-class* warnings as a tidy note.
    """

    def __init__(
        self,
        renderer: object,
        file_handler: logging.Handler | None = None,
    ) -> None:
        super().__init__(level=logging.WARNING)
        self._r = renderer
        self._file = file_handler
        self.setFormatter(logging.Formatter(
            "%(asctime)s %(levelname)s %(name)s: %(message)s",
        ))

    def emit(self, record: logging.LogRecord) -> None:
        try:
            if self._file is not None:
                self._file.emit(record)
            else:
                from apodex.run_layout import run_dir

                path = run_dir(os.environ["APODEX_SESSION_ID"]) / "engine.log"
                with path.open("a", encoding="utf-8") as stream:
                    stream.write(self.format(record) + "\n")
        except Exception:
            pass
        try:
            msg = record.getMessage()
        except Exception:
            return
        for needle, note in _LOG_NOTES:
            if needle in msg:
                with contextlib.suppress(Exception):
                    self._r.note(note)  # pyright: ignore[reportAttributeAccessIssue]
                return


def _route_engine_logs(renderer: object, session_id: str) -> None:
    """Send engine logs (WARNING+) to the active run's ``engine.log`` and keep
    the console clean. Best-effort — logging setup must never break a run."""
    try:
        os.environ["APODEX_SESSION_ID"] = session_id
        root = logging.getLogger()
        root.setLevel(logging.WARNING)
        # Drop any console StreamHandlers (FileHandler is a subclass, but ours
        # lives inside the router, not on root) so engine logs don't also hit
        # stderr; our router owns WARNING+ from here on.
        root.handlers = [h for h in root.handlers
                         if not isinstance(h, logging.StreamHandler)]
        root.addHandler(_EngineLogRouter(renderer))
    except Exception:
        pass


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="frontier-agent",
        description="A terminal-native FrontierAgent workflow client.",
    )
    p.add_argument("task", nargs="?", default=None, help="task to run (optional)")
    p.add_argument(
        "--mode", default="react",
        help="workflow: react | agent_team (default: react)",
    )
    p.add_argument(
        "--resume", nargs="?", const="", default=None, metavar="SESSION_ID",
        help="resume a session by id; without an id, list saved sessions",
    )
    p.add_argument("--model", default=None, help="model id (defaults to $OPENAI_MODEL / $APODEX_MODEL)")
    p.add_argument("--cwd", default=None, help="working directory the agent operates in (default: current)")
    p.add_argument(
        "--input", action="append", default=[], metavar="PATH",
        help=(
            "attach a file or directory as a read-only task input; relative "
            "paths start at --cwd (repeatable)"
        ),
    )
    p.add_argument("--max-turns", type=int, default=None, help="max agent turns per task (default: profile's, else 50)")
    p.add_argument(
        "--max-tokens", type=int, default=None,
        help="max output tokens per LLM call (default: the profile's value)",
    )
    p.add_argument("-y", "--yes", action="store_true", help="auto-approve all tool calls (no confirmation prompts)")
    p.add_argument("-p", "--print", dest="one_shot", action="store_true", help="run TASK once, print the result, exit")
    p.add_argument("--plan", action="store_true", help="start in plan mode: investigate + propose a plan; edits locked until you approve it")
    p.add_argument(
        "--theme", default="catppuccin", choices=CLI_THEME_NAMES,
        help="color theme; mono uses line mode (default: catppuccin)",
    )
    p.add_argument("--no-color", action="store_true", help="disable colored output (same as --theme mono)")
    p.add_argument("--no-tui", action="store_true", help="use the plain line-mode UI instead of the full-screen TUI")
    p.add_argument("--docker", action="store_true", help="require the whole CLI to run inside a container")
    p.add_argument(
        "--native", action="store_true",
        help="use the workspace-local native runtime (the default on Linux)",
    )
    p.add_argument(
        "--bwrap", action="store_true",
        help="require a bubblewrap filesystem jail (Linux, explicit opt-in)",
    )
    p.add_argument("--no-sandbox", action="store_true", help="run commands directly on this machine, with no namespace (approval gate only)")
    p.add_argument("--version", action="version", version=f"FrontierAgent {__version__}")
    return p


def apply_model_overrides(
    cfg: ModelConfig,
    *,
    model: str | None = None,
    max_tokens: int | None = None,
    resumed_model: str | None = None,
) -> ModelConfig:
    """Layer CLI flags and a resumed session over the profile's LLM config.

    Mutates and returns *cfg* — the caller already owns a copy of the profile's
    config, because a session rewrites the model on ``/model``.

    ``--max-tokens`` is applied here rather than at the call site because it was
    previously parsed and then never read: a user who capped output tokens
    silently got the profile's value instead (32768 for the shipped profiles).
    An explicit ``--model`` outranks a resumed session's model, so a flag can
    still redirect ``--resume`` at a different endpoint.
    """
    if model:
        cfg.model = model
    elif resumed_model and resumed_model.strip():
        cfg.model = resumed_model.strip()
    if max_tokens is not None:
        cfg.max_tokens = max_tokens
    return cfg


def publish_model_overrides(
    cfg: ModelConfig, *, environ: MutableMapping[str, str] | None = None,
) -> None:
    """Republish the resolved model settings into the environment.

    Reaching ``ModelConfig`` is not enough on its own. The native workflows
    build their own LLM config from their profile YAML, which interpolates
    ``${OPENAI_MODEL}`` and ``${OPENAI_MAX_TOKENS}`` from the environment
    (``workflows/stateful_react_agent/profiles/tui.yaml``), and never consults
    this object — so ``--model`` and ``--max-tokens`` were both invisible to the
    agent's own calls even after being applied here. Verified against a
    recording endpoint: before this, ``--max-tokens 24`` still sent
    ``max_completion_tokens: 32768``.

    Writing unconditionally is safe because both profile fields resolve *from*
    these same variables, so with no flag in play this restates the value that
    was already there.
    """
    env = os.environ if environ is None else environ
    if cfg.model:
        env["OPENAI_MODEL"] = cfg.model
    if cfg.max_tokens:
        env["OPENAI_MAX_TOKENS"] = str(cfg.max_tokens)


def _init_readline() -> None:
    """Enable arrow-key history + emacs line editing for the REPL ``input()``,
    persisting history to ``~/.apodex_history``. Best-effort:
    ``readline`` is a Unix stdlib module and absent/inert elsewhere.
    """
    try:
        import atexit
        import glob
        import os
        import readline
    except Exception:
        return
    hist = os.path.expanduser("~/.apodex_history")
    with contextlib.suppress(Exception):
        readline.read_history_file(hist)
    readline.set_history_length(1000)
    atexit.register(lambda: _safe_write_history(readline, hist))

    # ``@path`` Tab-completion — the dominant input action for a coding agent is
    # pointing it at a file. Completes only the ``@…`` token (so normal typing is
    # untouched); degrades to nothing where readline is inert.
    def _complete(text: str, state: int) -> str | None:
        if not text.startswith("@"):
            return None
        frag = text[1:]
        hits = sorted(glob.glob(frag + "*"))[:50]
        hits = [h + ("/" if os.path.isdir(h) else "") for h in hits]
        return ("@" + hits[state]) if state < len(hits) else None

    try:
        readline.set_completer(_complete)
        readline.set_completer_delims(" \t\n")
        readline.parse_and_bind("tab: complete")
    except Exception:
        pass


def _safe_write_history(readline: object, path: str) -> None:
    with contextlib.suppress(Exception):
        readline.write_history_file(path)  # pyright: ignore[reportAttributeAccessIssue]


async def _amain(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    # ``--resume`` on its own is deliberately a local, read-only operation:
    # it should work even when no model credentials or sandbox are available.
    if args.resume == "":
        from apodex.session import list_saved_sessions

        native_workspace = os.path.abspath(args.cwd or os.getcwd())
        native_sessions = [
            os.path.join(
                native_workspace, ".apodex", "runtime", "native", "home",
                ".apodex", "sessions",
            ),
            os.path.join(
                native_workspace, ".apodex", "native", "home", ".apodex", "sessions",
            ),
        ]
        existing_native_sessions = [
            root for root in native_sessions if os.path.isdir(root)
        ]
        # This branch answers before the ``--cwd`` chdir, so the run tree has
        # to be named explicitly instead of inferred from the launch dir.
        sessions = list_saved_sessions(
            existing_native_sessions, workspace=native_workspace,
        )
        if not sessions:
            print("No saved sessions.")
            return 0
        print("Saved sessions:")
        for saved in sessions:
            print(
                f"  {saved['session_id']}  "
                f"({saved['mode']}, {saved['message_count']} messages, "
                f"{saved['modified_at']})\n"
                f"    {saved['cwd']}"
            )
        return 0

    # Load .env from the launch dir BEFORE any chdir, so OPENAI_*/APODEX_*
    # keys are available (the standalone CLI isn't bootstrapped by the app).
    # (The fully-local toolchain guarantee — incl. dropping E2B_API_KEY — is
    # owned by TerminalSession._authorize_workspace.)
    _load_env()

    # Textual's Kitty keyboard negotiation drops IME commits in iTerm2.  Set
    # the compatibility fallback before either starting the native TUI or
    # constructing the Docker environment for the default macOS path.
    from apodex.terminal import configure_terminal_keyboard
    configure_terminal_keyboard(os.environ)

    if args.cwd:
        try:
            os.chdir(args.cwd)
        except Exception as exc:
            print(f"error: cannot chdir to {args.cwd!r}: {exc}", file=sys.stderr)
            return 2
    cwd = os.getcwd()

    if args.bwrap and (args.docker or args.native or args.no_sandbox):
        print(
            "error: --bwrap cannot be combined with --docker, --native, "
            "or --no-sandbox",
            file=sys.stderr,
        )
        return 2
    if args.docker and args.native:
        print("error: --docker and --native cannot be used together", file=sys.stderr)
        return 2
    if args.max_tokens is not None and args.max_tokens < 1:
        print("error: --max-tokens must be a positive integer", file=sys.stderr)
        return 2

    # macOS (and an explicit --docker) re-executes the whole CLI inside the
    # repo image; the container is the isolation boundary there. Skipped when
    # we already are that container, which is what stops it recursing.
    from apodex.sandbox import configured_backend

    in_container = os.environ.get("APODEX_IN_CONTAINER", "").strip() == "1"
    # Empty when nothing (APODEX_SANDBOX or SANDBOX_BACKEND) names a backend.
    sandbox_override, _ = configured_backend()
    no_other_boundary = (
        not in_container
        and not args.docker
        and not args.bwrap
        and not args.no_sandbox
    )
    explicit_env_native = sandbox_override == "native" and no_other_boundary
    # Linux host installs use the workspace-local native runtime by default.
    # An explicit backend still wins, as do --docker and the marker set by our
    # own outer container: ``prepare_native_runtime`` rewrites SANDBOX_BACKEND
    # and HOME, so letting this default fire over a configured backend would
    # silently dismantle the boundary the operator asked for.
    implicit_linux_native = (
        sys.platform.startswith("linux")
        and no_other_boundary
        and not sandbox_override
    )
    if implicit_linux_native or explicit_env_native:
        args.native = True
    # ``--bwrap`` is excluded alongside the other explicit boundaries: without
    # it, asking for a host bubblewrap jail on macOS built the image and entered
    # the container first, and only then failed inside it, because bubblewrap
    # cannot work there either way. Falling through instead reaches
    # resolve_strategy, whose macOS message already names the two paths that do
    # work — and it costs no image build to say so.
    implicit_macos_docker = (
        sys.platform == "darwin"
        and not args.no_sandbox
        and not args.native
        and not args.bwrap
    )
    want_docker = args.docker or implicit_macos_docker
    if want_docker and not in_container:
        from apodex.docker import docker_available, run_in_container
        passthrough = [a for a in (argv if argv is not None else sys.argv[1:])
                       if a != "--docker"]
        docker_ok, docker_reason = docker_available()
        if args.docker or docker_ok:
            return run_in_container(passthrough, cwd=cwd)
        print(
            f"apodex: Docker is unavailable ({docker_reason}); using native mode.",
            file=sys.stderr,
        )
        args.native = True

    if args.native and not in_container:
        from apodex.docker import _session_id_for_run
        from apodex.native import prepare_native_runtime

        native_argv = argv if argv is not None else sys.argv[1:]
        native_root = prepare_native_runtime(
            cwd, _session_id_for_run(list(native_argv)),
        )
        print(
            "apodex: native mode — mutable runtime files are under "
            f"{native_root}.\n"
            "        This is not a container or OS sandbox; approved commands "
            "run with your host user permissions.",
            file=sys.stderr,
        )

    # Decide isolation once, before any tool can run, and say so out loud: a
    # user who is told "bubblewrap jail" must never get host execution instead.
    from apodex.sandbox import (
        HOST,
        SandboxUnavailable,
        Strategy,
        resolve_strategy,
        set_active_strategy,
    )
    if args.native:
        from apodex.sandbox import NATIVE
        strategy = Strategy(NATIVE, "workspace-local native runtime")
    elif args.bwrap:
        from apodex.sandbox import BWRAP
        try:
            strategy = resolve_strategy(BWRAP)
        except SandboxUnavailable as exc:
            print(f"apodex: {exc}", file=sys.stderr)
            return 2
    elif args.no_sandbox:
        strategy = Strategy(HOST, "--no-sandbox")
    else:
        try:
            strategy = resolve_strategy()
        except SandboxUnavailable as exc:
            print(f"apodex: {exc}", file=sys.stderr)
            return 2
    set_active_strategy(strategy)
    if not strategy.isolated:
        print(f"apodex: {strategy.describe()} ({strategy.reason})", file=sys.stderr)

    terminal_ui = resolve_terminal_ui(
        stdin_tty=sys.stdin.isatty(),
        stdout_tty=sys.stdout.isatty(),
        one_shot=args.one_shot,
        no_tui=args.no_tui,
        no_color=args.no_color,
        requested_theme=args.theme,
        environ=os.environ,
    )
    theme = terminal_ui.theme
    renderer = Renderer(theme=theme)
    if terminal_ui.color_warning:
        # Surfaced up front: a correct palette rendered into 8 ANSI slots looks
        # like a broken palette, and the cause is entirely outside this process.
        print(f"apodex: {terminal_ui.color_warning}", file=sys.stderr)
    interactive = terminal_ui.interactive
    use_tui = terminal_ui.use_tui
    if interactive and not use_tui:
        _init_readline()  # line-mode REPL only — the TUI has its own input box

    if args.mode not in terminal_mode_names():
        print(f"error: unknown --mode {args.mode!r}; available: {', '.join(terminal_mode_names())}", file=sys.stderr)
        return 2

    # Resume: load a prior session's mode/cwd/history/journal if requested.
    resumed_state = None
    mode, session_id = args.mode, None
    if args.resume:
        from apodex.session import load_session_state
        resumed_state = load_session_state(args.resume)
        if resumed_state is None:
            print(f"error: no saved session {args.resume!r}", file=sys.stderr)
            return 2
        mode = resumed_state.get("mode", args.mode)
        session_id = resumed_state.get("session_id", args.resume)
        # Validate BEFORE the chdir, the same order TerminalSession.switch_session
        # keeps: a rejected resume must leave the process exactly as it found
        # it. Chdir-then-bail stranded the caller in the saved session's
        # directory — invisible to the CLI, which exits anyway, but every
        # in-process caller (the test suite included) inherits it.
        if mode not in terminal_mode_names():
            print(
                f"error: saved mode {mode!r} is unavailable; start a react or "
                "agent_team session",
                file=sys.stderr,
            )
            return 2
        resumed_cwd = resumed_state.get("cwd")
        if resumed_cwd and os.path.isdir(resumed_cwd):
            os.chdir(resumed_cwd)
            cwd = resumed_cwd

    # The profile (YAML) is the source of truth for model/tools/skills/prompt;
    # .env only holds the secrets it references. --model still overrides the
    # profile's model; --max-turns overrides its turn budget.
    import dataclasses
    try:
        profile = get_profile(mode)
    except (KeyError, FileNotFoundError, ValueError) as exc:
        print(f"error: could not load profile {mode!r}: {exc}", file=sys.stderr)
        return 2
    cfg = dataclasses.replace(profile.model_config)  # copy — session mutates on /model
    apply_model_overrides(
        cfg,
        model=args.model,
        max_tokens=args.max_tokens,
        resumed_model=(
            str(resumed_state.get("model") or "") if resumed_state is not None else None
        ),
    )
    # Must precede TerminalSession: the workflow profile is rendered from the
    # environment when a task runs, not from cfg.
    publish_model_overrides(cfg)
    max_turns = args.max_turns if args.max_turns is not None else (profile.max_turns or 50)

    # Purely local BYOK validation. This must remain before TerminalSession/TUI
    # construction so a missing key or malformed endpoint fails cleanly rather
    # than surfacing during the first LLM request.
    from apodex.config import format_preflight_errors
    runtime_config = profile.runtime_config(cfg, mode=mode)
    if not runtime_config.ok:
        print(format_preflight_errors(runtime_config), file=sys.stderr)
        return 2
    for warning in runtime_config.warnings:
        print(f"warning: {warning.message}", file=sys.stderr)

    session = TerminalSession(
        cfg=cfg,
        cwd=cwd,
        renderer=renderer,
        auto_approve=args.yes,
        max_turns=max_turns,
        interactive=interactive,
        mode=mode,
        session_id=session_id,
        plan_mode=args.plan,
    )
    if args.input:
        from apodex.attachments import AttachmentError
        try:
            session.attachments.attach_many(args.input)
        except AttachmentError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
    if resumed_state is not None:
        session.restore(resumed_state)

    # Full-screen TUI path: the app owns the terminal; engine logs route to its
    # widget sink, and it runs the given task (if any) on mount.
    if use_tui:
        from apodex.tui.app import FrontierAgentApp
        app = FrontierAgentApp(
            session, resumed=resumed_state is not None,
            initial_task=args.task, theme=theme,
        )
        _route_engine_logs(app.sink, session.session_id)
        await app.run_async()
        return 0

    # Keep the TUI clean: engine logs (incl. recovery warnings like
    # ``[LeakedToolCallRetry]``) go to a per-session file; only recovery-class
    # warnings surface as tidy notes instead of raw log lines.
    _route_engine_logs(renderer, session.session_id)

    if resumed_state is not None:
        renderer.note(f"resumed session {session.session_id} ({len(session.history)} prior messages)")

    if args.one_shot:
        if not args.task:
            print("error: -p/--print requires a TASK argument", file=sys.stderr)
            return 2
        await session.run_task(args.task)
        return 0

    if args.task:
        # Run the given task once, then drop into the REPL for follow-ups.
        renderer.banner(model=cfg.model, cwd=cwd, auto_approve=session.approver.auto_approve, mode=mode)
        renderer.rule()
        await session.run_task(args.task)
        if interactive:
            await session.repl(skip_banner=True)  # banner already shown
        return 0

    await session.repl()
    return 0


def main(argv: list[str] | None = None) -> int:
    try:
        return asyncio.run(_amain(argv))
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
