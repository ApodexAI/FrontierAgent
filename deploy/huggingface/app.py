"""Gradio front end for the FrontierAgent ``react`` demo.

The UI is deliberately thin. It knows nothing about the agent loop, tools, or
the model protocol — it consumes :class:`~deploy.huggingface.events.DemoEvent`
values from :class:`~deploy.huggingface.adapter.FrontierAgentAdapter` and paints
them. Everything agentic happens in the FrontierAgent runtime.

Run locally::

    OPENAI_BASE_URL=… OPENAI_API_KEY=… OPENAI_MODEL=… \
      .venv/bin/python -m deploy.huggingface.app
"""

from __future__ import annotations

import contextlib
import html
import logging
import os
import signal
import sys
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import gradio as gr

from deploy.huggingface.adapter import FrontierAgentAdapter
from deploy.huggingface.config import (
    DemoConfig,
    Preflight,
    load_config,
    preflight,
    runtime_preflight,
)
from deploy.huggingface.events import (
    ACTIVITY_FINISHED,
    ACTIVITY_STARTED,
    ARTIFACT_CREATED,
    ASSISTANT_DELTA,
    QUEUED,
    RUN_CANCELLED,
    RUN_COMPLETED,
    RUN_FAILED,
    RUN_STARTED,
    TASK_BOARD_UPDATED,
    WARNING,
    DemoEvent,
)
from deploy.huggingface.security import (
    Redactor,
    demo_safe_tool_names,
    install_log_redaction,
    list_output_files,
)
from deploy.huggingface.sessions import SessionStore
from frontier_agent.components.task_board_types import (
    BOARD_TOOLS,
    RESOLUTION_MARKS,
    count_resolutions,
)

logger = logging.getLogger(__name__)

#: Keep the activity feed bounded — a long run can emit hundreds of steps and
#: re-rendering an unbounded log makes the page crawl.
_MAX_ACTIVITY_LINES = 120

_PLACEHOLDER = (
    "Ask for something the agent can research and write up, for example:\n"
    "  • Summarise what the ReAct pattern is and save it as notes.md\n"
    "  • Compare two open-source vector databases and list the trade-offs"
)

_TITLE = "FrontierAgent Demo"


def _status(state: str, detail: str = "") -> str:
    icons = {
        "idle": "○", "queued": "◔", "running": "◉",
        "completed": "●", "failed": "✗", "cancelled": "■",
    }
    line = f"### {icons.get(state, '○')} {state.capitalize()}"
    return f"{line}\n\n{detail}" if detail else line


def _header(config: DemoConfig) -> str:
    tools = demo_safe_tool_names(
        config.allowed_tools, public_mode=config.public_mode,
    )
    model = (
        f"[{config.model_id}]({config.model_url})"
        if config.model_url else config.model_id
    )
    # ``HF_MODEL_ID`` is a display label while ``OPENAI_MODEL`` is what the
    # endpoint is actually asked for. When they differ, saying so is the
    # difference between a label and a claim about which model answered.
    if config.openai_model and config.openai_model not in config.model_id:
        model = f"{model} <sub>(served as `{config.openai_model}`)</sub>"
    lines = [
        f"# {_TITLE}",
        "",
        f"**Workflow** `{config.workflow}` · **Model** {model} · "
        f"**Turn limit** {config.max_turns} · "
        f"**Time limit** {int(config.task_timeout_s)}s",
        "",
        f"Tools available to the agent: {', '.join(f'`{t}`' for t in tools)}.",
    ]
    if config.public_mode:
        lines += [
            "",
            "_This is a public demo: the agent has no shell, cannot install "
            "packages, and can only read and write inside your own session "
            "directory. Runs are executed one at a time._",
        ]
    return "\n".join(lines)


def _activity_line(item: DemoEvent) -> str | None:
    data = item.data
    detail = str(data.get("detail") or "").strip()
    if item.type == ACTIVITY_STARTED:
        if data.get("activity") in BOARD_TOOLS:
            return None
        head = f"▸ **{data.get('activity', 'tool')}** · turn {data.get('turn', '?')}"
        return f"{head}\n  `{_clip(detail, 200)}`" if detail else head
    if item.type == ACTIVITY_FINISHED:
        if data.get("activity") in BOARD_TOOLS:
            return None
        ok = "✓" if data.get("ok") else "✗"
        ms = data.get("duration_ms")
        took = f" · {int(ms) / 1000:.1f}s" if isinstance(ms, (int, float)) else ""
        head = f"{ok} **{data.get('activity', 'tool')}**{took}"
        return f"{head}\n  `{_clip(detail, 200)}`" if detail else head
    if item.type == ARTIFACT_CREATED:
        name, size = data.get("name", "?"), data.get("size", 0)
        # ``generated`` marks the harness's own fallback file. Calling that
        # "produced" would credit the agent with a deliverable it never wrote.
        if data.get("generated"):
            return f"💾 saved your answer as **{name}** ({size} bytes)"
        return f"📄 produced **{name}** ({size} bytes)"
    if item.type == WARNING:
        return f"⚠ {data.get('message') or data.get('reason', 'warning')}"
    if item.type == QUEUED:
        return "⏳ queued"
    if item.type == RUN_STARTED:
        return "🚀 run started"
    return None


def _clip(text: str, limit: int) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _render_activity(lines: list[str]) -> str:
    if not lines:
        return "_No activity yet._"
    visible = lines[-_MAX_ACTIVITY_LINES:]
    elided = len(lines) - len(visible)
    prefix = f"_…{elided} earlier step(s) hidden._\n\n" if elided else ""
    return prefix + "\n\n".join(visible)


def _render_task_board(tasks: list[dict[str, Any]]) -> str:
    """Render the structured runtime board as compact, status-aware cards."""
    if not tasks:
        return (
            '<div class="task-board-empty">'
            '<span class="task-board-empty-icon">○</span>'
            "Tasks will appear here when the agent creates its plan."
            "</div>"
        )

    counts = count_resolutions(
        str(task.get("resolution") or "open") for task in tasks
    )
    summary = (
        '<div class="task-board-summary">'
        f"<strong>{counts.resolved}/{counts.active}</strong> resolved"
        f"<span>{counts.in_progress} active</span>"
        f"<span>{counts.open} open</span>"
        "</div>"
    )
    cards: list[str] = []
    for task in tasks:
        status = str(task.get("resolution") or "open")
        if status not in RESOLUTION_MARKS:
            status = "open"
        task_id = html.escape(str(task.get("id") or "?"))
        description = html.escape(str(task.get("description") or ""))
        owners = [html.escape(str(owner)) for owner in task.get("owners") or []]
        owner_line = (
            f'<div class="task-board-owner">{", ".join(owners)}</div>'
            if owners else ""
        )
        cards.append(
            f'<div class="task-card task-{status}">'
            f'<span class="task-icon">{RESOLUTION_MARKS[status]}</span>'
            '<div class="task-copy">'
            f'<div><code>{task_id}</code> {description}</div>{owner_line}'
            "</div></div>"
        )
    return summary + '<div class="task-board-list">' + "".join(cards) + "</div>"


_TASK_BOARD_CSS = """
.task-board-empty { color: var(--body-text-color-subdued); min-height: 92px;
  display: flex; align-items: center; justify-content: center; gap: 8px;
  text-align: center; padding: 18px; }
.task-board-empty-icon { font-size: 22px; }
.task-board-summary { display: flex; align-items: baseline; gap: 10px;
  margin: 2px 0 10px; color: var(--body-text-color-subdued); font-size: 12px; }
.task-board-summary strong { color: var(--body-text-color); font-size: 16px; }
.task-board-list { display: flex; flex-direction: column; gap: 7px; }
.task-card { display: flex; gap: 9px; align-items: flex-start; border: 1px solid
  var(--border-color-primary); border-radius: 9px; padding: 9px 10px;
  background: var(--background-fill-secondary); line-height: 1.35; }
.task-icon { width: 18px; height: 18px; border-radius: 50%; flex: 0 0 18px;
  display: inline-flex; align-items: center; justify-content: center;
  font-size: 11px; margin-top: 1px; }
.task-copy { min-width: 0; overflow-wrap: anywhere; font-size: 13px; }
.task-copy code { font-size: 11px; color: var(--body-text-color-subdued); }
.task-board-owner { color: var(--body-text-color-subdued); font-size: 11px;
  margin-top: 4px; }
.task-resolved { opacity: .72; }
.task-resolved .task-icon { color: #fff; background: #16a34a; }
.task-in_progress { border-color: #3b82f6; }
.task-in_progress .task-icon { color: #fff; background: #3b82f6; }
.task-open .task-icon { color: #64748b; border: 1px solid #94a3b8; }
.task-cancelled { opacity: .55; text-decoration: line-through; }
"""


def _files(session_outputs: Path, redactor: Redactor) -> list[str]:
    """Downloadable files, excluding any whose contents carry a secret."""
    return [
        str(path)
        for path in list_output_files(session_outputs, redactor=redactor)
    ]


def build_demo(config: DemoConfig | None = None) -> gr.Blocks:
    """Build the Gradio app. Pure construction — no network, no launch."""
    config = config or load_config()
    redactor = Redactor.for_secrets(config.secrets)
    install_log_redaction(redactor)
    store = SessionStore(config.runtime_root, ttl_s=config.session_ttl_s)
    adapter = FrontierAgentAdapter(config)

    def _new_session(_previous: str | None = None) -> tuple[str, str]:
        session = store.create()
        store.sweep(keep=adapter.busy_session_ids())
        return session.session_id, f"Session `{session.short_id}…`"

    async def _run(
        prompt: str, session_id: str,
    ) -> AsyncIterator[tuple[Any, ...]]:
        """Stream one run into the UI panes plus the run/session states."""
        session = store.get_or_create(session_id)
        activity: list[str] = []
        board: list[dict[str, Any]] = []
        answer = ""
        run_id = ""

        yield (
            _status("queued", "Submitting your task…"),
            _render_task_board(board), _render_activity(activity), answer,
            _files(session.outputs, redactor), run_id, session.session_id,
        )

        async for item in adapter.run(session=session, prompt=prompt):
            run_id = item.run_id or run_id
            line = _activity_line(item)
            if line:
                activity.append(line)
            if item.type == TASK_BOARD_UPDATED:
                board = list(item.data.get("tasks") or [])

            if item.type == ASSISTANT_DELTA:
                # Append unconditionally. Gating on ``attempt == 1`` looked like
                # it avoided duplicating a retried draft, but the runtime already
                # announces a discarded attempt (handled below) — and after that
                # reset the gate was false for every attempt-2 delta, so a
                # retried answer never appeared at all.
                answer += str(item.data.get("text", ""))
                # Streaming text is the hot path: repaint the answer only.
                yield (
                    _status("running", f"{len(answer)} characters so far…"),
                    gr.skip(), _render_activity(activity), answer,
                    gr.skip(), run_id, session.session_id,
                )
                continue

            if item.type == WARNING and item.data.get("discard_stream"):
                answer = ""  # the model's draft was thrown away and retried

            state, detail = _state_of(item, answer)
            if item.is_terminal:
                # ``partial_answer`` carries whatever the agent had when the
                # endpoint failed — better than discarding it.
                final = str(
                    item.data.get("answer") or item.data.get("partial_answer") or "",
                ).strip()
                if final:
                    answer = final
            yield (
                _status(state, detail), _render_task_board(board),
                _render_activity(activity), answer, _files(session.outputs, redactor),
                run_id, session.session_id,
            )

    def _stop(run_id: str) -> str:
        if not run_id:
            return _status("idle", "Nothing is running.")
        adapter.cancel(run_id)
        return _status(
            "running",
            "Stop requested — the agent finishes its current step and then "
            "returns whatever answer it has.",
        )

    def _clear(session_id: str) -> tuple[Any, ...]:
        """Wipe this session's files and reset the panes, keeping the session."""
        session = store.get_or_create(session_id)
        cleared = store.clear(session.session_id)
        return (
            _status("idle"), _render_task_board([]), _render_activity([]), "",
            _files(cleared.outputs, redactor), "",
        )

    # Gradio 6 takes ``theme`` on ``launch()``, not on the constructor.
    with gr.Blocks(title=_TITLE) as demo:
        session_state = gr.State("")
        run_state = gr.State("")

        gr.Markdown(_header(config))
        session_label = gr.Markdown("")

        with gr.Row():
            with gr.Column(scale=3):
                prompt = gr.Textbox(
                    label="Task", lines=4, placeholder=_PLACEHOLDER,
                    max_length=config.max_prompt_chars,
                )
                with gr.Row():
                    run_button = gr.Button("Run", variant="primary")
                    stop_button = gr.Button("Stop")
                    clear_button = gr.Button("Clear")
                    new_button = gr.Button("New session")
                status = gr.Markdown(_status("idle"))
                answer = gr.Markdown(
                    label="Final answer", show_label=True, value="",
                    container=True, min_height=160,
                )
            with gr.Column(scale=2):
                gr.Markdown("### Task board")
                task_board = gr.HTML(
                    value=_render_task_board([]),
                    css_template=_TASK_BOARD_CSS,
                    container=True,
                    min_height=130,
                    max_height=300,
                    autoscroll=True,
                )
                activity = gr.Markdown(
                    label="Activity", show_label=True,
                    value=_render_activity([]), container=True,
                    min_height=320, max_height=520,
                )
                files = gr.File(
                    label="Output files", file_count="multiple",
                    interactive=False, height=160,
                )

        outputs = [
            status, task_board, activity, answer, files, run_state, session_state,
        ]
        # gradio synthesises its event methods (.click / .submit / .load) at
        # class creation from each component's ``EVENTS`` list and
        # ``BLOCKS_EVENTS`` (see gradio/blocks_events.py), so they are not
        # statically declared on Button / Textbox / Blocks. gradio ships no
        # gradio/components/__init__.pyi, so the re-export in __init__.py binds
        # to button.py rather than button.pyi and the checker cannot see them.
        run_event = run_button.click(  # pyright: ignore[reportAttributeAccessIssue]
            _run, inputs=[prompt, session_state], outputs=outputs,
        )
        prompt.submit(  # pyright: ignore[reportAttributeAccessIssue]
            _run, inputs=[prompt, session_state], outputs=outputs,
        )
        # Deliberately NOT ``cancels=[run_event]``: killing the generator would
        # discard the partial answer. The adapter's cooperative stop lets the
        # agent land at its next turn boundary and still report what it found.
        stop_button.click(  # pyright: ignore[reportAttributeAccessIssue]
            _stop, inputs=[run_state], outputs=[status],
        )
        clear_button.click(  # pyright: ignore[reportAttributeAccessIssue]
            _clear, inputs=[session_state],
            outputs=[status, task_board, activity, answer, files, run_state],
        )
        new_button.click(  # pyright: ignore[reportAttributeAccessIssue]
            _new_session, inputs=[session_state],
            outputs=[session_state, session_label],
        ).then(
            _clear, inputs=[session_state],
            outputs=[status, task_board, activity, answer, files, run_state],
        )
        demo.load(  # pyright: ignore[reportAttributeAccessIssue]
            _new_session, outputs=[session_state, session_label],
        )

        _ = run_event  # kept for readability of the comment above

    # Gradio's own queue is the admission gate in front of the adapter's
    # single runner slot; the adapter still rejects anything past the bound.
    demo.queue(max_size=config.queue_size + 4, default_concurrency_limit=None)
    return demo


def _state_of(item: DemoEvent, answer: str) -> tuple[str, str]:
    data = item.data
    if item.type == QUEUED:
        return "queued", str(data.get("message") or "Waiting for a free slot…")
    if item.type == RUN_STARTED:
        return "running", "The agent is working…"
    if item.type == RUN_COMPLETED:
        return "completed", (
            f"{data.get('turns', 0)} turn(s), {data.get('tool_calls', 0)} tool "
            f"call(s), {_stopped_by(str(data.get('stopped_by') or ''))}."
        )
    if item.type == RUN_CANCELLED:
        return "cancelled", "Stopped. Any partial answer is shown above."
    if item.type == RUN_FAILED:
        return "failed", str(data.get("message") or data.get("reason") or "Run failed.")
    return ("running", "The agent is working…") if not answer else (
        "running", f"{len(answer)} characters so far…"
    )


#: How the loop's ``stopped_by`` reads to a visitor. Every one of these means
#: the run *landed* (the terminal event was ``run_completed``), so the wording
#: has to explain a bounded answer rather than sound like a failure. Unlisted
#: reasons fall through to the raw slug — new engine stop reasons appear
#: verbatim rather than being silently described as something they are not.
_STOPPED_BY_TEXT: dict[str, str] = {
    "": "finished with an answer",
    "finalize_answer": "finished with an answer",
    "max_turns": "stopped at the turn limit and answered with what it had",
    "wall_deadline": "stopped at the time limit and answered with what it had",
    "context_limit_reached": "ran out of context and answered with what it had",
    "budget_exhausted": "ran out of its token budget and answered with what it had",
    "repeated_tool_calls": "was stopped for repeating the same tool call",
    "max_attempts": "was stopped after too many re-planned steps",
    "no_tool": "stopped because it answered without using a tool",
}


def _stopped_by(reason: str) -> str:
    described = _STOPPED_BY_TEXT.get(reason)
    return described if described else f"stopped by `{reason}`"


def _install_signal_handlers() -> None:
    """Turn SIGTERM into the shutdown path Gradio already handles cleanly.

    A container stop sends SIGTERM. Gradio's ``block_thread`` loop catches
    ``KeyboardInterrupt`` and closes the server, so re-raising it here reuses
    the library's own tested teardown instead of a bespoke one.
    """
    def _handle(signum: int, _frame: Any) -> None:
        logger.info("received %s — shutting down", signal.Signals(signum).name)
        raise KeyboardInterrupt

    for sig in (signal.SIGTERM, signal.SIGINT):
        with contextlib.suppress(ValueError, OSError):
            signal.signal(sig, _handle)


def main(argv: list[str] | None = None) -> int:
    """Preflight the environment, then serve on 0.0.0.0:7860."""
    logging.basicConfig(
        level=os.environ.get("DEMO_LOG_LEVEL", "WARNING").upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    config = load_config()
    install_log_redaction(Redactor.for_secrets(config.secrets))

    checks = preflight(config)
    runtime_checks = runtime_preflight(config)
    checks = Preflight(
        errors=checks.errors + runtime_checks.errors,
        warnings=checks.warnings + runtime_checks.warnings,
    )
    if checks.warnings:
        print(checks.format(), file=sys.stderr)
    if not checks.ok:
        print(
            "\nRefusing to start. Fix the Space Variables/Secrets above.\n"
            "See deploy/huggingface/README.md for the full list.",
            file=sys.stderr,
        )
        return 2

    summary = config.public_summary()
    print(
        f"{_TITLE}: workflow={summary['workflow']} model={summary['model']} "
        f"served_model={summary['served_model']} endpoint={summary['endpoint']} "
        f"runtime_root={config.runtime_root}",
        file=sys.stderr,
    )

    demo = build_demo(config)
    _install_signal_handlers()
    store_root = config.runtime_root / "sessions"
    try:
        demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", "7860")),
            theme=gr.themes.Soft(),  # pyright: ignore[reportPrivateImportUsage]
            # Downloads are served only from session directories, whose ids are
            # cryptographically random and so unguessable by another visitor.
            allowed_paths=[str(store_root)],
            # Belt and braces on top of ``allowed_paths``: never serve the
            # service's own tree or a stray dotfile, whatever else is allowed.
            blocked_paths=["/app", "/etc", str(Path.home() / ".cache")],
            mcp_server=False,
            quiet=True,
        )
    except KeyboardInterrupt:
        print("shutting down", file=sys.stderr)
    finally:
        with contextlib.suppress(Exception):
            demo.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
