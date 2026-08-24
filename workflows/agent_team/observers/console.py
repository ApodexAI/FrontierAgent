"""Console observers — main-agent Rich panel + sub-agent log lines."""

from __future__ import annotations

import json
import logging
import sys

from rich.console import Console
from rich.markup import escape as _escape

from frontier_agent.core.loop_types import (
    AgentLoopResult,
    BaseObserver,
    LoopConfig,
    ToolResult,
    TurnContext,
)

logger = logging.getLogger("frontier_agent.components.agent_bus")

# Shared module-level console — thread-safe via Rich's internal lock.
# Writes to stderr so a protocol-stream deployment (which uses stdout for
# its JSONL event stream) doesn't see panel/box characters bleed into the event
# stream. width=200 avoids line-wrapping; force_terminal=True keeps ANSI
# in tee/logs.
_console = Console(file=sys.stderr, width=200, force_terminal=True)

_CONTENT_PREVIEW = 2000
_ARGS_PREVIEW = 400
_RESULT_PREVIEW = 400

_STOP_ICON: dict[str, str] = {
    "FinalizeAnswerObserver": "✅",
    "BudgetObserver":         "💰",
    "max_turns":              "⏱️",
    "no_tool_budget":         "⚠️",
}


class RichConsoleObserver(BaseObserver):
    """Prints agent-team main-agent execution to terminal in real time."""

    critical: bool = False

    async def on_loop_start(self, config: LoopConfig) -> None:
        _console.print(
            f"\n[bold magenta]{'━' * 60}[/bold magenta]"
            f"\n[bold magenta]🚀 [MAIN] Swarm task started[/bold magenta]"
            f"  task_id=[cyan]{_escape(config.task_id)}[/cyan]"
            f"  max_turns={config.max_turns}"
        )

    async def on_llm_response(self, ctx: TurnContext) -> None:
        tokens = ""
        if ctx.usage:
            inp = ctx.usage.get("input_tokens") or ctx.usage.get("prompt_tokens") or 0
            out = ctx.usage.get("output_tokens") or ctx.usage.get("completion_tokens") or 0
            tokens = f"  [dim]tokens: {inp}→{out}[/dim]"

        _console.print(
            f"\n[bold cyan]═══ [MAIN] Turn {ctx.turn}/{ctx.max_turns}[/bold cyan]"
            f"  [dim]{_escape(ctx.role_id)}[/dim]{tokens}"
        )

        text = ctx.ai_text.strip()
        if text:
            _console.print(_escape(text[:_CONTENT_PREVIEW]))
            if len(text) > _CONTENT_PREVIEW:
                _console.print(f"[dim]... ({len(text)} chars total)[/dim]")

        for tc in ctx.tool_calls or []:
            raw = json.dumps(tc.get("args", {}), ensure_ascii=False)
            preview = raw[:_ARGS_PREVIEW] + ("…" if len(raw) > _ARGS_PREVIEW else "")
            _console.print(
                f"  [yellow]🔧 {_escape(tc.get('name', '?'))}[/yellow]"
                f"  [dim]{_escape(preview)}[/dim]"
            )

    async def on_tool_result(self, ctx: TurnContext, result: ToolResult) -> None:
        preview = (
            result.result[:_RESULT_PREVIEW]
            + ("…" if len(result.result) > _RESULT_PREVIEW else "")
        )
        style = "red" if result.is_error else "green"
        prefix = "ERROR: " if result.is_error else ""
        _console.print(f"  [{style}]↳ {_escape(prefix + preview)}[/{style}]")

    async def on_loop_end(self, result: AgentLoopResult) -> None:
        icon = _STOP_ICON.get(result.stopped_by, "⏹")
        _console.print(
            f"\n[bold magenta]{icon} [MAIN] swarm done[/bold magenta]"
            f"  turns={result.turns_used}"
            f"  tool_calls={result.tool_calls_count}"
            f"  stopped_by={_escape(result.stopped_by)}"
        )


class SubAgentLoggingObserver(BaseObserver):
    """Per-turn ``logger.info`` writer for one sub-agent session."""

    critical: bool = False
    _PREVIEW_CHARS = 100

    def __init__(self, session_name: str) -> None:
        self._name = session_name

    async def on_llm_response(self, ctx: TurnContext) -> None:
        tools = [tc.get("name", "?") for tc in (ctx.tool_calls or [])]
        logger.info(
            "[sub/%s] turn %d → %s",
            self._name, ctx.turn, tools or ["<think>"],
        )

    async def on_tool_result(self, ctx: TurnContext, result: ToolResult) -> None:
        body = (result.result or "").replace("\n", " ").strip()
        if len(body) > self._PREVIEW_CHARS:
            body = body[: self._PREVIEW_CHARS] + "…"
        logger.info(
            "[sub/%s] %s %s (%dms)%s",
            self._name,
            "✗" if result.is_error else "✓",
            result.name,
            result.duration_ms or 0,
            f" → {body}" if body else "",
        )

    async def on_loop_end(self, result: AgentLoopResult) -> None:
        logger.info(
            "[sub/%s] done — stopped_by=%s turns=%d tool_calls=%d",
            self._name, result.stopped_by, result.turns_used,
            result.tool_calls_count,
        )
