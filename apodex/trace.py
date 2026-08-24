"""TraceObserver — append a local JSONL trace of everything the agent does.

One line per event (LLM turn / tool call+result / final), written with flush
so a crash still leaves a valid partial trace. Gives the OKR's "all shell
commands and file operations enter a local trace" + a `/log` to review and
diagnose flaky runs — as a plug-in observer, no engine changes.
"""

from __future__ import annotations

import json
from typing import Any

from frontier_agent.core.loop_types import (
    AgentLoopResult,
    BaseObserver,
    Intervention,
    ToolResult,
    TurnContext,
)


def default_trace_path(session_id: str) -> str:
    """Return ``<cwd>/.apodex/runs/<session-id>/trace.jsonl``."""
    from apodex.run_layout import run_dir

    return str(run_dir(session_id) / "trace.jsonl")


class TraceObserver(BaseObserver):
    """Awaited (critical) so trace lines stay ordered; never returns an
    intervention. Append-only JSONL; failures are swallowed (a broken trace
    must not break a run)."""

    critical = True

    def __init__(self, path: str, *, mode: str = "", cwd: str = "") -> None:
        self.path = path
        self._write({"t": "start", "mode": mode, "cwd": cwd})

    def _write(self, rec: dict[str, Any]) -> None:
        try:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
                f.flush()
        except Exception:
            pass

    async def on_llm_response(self, ctx: TurnContext) -> Intervention | None:
        self._write({
            "t": "llm", "turn": ctx.turn,
            "text": ctx.ai_text or "",
            "tool_calls": list(ctx.tool_calls or []),
        })
        return None

    async def on_tool_result(
        self, ctx: TurnContext, result: ToolResult,
    ) -> ToolResult | None:
        self._write({
            "t": "tool", "turn": ctx.turn, "name": result.name,
            "args": dict(result.args or {}),
            "ms": result.duration_ms, "is_error": result.is_error,
            "result": result.result or "",
        })
        return None

    async def on_loop_end(self, result: AgentLoopResult) -> None:
        self._write({
            "t": "end", "stopped_by": result.stopped_by,
            "turns": result.turns_used, "tool_calls": result.tool_calls_count,
        })
        return None
