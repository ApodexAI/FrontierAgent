"""SSEObserver — passive observer that forwards loop events to EventStore.

Emits AGENT_ACTION events so the SSE stream and frontend receive real-time
updates for every LLM turn and tool call inside the agent loop engine.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any

from frontier_agent.core.events import EventType
from frontier_agent.core.loop_types import ToolResult, TurnContext
from frontier_agent.core.runtime.loop.model_profile import HistoryPolicy

logger = logging.getLogger(__name__)

_SKILL_PATH_RE = re.compile(r"(?:^|/)skills/([^/]+)/SKILL\.md$")


class SSEObserver:
    """Passive observer that persists react_think / react_tool_call events.

    Designed to be registered with the agent loop engine and fire-and-forgot
    (critical=False).  All errors are swallowed by the base notify_observers
    helper — this observer must never crash the loop.
    """

    critical = False

    def __init__(
        self,
        event_store: Any,
        task_id: str,
        history_policy: HistoryPolicy | None = None,
        *,
        run_id: str = "",
        run_type: str = "",
    ) -> None:
        self._es = event_store
        self._task_id = task_id
        self._policy = history_policy or HistoryPolicy()
        self._skills_announced: set[str] = set()
        # Heavy-mode tags. K parallel main_agent runs all share the
        # same root ``task_id`` for SSE; the ``run_id`` field on every
        # emitted payload lets the frontend disambiguate which heavy
        # run an event belongs to. Empty strings mean "not in heavy
        # mode" — the keys are simply absent from the payload.
        self._run_id = run_id
        self._run_type = run_type

    def _annotate(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Add heavy-mode run_id / run_type fields when set."""
        if self._run_id:
            payload["run_id"] = self._run_id
        if self._run_type:
            payload["run_type"] = self._run_type
        return payload

    async def on_loop_start(self, config: Any) -> None:
        pass

    async def on_turn_end(self, ctx: TurnContext) -> None:
        pass

    async def on_loop_end(self, result: Any) -> None:
        pass

    async def on_llm_response(self, ctx: TurnContext) -> None:
        """Emit a ``react_think`` AGENT_ACTION event for the completed turn."""
        payload: dict[str, Any] = {
            "trace_type": "react_think",
            "agent": ctx.role_id,
            "turn": ctx.turn,
            "action": ctx.ai_text,
            "detail": ctx.ai_text[:200] if ctx.ai_text else "",
        }
        if self._policy.thinking_in_sse and ctx.thinking:
            payload["thinking"] = ctx.thinking
        # Reasoning recovered from leaked private tags (e.g.
        # ``<think_never_used_…>``) — gated on the same flag as the native
        # thinking stream so UIs that hide thinking also hide this.
        if self._policy.thinking_in_sse and ctx.leaked_reasoning:
            payload["leaked_reasoning"] = ctx.leaked_reasoning

        await self._es.append(
            self._task_id, EventType.AGENT_ACTION, self._annotate(payload),
        )

    async def on_tool_result(self, ctx: TurnContext, result: ToolResult) -> None:
        """Emit a ``react_tool_call`` AGENT_ACTION event for the tool execution.

        Also emits a one-shot ``skill_loaded`` trace event when the call is a
        successful ``read_text`` on a ``skills/<id>/SKILL.md`` path — gives the
        frontend an explicit signal for which skill the agent activated instead
        of having to pattern-match on tool args.
        """
        try:
            tool_args_str = json.dumps(result.args, ensure_ascii=False)[:2000]
        except (TypeError, ValueError):
            tool_args_str = str(result.args)[:2000]
        payload: dict[str, Any] = {
            "trace_type": "react_tool_call",
            "agent": ctx.role_id,
            "turn": ctx.turn,
            "tool_name": result.name,
            "tool_args": tool_args_str,
            "detail": result.result[:200] if result.result else "",
            "duration_ms": result.duration_ms,
            "is_error": result.is_error,
        }
        await self._es.append(
            self._task_id, EventType.AGENT_ACTION, self._annotate(payload),
        )

        await self._maybe_emit_skill_loaded(ctx, result)

    async def _maybe_emit_skill_loaded(
        self, ctx: TurnContext, result: ToolResult
    ) -> None:
        if result.name != "read_text" or result.is_error:
            return
        path = (result.args or {}).get("path")
        if not isinstance(path, str):
            return
        match = _SKILL_PATH_RE.search(path)
        if not match:
            return
        skill_id = match.group(1)
        if skill_id in self._skills_announced:
            return
        self._skills_announced.add(skill_id)

        skill_name = self._lookup_skill_name(skill_id) or skill_id
        await self._es.append(
            self._task_id,
            EventType.AGENT_ACTION,
            self._annotate({
                "trace_type": "skill_loaded",
                "agent": ctx.role_id,
                "turn": ctx.turn,
                "skill_id": skill_id,
                "skill_name": skill_name,
                "action": "skill_loaded",
                "detail": f"Loaded skill: {skill_name}",
            }),
        )

    @staticmethod
    def _lookup_skill_name(skill_id: str) -> str | None:
        """Skill loading is not part of the trimmed OSS distribution."""
        return None
