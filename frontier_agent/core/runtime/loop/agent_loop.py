"""FrontierAgent adapter for AgentCore's shared agent-loop engine.

The ReAct loop itself lives in :mod:`agent_core.runtime.loop.agent_loop`. This
module only injects the product runtime decisions through
:class:`AgentLoopHooks`: sticky-session binding, the execution scope, the wall
deadline, spill-file detection, and the tool execution policies.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import agent_core.runtime.loop.agent_loop as _shared
from agent_core.runtime.loop.agent_loop import (
    AgentLoopHooks,
    PauseCheckHook,
    TurnCompleteHook,
)

from frontier_agent.core.execution_context import (
    ExecutionScope,
    chain_fallback_active,
    reset_current_execution_scope,
    set_current_execution_scope,
)
from frontier_agent.core.llm import LLMClient
from frontier_agent.core.loop_types import (
    AgentLoopResult,
    LoopConfig,
    wall_deadline_remaining_s,
)
from frontier_agent.core.messages import Message
from frontier_agent.core.runtime.loop import _bind
from frontier_agent.core.runtime.loop.model_profile import HistoryPolicy, ModelProfile
from frontier_agent.core.runtime.loop.tool_call_parser import ToolCallParser
from frontier_agent.core.runtime.loop.tool_exec import TOOL_EXECUTION_HOOKS
from frontier_agent.core.tool import Tool

__all__ = ["RUNTIME_HOOKS", "PauseCheckHook", "TurnCompleteHook", "run_agent_loop"]


def _enter_scope(
    cfg: LoopConfig, phase_id: str, metadata: dict[str, Any]
) -> tuple[ExecutionScope, Any]:
    scope = ExecutionScope(
        task_id=cfg.task_id,
        role_id=cfg.role_id,
        phase_id=phase_id,
        metadata=metadata,
    )
    return scope, set_current_execution_scope(scope)


def _body_has_spill_reference(body: str) -> bool:
    from plugins.tools._overflow import body_names_a_spill_file

    return body_names_a_spill_file(body)


def _with_recovery_handle(body: str, result: Any, turn: int, *, enabled: bool) -> str:
    """The shared recovery footer with the product spill-file detector bound."""
    return _shared._with_recovery_handle(
        body, result, turn, enabled=enabled, body_has_spill_reference=_body_has_spill_reference
    )


RUNTIME_HOOKS = AgentLoopHooks(
    # Late-bound so tests can monkeypatch ``_bind.bind_session_id``.
    bind_session=lambda llm, session_id: _bind.bind_session_id(llm, session_id),
    wall_deadline_remaining=wall_deadline_remaining_s,
    chain_fallback_active=chain_fallback_active,
    enter_scope=_enter_scope,
    exit_scope=reset_current_execution_scope,
    body_has_spill_reference=_body_has_spill_reference,
    tool_execution=TOOL_EXECUTION_HOOKS,
)


async def run_agent_loop(
    *,
    system_prompt: str,
    user_message: str,
    llm: LLMClient,
    tools: list[Tool],
    config: LoopConfig | None = None,
    observers: list[Any] | None = None,
    parser: ToolCallParser | None = None,
    model_profile: ModelProfile | None = None,
    history_policy: HistoryPolicy | None = None,
    initial_messages: list[Message] | None = None,
    on_turn_complete: TurnCompleteHook | None = None,
    pause_check: PauseCheckHook | None = None,
    scope_metadata: dict[str, Any] | None = None,
) -> AgentLoopResult:
    """Run AgentCore's ReAct loop with the FrontierAgent runtime hooks."""
    cfg = config or LoopConfig()
    if cfg.tool_result_max_chars is None and history_policy is not None:
        # AgentCore applies ``HistoryPolicy.tool_result_max_chars`` when the
        # loop config leaves it unset; FrontierAgent caps tool results in the
        # executor (``TOOL_EXECUTION_HOOKS``) instead, so opt out explicitly.
        cfg = replace(cfg, tool_result_max_chars=0)
    return await _shared.run_agent_loop(
        system_prompt=system_prompt,
        user_message=user_message,
        llm=llm,
        tools=tools,
        config=cfg,
        observers=observers,
        parser=parser,
        model_profile=model_profile,
        history_policy=history_policy,
        initial_messages=initial_messages,
        on_turn_complete=on_turn_complete,
        pause_check=pause_check,
        scope_metadata=scope_metadata,
        runtime_hooks=RUNTIME_HOOKS,
    )


def __getattr__(name: str) -> Any:
    """Read-through to the shared loop module for private helpers."""
    return getattr(_shared, name)
