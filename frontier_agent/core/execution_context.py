"""Shared execution context carrier for phase, LLM, and tool calls.

Execution metadata is stored durably in pipeline state under the
``execution_context`` key and exposed at runtime via a ContextVar so
LLM/tool middleware can read it without changing every call site.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from typing import Any

from frontier_agent.core.types import new_prompt_id, new_session_id, new_step_id


@dataclass
class ExecutionScope:
    """Runtime execution scope for the current phase."""

    task_id: str = ""
    phase_id: str = ""
    role_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


_CURRENT_SCOPE: ContextVar[ExecutionScope | None] = ContextVar(
    "frontier_agent_execution_scope", default=None
)

# Per-tool-call contextvar — set inside each parallel ``_run_one`` task
# in ``tool_exec.execute_tools``. ``asyncio.gather`` gives each task its
# own Context copy, so concurrent tools see distinct values. Tools fired
# from inside the loop (delegate_subtask / assign_task) read this to
# stamp ``spawn_context.spawned_by_tool_call_id`` on the new sub-agent.
_CURRENT_TOOL_CALL_ID: ContextVar[str] = ContextVar(
    "frontier_agent_current_tool_call_id", default=""
)

# Seconds the CURRENT tool call may run before ``execute_tools``' outer
# ``asyncio.wait_for`` cancels it. Set per ``_run_one`` task, so a tool that
# also enforces its own deadline can read the loop's configured budget instead
# of a module constant and fail with its own diagnosis just inside the outer
# wait. ``None`` means "no loop budget in scope" (a tool invoked directly by a
# script or a test), and the tool keeps its own default.
_CURRENT_TOOL_BUDGET: ContextVar[float | None] = ContextVar(
    "frontier_agent_current_tool_budget", default=None
)

# Whether the current async context runs *under* an outer provider-chain
# runner (a workflow's provider-chain wrapper) that will catch an exception
# escaping ``run_agent_loop`` and rotate to the next leg. Set narrowly
# around the chain's ``attempt_fn``
# invocation, so it is True exactly while the wrapped loop runs and False
# again by the time control returns to the chain's own except handler.
_CHAIN_FALLBACK_ACTIVE: ContextVar[bool] = ContextVar(
    "frontier_agent_chain_fallback_active", default=False
)


def normalize_execution_context(value: Any) -> dict[str, Any]:
    """Return a mutable execution-context dict."""
    if isinstance(value, dict):
        return dict(value)
    return {}


def build_execution_scope(
    *,
    task_id: str,
    phase_id: str,
    role_id: str,
    state: dict[str, Any] | None = None,
) -> ExecutionScope:
    """Build a scope from task/phase identity plus state metadata."""
    metadata = normalize_execution_context((state or {}).get("execution_context"))
    metadata.setdefault("agent_id", role_id)
    return ExecutionScope(
        task_id=task_id,
        phase_id=phase_id,
        role_id=role_id,
        metadata=metadata,
    )


def set_current_execution_scope(scope: ExecutionScope) -> Token:
    """Set the current execution scope for this async context."""
    return _CURRENT_SCOPE.set(scope)


def get_current_execution_scope() -> ExecutionScope | None:
    """Return the current execution scope if one is active."""
    return _CURRENT_SCOPE.get()


def reset_current_execution_scope(token: Token) -> None:
    """Restore the previous execution scope."""
    _CURRENT_SCOPE.reset(token)


def set_current_tool_call_id(tool_call_id: str) -> Token:
    """Stash the active tool_call_id on this asyncio Task's context.

    ``asyncio.gather`` runs each coroutine as its own Task with a copy of
    the current Context, so each parallel tool sees its own id.
    """
    return _CURRENT_TOOL_CALL_ID.set(tool_call_id)


def get_current_tool_call_id() -> str:
    """Return the active tool_call_id, or ``''`` outside tool execution."""
    return _CURRENT_TOOL_CALL_ID.get()


def reset_current_tool_call_id(token: Token) -> None:
    """Restore the prior tool_call_id contextvar value."""
    _CURRENT_TOOL_CALL_ID.reset(token)


def set_current_tool_budget(seconds: float | None) -> Token:
    """Publish the wall-clock budget for the tool call running in this Task.

    Same per-Task isolation as :func:`set_current_tool_call_id`: parallel tool
    calls each get their own value.
    """
    return _CURRENT_TOOL_BUDGET.set(seconds)


def get_current_tool_budget() -> float | None:
    """Seconds the active tool call may run, or ``None`` outside the loop.

    A tool that enforces its own internal deadline should prefer this over a
    module constant, and must stay at or under it — overshooting only trades
    the tool's own structured error for the loop's bare "timed out" cancel.
    """
    return _CURRENT_TOOL_BUDGET.get()


def reset_current_tool_budget(token: Token) -> None:
    """Restore the prior tool-budget contextvar value."""
    _CURRENT_TOOL_BUDGET.reset(token)


def chain_fallback_active() -> bool:
    """Whether the current async context runs under an outer provider-chain
    runner that will catch a surfaced exception and rotate to the next leg.

    ``run_agent_loop`` reads this to decide its turn-1 exhaustion policy:
    when ``True`` it re-raises so the outer chain can advance; when
    ``False`` (benchmark single-provider, or a caller whose own chain
    rotation already finished *inside* ``call_llm``) it degrades gracefully
    to an ``llm_error`` stop instead of crashing the run.
    """
    return _CHAIN_FALLBACK_ACTIVE.get()


@contextmanager
def chain_fallback_scope() -> Iterator[None]:
    """Mark the current async context as running under an outer chain runner.

    Nesting-safe via token reset, so the L3 recursion in ``run_with_chain``
    can re-enter without clobbering the outer reset.
    """
    token = _CHAIN_FALLBACK_ACTIVE.set(True)
    try:
        yield
    finally:
        _CHAIN_FALLBACK_ACTIVE.reset(token)


def ensure_trace_metadata(
    metadata: dict[str, Any],
    *,
    default_step_id: str | None = None,
    refresh_prompt_id: bool = False,
) -> dict[str, Any]:
    """Ensure trace-chain identifiers exist in execution metadata."""
    metadata.setdefault("session_id", str(new_session_id()))
    if default_step_id:
        metadata.setdefault("step_id", default_step_id)
    else:
        metadata.setdefault("step_id", str(new_step_id()))
    if refresh_prompt_id or not metadata.get("prompt_id"):
        metadata["prompt_id"] = str(new_prompt_id())
    return metadata
