"""Task-scoped pause_check helper for research-mode ReAct loops."""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from typing import Any

from frontier_agent.core.errors import TaskNotFoundError
from frontier_agent.core.runtime.registries import services as registry
from frontier_agent.core.types import TaskId, TaskStatus

logger = logging.getLogger(__name__)

PauseCheckFn = Callable[[], Awaitable[bool]]

# Statuses that should stop an in-flight agent loop at the next turn.
# ``FAILED`` is omitted — a transition to FAILED usually means the
# pipeline itself set that status after the loop returned, so stopping
# on it would create a feedback cycle.
_STOP_STATUSES = {TaskStatus.SUSPENDED, TaskStatus.ABORTED}


def make_task_pause_check(task_id: str | TaskId) -> PauseCheckFn:
    """Return a ``pause_check`` closure bound to one research task.

    The closure is **safe to call from inside the kernel loop**:
    - never raises; unexpected errors log at WARNING and return False
      (i.e. keep running rather than stop on a read hiccup);
    - returns True only when the task's current status is in the
      stop set (``suspended`` or ``aborted``).
    """
    tid = str(task_id)

    async def _check() -> bool:
        # Lazy import keeps ``core/runtime`` free of a top-level
        # ``scheduling`` dep — runtime is a peer, not a downstream, of
        # ``scheduling/`` (see ``test_kernel_purity``).
        from frontier_agent.scheduling.process_manager import ProcessManager

        pm = registry.get_optional(ProcessManager)
        if pm is None:
            return False
        try:
            task = await pm.get_task(TaskId(tid))
        except TaskNotFoundError:
            # Sub-runs use synthetic ids (``<root>.<sub>`` fan-out ids,
            # AgentBus job suffixes that survived strip) which are intentionally
            # not registered with ProcessManager. Treat as "no pause signal"
            # silently rather than emitting a warning every turn.
            return False
        except Exception as exc:
            logger.warning(
                "pause_check: get_task(%s) failed: %s", tid, exc,
            )
            return False
        return getattr(task, "status", None) in _STOP_STATUSES

    return _check


def pause_check_from_state(state: dict[str, Any] | None) -> PauseCheckFn | None:
    """Pull the ``pause_check`` closure out of ``state.metadata``.

    Research / agent runners stash the closure on
    ``state["metadata"]["pause_check"]`` so every node downstream can
    forward it to ``run_agent_loop`` without re-importing
    :func:`make_task_pause_check`. SDK paths inject their own closure
    the same way. ``None`` means "no pause hook wired for this call".
    """
    if not state:
        return None
    metadata = state.get("metadata") or {}
    return metadata.get("pause_check")


__all__ = ["PauseCheckFn", "make_task_pause_check", "pause_check_from_state"]
