# pyright: reportWildcardImportFromLibrary=false
"""Product composition for the shared AgentBus implementation."""

import sys
from typing import Any

import agent_core.components.agent_bus.bus as _implementation
from agent_core.components.agent_bus.bus import *  # noqa: F403
from agent_core.components.agent_bus.bus import PauseCheckFn

from frontier_agent.core.protocols import EventSink
from frontier_agent.core.runtime.registries import services as registry


def _pause_check(task_id: str) -> PauseCheckFn:
    from frontier_agent.core.runtime.pause_check import make_task_pause_check

    return make_task_pause_check(task_id)


def _event_sink() -> EventSink | None:
    return registry.get_optional(EventSink)


def _runtime_hooks() -> Any:
    """Give sub-agents the same ``AgentLoopHooks`` the main agent gets.

    Imported lazily: this module sits below the loop package in the layer
    stack, and ``tests/test_kernel_purity.py`` enforces that.
    """
    from frontier_agent.core.runtime.loop.agent_loop import RUNTIME_HOOKS

    return RUNTIME_HOOKS


_implementation.configure_default_pause_check_factory(_pause_check)
_implementation.configure_default_event_sink_resolver(_event_sink)
_implementation.configure_default_session_activity(True)
_implementation.configure_default_runtime_hooks(_runtime_hooks)
sys.modules[__name__] = _implementation
