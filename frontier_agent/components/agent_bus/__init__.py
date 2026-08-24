"""AgentBus — multi-agent dispatch, communication, and spawn control.

Also serves as the back-compat entry point for ``frontier_agent.components.agent_bus``
(the former flat-file module). Re-exports the public API from submodules.
"""

from frontier_agent.components.agent_bus.agent_comm import AgentComm, DeliveryMode
from frontier_agent.components.agent_bus.bus import AgentBus
from frontier_agent.components.agent_bus.fan_in import (
    COMPLETE_STOP_REASONS,
    INCOMPLETE_STOP_REASONS,
    ORCHESTRATOR_AGENT_NAME,
    CompletionInfo,
    CompletionPolicy,
    FanInBatch,
    classify_completion,
    format_report_block,
    format_status_line,
    format_status_report_block,
    process_collected,
    session_state,
)
from frontier_agent.components.agent_bus.models import (
    CollectResult,
    DepthLimitExceeded,
    JobEntry,
    PendingSessionTask,
    SessionWaitOutcome,
    SubAgentResult,
    SubAgentRuntimeSpec,
    SubAgentSession,
    SubTask,
)
from frontier_agent.components.agent_bus.runtime import (
    adapt_default_session_result,
    adapt_default_subagent_result,
    build_default_subagent_loop_config,
    build_default_subagent_observers,
    build_session_loop_config,
    close_session_boundary_aborted,
    resolve_session_observers,
)
from frontier_agent.components.agent_bus.shared_pool import SharedArtifactPool
from frontier_agent.components.agent_bus.spawn_guard import (
    BudgetExhausted,
    SpawnDepthExceeded,
    SpawnGuard,
    SpawnReservation,
)

__all__ = [
    "COMPLETE_STOP_REASONS",
    "INCOMPLETE_STOP_REASONS",
    "ORCHESTRATOR_AGENT_NAME",
    "AgentBus",
    "AgentComm",
    "BudgetExhausted",
    "CollectResult",
    "CompletionInfo",
    "CompletionPolicy",
    "DeliveryMode",
    "DepthLimitExceeded",
    "FanInBatch",
    "JobEntry",
    "PendingSessionTask",
    "SessionWaitOutcome",
    "SharedArtifactPool",
    "SpawnDepthExceeded",
    "SpawnGuard",
    "SpawnReservation",
    "SubAgentResult",
    "SubAgentRuntimeSpec",
    "SubAgentSession",
    "SubTask",
    "adapt_default_session_result",
    "adapt_default_subagent_result",
    "build_default_subagent_loop_config",
    "build_default_subagent_observers",
    "build_session_loop_config",
    "classify_completion",
    "close_session_boundary_aborted",
    "format_report_block",
    "format_status_line",
    "format_status_report_block",
    "process_collected",
    "resolve_session_observers",
    "session_state",
]
