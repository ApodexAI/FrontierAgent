"""Compatibility names for the shared AgentCore exception hierarchy."""

from agent_core.errors import (
    AgentCoreError as FrontierAgentError,
)
from agent_core.errors import (
    InvalidStateTransition,
    KernelError,
    LLMCallExhausted,
    LLMDeadlineExceeded,
    LLMError,
    LLMReasoningRunaway,
    LLMStreamStalled,
    PermissionDenied,
    ServiceNotRegistered,
    TaskNotFoundError,
)

__all__ = [
    "FrontierAgentError",
    "InvalidStateTransition",
    "KernelError",
    "LLMCallExhausted",
    "LLMDeadlineExceeded",
    "LLMError",
    "LLMReasoningRunaway",
    "LLMStreamStalled",
    "PermissionDenied",
    "ServiceNotRegistered",
    "TaskNotFoundError",
]
