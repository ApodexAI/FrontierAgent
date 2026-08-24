"""Exception hierarchy for FrontierAgent."""

from __future__ import annotations

from typing import Any


class FrontierAgentError(Exception):
    """Base exception for all FrontierAgent errors."""


# ── Kernel errors ───────────────────────────────────────────────────────────


class KernelError(FrontierAgentError):
    """Errors originating from the OS kernel layer."""


class TaskNotFoundError(KernelError):
    def __init__(self, task_id: str) -> None:
        super().__init__(f"Task not found: {task_id}")
        self.task_id = task_id


class InvalidStateTransition(KernelError):
    def __init__(self, task_id: str, current: str, target: str) -> None:
        super().__init__(f"Invalid transition for {task_id}: {current} → {target}")


class ServiceNotRegistered(KernelError):
    def __init__(self, service_type: type) -> None:
        super().__init__(f"Service not registered: {service_type.__name__}")


class PermissionDenied(KernelError):
    def __init__(self, role: str, tool: str) -> None:
        super().__init__(f"Role '{role}' has no permission for tool '{tool}'")

# LLM request errors

class LLMError(FrontierAgentError):
    """Errors from the LLM/provider layer."""


class LLMReasoningRunaway(LLMError):
    """A live stream spent its semantic budget on reasoning-only output.

    Unlike :class:`LLMStreamStalled`, the provider is healthy and actively
    emitting chunks. The failure is semantic: no non-whitespace visible text
    or tool-call delta appeared before the configured time/token guard fired.

    ``partial_response`` is intentionally carried separately from provider
    usage. Early stream cancellation often happens before the terminal usage
    chunk arrives, so its estimated reasoning tokens must never be presented
    as authoritative billing data.
    """

    def __init__(
        self,
        *,
        elapsed_s: float,
        estimated_tokens: int,
        trigger: str,
        partial_response: Any,
    ) -> None:
        self.elapsed_s = float(elapsed_s)
        self.estimated_tokens = int(estimated_tokens)
        self.trigger = trigger
        self.partial_response = partial_response
        super().__init__(
            "reasoning-only stream exceeded "
            f"{trigger} guard (elapsed={self.elapsed_s:.1f}s, "
            f"estimated_tokens={self.estimated_tokens})",
        )


class LLMStreamStalled(LLMError, TimeoutError):
    """A streaming LLM call went silent mid-flight.

    Subclasses ``asyncio.TimeoutError`` so every existing transient-
    timeout handler (retry/backoff in ``call_llm``, chain wrappers,
    classification) treats it identically without changes; carried
    fields make the distinct failure mode visible in logs and traces.
    """

    def __init__(
        self, stall_s: float, chunks_seen: int, elapsed_s: float,
    ) -> None:
        self.stall_s = stall_s
        self.chunks_seen = chunks_seen
        self.elapsed_s = elapsed_s
        super().__init__(
            f"stream stalled: no chunks for {stall_s:.0f}s "
            f"(chunks_seen={chunks_seen}, elapsed={elapsed_s:.0f}s)",
        )

class LLMCallExhausted(LLMError, RuntimeError):
    """Raised by ``call_llm`` when retries are exhausted or the error is
    structurally unrecoverable (4xx without proxy-wrap, or a chain-aware
    fallback signal like ``model_not_found``).

    Wraps the last exception encountered so the caller (typically
    ``run_agent_loop``) can surface it to a chain wrapper for leg
    rotation. Carries ``last_exc`` separately because ``raise from`` is
    too opaque for chain-aware classification — ``provider_chain`` calls
    ``classify_error(last_exc)`` directly.
    """

    def __init__(self, last_exc: BaseException, reason: str) -> None:
        self.last_exc = last_exc
        self.reason = reason
        super().__init__(f"call_llm {reason}: {last_exc!r}")
