"""Immutable kernel event model."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from frontier_agent.core.types import EventId, PromptId, SessionId, StepId, TaskId

# SDK Agent Protocol namespace prefix — must stay in sync with
# the protocol layer's own event namespace (not imported from there: that
# layer sits above frontier_agent in the stack).
_SDK_EVENT_NS: str = "response.swarm"


class KernelEvent(BaseModel):
    """An immutable event in the OS event log.

    Every action in FrontierAgent produces an event. Events are append-only
    and support replay for debugging and audit.

    ``event_type`` is stored as a plain string. Callers can pass any
    ``str``-Enum member (``EventType``, or any domain-specific event enum a
    caller defines) and pydantic will coerce it to its underlying
    string value, so the kernel does not need to know about every
    domain enum that writes to the event log.
    """

    model_config = ConfigDict(frozen=True)

    id: EventId = Field(default=EventId(""))
    task_id: TaskId
    session_id: SessionId | None = None
    prompt_id: PromptId | None = None
    step_id: StepId | None = None
    event_type: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    payload: dict[str, Any] = Field(default_factory=dict)
    from_agent: str | None = None
    to_agent: str | None = None
    message_type: str | None = None
    correlation_id: str | None = None
    retry_count: int = 0
    degrade_from: str | None = None
    degrade_to: str | None = None

    # Internal events not shown in the frontend activity feed
    _SKIP_EVENTS = {"task_created", "task_status_changed", "tool_called", "tool_result"}

    # Mapping from kernel/domain event strings to frontend SSEEventType
    _TYPE_MAP: dict[str, str] = {
        "phase_transition": "phase_change",
        "agent_action": "agent_action",
        "agent_tool_call": "agent_action",
        "agent_message": "agent_action",
        "error": "error",
        "report_generated": "completed",
    }

    # Protocol envelope event types — emitted by the optional external
    # protocol observer (see ``workflows/_shared/sdk_shim``) and bridged into
    # the API by whatever emitter that deployment wires up. They carry the
    # full envelope in ``payload``; SSE surfaces them verbatim so the
    # frontend can drive the DAG / log view directly off the protocol
    # contract without an intermediate translation table.
    #
    # All events now share the ``response.swarm.*`` namespace, so the
    # prefix check below is sufficient — the legacy bare-type set is
    # empty (kept as a frozenset for forward compat if a non-namespaced
    # type ever shows up again).
    _SDK_PROTOCOL_RESERVED: frozenset[str] = frozenset()

    def _is_sdk_protocol_event(self, evt_value: str) -> bool:
        if evt_value.startswith(_SDK_EVENT_NS + "."):
            return True
        return evt_value in self._SDK_PROTOCOL_RESERVED

    def to_sse_event(self) -> dict[str, Any] | None:
        """Convert to SSE-friendly dict matching frontend event interface.

        Frontend expects: { type: string, timestamp: string, data: {...} }
        Returns None for internal events that should not be streamed.
        """
        evt_value = str(self.event_type)

        # ── SDK Agent Protocol v1 envelope — pass through verbatim ─────
        if self._is_sdk_protocol_event(evt_value):
            data = dict(self.payload)
            data.setdefault("source_task_id", str(self.task_id))
            return {
                "type": evt_value,
                "timestamp": self.timestamp.isoformat(),
                "data": data,
            }

        if evt_value == "task_status_changed" and self.payload.get("new_status") == "suspended":
            return {
                "type": "suspended",
                "timestamp": self.timestamp.isoformat(),
                "data": {
                    "task_id": str(self.task_id),
                    "summary": self.payload.get("message", "Task paused"),
                    "status": "suspended",
                    "source_task_id": self.payload.get("source_task_id", str(self.task_id)),
                },
            }
        if evt_value == "task_status_changed" and self.payload.get("new_status") == "aborted":
            return {
                "type": "aborted",
                "timestamp": self.timestamp.isoformat(),
                "data": {
                    "task_id": str(self.task_id),
                    "summary": self.payload.get("message", "Task aborted"),
                    "status": "aborted",
                    "source_task_id": self.payload.get("source_task_id", str(self.task_id)),
                },
            }

        # Skip internal lifecycle events
        if evt_value in self._SKIP_EVENTS:
            return None

        sse_type = self._TYPE_MAP.get(evt_value, "agent_action")

        # Build data according to the expected frontend interface
        source_task_id = self.payload.get("source_task_id", str(self.task_id))

        if sse_type == "phase_change":
            data = {
                "phase": self.payload.get("phase", ""),
                "message": self.payload.get("message", f"Entering {self.payload.get('phase', 'unknown')} phase"),
                "source_task_id": source_task_id,
            }
        elif sse_type == "error":
            data = {
                "message": self.payload.get("error", str(self.payload)),
                "source_task_id": source_task_id,
            }
        elif sse_type == "completed":
            data = {
                "task_id": str(self.task_id),
                "summary": self.payload.get("summary", "Research completed"),
                "status": self.payload.get("status", "completed"),
                "source_task_id": source_task_id,
            }
        elif evt_value == "routing_decision":
            data = {
                "agent": self.payload.get("agent", "system"),
                "action": "routing_decision",
                "detail": self.payload.get("hints", {}).get("reason", "routing_decision"),
                "initial_macro": self.payload.get("initial_macro"),
                "budget": self.payload.get("budget", {}),
                "escalation_policy": self.payload.get("escalation_policy", {}),
                "hints": self.payload.get("hints", {}),
                "features": self.payload.get("features", {}),
                "source_task_id": source_task_id,
            }
        else:
            # Special frontend event types based on trace_type
            trace_type = self.payload.get("trace_type", "")
            # Whitelist: events whose payload should be preserved verbatim
            # (top-level ``type`` mirrors ``trace_type``). Without this, the
            # generic agent_action branch below collapses the payload into a
            # human-readable detail and drops the structured fields.
            #
            # Fan-out phase entries: the run phase emits
            # ``dag_finalize_start`` + ``heavy_converge_start``; the
            # reporter emits user-safe progress events.
            if trace_type in (
                # ReAct + verification + skill traces.
                "verification_step", "verification_complete",
                "react_think", "react_tool_call", "skill_loaded",
                # swarm_heavy: user-safe phase boundary + reporter status events.
                # ``dag.phase_started`` / ``dag.phase_cancelled`` are the
                # pipeline-agnostic phase brackets;
                # ``dag_finalize_*`` brackets the DAG
                # finalize sub-phase.
                "dag.phase_started", "dag.phase_cancelled",
                "dag.phase_timeout",
                "dag_finalize_start",
                "dag_finalize.progress", "heavy_converge_start",
                "verify.started", "verify.progress", "verify.done",
                "verify.degraded",
                "outline.started", "outline.progress", "outline.submitted",
                "report.started", "report.progress", "report.submitted",
                "report.degraded",
                "report.citations.started", "report.citations.refining",
                "report.citations.ready", "stt.progress",
            ):
                trace_data = dict(self.payload)
                trace_data.setdefault("source_task_id", source_task_id)
                return {
                    "type": trace_type,
                    "timestamp": self.timestamp.isoformat(),
                    "data": trace_data,
                }

            # agent_action — build a human-readable detail
            agent = self.payload.get("agent", self.from_agent or self.payload.get("from_agent", "system"))
            action = self.payload.get("action", evt_value)
            # Prefer specific detail fields over raw payload dump
            if "detail" in self.payload:
                detail = self.payload["detail"]
            elif self.payload.get("tool_name"):
                detail = f"Called {self.payload['tool_name']}"
                if self.payload.get("input_data"):
                    detail += f": {self.payload['input_data'][:100]}"
            elif self.message_type or self.payload.get("message_type"):
                from_a = self.from_agent or self.payload.get("from_agent", "?")
                to_a = self.to_agent or self.payload.get("to_agent", "?")
                msg_type = self.message_type or self.payload.get("message_type", evt_value)
                detail = f"{from_a} → {to_a}: {msg_type}"
            elif "output_preview" in self.payload:
                detail = self.payload["output_preview"][:2000]
            else:
                detail = action
            data = {
                "agent": agent,
                "action": action,
                "detail": detail,
                "source_task_id": source_task_id,
            }

        return {
            "type": sse_type,
            "timestamp": self.timestamp.isoformat(),
            "data": data,
        }
