"""No-op replacements for the optional SDK protocol/telemetry layer.

A serve or CLI deployment can bolt an external protocol + telemetry layer onto
the shipped workflows: a stdout event stream, per-worker trace files, and a
cross-agent usage aggregator. It plugs in through ``metadata`` — its observers
arrive in ``sdk_extra_observers``, its emitter and aggregator in
``sdk_protocol_emitter`` / ``sdk_protocol_usage_aggregator`` — so nothing in
this repository imports it.

That layer is NOT part of this repository. Every hook below degrades to a
no-op, which is what lets the workflows run unchanged without it: a call that
would have installed a trace observer gets ``None`` back and skips the block.

Comments elsewhere therefore describe its pieces by role — "the protocol
stream observer", "a worker-trace observer" — rather than by class name,
because those classes live in the external layer and cannot be looked up here.
"""

from __future__ import annotations

from typing import Any


class ReporterDeltaEmitter:
    """Keep reporter call sites uniform when no protocol stream is installed."""

    def __init__(self, emitter: Any | None, **_: Any) -> None:
        self.emitter = emitter

    def start(self, *, max_turns: int = 0) -> None:
        return None

    def finish(self, **_: Any) -> None:
        return None

    def reasoning(self, text: str) -> None:
        return None

    def output(self, text: str) -> None:
        return None

    def stream_output(self, text: str) -> None:
        return None


def record_reporter_usage(*_: Any, **__: Any) -> bool:
    return False


def record_loop_usage(*_: Any, **__: Any) -> bool:
    return False


def record_language_detect_usage(*_: Any, **__: Any) -> bool:
    return False


def find_trace_observer(_: Any) -> None:
    return None


def make_subagent_trace_observer(**_: Any) -> None:
    return None


def start_heartbeat(*_: Any, **__: Any) -> None:
    return None


def stop_heartbeat(_: Any) -> None:
    return None
