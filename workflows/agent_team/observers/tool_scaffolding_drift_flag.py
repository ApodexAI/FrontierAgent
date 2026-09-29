"""ToolScaffoldingDriftFlag — logging-only observer that flags when the
cumulative volume of tool-call scaffolding (argument JSON) an agent has
produced grows to dominate the real, actual task content it has read.

Real, added 2026-09-08. This is a real, distinct, THEORETICAL mechanism --
like PartialReadContaminationFlag, this targets a hypothesized cause that
has NOT been directly confirmed as the actual mechanism behind the
original doc2 fabrication. Nobody has measured scaffolding volume in that
real trace; this is a real, plausible hypothesis worth instrumenting, not
a demonstrated mechanism, and should be read that way until it fires on a
real, live case and correlates with a real fabrication/self-consistency
event.

This observer does NOT change behavior -- matching the established,
deliberate design of the other real observers built this session: it
only logs a flagged anomaly.
"""

from __future__ import annotations

import json
import logging

from frontier_agent.core.loop_types import BaseObserver, ToolResult, TurnContext

logger = logging.getLogger(__name__)

__all__ = ["ToolScaffoldingDriftFlag"]


class ToolScaffoldingDriftFlag(BaseObserver):
    """Real, logging-only: tracks cumulative tool-call argument bytes
    (scaffolding proxy) against cumulative real read_file content bytes
    (task-content proxy) per agent; flags when the ratio crosses
    ``ratio_threshold`` and at least ``min_scaffolding_bytes`` of real
    scaffolding has genuinely accumulated (avoiding a false flag on a
    single small early call)."""

    def __init__(
        self, *, ratio_threshold: float = 5.0, min_scaffolding_bytes: int = 2000,
    ) -> None:
        self.ratio_threshold = max(1.0, float(ratio_threshold))
        self.min_scaffolding_bytes = max(0, int(min_scaffolding_bytes))
        self._scaffolding_bytes: dict[str, int] = {}
        self._content_bytes: dict[str, int] = {}
        self._flagged: set[str] = set()

    async def on_tool_call(
        self, ctx: TurnContext, tool_call: dict,
    ):
        args = tool_call.get("args") or {}
        try:
            payload = json.dumps(args, default=str)
        except (TypeError, ValueError):
            payload = repr(args)
        self._scaffolding_bytes[ctx.task_id] = (
            self._scaffolding_bytes.get(ctx.task_id, 0) + len(payload)
        )
        self._check(ctx)
        return None

    async def on_tool_result(
        self, ctx: TurnContext, result: ToolResult,
    ) -> ToolResult | None:
        if result.name == "read_file" and not result.is_error:
            body = result.result if isinstance(result.result, str) else str(result.result)
            self._content_bytes[ctx.task_id] = (
                self._content_bytes.get(ctx.task_id, 0) + len(body)
            )
        return None

    def _check(self, ctx: TurnContext) -> None:
        if ctx.task_id in self._flagged:
            return
        scaffolding = self._scaffolding_bytes.get(ctx.task_id, 0)
        content = self._content_bytes.get(ctx.task_id, 0)
        if scaffolding < self.min_scaffolding_bytes:
            return
        # content == 0 with real, accumulated scaffolding is itself a real,
        # maximal-ratio case (all scaffolding, no real task content read
        # yet) -- treat it as exceeding the threshold rather than dividing
        # by zero.
        ratio = (scaffolding / content) if content > 0 else float("inf")
        if ratio < self.ratio_threshold:
            return
        self._flagged.add(ctx.task_id)
        # Real, print()-based (not logger.warning()), matching the
        # established, confirmed-necessary pattern from earlier this same
        # session -- logger output has nowhere real to go here.
        print(
            f"[TOOL_SCAFFOLDING_DRIFT_FLAG] task={ctx.task_id} "
            f"role={ctx.role_id} turn={ctx.turn} -- cumulative tool-call "
            f"scaffolding ({scaffolding} bytes) has grown to "
            f"{'infinitely ' if content == 0 else ''}exceed real read_file "
            f"content ({content} bytes) by a real ratio "
            f"{'(no content read yet)' if content == 0 else f'of {ratio:.1f}x'}. "
            f"This is a real, plausible risk signal, not a confirmed "
            f"fabrication cause.",
            flush=True,
        )


__all__ = ["ToolScaffoldingDriftFlag"]
