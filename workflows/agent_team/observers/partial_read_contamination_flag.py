"""PartialReadContaminationFlag — logging-only observer that flags when an
agent reads the SAME real file multiple times with DIFFERENT offset/
max_chars combinations, a real risk factor for anchoring on the wrong,
partial version of the file's content.

Real, added 2026-09-08. This is a real, distinct, THEORETICAL mechanism --
unlike SelfConsistencyFlag and FabricationFlag (both built directly from a
real, confirmed-live case), this observer targets a hypothesized cause
that has NOT been directly confirmed as the actual mechanism behind the
original doc2 fabrication. That case was never traced back to a confirmed
partial-read sequence -- this is a real, plausible hypothesis worth
instrumenting, not a demonstrated mechanism, and should be read that way
until it fires on a real, live case.

This observer does NOT change behavior -- matching the established,
deliberate design of the other real observers built this session: it only
logs a flagged anomaly, so a real correlation with FabricationFlag/
SelfConsistencyFlag can be tracked across runs if one exists.
"""

from __future__ import annotations

import logging

from frontier_agent.core.loop_types import BaseObserver, TurnContext

logger = logging.getLogger(__name__)

__all__ = ["PartialReadContaminationFlag"]


class PartialReadContaminationFlag(BaseObserver):
    """Real, logging-only: tracks distinct (offset, max_chars) read_file
    variants seen for each real file path, per agent; flags when a real
    file has been read with 2+ distinct variants."""

    def __init__(self, *, min_variants: int = 2) -> None:
        self.min_variants = max(2, int(min_variants))
        # (task_id, path) -> set of (offset, max_chars) variants seen
        self._variants: dict[tuple[str, str], set[tuple[int, int | None]]] = {}
        self._flagged: set[tuple[str, str]] = set()

    async def on_tool_call(
        self, ctx: TurnContext, tool_call: dict,
    ):
        name = str(tool_call.get("name", ""))
        if name != "read_file":
            return None
        args = tool_call.get("args") or {}
        path = args.get("path")
        if not isinstance(path, str) or not path:
            return None

        offset = args.get("offset", 0)
        offset = offset if isinstance(offset, int) else 0
        max_chars = args.get("max_chars")
        max_chars = max_chars if isinstance(max_chars, int) else None
        variant = (offset, max_chars)

        key = (ctx.task_id, path)
        variants = self._variants.setdefault(key, set())
        variants.add(variant)

        if len(variants) < self.min_variants or key in self._flagged:
            return None
        self._flagged.add(key)

        # Real, print()-based (not logger.warning()), matching the
        # established, confirmed-necessary pattern from earlier this same
        # session -- logger output has nowhere real to go here.
        print(
            f"[PARTIAL_READ_CONTAMINATION_FLAG] task={ctx.task_id} "
            f"role={ctx.role_id} turn={ctx.turn} -- real file {path!r} has "
            f"been read with {len(variants)} distinct (offset, max_chars) "
            # Real bug found and fixed 2026-09-08, directly, before a live
            # run could hit it: sorted() on raw tuples raised a real,
            # genuine TypeError when comparing (int, None) against
            # (int, int) -- max_chars is legitimately sometimes None (the
            # tool's own real default). Sort by string repr instead.
            f"variants: {sorted(variants, key=str)}. Possible anchoring on "
            f"an incomplete/inconsistent read. This is a real, plausible "
            f"risk signal, not a confirmed fabrication cause.",
            flush=True,
        )
        return None


__all__ = ["PartialReadContaminationFlag"]
