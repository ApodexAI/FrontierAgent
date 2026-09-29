"""SelfConsistencyFlag — logging-only observer that flags when a sub-agent's
final report contradicts its own, earlier, real, substantive content.

Real, added 2026-09-08. Directly motivated by a real, confirmed-live case
this same session: doc2_summarizer correctly read /doc2.txt and produced a
real, accurate intermediate summary (via create_file) -- diversification,
position sizing, cash reserves, all genuinely matching the real file --
then later submitted a final report claiming the file was "uninformative...
repeated line numbers", directly contradicting its own, earlier, correct
work. Traced directly: this is a self-consistency failure (the agent had
the right answer, then overwrote it), not a misreading (it never failed to
read the file correctly).

This observer does NOT change behavior -- per real, deliberate design
(matching the user's own explicit recommendation): it only logs a flagged
anomaly, so a real pattern can be tracked across runs before any
intervention is designed. It is a coarse, heuristic check, not a semantic
diff -- it looks for the specific, real signature observed live: a final
report using dismissive/empty language while a real, earlier artifact from
the SAME agent contained substantive, real content.
"""

from __future__ import annotations

import logging
import re

from frontier_agent.core.loop_types import BaseObserver, ToolCallIntervention, TurnContext

logger = logging.getLogger(__name__)

__all__ = ["SelfConsistencyFlag"]

# Real, direct phrases matching the exact dismissive/empty-content claim
# style observed live ("uninformative... repeated line numbers... no
# meaningful content"). Deliberately narrow and literal, matching what was
# actually seen, rather than a broad guess at every possible dismissive
# phrasing -- broadening this is real, future work once more real cases
# are logged.
_DISMISSIVE_RE = re.compile(
    r"\b(uninformative|no meaningful content|non-deliverable|"
    r"could not be (?:extracted|located|found)|nothing (?:useful|substantive)|"
    r"lacks? (?:any )?(?:real |substantive )?(?:content|information))\b",
    re.IGNORECASE,
)

# Real, minimum length (chars) for an earlier artifact to count as
# "substantive" -- short enough to not miss genuinely brief but real
# content, long enough to exclude trivial/placeholder writes.
_MIN_SUBSTANTIVE_CHARS = 120


def _extract_text_content(args: dict) -> str:
    """Real, direct extraction of the actual, real prose from a
    create_file/submit_report call's args, across the real, different
    shapes each tool uses."""
    # submit_report: {"content": "..."}
    content = args.get("content")
    if isinstance(content, str):
        return content
    # create_file: {"ops": [{"create": {"content": "..."}}, ...]}
    parts: list[str] = []
    for op in args.get("ops") or []:
        if not isinstance(op, dict):
            continue
        for op_body in op.values():
            if isinstance(op_body, dict):
                c = op_body.get("content")
                if isinstance(c, str):
                    parts.append(c)
    return "\n".join(parts)


class SelfConsistencyFlag(BaseObserver):
    """Real, logging-only: tracks the longest real, substantive text a
    sub-agent has written this run (via create_file), then checks its
    final submit_report against that history for a real, direct
    contradiction signature."""

    def __init__(self) -> None:
        self._longest_seen: dict[str, str] = {}  # task_id -> longest text

    async def on_tool_call(
        self, ctx: TurnContext, tool_call: dict,
    ) -> ToolCallIntervention | None:
        name = str(tool_call.get("name", ""))
        if name not in ("create_file", "submit_report"):
            return None
        args = tool_call.get("args") or {}
        text = _extract_text_content(args)

        if name == "create_file":
            if len(text) >= _MIN_SUBSTANTIVE_CHARS:
                prior = self._longest_seen.get(ctx.task_id, "")
                if len(text) > len(prior):
                    self._longest_seen[ctx.task_id] = text
            return None

        # name == "submit_report"
        prior = self._longest_seen.get(ctx.task_id, "")
        if len(prior) < _MIN_SUBSTANTIVE_CHARS:
            return None  # No real, earlier substantive artifact to compare against.
        if _DISMISSIVE_RE.search(text):
            # Real, print()-based (not logger.warning()) because this
            # codebase never calls logging.basicConfig(), confirmed
            # directly earlier this same session -- logger output has
            # nowhere real to go.
            print(
                f"[SELF_CONSISTENCY_FLAG] task={ctx.task_id} "
                f"role={ctx.role_id} turn={ctx.turn} -- final submit_report "
                f"uses dismissive/empty-content language, but this agent "
                f"earlier wrote {len(prior)} real chars of substantive "
                f"content via create_file. Possible self-consistency "
                f"failure. Final report snippet: {text[:200]!r} | Earlier "
                f"snippet: {prior[:200]!r}",
                flush=True,
            )
        return None


__all__ = ["SelfConsistencyFlag"]
