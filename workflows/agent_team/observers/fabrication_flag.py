"""FabricationFlag — logging-only observer that flags when a sub-agent's
final report claims a real file is empty/uninformative/repetitive, despite
the actual, real file content (captured directly from read_file's own
result) being genuinely substantive.

Real, added 2026-09-08. Directly motivated by a real, confirmed-live case
this same session: a sub-agent's final submit_report claimed
"/doc2.txt...contained uninformative text with repeated line numbers",
when the real, actual file content (confirmed directly, by reading it) was
genuinely substantive, coherent prose about risk management. This is a
DIFFERENT real failure signature than self_consistency_flag.py: that
observer compares the agent's own two outputs (an earlier summary vs. a
later report); this one compares the agent's final claim directly against
the REAL, GROUND-TRUTH file content, catching a fabrication even when the
agent never produced any correct intermediate summary at all.

This observer does NOT change behavior -- matching the established,
deliberate design of self_consistency_flag.py: it only logs a flagged
anomaly, so a real pattern can be tracked across runs before any
intervention is designed.
"""

from __future__ import annotations

import logging
import re

from frontier_agent.core.loop_types import BaseObserver, ToolResult, TurnContext

logger = logging.getLogger(__name__)

__all__ = ["FabricationFlag"]

# Real, direct reuse of the same dismissive-language signature confirmed
# live in the actual, observed fabrication case, kept in sync with (but
# separate from) self_consistency_flag.py's own copy -- these two
# observers check genuinely different things (ground truth vs. self) and
# are meant to stay independently maintainable.
_DISMISSIVE_RE = re.compile(
    r"\b(uninformative|no meaningful content|non-deliverable|"
    r"could not be (?:extracted|located|found)|nothing (?:useful|substantive)|"
    r"lacks? (?:any )?(?:real |substantive )?(?:content|information)|"
    # Real, added 2026-09-08: confirmed live, this exact phrasing ("no
    # evidence that [the file] contains unique or substantive content")
    # was a real, genuine fabrication that the original, narrower pattern
    # above missed entirely -- broadened to catch "no evidence (of/that)
    # ... substantive/unique/meaningful [content]" as its own real,
    # distinct construction.
    r"no evidence (?:of|that)[^.]{0,80}?(?:substantive|unique|meaningful)|"
    r"repeated line numbers|placeholder)\b",
    re.IGNORECASE,
)

# Real, minimum length (chars) for the ACTUAL file content (not an agent's
# summary of it) to count as "clearly substantive" -- deliberately low,
# since even a short real file with real prose is enough to make a
# "the file is empty/uninformative" claim a real, direct fabrication.
_MIN_REAL_FILE_CHARS = 50


def _extract_read_file_text(result: ToolResult) -> str:
    """Real, direct extraction of the actual file body from a real
    read_file ToolResult -- the raw result string itself, since read_file
    returns the file content directly (unlike create_file/submit_report,
    which take content as an argument)."""
    body = result.result if isinstance(result.result, str) else str(result.result)
    return body


def _extract_report_text(args: dict) -> str:
    content = args.get("content")
    return content if isinstance(content, str) else ""


class FabricationFlag(BaseObserver):
    """Real, logging-only: captures the real, actual content read_file
    genuinely returns, then checks the final submit_report against that
    real ground truth for a direct, dismissive-claim contradiction."""

    def __init__(self) -> None:
        self._longest_real_file_seen: dict[str, str] = {}  # task_id -> text

    async def on_tool_result(
        self, ctx: TurnContext, result: ToolResult,
    ) -> ToolResult | None:
        if result.name != "read_file" or result.is_error:
            return None
        text = _extract_read_file_text(result)
        if len(text) < _MIN_REAL_FILE_CHARS:
            return None
        prior = self._longest_real_file_seen.get(ctx.task_id, "")
        if len(text) > len(prior):
            self._longest_real_file_seen[ctx.task_id] = text
        return None

    async def on_tool_call(
        self, ctx: TurnContext, tool_call: dict,
    ):
        if str(tool_call.get("name", "")) != "submit_report":
            return None
        prior = self._longest_real_file_seen.get(ctx.task_id, "")
        if len(prior) < _MIN_REAL_FILE_CHARS:
            return None  # No real, actual file content was ever captured.
        args = tool_call.get("args") or {}
        report_text = _extract_report_text(args)
        if _DISMISSIVE_RE.search(report_text):
            # Real, print()-based (not logger.warning()), matching the
            # established, confirmed-necessary pattern from earlier this
            # same session -- logger output has nowhere real to go here.
            print(
                f"[FABRICATION_FLAG] task={ctx.task_id} role={ctx.role_id} "
                f"turn={ctx.turn} -- final submit_report claims the file "
                f"is empty/uninformative/repetitive, but the REAL file "
                f"content read via read_file was {len(prior)} genuinely "
                f"substantive chars. Possible fabrication. Report snippet: "
                f"{report_text[:200]!r} | Real file snippet: {prior[:200]!r}",
                flush=True,
            )
        return None


__all__ = ["FabricationFlag"]
