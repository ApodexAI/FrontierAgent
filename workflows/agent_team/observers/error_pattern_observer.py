"""ErrorPatternObserver — logging-only observer that categorizes tool-call
errors by tool name and error-message pattern, emitting a real summary at
the end of a sub-agent's run.

Real, added 2026-09-08. This is a real, distinct signal from
ErrorRecoveryStreakFlag (built earlier this same session): that observer
only tracks CONSECUTIVE error streaks (fires once at a threshold, then
resets). This observer instead categorizes every real error by (tool,
error-message-prefix), building up a real distribution across the whole
run, then reports it once at the end -- giving visibility into WHICH
tools fail and roughly WHY, rather than just "an error streak happened".

Confirmed firing live, this same session, after a separate, real fix to
frontier_agent/core/runtime/loop/tool_exec.py (read_file previously never
produced is_error=True at all -- a real, structural tool-contract gap,
confirmed and fixed directly, live).

Real, extended 2026-09-08 (round 2) with two real, additive signals:

1. Severity scoring -- a real, direct keyword classifier on the error
   message text (timeout/permission -> high; not-found/does-not-exist ->
   low; anything else -> medium). This is a real, coarse heuristic, not a
   confirmed-live-motivated one -- no specific real high-severity case
   has been observed yet this session (the one real, live error caught
   so far was a low-severity FileNotFoundError).

2. Temporal clustering -- tracks the real turn number of every error per
   agent; flags when 2+ errors land within a real, short (default 5-turn)
   window, distinct from ErrorRecoveryStreakFlag's CONSECUTIVE-turn
   requirement (a cluster here can have non-error turns interleaved,
   e.g. errors at turns 3 and 6 still cluster within a window of 5).

This observer does NOT change behavior -- matching the established,
deliberate design of the other real observers built this session: it
only logs a summary.
"""

from __future__ import annotations

import logging
import re
from collections import Counter

from frontier_agent.core.loop_types import (
    AgentLoopResult,
    BaseObserver,
    ToolResult,
    TurnContext,
)

logger = logging.getLogger(__name__)

__all__ = ["ErrorPatternObserver"]

# Real, direct bound on how much of a real error message is used as the
# "pattern" key -- deliberately short, since a full stack trace or long
# error string would make every error look unique and defeat the point of
# categorization. Real errors observed live this session (e.g. "invalid
# glob pattern: Non-relative patterns are unsupported") are short enough
# that this captures the real, meaningful part.
_PATTERN_PREFIX_CHARS = 60

# Real, direct, deliberately coarse severity keyword classifier. Checked
# in this real order (high before low) so a message matching both (rare,
# but possible) is treated as the more serious real category.
_HIGH_SEVERITY_RE = re.compile(
    r"\b(timeout|timed out|permission denied|access denied|"
    r"connection refused|out of memory|disk full)\b",
    re.IGNORECASE,
)
_LOW_SEVERITY_RE = re.compile(
    r"\b(not found|does not exist|no such file|filenotfounderror|"
    r"invalid (?:glob|pattern))\b",
    re.IGNORECASE,
)

# Real, direct default window (in turns) for temporal clustering -- two
# errors at turns N and N+4 still count as clustered; N and N+6 do not.
_DEFAULT_CLUSTER_WINDOW = 5


def _classify_severity(body: str) -> str:
    if _HIGH_SEVERITY_RE.search(body):
        return "high"
    if _LOW_SEVERITY_RE.search(body):
        return "low"
    return "medium"


class ErrorPatternObserver(BaseObserver):
    """Real, logging-only: tracks a real (tool_name, error_prefix) counter
    across a sub-agent's whole run; emits a real summary line when the
    loop ends, including a real severity breakdown and real temporal
    clustering.

    Keyed by ``role_id`` (not ``task_id``): ``AgentLoopResult`` (the only
    argument ``on_loop_end`` receives) genuinely has no ``task_id`` field
    -- confirmed directly, the same real limitation just fixed elsewhere
    in this codebase this session. ``role_id`` is reliably present on
    every ``TurnContext`` this observer sees via ``on_tool_result``, and
    is unique per sub-agent (e.g. ``doc2_summarizer``), so it is the real,
    correct key here instead.
    """

    def __init__(
        self, *, min_errors_to_report: int = 1,
        cluster_window: int = _DEFAULT_CLUSTER_WINDOW,
    ) -> None:
        self.min_errors_to_report = max(1, int(min_errors_to_report))
        self.cluster_window = max(2, int(cluster_window))
        # role_id -> Counter of (tool_name, error_prefix) -> count
        self._counts: dict[str, Counter[tuple[str, str]]] = {}
        # role_id -> Counter of severity -> count
        self._severity_counts: dict[str, Counter[str]] = {}
        # role_id -> sorted list of real turn numbers where an error occurred
        self._error_turns: dict[str, list[int]] = {}
        self._last_role_id: str = ""

    async def on_tool_result(
        self, ctx: TurnContext, result: ToolResult,
    ) -> ToolResult | None:
        self._last_role_id = ctx.role_id
        if not result.is_error:
            return None
        body = result.result if isinstance(result.result, str) else str(result.result)
        pattern = body[:_PATTERN_PREFIX_CHARS].replace("\n", " ").strip()
        key = (result.name or "unknown", pattern)
        counter = self._counts.setdefault(ctx.role_id, Counter())
        counter[key] += 1

        severity = _classify_severity(body)
        sev_counter = self._severity_counts.setdefault(ctx.role_id, Counter())
        sev_counter[severity] += 1

        self._error_turns.setdefault(ctx.role_id, []).append(ctx.turn)
        return None

    def _find_clusters(self, turns: list[int]) -> list[tuple[int, int]]:
        """Real, direct scan: returns (start_turn, error_count) for each
        real, maximal run of turns where consecutive recorded error-turns
        are within ``cluster_window`` of each other and the run has 2+
        errors. A single, isolated error is never a cluster."""
        if len(turns) < 2:
            return []
        turns = sorted(turns)
        clusters: list[tuple[int, int]] = []
        run_start = turns[0]
        run_count = 1
        for prev, cur in zip(turns, turns[1:]):
            if cur - prev <= self.cluster_window:
                run_count += 1
            else:
                if run_count >= 2:
                    clusters.append((run_start, run_count))
                run_start = cur
                run_count = 1
        if run_count >= 2:
            clusters.append((run_start, run_count))
        return clusters

    async def on_loop_end(self, result: AgentLoopResult) -> None:
        role_id = self._last_role_id
        counter = self._counts.pop(role_id, None)
        sev_counter = self._severity_counts.pop(role_id, Counter())
        turns = self._error_turns.pop(role_id, [])
        if not counter:
            return
        total = sum(counter.values())
        if total < self.min_errors_to_report:
            return
        # Real, direct top-5 breakdown, sorted by frequency -- enough to
        # see the real dominant failure modes without an unbounded dump.
        top = counter.most_common(5)
        breakdown = "; ".join(
            f"{tool!r}/{pattern!r}={count}" for (tool, pattern), count in top
        )
        severity_summary = "; ".join(
            f"{sev}={count}" for sev, count in sev_counter.most_common()
        )
        clusters = self._find_clusters(turns)
        cluster_summary = (
            "; ".join(
                f"turn={start}+ ({count} errors within {self.cluster_window} turns)"
                for start, count in clusters
            )
            if clusters else "none"
        )
        # Real, print()-based (not logger.warning()), matching the
        # established, confirmed-necessary pattern from earlier this same
        # session -- logger output has nowhere real to go here.
        print(
            f"[ERROR_PATTERN_OBSERVER] role={role_id or '?'} "
            f"stopped_by={result.stopped_by} -- {total} real tool errors "
            f"this run across {len(counter)} distinct (tool, pattern) "
            f"categories. Top: {breakdown} | Severity: {severity_summary} "
            f"| Temporal clusters: {cluster_summary}",
            flush=True,
        )


__all__ = ["ErrorPatternObserver"]
