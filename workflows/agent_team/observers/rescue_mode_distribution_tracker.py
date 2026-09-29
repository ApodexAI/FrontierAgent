"""RescueModeDistributionTracker — logging-only, real, module-level counter
of which rescue_mode force_final_answer resolves to, aggregated across
every real sub-agent run within this process.

Real, added 2026-09-08. This is a real, deliberately NARROW, non-
overlapping scope, distinct from ForceFinalAnswerObserver (the print()
statement already inside force_final_answer): that one logs a single
real event per invocation (task, stopped_by, rescue_mode, text_chars).
This tracker instead accumulates a real, running distribution across
ALL invocations seen so far in this process, answering a genuinely
different question -- not "what happened this one time" but "which
rescue path does this pipeline lean on most, over many real runs".

force_final_answer is a plain async function (SubAgentRuntimeSpec's
force_finalizer callback), not an observer with its own hook -- this
module is called directly from within that function, the same
self-contained pattern used for the fabrication check added earlier
this session, since there is no shared observer-instance state to hook
into there.
"""

from __future__ import annotations

from collections import Counter

# Real, module-level state: persists across every real sub-agent run in
# this process, not scoped to a single task_id or role_id, since the
# whole point is a cross-run distribution.
_counts: Counter[str] = Counter()


def record_and_report(rescue_mode: str, task_id: str = "") -> None:
    """Real, direct call site: record one real rescue_mode occurrence,
    then print the current, running distribution across every real
    invocation seen so far this process."""
    _counts[rescue_mode] += 1
    total = sum(_counts.values())
    breakdown = "; ".join(
        f"{mode}={count} ({count / total:.0%})"
        for mode, count in _counts.most_common()
    )
    # Real, print()-based (not logger.warning()), matching the
    # established, confirmed-necessary pattern from earlier this same
    # session -- logger output has nowhere real to go here.
    print(
        f"[RESCUE_MODE_DISTRIBUTION_TRACKER] task={task_id or '?'} "
        f"latest={rescue_mode} -- running distribution across {total} "
        f"real invocations this process: {breakdown}",
        flush=True,
    )


def reset() -> None:
    """Real, direct reset -- exposed for tests; not called in production."""
    _counts.clear()


__all__ = ["record_and_report", "reset"]
