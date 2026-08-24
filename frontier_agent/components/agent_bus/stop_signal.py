"""Process-wide cooperative stop-signal registry for sub-agents."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class SubAgentStopRegistry:
    """Pending cooperative-stop requests keyed by ``session_id``.

    Each entry records the ``job_id`` the stop was requested against. Sessions
    are reusable (``assign_task`` again after a stop), so a stop that outlives
    its job — the job finished before the observer consumed it — must NOT roll
    back a later task on the same session. :meth:`clear_stale` drops such
    orphans at the next task's start.

    Single-event-loop access, so a plain ``dict`` is race-free for our needs
    (assignment / ``pop`` / membership are atomic w.r.t. the loop).
    """

    def __init__(self) -> None:
        # session_id -> job_id the stop was requested against ("" if unknown).
        self._requested: dict[str, str] = {}

    def request_stop(self, session_id: str, job_id: str = "") -> None:
        sid = (session_id or "").strip()
        if not sid:
            return
        self._requested[sid] = job_id or ""
        logger.info(
            "Cooperative stop requested for sub-agent session=%s (job=%s)",
            sid, job_id or "?",
        )

    def consume(self, session_id: str) -> bool:
        """Return True exactly once per request, clearing the flag.

        Called by the observer at each turn boundary so the stop prompt is
        injected only once even if more turns follow.
        """
        sid = (session_id or "").strip()
        if sid in self._requested:
            del self._requested[sid]
            return True
        return False

    def clear_stale(self, session_id: str, current_job_id: str) -> bool:
        """Drop a pending stop requested against a DIFFERENT job.

        Called at the start of each new task on a (reusable) session so a stop
        that outlived its job never fires on turn 1 of the next task. A stop
        for the *current* job is preserved. Returns True if one was cleared.
        """
        sid = (session_id or "").strip()
        pending = self._requested.get(sid)
        if pending is not None and pending != (current_job_id or ""):
            del self._requested[sid]
            logger.info(
                "Dropped stale stop for session=%s (queued for job=%s, now job=%s)",
                sid, pending or "?", current_job_id or "?",
            )
            return True
        return False

    def is_requested(self, session_id: str) -> bool:
        return (session_id or "").strip() in self._requested


_REGISTRY = SubAgentStopRegistry()


def get_stop_registry() -> SubAgentStopRegistry:
    """Return the process-wide stop registry singleton."""
    return _REGISTRY


__all__ = ["SubAgentStopRegistry", "get_stop_registry"]
