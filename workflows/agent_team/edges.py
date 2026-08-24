"""Conditional routing for the optional agent-team reporter."""

from __future__ import annotations

from typing import Any


def should_run_reporter(state: dict[str, Any]) -> str:
    """Run the shared reporter node only when the resolved profile enables it.

    ``reporter_enabled`` is a MANDATORY output of ``main_agent_node`` (resolved
    there by ``_resolve_reporter_enabled``, which owns the per-pipeline
    defaults). The fallback below is fail-safe, not a default: a missing key
    means the edge cannot tell an ``agent.reporter=false`` request from a node
    that forgot to publish its decision, and skipping the reporter is the
    cheaper of the two wrong answers. Backend selection is independent and
    travels in ``reporter_backend``. So any new code path returning from
    ``main_agent_node`` must carry the key forward, or ``agent_team_report``
    silently loses its reporter — see the regression test in
    ``tests/workflows/agent_team/test_reporter_config.py``.
    """
    if bool(state.get("reporter_enabled", False)):
        return "agent_team_reporter"
    return "__END__"


__all__ = ["should_run_reporter"]
