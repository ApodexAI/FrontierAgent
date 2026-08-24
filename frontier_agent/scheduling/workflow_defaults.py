"""Workflow-scoped default lookup."""

from __future__ import annotations

import importlib
import logging
from typing import Any

logger = logging.getLogger(__name__)


# pipeline_id → dotted path of the workflow's defaults module.
# Add a row when a new workflow ships a ``defaults.py``.
_PIPELINE_TO_DEFAULTS: dict[str, str] = {
    # agent_team owns a phase-aware research wall and must not have its
    # downstream reporter cancelled by the scheduler's graph-wide ceiling.
    "agent_team": "workflows.agent_team.defaults",
    "agent_team_report": "workflows.agent_team.defaults",
    "agent-team": "workflows.agent_team.defaults",
    "agent-team-report": "workflows.agent_team.defaults",
    # Stateful report synthesis runs as an on_loop_end phase inside its single
    # DAG node, so it likewise needs the workflow-owned research wall.
    "stateful-react-agent": "workflows.stateful_react_agent.defaults",
    "stateful_react_agent": "workflows.stateful_react_agent.defaults",
    "react_base": "workflows.stateful_react_agent.defaults",
    "react-base": "workflows.stateful_react_agent.defaults",
}


def get_workflow_default(pipeline_id: str | None, attr: str) -> Any:
    """Look up ``attr`` from ``pipeline_id``'s workflow defaults module.

    Returns ``None`` when:

    - ``pipeline_id`` is falsy or unrecognised;
    - the registered defaults module is missing or fails to import;
    - the module does not define ``attr``.

    Callers should treat ``None`` as "no workflow default" and continue
    their existing fallback chain (env var, then no-deadline / default).
    """
    if not pipeline_id:
        return None
    module_path = _PIPELINE_TO_DEFAULTS.get(pipeline_id)
    if module_path is None:
        return None
    try:
        module = importlib.import_module(module_path)
    except ImportError as e:
        logger.debug(
            "workflow_defaults: failed to import %s (pipeline=%s): %s",
            module_path, pipeline_id, e,
        )
        return None
    return getattr(module, attr, None)
