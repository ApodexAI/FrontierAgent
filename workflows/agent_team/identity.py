"""Canonical identifiers for the Agent Team workflow."""

from urllib.parse import quote

PIPELINE_ID = "agent_team"
REPORT_PIPELINE_ID = "agent_team_report"

# Input-only compatibility aliases. New configuration and emitted metadata
# must use the canonical underscore ids above.
LEGACY_PIPELINE_ID = "agent-team"
LEGACY_REPORT_PIPELINE_ID = "agent-team-report"

MAIN_ROLE_ID = "agent_team_main"
SUB_ROLE_ID = "agent_team_sub"

# The coordinator is a singleton, so its instance identity is intentionally
# identical to its role identity.
MAIN_AGENT_ID = MAIN_ROLE_ID


def llm_session_id(task_id: str, agent_id: str) -> str:
    """Return a stable, per-agent upstream affinity key.

    AgentBus and event plumbing continue to use the unmodified root task id;
    only the upstream LLM routing key is split. Quoting keeps free-form,
    Unicode-capable agent-team names safe in an HTTP header.
    """
    safe_agent_id = quote(str(agent_id or "unknown"), safe="-_.")
    return f"{task_id}.agent_team.{safe_agent_id}"


__all__ = [
    "LEGACY_PIPELINE_ID",
    "LEGACY_REPORT_PIPELINE_ID",
    "MAIN_AGENT_ID",
    "MAIN_ROLE_ID",
    "PIPELINE_ID",
    "REPORT_PIPELINE_ID",
    "SUB_ROLE_ID",
    "llm_session_id",
]
