"""Stateful ReAct Agent - single-agent workflow."""

from __future__ import annotations

import os

from frontier_agent.core.runtime.registries.workflows import WorkflowContext
from frontier_agent.models.agent_definition import AgentDefinition
from workflows.stateful_react_agent.spec import REACT_SPEC

_NO_WEB = os.environ.get("REACT_NO_WEB", "").lower() in (
    "1",
    "true",
    "yes",
    "on",
)

# The tools a closed-book run must not bind, named once so the role pool and
# the profile-override path in nodes/main_agent.py cannot disagree.
WEB_TOOL_NAMES = frozenset({"web_search", "web_fetch", "download_file"})

# ``recover_result`` is deliberately outside the web gate: it reads this run's
# own transcript, so a closed-book run keeps it.
_TOOLS = ["bash", "grep_search", "glob_search", "recover_result"]
if not _NO_WEB:
    _TOOLS = ["web_search", "web_fetch", "download_file", *_TOOLS]


REACT_AGENT_DEF = AgentDefinition(
    role_id="stateful_react",
    display_name="Stateful ReAct Agent",
    system_prompt="Computed per-task.",
    allowed_tools=_TOOLS,
    color="#0ea5e9",
    icon="bot",
    description=(
        "Single stateful ReAct agent: works in a per-task workspace, uses "
        "search/fetch/bash/grep/glob tools, and returns a plain-text "
        "final answer when it stops calling tools."
    ),
)


def register(ctx: WorkflowContext) -> None:
    ctx.register_agent(REACT_AGENT_DEF)
    ctx.register_pipeline(REACT_SPEC)
