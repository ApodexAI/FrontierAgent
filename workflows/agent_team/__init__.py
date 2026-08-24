"""Agent-Team — async multi-agent file-sandbox workflow."""

from __future__ import annotations

import os

from frontier_agent.core.runtime.registries.workflows import WorkflowContext
from frontier_agent.models.agent_definition import AgentDefinition
from workflows.agent_team.identity import (
    LEGACY_PIPELINE_ID,
    LEGACY_REPORT_PIPELINE_ID,
    MAIN_ROLE_ID,
    SUB_ROLE_ID,
)
from workflows.agent_team.spec import SWARM_SPEC
from workflows.agent_team.spec_report import AGENT_TEAM_REPORT_SPEC

# Closed-book toggle. When ``SWARM_NO_WEB=1`` is set in the worker env at
# import time, the sub-agent role loses its ``web_search`` + ``web_fetch``
# permissions. Used for fair-comparison evals on benchmarks whose problems
# are likely findable online (e.g. IMO-ProofBench). Default = web on.
_NO_WEB = os.environ.get("SWARM_NO_WEB", "").lower() in ("1", "true", "yes", "on")
# Filesystem/exec tools available to both roles. They run inside each agent's
# per-task bwrap sandbox (bash via sandbox.commands.run; grep/glob routed
# through the sandbox when a task sandbox is active), so they
# only see that agent's /workspace + /inputs (ro) + /outputs.
# Named once so the role pool and the profile-override path in
# nodes/main_agent.py cannot disagree about what "no web" means.
WEB_TOOL_NAMES = frozenset({"web_search", "web_fetch", "download_file"})

_FS_TOOLS = ["bash", "grep_search", "glob_search"]

_SUB_TOOLS: list[str] = []
if not _NO_WEB:
    _SUB_TOOLS += ["web_search", "web_fetch", "download_file"]
# ``read_file`` = the sandbox-aware structured reader (office/pdf/csv -> markdown);
# sub-agents read fetched/input files with it.
_SUB_TOOLS += ["submit_report", "read_file", "recover_result", *_FS_TOOLS]

# Placeholders — real prompts are built at runtime by main_agent_node /
# create_subagent so they always reflect the current date and
# the agent's name-based specialisation.
_MAIN_PLACEHOLDER = "Computed per-task."
_SUB_PLACEHOLDER = "Routed per-spawn by name."


MAIN_AGENT_DEF = AgentDefinition(
    role_id=MAIN_ROLE_ID,
    display_name="Agent-Team Coordinator",
    system_prompt=_MAIN_PLACEHOLDER,
    allowed_tools=[
        "create_subagent",
        "assign_task",
        "collect_reports",
        # Cooperative stop: ask an off-track / looping / no-longer-needed
        # sub-agent to wrap up (kept alive, re-assignable). NOT a hard cancel.
        "stop_subagent",
        # Task board (Planning Mode): the coordinator plans on the board and
        # finishes by ending a turn with plain text — there is NO finalize_answer
        # tool (BareTextFinalizeObserver gates + latches the answer).
        "add_task",
        "update_task",
        "finish_planning",
        # Coordinator is READ-ONLY: it inspects to plan via grep/glob (sandbox-
        # aware, reach /inputs) and may web_search to clarify a term, but cannot
        # execute code / write files — production is delegated to sub-agents.
        "web_search",
        "grep_search",
        "glob_search",
    ],
    color="#6366f1",
    icon="git-branch",
    description=(
        "Coordinator: decomposes questions, spawns persistent sub-agents, "
        "fan-ins reports asynchronously, verifies, then synthesises."
    ),
)

SUB_AGENT_DEF = AgentDefinition(
    role_id=SUB_ROLE_ID,
    display_name="Agent-Team Sub",
    system_prompt=_SUB_PLACEHOLDER,
    allowed_tools=_SUB_TOOLS,
    color="#8b5cf6",
    icon="search",
    description=(
        "Research sub-agent: executes one focused task (search, extract, "
        "compute) and returns a structured Scope/Finding/Evidence report."
    ),
)


def register(ctx: WorkflowContext) -> None:
    ctx.register_agent(MAIN_AGENT_DEF)
    ctx.register_agent(SUB_AGENT_DEF)
    ctx.register_pipeline(SWARM_SPEC)
    # Compatibility report-oriented variant. Both specs gate the same stable
    # reporter node on ``agent.reporter``; ``agent.reporter_backend`` selects
    # fast (default) or heavy behind that node.
    # Reuses swarm's ``swarm_reporter`` role (always loaded — every workflow's
    # register() runs into one shared registry at bootstrap).
    ctx.register_pipeline(AGENT_TEAM_REPORT_SPEC)
    # Input-only compatibility for existing deploy env and SDK callers. Hidden
    # aliases keep old selectors working without advertising them as canonical.
    ctx.register_pipeline(SWARM_SPEC.model_copy(update={
        "pipeline_id": LEGACY_PIPELINE_ID,
        "hidden": True,
    }))
    ctx.register_pipeline(AGENT_TEAM_REPORT_SPEC.model_copy(update={
        "pipeline_id": LEGACY_REPORT_PIPELINE_ID,
        "hidden": True,
    }))
