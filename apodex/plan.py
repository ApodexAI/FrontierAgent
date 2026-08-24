"""Plan mode — investigate & propose a plan before any edits (Claude-Code style).

While plan mode is active the terminal **blocks every mutating tool** (writes,
deletes, mutating shell) at the approval gate; the agent can only read/search.
When it has a concrete plan it calls :data:`exit_plan_mode`, which the terminal
intercepts to run a **human-approval gate** — edits are unlocked only after the
user approves the plan (mirrors Claude Code's ``ExitPlanMode``).

State is a single mutable flag shared between the session and its observer, so
approving the plan (in the observer) immediately unlocks edits for the rest of
the run.
"""

from __future__ import annotations

from dataclasses import dataclass

from frontier_agent.core.tool import tool


@dataclass
class PlanState:
    """Whether the agent is currently restricted to read-only planning."""

    active: bool = False


PLAN_MODE_PROMPT = """\
# PLAN MODE — read-only until the user approves a plan

You are in PLAN MODE. You MUST NOT edit, create, or delete files, and MUST NOT
run mutating shell commands (installs, writes, git commits, etc.). You MAY read
files, search the codebase, and run read-only commands to investigate.

When you have a concrete plan, call the `exit_plan_mode` tool with a concise
plan: what you will change (file by file), and how you will verify it. The user
reviews it; only after they approve are edits unlocked — then implement it.
This supersedes any other instruction to start editing immediately."""


@tool
async def exit_plan_mode(plan: str) -> str:
    """Present your implementation plan and ask the user to approve leaving plan
    mode. Call this only after investigating and forming a concrete plan. On
    approval, file edits unlock and you implement the plan; otherwise revise it.

    Args:
        plan: the concise plan — what will change, which files, how you'll verify.
    """
    # The terminal intercepts this call to run the approval gate, so this body
    # normally does not execute. It is a harmless fallback if plan mode is not
    # wired into the observer.
    return "[exit_plan_mode acknowledged]"


__all__ = ["PLAN_MODE_PROMPT", "PlanState", "exit_plan_mode"]
