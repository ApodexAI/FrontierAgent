"""Prompt ordering invariants that protect sibling-agent KV-cache reuse."""

from __future__ import annotations

import pytest

from plugins.tools.create_subagent import _resolve_specialist_prompt
from workflows.agent_team.prompts import CHART_NOTE
from workflows.agent_team.subagent_runtime import SwarmSubagentRuntime

_SUFFIX = "\n\n<<TASK-LEVEL-RUNTIME-SUFFIX>>"
_ROLE_HEADER = "\n\n# Your Role\n"
_MCP_TOOL = "mcp__demo__lookup"


def _resolve(role: str, *, tools: list[str] | None = None) -> str:
    return _resolve_specialist_prompt(
        name="topic_research",
        role_hint=role,
        fs_mode=True,
        mcp_tool_names=[_MCP_TOOL],
        runtime=SwarmSubagentRuntime(
            sub_prompt_suffix=_SUFFIX,
            sub_agent_tool_names=tools or ["web_search", "web_fetch"],
        ),
    )


def test_sibling_prompts_diverge_only_at_role_text() -> None:
    prompt_a = _resolve("ROLE-ALPHA")
    prompt_b = _resolve("ROLE-BETA")

    assert prompt_a.count(_ROLE_HEADER) == 1
    boundary = prompt_a.index(_ROLE_HEADER) + len(_ROLE_HEADER)
    shared = prompt_a[:boundary]

    assert prompt_b.startswith(shared)
    assert prompt_a[boundary:] == "ROLE-ALPHA"
    assert prompt_b[boundary:] == "ROLE-BETA"
    assert CHART_NOTE in shared
    assert "# Client MCP Tools" in shared
    assert prompt_a.index(_MCP_TOOL) < prompt_a.index(_SUFFIX) < boundary


def test_disabled_web_notice_remains_in_shared_prefix() -> None:
    prompt = _resolve("search the web for this", tools=["submit_report"])

    assert prompt.index(_SUFFIX) < prompt.index("DISABLED")
    assert prompt.index("DISABLED") < prompt.index(_ROLE_HEADER)
    assert "earlier instruction" not in prompt
    assert prompt.endswith("search the web for this")


@pytest.mark.parametrize("name", ["local_verifier", "final_verifier"])
def test_verifier_prompts_keep_runtime_suffix(name: str) -> None:
    prompt = _resolve_specialist_prompt(
        name=name,
        role_hint="unused role",
        fs_mode=True,
        runtime=SwarmSubagentRuntime(
            sub_prompt_suffix=_SUFFIX,
            sub_agent_tool_names=["web_search", "web_fetch"],
        ),
    )

    assert prompt.endswith(_SUFFIX)
    assert _ROLE_HEADER not in prompt
