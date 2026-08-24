"""The guards are only worth their code if they are actually mounted/configured.

Covers the two wiring failures this change fixes: the reasoning-runaway
watchdog shipped as dead configuration (engine present, profile keys absent),
and the repetition guards existing but mounted on no shipped workflow.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from frontier_agent.components.observers.duplicate_query_rollback import (
    DuplicateQueryRollbackObserver,
)
from frontier_agent.components.observers.repetition_guard import RepetitionGuard
from frontier_agent.components.observers.text_repetition_guard import (
    TextRepetitionGuard,
)
from frontier_agent.core.loop_types import LoopConfig
from workflows.agent_team.nodes.main_agent import _build_observers
from workflows.agent_team.profile import load_swarm_profile
from workflows.agent_team.subagent_runtime import (
    SwarmSubagentRuntime,
    _swarm_observers,
)
from workflows.stateful_react_agent.nodes.main_agent import (
    _resolve_runaway_guardrails,
)
from workflows.stateful_react_agent.profile import load_react_profile

_GUARDRAILS = {
    "reasoning_only_timeout_s": 120,
    "reasoning_only_max_tokens": 16384,
    "logical_call_timeout_s": 900,
}

_SUB_TOOLS = [
    "web_search", "web_fetch", "submit_report", "bash", "read_file",
]


@pytest.fixture(autouse=True)
def _endpoint_env(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_MODEL", "test-model")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://example.test/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")


# ── Reasoning-runaway guardrails are configured, not just supported ──


@pytest.mark.parametrize("name", ["simple", "benchmark", "tui"])
def test_agent_team_profiles_arm_the_runaway_watchdog(name: str) -> None:
    agent_cfg = load_swarm_profile(name)["agent"]

    for key, value in _GUARDRAILS.items():
        assert agent_cfg[key] == value, f"{name}.{key}"


@pytest.mark.parametrize("name", ["simple", "benchmark", "tui"])
def test_stateful_profiles_arm_the_runaway_watchdog(name: str) -> None:
    agent_cfg = load_react_profile(name)["agent"]

    for key, value in _GUARDRAILS.items():
        assert agent_cfg[key] == value, f"{name}.{key}"


def test_every_shipped_profile_is_covered_by_the_assertions_above() -> None:
    """A new profile must not silently ship without the guardrails."""
    root = Path(__file__).parents[1] / "workflows"
    team = {p.stem for p in (root / "agent_team" / "profiles").glob("*.yaml")}
    react = {p.stem for p in (root / "stateful_react_agent" / "profiles").glob("*.yaml")}

    assert team == {"simple", "benchmark", "tui"}
    assert react == {"simple", "benchmark", "tui"}


def test_stateful_resolver_maps_the_profile_onto_loop_config() -> None:
    timeout_s, max_tokens, logical_s = _resolve_runaway_guardrails(
        load_react_profile("tui")["agent"],
    )
    config = LoopConfig(
        reasoning_only_timeout_s=timeout_s,
        reasoning_only_max_tokens=max_tokens,
        logical_call_timeout_s=logical_s,
    )

    assert config.reasoning_only_timeout_s == 120
    assert config.reasoning_only_max_tokens == 16384
    assert config.logical_call_timeout_s == 900


def test_stateful_resolver_treats_absent_and_zero_as_off() -> None:
    assert _resolve_runaway_guardrails({}) == (None, None, None)
    assert _resolve_runaway_guardrails({
        "reasoning_only_timeout_s": 0,
        "reasoning_only_max_tokens": 0,
        "logical_call_timeout_s": 0,
    }) == (None, None, None)


# ── Repetition guards are mounted ────────────────────────────────────


def _types(observers: list[object]) -> set[type]:
    return {type(o) for o in observers}


def test_subagent_stack_mounts_all_three_guards(tmp_path) -> None:
    runtime = SwarmSubagentRuntime(sub_agent_tool_names=_SUB_TOOLS)
    del tmp_path

    observers = _swarm_observers(
        runtime, task_id="task", session_name="researcher",
    )
    types = _types(observers)

    assert DuplicateQueryRollbackObserver in types
    assert RepetitionGuard in types
    assert TextRepetitionGuard in types


def test_subagent_text_guard_may_stop_but_the_coordinator_may_not() -> None:
    """A looping sub-agent burns an allowance; the coordinator IS the run."""
    sub_guard = next(
        o for o in _swarm_observers(
            SwarmSubagentRuntime(sub_agent_tool_names=_SUB_TOOLS),
            task_id="task",
            session_name="researcher",
        )
        if isinstance(o, TextRepetitionGuard)
    )
    main_guard = next(
        o for o in _build_observers(
            traj_dir=Path("/tmp/does-not-need-to-exist"),
            task_id="task",
            budget_tokens=None,
            max_input_tokens=None,
            event_store=None,
            tool_names=["create_subagent", "assign_task", "collect_reports"],
        )
        if isinstance(o, TextRepetitionGuard)
    )

    assert sub_guard.enable_stop is True
    assert main_guard.enable_stop is False
    # And the coordinator's hint must not tell a correctly-waiting agent to
    # finalize — the default template ends with "call your terminal/finalize
    # action", which is the pathology the mount above avoids in the tool
    # channel.
    assert main_guard.hint_message
    assert "do not finalize" in main_guard.hint_message.lower()
    assert "collect_reports" in main_guard.hint_message
    assert not sub_guard.hint_message


def test_only_the_subagent_repetition_guard_can_stop_the_loop() -> None:
    sub = next(
        o for o in _swarm_observers(
            SwarmSubagentRuntime(sub_agent_tool_names=_SUB_TOOLS),
            task_id="task",
            session_name="researcher",
        )
        if isinstance(o, RepetitionGuard)
    )

    assert sub.stop_after > sub.threshold


def test_a_searchless_subagent_skips_the_query_rollback() -> None:
    runtime = SwarmSubagentRuntime(
        sub_agent_tool_names=["read_file", "bash", "submit_report"],
    )

    observers = _swarm_observers(
        runtime, task_id="task", session_name="local",
    )

    assert DuplicateQueryRollbackObserver not in _types(observers)


def test_coordinator_mounts_the_rollback_only_when_it_can_search() -> None:
    def coordinator_types(tool_names: list[str]) -> set[type]:
        return _types(_build_observers(
            traj_dir=Path("/tmp/does-not-need-to-exist"),
            task_id="task",
            budget_tokens=None,
            max_input_tokens=None,
            event_store=None,
            tool_names=tool_names,
        ))

    searching = coordinator_types(["assign_task", "collect_reports", "web_search"])
    delegating = coordinator_types(["assign_task", "collect_reports"])

    assert DuplicateQueryRollbackObserver in searching
    assert DuplicateQueryRollbackObserver not in delegating
    # RepetitionGuard is deliberately absent from the coordinator: polling
    # collect_reports with identical arguments is correct behaviour there.
    assert RepetitionGuard not in searching
