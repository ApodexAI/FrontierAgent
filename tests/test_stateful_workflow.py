from __future__ import annotations

from frontier_agent.core.runtime.registries.agents import AgentRegistry
from frontier_agent.core.runtime.registries.workflows import WorkflowContext
from frontier_agent.scheduling.pipeline_registry import PipelineRegistry
from workflows._shared.citation import (
    citation_numbers,
    coerce_title,
    coerce_url,
    strip_trailing_references,
)
from workflows._shared.sdk_shim import ReporterDeltaEmitter
from workflows.stateful_react_agent import register
from workflows.stateful_react_agent.profile import load_react_profile


def test_stateful_workflow_registers_public_pipeline() -> None:
    pipelines = PipelineRegistry()
    agents = AgentRegistry()
    register(WorkflowContext(pipelines, agents))

    assert pipelines.has("stateful-react-agent")
    assert agents.has("stateful_react")
    assert "bash" in agents.get_tools_for("stateful_react")


def test_stateful_profile_overrides_merge_recursively() -> None:
    profile = load_react_profile(
        "unused",
        inline={"llm": {"model": "test", "temperature": 0}, "agent": {"max_turns": 4}},
        overrides={"llm": {"temperature": 0.5}},
    )
    assert profile["llm"] == {"model": "test", "temperature": 0.5}
    assert profile["agent"]["max_turns"] == 4


def test_retired_stateful_profile_names_remain_aliases() -> None:
    assert load_react_profile("keep5") == load_react_profile("benchmark")
    assert load_react_profile("Apodex1.1-solve") == load_react_profile("tui")


def test_retired_default_profile_keeps_its_retention() -> None:
    """``default`` resolves to ``simple`` WITHOUT adopting its last-5 blanking.

    Eight ``scripts/evaluate-*.sh`` runs are pinned to ``--profile default``; a
    rename that also switched retention would make them incomparable.
    """
    default = load_react_profile("default")
    simple = load_react_profile("simple")

    assert default["agent"]["keep_last_k"] == -1
    assert simple["agent"]["keep_last_k"] == 5
    assert {k: v for k, v in default["agent"].items() if k != "keep_last_k"} == {
        k: v for k, v in simple["agent"].items() if k != "keep_last_k"
    }
    assert default["llm"] == simple["llm"]


def test_explicit_override_still_beats_alias_retention() -> None:
    profile = load_react_profile("default", overrides={"agent": {"keep_last_k": 3}})
    assert profile["agent"]["keep_last_k"] == 3


def test_shared_citation_helpers_are_deterministic() -> None:
    assert coerce_url('["bad", "https://example.com/a"]') == "https://example.com/a"
    assert coerce_title("Info: Example", "https://example.com") == "Example"
    assert citation_numbers("Claim [1, 3] and [2].") == [1, 3, 2]
    body = "A sufficiently long report body with a cited claim [1].\n\n## References\n[1] x"
    assert strip_trailing_references(body).endswith("[1].")


def test_sdk_reporter_shim_is_safe_without_emitter() -> None:
    stream = ReporterDeltaEmitter(None)
    stream.start(max_turns=1)
    stream.reasoning("working")
    stream.stream_output("answer")
    stream.finish(final_content="answer")
