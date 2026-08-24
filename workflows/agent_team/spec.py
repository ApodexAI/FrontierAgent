"""Pipeline spec: ``agent_team``."""

from __future__ import annotations

from frontier_agent.models.pipeline_spec import (
    CompressionConfig,
    ContextPolicy,
    NodeDefinition,
    PipelineSpec,
    TransitionSpec,
)
from workflows.agent_team.identity import MAIN_ROLE_ID, PIPELINE_ID

SWARM_SPEC = PipelineSpec(
    pipeline_id=PIPELINE_ID,
    name="Agent Team",
    description=(
        "Main agent decomposes the question, spawns parallel sub-agents, "
        "fan-ins their reports, verifies, then synthesises a final answer. "
        "Profiles may enable the stable fast reporter node."
    ),
    entry_point="main_agent",
    terminal_nodes=["main_agent", "agent_team_reporter"],
    nodes=[
        NodeDefinition(
            node_id="main_agent",
            role_id=MAIN_ROLE_ID,
            node_function=(
                "workflows.agent_team.nodes.main_agent.main_agent_node"
            ),
            context_policy=ContextPolicy(
                include_fields=[
                    "original_question", "current_query", "language",
                    "task_id", "metadata",
                ],
            ),
            compression=CompressionConfig(enabled=False),
            output_fields=[
                "final_answer", "final_content", "answer_confidence",
                "evidence_cards", "assertions",
                "clarified_questions",
                "react_steps",
                "session_turn",
                "live_followups", "effective_question",
                "reporter_enabled", "reporter_backend", "reporter_wall_time_s",
                "reporter_deadline_monotonic_s",
                "answer_status", "answer_sentinel",
                "final_answer_rescued", "final_answer_rescue_mode",
                "final_answer_source", "stopped_by",
            ],
        ),
        NodeDefinition(
            node_id="agent_team_reporter",
            role_id="swarm_reporter",
            node_function=(
                "workflows.agent_team.nodes.reporter.agent_team_reporter"
            ),
            context_policy=ContextPolicy(
                include_fields=[
                    "original_question", "current_query", "language",
                    "task_id", "metadata",
                    "final_answer", "final_content",
                    "evidence_cards", "assertions",
                    "live_followups", "effective_question",
                    "reporter_backend", "reporter_wall_time_s",
                    "reporter_deadline_monotonic_s",
                ],
            ),
            compression=CompressionConfig(enabled=False),
            output_fields=[
                "final_answer",
                "final_content",
                "report_markdown",
                "answer_status",
                "answer_sentinel",
                "final_answer_source",
                "final_answer_rescued",
                "final_answer_rescue_mode",
            ],
        ),
    ],
    transitions=[
        TransitionSpec(
            from_phase="main_agent",
            to_phase="agent_team_reporter",
            condition="workflows.agent_team.edges.should_run_reporter",
        ),
        TransitionSpec(from_phase="main_agent", to_phase="__END__"),
        TransitionSpec(from_phase="agent_team_reporter", to_phase="__END__"),
    ],
)
