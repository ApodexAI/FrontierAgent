"""Report-oriented agent-team pipeline with an optional fast reporter.

The reporter consumes state evidence and fails open, so the coordinator's
answer remains available if report generation fails.
"""

from __future__ import annotations

from frontier_agent.models.pipeline_spec import (
    CompressionConfig,
    ContextPolicy,
    NodeDefinition,
    PipelineSpec,
    TransitionSpec,
)
from workflows.agent_team.identity import MAIN_ROLE_ID, REPORT_PIPELINE_ID

AGENT_TEAM_REPORT_SPEC = PipelineSpec(
    pipeline_id=REPORT_PIPELINE_ID,
    name="Agent Team + Reporter",
    description=(
        "Agent-Team coordinator + sub-agents (identical to ``agent_team``) "
        "followed by a stable reporter node. The resolved profile's "
        "agent.reporter flag controls whether the branch runs, and "
        "agent.reporter_backend keeps a stable dispatch seam; the OSS build "
        "ships the fast implementation."
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
                include_fields=["original_question", "language", "task_id", "metadata"],
            ),
            compression=CompressionConfig(enabled=False),
            output_fields=[
                "final_answer", "final_content", "answer_confidence",
                "evidence_cards", "assertions",
                "clarified_questions",
                "react_steps",
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


__all__ = ["AGENT_TEAM_REPORT_SPEC"]
