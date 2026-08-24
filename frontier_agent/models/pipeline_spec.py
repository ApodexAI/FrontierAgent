"""Declarative pipeline specification models.

A PipelineSpec describes the complete topology of a pipeline DAG:
which nodes exist, what agent roles they use, and how they connect.
Each spec can also bundle the agent definitions it needs.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from frontier_agent.models.agent_definition import AgentDefinition

# ---------------------------------------------------------------------------
# Node-level declarative models (pluggable node architecture)
# ---------------------------------------------------------------------------


class ContextPolicy(BaseModel):
    """Declares what subset of pipeline state a node receives."""

    include_fields: list[str] | None = Field(
        default=None,
        description="Whitelist of state fields. None = full state.",
    )
    filter_fn: str | None = Field(
        default=None,
        description=(
            "Dotted import path to (state: dict) -> dict. "
            "Takes precedence over include_fields."
        ),
    )
    inject_fields: list[str] = Field(
        default_factory=list,
        description=(
            "Fields always injected by framework regardless of "
            "include_fields."
        ),
    )


class CompressionConfig(BaseModel):
    """Per-node compression strategy for SummarizationMiddleware."""

    enabled: bool = Field(
        default=True,
        description="Whether compression is active for this node.",
    )
    threshold: int = Field(
        default=80000,
        description="Token count that triggers auto-compact.",
    )
    keep_recent: int = Field(
        default=6,
        description="Recent messages to preserve during compression.",
    )
    max_field_tokens: dict[str, int] = Field(
        default_factory=dict,
        description=(
            "Truncate state fields before prompt. "
            "E.g. {'evidence_cards': 8000}."
        ),
    )


class NodeExecutionPolicy(BaseModel):
    """Declares tool gating owned by a node/workflow, not by the kernel."""

    allow_tools: list[str] | None = Field(
        default=None,
        description=(
            "Allowlist of tool names for this node. None = all role-permitted "
            "tools remain available."
        ),
    )
    deny_tools: list[str] = Field(
        default_factory=list,
        description="Explicit tool-name denials for this node.",
    )
    deny_tool_prefixes: list[str] = Field(
        default_factory=list,
        description="Prefix-based tool denials for this node.",
    )


class SubAgentProfile(BaseModel):
    """Named template for sub-agents spawned by a node."""

    role_id: str = Field(
        ...,
        description="Agent role -> AgentDefinition (prompt, tools, model)",
    )
    context_policy: ContextPolicy = Field(default_factory=ContextPolicy)
    max_turns: int = Field(
        default=8,
        description="Max ReAct turns for this sub-agent",
    )
    budget_fraction: float = Field(
        default=0.1,
        description=(
            "Fraction of parent node's remaining budget allocated "
            "to this sub-agent"
        ),
    )


class NodeDefinition(BaseModel):
    """Declares a node's identity, behavior, and context requirements."""

    node_id: str = Field(
        ...,
        description="Unique within pipeline, e.g. 'clarify'",
    )
    role_id: str = Field(
        ...,
        description="Agent role -> AgentDefinition (prompt, tools, model)",
    )
    context_policy: ContextPolicy = Field(default_factory=ContextPolicy)
    execution_policy: NodeExecutionPolicy | None = Field(
        default=None,
        description=(
            "Node-owned runtime policy such as tool gating. "
            "None = use workflow/framework defaults."
        ),
    )
    compression: CompressionConfig | None = Field(
        default=None,
        description=(
            "Per-node compression config. None = framework defaults."
        ),
    )
    node_function: str | None = Field(
        default=None,
        description="Dotted import path to async def(state: dict) -> dict.",
    )
    prompt_template: str = Field(
        default="",
        description=(
            "Jinja2 template. Used by generic executor when "
            "node_function is None."
        ),
    )
    output_fields: list[str] = Field(default_factory=list)
    sub_agent_profiles: dict[str, SubAgentProfile] = Field(
        default_factory=dict,
    )
    display_label: str = Field(
        default="",
        description=(
            "Human-readable label emitted on the PHASE_TRANSITION event "
            "(e.g. 'Searching and collecting evidence'). Empty falls back "
            "to f'Entering {node_id} phase'. Replaces the hardcoded "
            "_PHASE_LABELS dict in EventEmitterMiddleware."
        ),
    )
    metadata: dict = Field(default_factory=dict)


class TransitionSpec(BaseModel):
    """Describes an edge in the pipeline DAG."""

    from_phase: str
    to_phase: str  # "__END__" for terminal edges
    condition: str | None = Field(
        default=None,
        description="Dotted import path to a condition function, or None for unconditional",
    )


class PipelineSpec(BaseModel):
    """Complete declarative description of a pipeline topology."""

    pipeline_id: str = Field(..., description="Unique ID, e.g. 'deep_research'")
    name: str = Field(default="")
    description: str = Field(default="")
    required_roles: list[str] = Field(default_factory=list)
    agent_definitions: list[AgentDefinition] = Field(
        default_factory=list,
        description="Agent roles bundled with this pipeline. Auto-registered at execution time.",
    )
    nodes: list[NodeDefinition] = Field(default_factory=list)
    transitions: list[TransitionSpec] = Field(default_factory=list)
    entry_point: str = Field(..., description="node_id of the first node")
    terminal_nodes: list[str] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict, description="Pipeline-level metadata")
    hidden: bool = Field(
        default=False,
        description=(
            "If True, this pipeline is omitted from the default /pipelines "
            "list (so it does not clutter the UI dropdown). It remains "
            "registered and selectable via explicit pipeline_id in API "
            "requests. Use for benchmark-only, backwards-compat, or "
            "developer-template pipelines."
        ),
    )
    state_type: str | None = Field(
        default=None,
        description="Dotted import path to the state TypedDict class. If None, uses BaseTaskState.",
    )
