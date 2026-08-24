"""AgentDefinition — declarative, runtime-configurable agent role."""

from __future__ import annotations

from pydantic import BaseModel, Field


class AgentDefinition(BaseModel):
    """Describes an agent role that can be dynamically registered.

    This replaces the hardcoded AgentRole enum — new roles can be added
    at runtime via AgentRegistry.register() without modifying any kernel code.
    """

    role_id: str = Field(..., description="Unique identifier, e.g. 'researcher', 'critic', or any custom role")
    display_name: str = Field(..., description="Human-readable name for UI display")
    system_prompt: str = Field(default="", description="System prompt sent to LLM when this agent acts")
    allowed_tools: list[str] = Field(default_factory=list, description="Tool names this agent can use")
    # Per-role LLM overrides.
    model: str | None = Field(default=None, description="Override LLM model for this role (None = use default)")
    temperature: float = Field(default=0.3, description="LLM temperature for this role")
    max_tokens: int = Field(default=4096, description="Max output tokens for this role")

    color: str = Field(default="#6b7280", description="Hex color for frontend visualization")
    icon: str = Field(default="agent", description="Icon identifier for frontend")
    description: str = Field(default="", description="Brief description of this agent's purpose")
    metadata: dict = Field(default_factory=dict, description="Extensible metadata")

    # Whether SkillInjectionMiddleware should inject the available-skills
    # metadata into this role's system prompt. Off by default so adding a
    # new role does not silently start consuming skills. Roles that opt in
    # are also responsible for declaring `read_text` in `allowed_tools` —
    # without it the LLM can see skill metadata but cannot load SKILL.md.
    enable_skills: bool = Field(
        default=False,
        description=(
            "If True, SkillInjectionMiddleware injects available-skills "
            "metadata into this role's system prompt. Caller must also "
            "ensure read_text is in allowed_tools for the LLM to load "
            "the referenced SKILL.md files."
        ),
    )
