"""Skill data models — Pydantic types for skill configuration and metadata.

References:
- DeerFlow: skills/loader.py (SKILL.md format)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class SkillConfig(BaseModel):
    """Configuration and metadata for a single skill.

    Parsed from a SKILL.md file with YAML frontmatter.
    """

    # Identity
    skill_id: str  # Directory name (unique identifier)
    name: str  # Display name from frontmatter
    description: str = ""

    # Metadata
    version: str = "1.0.0"
    author: str = ""
    license: str = ""
    tags: list[str] = Field(default_factory=list)
    allowed_tools: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    # Content
    content: str = ""  # Markdown body (after frontmatter)

    # Paths
    root_dir: str = ""  # Absolute path to skill directory
    scripts: list[str] = Field(default_factory=list)  # Script files
    resources: list[str] = Field(default_factory=list)  # Resource files

    # State
    enabled: bool = True

    @property
    def skill_md_path(self) -> Path:
        return Path(self.root_dir) / "SKILL.md"
