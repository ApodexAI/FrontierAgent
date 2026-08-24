"""Skills runtime — filesystem-backed implementation of the SkillLoader Protocol.

Three layers, deliberately separate:

- **Protocol** lives in ``frontier_agent.core.protocols`` (``Skill``, ``SkillLoader``).
- **Implementation** lives here (``FileSystemSkillLoader``, ``ExtensionsConfig``).
- **Data** lives under top-level ``plugins/skills/<skill_id>/SKILL.md``.

No skills are bundled; drop a ``SKILL.md`` under ``plugins/skills/`` and a
profile's ``skills:`` list picks it up.
"""

from __future__ import annotations

from frontier_agent.components.skills.allowlist_loader import AllowlistSkillLoader
from frontier_agent.components.skills.config import SkillConfig
from frontier_agent.components.skills.extensions_config import ExtensionsConfig
from frontier_agent.components.skills.file_system_loader import (
    FileSystemSkillLoader,
)

__all__ = [
    "AllowlistSkillLoader",
    "ExtensionsConfig",
    "FileSystemSkillLoader",
    "SkillConfig",
]
