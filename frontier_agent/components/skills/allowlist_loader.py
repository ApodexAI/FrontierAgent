"""Allowlist filter wrapper for the ``SkillLoader`` Protocol.

Generic — no workflow / domain coupling. Workflows that need to scope
skill injection to a subset of IDs (rather than relying on the global
``ExtensionsConfig`` enabled flag, which affects every workflow) wrap
the registered loader with this and pass it to a workflow-local
``SkillInjectionMiddleware`` instance.

See ``workflows/apodex_react_skills/nodes/main_agent.py:_enable_scoped_skills``
for the canonical use site.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from frontier_agent.core.protocols import Skill, SkillLoader


@dataclass(frozen=True)
class AllowlistSkillLoader:
    """``SkillLoader`` wrapper exposing only an explicit set of skill IDs.

    Implements :class:`frontier_agent.core.protocols.SkillLoader`
    structurally. Read methods intersect with ``allowed_ids``; mutation
    and reload pass through so an upstream toggle / disk reload of the
    base loader is still honoured for ids that are inside the allowlist.
    """

    inner: SkillLoader
    allowed_ids: frozenset[str]

    def list_skills(self) -> list[Skill]:
        return [
            s for s in self.inner.list_skills()
            if s.skill_id in self.allowed_ids
        ]

    def get_skill(self, skill_id: str) -> Skill | None:
        if skill_id not in self.allowed_ids:
            return None
        return self.inner.get_skill(skill_id)

    def get_enabled_skills(self) -> list[Skill]:
        return [
            s for s in self.inner.get_enabled_skills()
            if s.skill_id in self.allowed_ids
        ]

    def toggle_skill(self, skill_id: str, enabled: bool) -> bool:
        if skill_id not in self.allowed_ids:
            return False
        return self.inner.toggle_skill(skill_id, enabled)

    def reload(self) -> None:
        self.inner.reload()


__all__ = ["AllowlistSkillLoader"]
