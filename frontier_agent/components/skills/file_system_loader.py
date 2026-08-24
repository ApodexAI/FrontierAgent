"""Filesystem-backed implementation of the ``SkillLoader`` Protocol.

Scans skill directories for SKILL.md files, parses YAML frontmatter,
and serves skill content for agent prompt injection.

Skill data lives under ``plugins/skills/`` (the default search path here).

References:
- DeerFlow: skills/loader.py (filesystem scan + frontmatter parsing)
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

from frontier_agent.components.skills.config import SkillConfig
from frontier_agent.components.skills.extensions_config import ExtensionsConfig

logger = logging.getLogger(__name__)

# ── YAML frontmatter parsing ─────────────────────────────────────────────

_FRONTMATTER_RE = re.compile(
    r"^---\s*\n(.*?)\n---\s*\n(.*)",
    re.DOTALL,
)


def _parse_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    """Parse YAML frontmatter from a SKILL.md file.

    Returns (metadata_dict, body_content).
    Uses PyYAML for robust parsing (supports quoted values, multi-line, etc.).
    """
    match = _FRONTMATTER_RE.match(text)
    if not match:
        return {}, text

    front = match.group(1)
    body = match.group(2).strip()

    try:
        import yaml
        metadata = yaml.safe_load(front)
        if not isinstance(metadata, dict):
            metadata = {}
    except Exception:
        # Fallback: return empty metadata on parse error
        metadata = {}

    return metadata, body


def _list_field(metadata: dict[str, Any], key: str) -> list:
    """Return ``metadata[key]`` if it is a list; ``[]`` otherwise.

    Frontmatter sometimes ships these fields as scalars or strings; the
    loader treats anything non-list as missing rather than raising.
    """
    value = metadata.get(key)
    return value if isinstance(value, list) else []


def _collect_files(directory: Path) -> list[str]:
    """Return absolute paths of regular files directly under ``directory``."""
    if not directory.is_dir():
        return []
    return [str(f) for f in directory.iterdir() if f.is_file()]


# ── Skill loader ─────────────────────────────────────────────────────────


class FileSystemSkillLoader:
    """Loads skills from filesystem directories.

    Implements ``frontier_agent.core.protocols.SkillLoader``. Scans configured
    skill directories for SKILL.md files, parses their frontmatter and
    content, and manages enable/disable state.
    """

    def __init__(
        self,
        skill_dirs: list[str | Path] | None = None,
        extensions_config: ExtensionsConfig | None = None,
    ) -> None:
        self._skill_dirs = [
            Path(d) for d in (skill_dirs or [Path.cwd() / "plugins" / "skills"])
        ]
        self._extensions_config = extensions_config or ExtensionsConfig.from_file()
        self._skills: dict[str, SkillConfig] = {}
        self._discovered = False

    def discover(self) -> dict[str, SkillConfig]:
        """Scan skill directories and discover all skills.

        Returns dict mapping skill_id -> SkillConfig.
        """
        if self._discovered:
            return self._skills

        for skill_dir in self._skill_dirs:
            if not skill_dir.is_dir():
                logger.debug("Skill directory not found: %s", skill_dir)
                continue

            for entry in sorted(skill_dir.iterdir()):
                if not entry.is_dir():
                    continue
                skill_md = entry / "SKILL.md"
                if not skill_md.is_file():
                    continue

                try:
                    skill = self._load_skill(entry)
                    self._skills[skill.skill_id] = skill
                    logger.debug("Discovered skill: %s", skill.name)
                except Exception as e:
                    logger.warning("Failed to load skill from %s: %s", entry, e)

        self._discovered = True
        logger.info("Discovered %d skills", len(self._skills))
        return self._skills

    def _load_skill(self, skill_dir: Path) -> SkillConfig:
        """Load a single skill from its directory."""
        skill_md = skill_dir / "SKILL.md"
        text = skill_md.read_text(encoding="utf-8")
        metadata, body = _parse_frontmatter(text)

        skill_id = skill_dir.name

        return SkillConfig(
            skill_id=skill_id,
            name=metadata.get("name", skill_id),
            description=metadata.get("description", ""),
            version=metadata.get("version", "1.0.0"),
            author=metadata.get("author", ""),
            license=metadata.get("license", ""),
            tags=_list_field(metadata, "tags"),
            allowed_tools=_list_field(metadata, "allowed-tools"),
            metadata={k: v for k, v in metadata.items()
                      if k not in ("name", "description", "version", "author",
                                   "license", "tags", "allowed-tools")},
            content=body,
            root_dir=str(skill_dir),
            scripts=_collect_files(skill_dir / "scripts"),
            resources=_collect_files(skill_dir / "resources"),
            enabled=self._extensions_config.is_skill_enabled(skill_id),
        )

    # ── Queries ───────────────────────────────────────────────────────

    def list_skills(self) -> list[SkillConfig]:
        """Return all discovered skills (sorted by skill_id for stable order)."""
        if not self._discovered:
            self.discover()
        return sorted(self._skills.values(), key=lambda s: s.skill_id)

    def get_skill(self, skill_id: str) -> SkillConfig | None:
        """Get a specific skill by ID."""
        if not self._discovered:
            self.discover()
        return self._skills.get(skill_id)

    def get_enabled_skills(self) -> list[SkillConfig]:
        """Return only enabled skills.

        Auto-reloads extensions config if the backing file has changed.
        """
        if self._extensions_config.has_changed():
            logger.info("Extensions config changed on disk — reloading skill state")
            self._extensions_config = ExtensionsConfig.from_file()
            # Re-apply enabled state to already-discovered skills
            for skill in self._skills.values():
                skill.enabled = self._extensions_config.is_skill_enabled(skill.skill_id)
        return [s for s in self.list_skills() if s.enabled]

    # ── Mutations ─────────────────────────────────────────────────────

    def toggle_skill(self, skill_id: str, enabled: bool) -> bool:
        """Enable or disable a skill. Returns True if skill exists."""
        skill = self._skills.get(skill_id)
        if skill is None:
            return False
        skill.enabled = enabled
        return True

    def reload(self) -> None:
        """Force re-discovery of skills."""
        self._skills.clear()
        self._discovered = False
        self._extensions_config = ExtensionsConfig.from_file()
        self.discover()
