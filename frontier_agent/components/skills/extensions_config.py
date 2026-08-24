"""Skill state configuration — persists enable/disable state for skills.

Simplified from FrontierAgent's MCP extensions config — only skill state management,
no MCP server configuration.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, PrivateAttr

logger = logging.getLogger(__name__)

_CONFIG_FILENAMES = ["extensions_config.json", "mcp_config.json"]
_ENV_VAR = "FRONTIER_AGENT_EXTENSIONS_CONFIG_PATH"


def _find_config_file() -> Path | None:
    """Search for extensions config in standard locations."""
    env_path = os.getenv(_ENV_VAR)
    if env_path:
        p = Path(env_path)
        if p.is_file():
            return p

    for directory in [Path.cwd(), Path.cwd().parent]:
        for name in _CONFIG_FILENAMES:
            p = directory / name
            if p.is_file():
                return p

    return None


class SkillStateConfig(BaseModel):
    """Enable/disable state for a skill."""
    enabled: bool = True


class ExtensionsConfig(BaseModel):
    """Skill state configuration (loaded from extensions_config.json)."""

    skills: dict[str, SkillStateConfig] = Field(default_factory=dict)
    _file_path: Path | None = PrivateAttr(default=None)
    _file_mtime: float = PrivateAttr(default=0.0)

    model_config = {"populate_by_name": True}

    @classmethod
    def from_file(cls, config_path: str | Path | None = None) -> ExtensionsConfig:
        """Load config from JSON file with environment variable resolution."""
        resolved = Path(config_path) if config_path else _find_config_file()

        if resolved is None or not resolved.is_file():
            logger.debug("No extensions config found — using empty defaults")
            return cls()

        try:
            with open(resolved, encoding="utf-8") as f:
                data = json.load(f)
            _resolve_env_variables(data)
            logger.info("Loaded extensions config from %s", resolved)
            instance = cls.model_validate(data)
            instance._file_path = resolved
            with contextlib.suppress(OSError):
                instance._file_mtime = resolved.stat().st_mtime
            return instance
        except Exception as e:
            logger.warning("Failed to load extensions config %s: %s", resolved, e)
            return cls()

    def has_changed(self) -> bool:
        """Return True if the backing file has been modified since load."""
        if self._file_path is None or not self._file_path.is_file():
            return False
        try:
            return self._file_path.stat().st_mtime > self._file_mtime
        except OSError:
            return False

    def is_skill_enabled(self, skill_name: str) -> bool:
        """Check if a skill is enabled (default: True if not listed)."""
        state = self.skills.get(skill_name)
        return state.enabled if state else True


def _resolve_env_variables(obj: Any) -> Any:
    """Recursively replace $VAR_NAME with environment variable values."""
    if isinstance(obj, str) and obj.startswith("$"):
        var_name = obj[1:]
        value = os.getenv(var_name, "")
        if not value:
            logger.debug("Env var %s not set, using empty string", var_name)
        return value
    elif isinstance(obj, dict):
        for key in obj:
            obj[key] = _resolve_env_variables(obj[key])
        return obj
    elif isinstance(obj, list):
        return [_resolve_env_variables(item) for item in obj]
    return obj
