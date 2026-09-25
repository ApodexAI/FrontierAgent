# pyright: reportWildcardImportFromLibrary=false
"""Skill data models — Pydantic types for skill configuration and metadata (implemented by ``agent_core.components.skills.config``)."""

import sys

import agent_core.components.skills.config as _implementation
from agent_core.components.skills.config import *  # noqa: F403

sys.modules[__name__] = _implementation
