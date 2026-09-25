# pyright: reportWildcardImportFromLibrary=false
"""Skill state configuration — persists enable/disable state for skills (implemented by ``agent_core.components.skills.extensions_config``)."""

import sys

import agent_core.components.skills.extensions_config as _implementation
from agent_core.components.skills.extensions_config import *  # noqa: F403

sys.modules[__name__] = _implementation
