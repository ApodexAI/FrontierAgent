# pyright: reportWildcardImportFromLibrary=false
"""Alias (implemented by ``agent_core.components.middleware.llm.skill_injection``)."""

import sys

import agent_core.components.middleware.llm.skill_injection as _implementation
from agent_core.components.middleware.llm.skill_injection import *  # noqa: F403

sys.modules[__name__] = _implementation
