# pyright: reportWildcardImportFromLibrary=false
"""Skills runtime — filesystem-backed implementation of the SkillLoader Protocol (implemented by ``agent_core.components.skills``)."""

import sys

import agent_core.components.skills as _implementation
from agent_core.components.skills import *  # noqa: F403

sys.modules[__name__] = _implementation
