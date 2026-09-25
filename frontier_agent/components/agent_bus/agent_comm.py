# pyright: reportWildcardImportFromLibrary=false
"""Persist typed inter-agent messages and optionally publish them live (implemented by ``agent_core.components.agent_bus.agent_comm``)."""

import sys

import agent_core.components.agent_bus.agent_comm as _implementation
from agent_core.components.agent_bus.agent_comm import *  # noqa: F403

sys.modules[__name__] = _implementation
