# pyright: reportWildcardImportFromLibrary=false
"""Generic runtime helpers for AgentBus (implemented by ``agent_core.components.agent_bus.runtime``)."""

import sys

import agent_core.components.agent_bus.runtime as _implementation
from agent_core.components.agent_bus.runtime import *  # noqa: F403

sys.modules[__name__] = _implementation
