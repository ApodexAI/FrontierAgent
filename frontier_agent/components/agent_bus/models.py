# pyright: reportWildcardImportFromLibrary=false
"""Data models for AgentBus (implemented by ``agent_core.components.agent_bus.models``)."""

import sys

import agent_core.components.agent_bus.models as _implementation
from agent_core.components.agent_bus.models import *  # noqa: F403

sys.modules[__name__] = _implementation
