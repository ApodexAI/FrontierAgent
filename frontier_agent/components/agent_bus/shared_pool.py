# pyright: reportWildcardImportFromLibrary=false
"""Shared artifact pool for parallel sub-agent execution (implemented by ``agent_core.components.agent_bus.shared_pool``)."""

import sys

import agent_core.components.agent_bus.shared_pool as _implementation
from agent_core.components.agent_bus.shared_pool import *  # noqa: F403

sys.modules[__name__] = _implementation
