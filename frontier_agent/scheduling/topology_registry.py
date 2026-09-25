# pyright: reportWildcardImportFromLibrary=false
"""Topology factory registry — workflow contributes, kernel dispatches (implemented by ``agent_core.scheduling.topology_registry``)."""

import sys

import agent_core.scheduling.topology_registry as _implementation
from agent_core.scheduling.topology_registry import *  # noqa: F403

sys.modules[__name__] = _implementation
