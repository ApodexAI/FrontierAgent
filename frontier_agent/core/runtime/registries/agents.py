# pyright: reportWildcardImportFromLibrary=false
"""AgentRegistry — dynamic agent role registration and lookup (implemented by ``agent_core.runtime.registries.agents``)."""

import sys

import agent_core.runtime.registries.agents as _implementation
from agent_core.runtime.registries.agents import *  # noqa: F403

sys.modules[__name__] = _implementation
