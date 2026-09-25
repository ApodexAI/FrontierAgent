# pyright: reportWildcardImportFromLibrary=false
"""Guard sub-agent spawning by depth, concurrency, budget, and wall time (implemented by ``agent_core.components.agent_bus.spawn_guard``)."""

import sys

import agent_core.components.agent_bus.spawn_guard as _implementation
from agent_core.components.agent_bus.spawn_guard import *  # noqa: F403

sys.modules[__name__] = _implementation
