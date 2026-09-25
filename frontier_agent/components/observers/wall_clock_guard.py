# pyright: reportWildcardImportFromLibrary=false
"""Stop sub-agent loops cleanly before their hard wall-time cancellation (implemented by ``agent_core.components.observers.wall_clock_guard``)."""

import sys

import agent_core.components.observers.wall_clock_guard as _implementation
from agent_core.components.observers.wall_clock_guard import *  # noqa: F403

sys.modules[__name__] = _implementation
