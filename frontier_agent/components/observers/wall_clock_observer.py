# pyright: reportWildcardImportFromLibrary=false
"""WallClockDeadlineObserver — stop the loop gracefully before a hard (implemented by ``agent_core.components.observers.wall_clock_observer``)."""

import sys

import agent_core.components.observers.wall_clock_observer as _implementation
from agent_core.components.observers.wall_clock_observer import *  # noqa: F403

sys.modules[__name__] = _implementation
