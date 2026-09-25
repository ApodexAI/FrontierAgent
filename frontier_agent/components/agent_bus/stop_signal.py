# pyright: reportWildcardImportFromLibrary=false
"""Process-wide cooperative stop-signal registry for sub-agents (implemented by ``agent_core.components.agent_bus.stop_signal``)."""

import sys

import agent_core.components.agent_bus.stop_signal as _implementation
from agent_core.components.agent_bus.stop_signal import *  # noqa: F403

sys.modules[__name__] = _implementation
