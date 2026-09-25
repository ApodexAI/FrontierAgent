# pyright: reportWildcardImportFromLibrary=false
"""Cooperative stop-signal observer for sub-agents (implemented by ``agent_core.components.observers.stop_signal_observer``)."""

import sys

import agent_core.components.observers.stop_signal_observer as _implementation
from agent_core.components.observers.stop_signal_observer import *  # noqa: F403

sys.modules[__name__] = _implementation
