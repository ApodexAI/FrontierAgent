# pyright: reportWildcardImportFromLibrary=false
"""Warn the LLM one turn before the loop closes (implemented by ``agent_core.components.observers.last_turn_forcer``)."""

import sys

import agent_core.components.observers.last_turn_forcer as _implementation
from agent_core.components.observers.last_turn_forcer import *  # noqa: F403

sys.modules[__name__] = _implementation
