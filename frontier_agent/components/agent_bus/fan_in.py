# pyright: reportWildcardImportFromLibrary=false
"""Shared report-formatting helpers for sub-agent fan-in paths (implemented by ``agent_core.components.agent_bus.fan_in``)."""

import sys

import agent_core.components.agent_bus.fan_in as _implementation
from agent_core.components.agent_bus.fan_in import *  # noqa: F403

sys.modules[__name__] = _implementation
