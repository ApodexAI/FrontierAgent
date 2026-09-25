# pyright: reportWildcardImportFromLibrary=false
"""Alias (implemented by ``agent_core.runtime.loop._runaway``)."""

import sys

import agent_core.runtime.loop._runaway as _implementation
from agent_core.runtime.loop._runaway import *  # noqa: F403

sys.modules[__name__] = _implementation
