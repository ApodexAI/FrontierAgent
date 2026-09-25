# pyright: reportWildcardImportFromLibrary=false
"""Alias (implemented by ``agent_core.runtime.loop._streaming``)."""

import sys

import agent_core.runtime.loop._streaming as _implementation
from agent_core.runtime.loop._streaming import *  # noqa: F403

sys.modules[__name__] = _implementation
