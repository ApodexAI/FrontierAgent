# pyright: reportWildcardImportFromLibrary=false
"""Kernel-generic event identifiers (implemented by ``agent_core.events``)."""

import sys

import agent_core.events as _implementation
from agent_core.events import *  # noqa: F403

sys.modules[__name__] = _implementation
