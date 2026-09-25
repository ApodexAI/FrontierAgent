# pyright: reportWildcardImportFromLibrary=false
"""Immutable kernel event model (implemented by ``agent_core.models.event``)."""

import sys

import agent_core.models.event as _implementation
from agent_core.models.event import *  # noqa: F403

sys.modules[__name__] = _implementation
