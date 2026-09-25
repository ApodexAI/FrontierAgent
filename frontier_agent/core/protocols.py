# pyright: reportWildcardImportFromLibrary=false
"""Shared structural contracts used across runtime components (implemented by ``agent_core.protocols``)."""

import sys

import agent_core.protocols as _implementation
from agent_core.protocols import *  # noqa: F403

sys.modules[__name__] = _implementation
