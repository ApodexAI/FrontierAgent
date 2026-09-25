# pyright: reportWildcardImportFromLibrary=false
"""Base types for FrontierAgent kernel and application layers (implemented by ``agent_core.types``)."""

import sys

import agent_core.types as _implementation
from agent_core.types import *  # noqa: F403

sys.modules[__name__] = _implementation
