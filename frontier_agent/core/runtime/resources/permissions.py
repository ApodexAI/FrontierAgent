# pyright: reportWildcardImportFromLibrary=false
"""Tool-permission helpers for ResourceManager (implemented by ``agent_core.runtime.resources.permissions``)."""

import sys

import agent_core.runtime.resources.permissions as _implementation
from agent_core.runtime.resources.permissions import *  # noqa: F403

sys.modules[__name__] = _implementation
