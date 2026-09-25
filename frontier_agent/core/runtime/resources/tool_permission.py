# pyright: reportWildcardImportFromLibrary=false
"""Tool permission context for filtering available tools (implemented by ``agent_core.runtime.tool_permission``)."""

import sys

import agent_core.runtime.tool_permission as _implementation
from agent_core.runtime.tool_permission import *  # noqa: F403

sys.modules[__name__] = _implementation
