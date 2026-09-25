# pyright: reportWildcardImportFromLibrary=false
"""Generic tool call parser: native function-calling + JSON text fallback (implemented by ``agent_core.runtime.loop.tool_call_parser``)."""

import sys

import agent_core.runtime.loop.tool_call_parser as _implementation
from agent_core.runtime.loop.tool_call_parser import *  # noqa: F403

sys.modules[__name__] = _implementation
