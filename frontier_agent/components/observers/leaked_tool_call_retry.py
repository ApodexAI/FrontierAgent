# pyright: reportWildcardImportFromLibrary=false
"""Observer that retries leaked-text tool calls with escalating temperature (implemented by ``agent_core.components.observers.leaked_tool_call_retry``)."""

import sys

import agent_core.components.observers.leaked_tool_call_retry as _implementation
from agent_core.components.observers.leaked_tool_call_retry import *  # noqa: F403

sys.modules[__name__] = _implementation
