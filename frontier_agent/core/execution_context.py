# pyright: reportWildcardImportFromLibrary=false
"""Shared execution context carrier for phase, LLM, and tool calls (implemented by ``agent_core.execution_context``)."""

import sys

import agent_core.execution_context as _implementation
from agent_core.execution_context import *  # noqa: F403

sys.modules[__name__] = _implementation
