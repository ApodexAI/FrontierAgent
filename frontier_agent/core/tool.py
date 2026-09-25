# pyright: reportWildcardImportFromLibrary=false
"""Tool definitions — function + OpenAI function-schema (implemented by ``agent_core.tool``)."""

import sys

import agent_core.tool as _implementation
from agent_core.tool import *  # noqa: F403

sys.modules[__name__] = _implementation
