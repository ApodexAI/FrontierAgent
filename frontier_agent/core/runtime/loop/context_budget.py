# pyright: reportWildcardImportFromLibrary=false
"""Token estimation and text truncation for context compression (implemented by ``agent_core.runtime.loop.context_budget``)."""

import sys

import agent_core.runtime.loop.context_budget as _implementation
from agent_core.runtime.loop.context_budget import *  # noqa: F403

sys.modules[__name__] = _implementation
