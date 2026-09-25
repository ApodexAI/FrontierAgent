# pyright: reportWildcardImportFromLibrary=false
"""ContextSizeGuard — pre-empt LLM context-window overflow (implemented by ``agent_core.components.observers.context_size_guard``)."""

import sys

import agent_core.components.observers.context_size_guard as _implementation
from agent_core.components.observers.context_size_guard import *  # noqa: F403

sys.modules[__name__] = _implementation
