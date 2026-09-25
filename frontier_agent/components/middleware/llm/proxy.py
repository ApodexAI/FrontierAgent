# pyright: reportWildcardImportFromLibrary=false
"""``LLMProxy`` — transparent :class:`LLMClient` wrapping a middleware chain (implemented by ``agent_core.components.middleware.llm.proxy``)."""

import sys

import agent_core.components.middleware.llm.proxy as _implementation
from agent_core.components.middleware.llm.proxy import *  # noqa: F403

sys.modules[__name__] = _implementation
