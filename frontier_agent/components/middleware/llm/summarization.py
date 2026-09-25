# pyright: reportWildcardImportFromLibrary=false
"""Alias (implemented by ``agent_core.components.middleware.llm.summarization``)."""

import sys

import agent_core.components.middleware.llm.summarization as _implementation
from agent_core.components.middleware.llm.summarization import *  # noqa: F403

sys.modules[__name__] = _implementation
