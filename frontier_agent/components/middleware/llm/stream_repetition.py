# pyright: reportWildcardImportFromLibrary=false
"""Stream-level repetition detector — kills degenerate LLM output loops (implemented by ``agent_core.components.middleware.llm.stream_repetition``)."""

import sys

import agent_core.components.middleware.llm.stream_repetition as _implementation
from agent_core.components.middleware.llm.stream_repetition import *  # noqa: F403

sys.modules[__name__] = _implementation
