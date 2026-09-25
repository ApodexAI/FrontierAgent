# pyright: reportWildcardImportFromLibrary=false
"""LLM client contracts — provider-agnostic chat completion interface (implemented by ``agent_core.llm``)."""

import sys

import agent_core.llm as _implementation
from agent_core.llm import *  # noqa: F403

sys.modules[__name__] = _implementation
