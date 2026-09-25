# pyright: reportWildcardImportFromLibrary=false
"""OpenAI-compatible message types — provider-agnostic chat representation (implemented by ``agent_core.messages``)."""

import sys

import agent_core.messages as _implementation
from agent_core.messages import *  # noqa: F403

sys.modules[__name__] = _implementation
