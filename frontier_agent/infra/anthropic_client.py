# pyright: reportWildcardImportFromLibrary=false
"""Anthropic LLMClient — wraps :class:`anthropic.AsyncAnthropic` (implemented by ``agent_core.providers.anthropic``)."""

import sys

import agent_core.providers.anthropic as _implementation
from agent_core.providers.anthropic import *  # noqa: F403

sys.modules[__name__] = _implementation
