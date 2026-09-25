# pyright: reportWildcardImportFromLibrary=false
"""Anthropic prompt-cache adapter — opt-in cache_control injection (implemented by ``agent_core.providers.prompt_cache``)."""

import sys

import agent_core.providers.prompt_cache as _implementation
from agent_core.providers.prompt_cache import *  # noqa: F403

sys.modules[__name__] = _implementation
