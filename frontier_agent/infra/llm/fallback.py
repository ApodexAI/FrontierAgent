# pyright: reportWildcardImportFromLibrary=false
"""``LLMFallbackChain`` — provider-failover :class:`LLMClient` wrapper (implemented by ``agent_core.providers.fallback``)."""

import sys

import agent_core.providers.fallback as _implementation
from agent_core.providers.fallback import *  # noqa: F403

sys.modules[__name__] = _implementation
