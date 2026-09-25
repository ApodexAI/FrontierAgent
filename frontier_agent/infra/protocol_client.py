# pyright: reportWildcardImportFromLibrary=false
"""Per-profile ``protocol`` → native LLM client selection (implemented by ``agent_core.providers.protocol_client``)."""

import sys

import agent_core.providers.protocol_client as _implementation
from agent_core.providers.protocol_client import *  # noqa: F403

sys.modules[__name__] = _implementation
