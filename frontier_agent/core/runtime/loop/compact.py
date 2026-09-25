# pyright: reportWildcardImportFromLibrary=false
"""Message-history compaction for the agent loop (implemented by ``agent_core.runtime.loop.compact``)."""

import sys

import agent_core.runtime.loop.compact as _implementation
from agent_core.runtime.loop.compact import *  # noqa: F403

sys.modules[__name__] = _implementation
