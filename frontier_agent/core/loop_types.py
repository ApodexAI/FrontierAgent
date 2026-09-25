# pyright: reportWildcardImportFromLibrary=false
"""Loop type contracts for the agent-loop engine (implemented by ``agent_core.loop_types``)."""

import sys

import agent_core.loop_types as _implementation
from agent_core.loop_types import *  # noqa: F403

sys.modules[__name__] = _implementation
