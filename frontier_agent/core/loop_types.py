# pyright: reportWildcardImportFromLibrary=false
"""Loop type contracts for the agent-loop engine (implemented by ``agent_core.loop_types``)."""

import sys

import agent_core.loop_types as _implementation
from agent_core.loop_types import *  # noqa: F403
from agent_core.loop_types import (  # not in __all__; named for static checkers
    wall_deadline_remaining_s as wall_deadline_remaining_s,
)

sys.modules[__name__] = _implementation
