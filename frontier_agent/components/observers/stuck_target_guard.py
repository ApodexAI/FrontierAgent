# pyright: reportWildcardImportFromLibrary=false
"""Detect and quarantine repeatedly failing network targets (implemented by ``agent_core.components.observers.stuck_target_guard``)."""

import sys

import agent_core.components.observers.stuck_target_guard as _implementation
from agent_core.components.observers.stuck_target_guard import *  # noqa: F403

sys.modules[__name__] = _implementation
