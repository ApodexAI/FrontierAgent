# pyright: reportWildcardImportFromLibrary=false
"""Record bounded, JSON-safe tool-call previews in ``react_steps`` (implemented by ``agent_core.components.observers.react_step_tracker``)."""

import sys

import agent_core.components.observers.react_step_tracker as _implementation
from agent_core.components.observers.react_step_tracker import *  # noqa: F403

sys.modules[__name__] = _implementation
