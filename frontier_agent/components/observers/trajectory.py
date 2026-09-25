# pyright: reportWildcardImportFromLibrary=false
"""Write workflow-neutral trajectories in JSON and/or JSONL formats (implemented by ``agent_core.components.observers.trajectory``)."""

import sys

import agent_core.components.observers.trajectory as _implementation
from agent_core.components.observers.trajectory import *  # noqa: F403

sys.modules[__name__] = _implementation
