# pyright: reportWildcardImportFromLibrary=false
"""Budget limits for the orchestrator runtime (implemented by ``agent_core.models.task_budget``)."""

import sys

import agent_core.models.task_budget as _implementation
from agent_core.models.task_budget import *  # noqa: F403

sys.modules[__name__] = _implementation
