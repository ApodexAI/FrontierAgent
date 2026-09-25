# pyright: reportWildcardImportFromLibrary=false
"""Task and research request models (implemented by ``agent_core.models.task``)."""

import sys

import agent_core.models.task as _implementation
from agent_core.models.task import *  # noqa: F403

sys.modules[__name__] = _implementation
