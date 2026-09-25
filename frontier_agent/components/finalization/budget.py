# pyright: reportWildcardImportFromLibrary=false
"""Wall-clock arithmetic for workflows with a research-only deadline (implemented by ``agent_core.components.finalization.budget``)."""

import sys

import agent_core.components.finalization.budget as _implementation
from agent_core.components.finalization.budget import *  # noqa: F403

sys.modules[__name__] = _implementation
