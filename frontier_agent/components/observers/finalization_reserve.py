# pyright: reportWildcardImportFromLibrary=false
"""Reserve several tool-enabled turns for deliverables and final synthesis (implemented by ``agent_core.components.observers.finalization_reserve``)."""

import sys

import agent_core.components.observers.finalization_reserve as _implementation
from agent_core.components.observers.finalization_reserve import *  # noqa: F403

sys.modules[__name__] = _implementation
