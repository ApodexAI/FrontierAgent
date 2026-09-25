# pyright: reportWildcardImportFromLibrary=false
"""Build a protocol-clean finalization request from a damaged history (implemented by ``agent_core.components.finalization.recovery``)."""

import sys

import agent_core.components.finalization.recovery as _implementation
from agent_core.components.finalization.recovery import *  # noqa: F403

sys.modules[__name__] = _implementation
