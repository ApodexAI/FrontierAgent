# pyright: reportWildcardImportFromLibrary=false
"""Token estimation for messages (implemented by ``agent_core.tokens``)."""

import sys

import agent_core.tokens as _implementation
from agent_core.tokens import *  # noqa: F403

sys.modules[__name__] = _implementation
