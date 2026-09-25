# pyright: reportWildcardImportFromLibrary=false
"""Cross-execution conversation history for caller-level sessions (implemented by ``agent_core.runtime.session_history``)."""

import sys

import agent_core.runtime.session_history as _implementation
from agent_core.runtime.session_history import *  # noqa: F403

sys.modules[__name__] = _implementation
