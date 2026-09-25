# pyright: reportWildcardImportFromLibrary=false
"""SSEObserver — passive observer that forwards loop events to EventStore (implemented by ``agent_core.components.observers.sse_observer``)."""

import sys

import agent_core.components.observers.sse_observer as _implementation
from agent_core.components.observers.sse_observer import *  # noqa: F403

sys.modules[__name__] = _implementation
