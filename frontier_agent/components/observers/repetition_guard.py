# pyright: reportWildcardImportFromLibrary=false
"""RepetitionGuard — hint when consecutive turns repeat the same tool call (implemented by ``agent_core.components.observers.repetition_guard``)."""

import sys

import agent_core.components.observers.repetition_guard as _implementation
from agent_core.components.observers.repetition_guard import *  # noqa: F403

sys.modules[__name__] = _implementation
