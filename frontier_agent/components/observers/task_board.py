# pyright: reportWildcardImportFromLibrary=false
"""Task-board reminders (implemented by ``agent_core.components.observers.task_board``)."""

import sys

import agent_core.components.observers.task_board as _implementation
from agent_core.components.observers.task_board import *  # noqa: F403

sys.modules[__name__] = _implementation
