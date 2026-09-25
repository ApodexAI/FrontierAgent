# pyright: reportWildcardImportFromLibrary=false
"""BudgetObserver — critical observer that tracks token usage and stops the loop (implemented by ``agent_core.components.observers.budget_observer``)."""

import sys

import agent_core.components.observers.budget_observer as _implementation
from agent_core.components.observers.budget_observer import *  # noqa: F403

sys.modules[__name__] = _implementation
