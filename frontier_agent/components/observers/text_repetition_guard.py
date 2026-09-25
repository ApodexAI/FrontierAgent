# pyright: reportWildcardImportFromLibrary=false
"""Detect near-verbatim repetition across recent assistant turns (implemented by ``agent_core.components.observers.text_repetition_guard``)."""

import sys

import agent_core.components.observers.text_repetition_guard as _implementation
from agent_core.components.observers.text_repetition_guard import *  # noqa: F403

sys.modules[__name__] = _implementation
