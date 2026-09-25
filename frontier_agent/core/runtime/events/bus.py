# pyright: reportWildcardImportFromLibrary=false
"""Async pub/sub event bus for kernel-level communication (implemented by ``agent_core.runtime.events``)."""

import sys

import agent_core.runtime.events as _implementation
from agent_core.runtime.events import *  # noqa: F403

sys.modules[__name__] = _implementation
