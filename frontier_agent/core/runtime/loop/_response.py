# pyright: reportWildcardImportFromLibrary=false
"""Alias (implemented by ``agent_core.runtime.loop._response``)."""

import sys

import agent_core.runtime.loop._response as _implementation
from agent_core.runtime.loop._response import *  # noqa: F403

sys.modules[__name__] = _implementation
