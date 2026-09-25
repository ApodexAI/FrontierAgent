# pyright: reportWildcardImportFromLibrary=false
"""Pluggable message trimmers for sub-agent session history (implemented by ``agent_core.runtime.loop.message_trimmer``)."""

import sys

import agent_core.runtime.loop.message_trimmer as _implementation
from agent_core.runtime.loop.message_trimmer import *  # noqa: F403

sys.modules[__name__] = _implementation
