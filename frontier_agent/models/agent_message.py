# pyright: reportWildcardImportFromLibrary=false
"""AgentMessage — typed inter-agent communication protocol (implemented by ``agent_core.models.agent_message``)."""

import sys

import agent_core.models.agent_message as _implementation
from agent_core.models.agent_message import *  # noqa: F403

sys.modules[__name__] = _implementation
