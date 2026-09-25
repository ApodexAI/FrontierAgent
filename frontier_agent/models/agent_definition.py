# pyright: reportWildcardImportFromLibrary=false
"""AgentDefinition — declarative, runtime-configurable agent role (implemented by ``agent_core.models.agent_definition``)."""

import sys

import agent_core.models.agent_definition as _implementation
from agent_core.models.agent_definition import *  # noqa: F403

sys.modules[__name__] = _implementation
