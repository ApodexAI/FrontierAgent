# pyright: reportWildcardImportFromLibrary=false
"""NodeContext — minimal facade between a DAG node and the framework (implemented by ``agent_core.models.node_context``)."""

import sys

import agent_core.models.node_context as _implementation
from agent_core.models.node_context import *  # noqa: F403

sys.modules[__name__] = _implementation
