# pyright: reportWildcardImportFromLibrary=false
"""DynamicGraphBuilder — builds MiniDAG from PipelineSpec (implemented by ``agent_core.runtime.dag.graph_builder``)."""

import sys

import agent_core.runtime.dag.graph_builder as _implementation
from agent_core.runtime.dag.graph_builder import *  # noqa: F403

sys.modules[__name__] = _implementation
