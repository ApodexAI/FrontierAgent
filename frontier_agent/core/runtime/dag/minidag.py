# pyright: reportWildcardImportFromLibrary=false
"""MiniDAG — lightweight DAG execution engine replacing LangGraph (implemented by ``agent_core.runtime.dag.minidag``)."""

import sys

import agent_core.runtime.dag.minidag as _implementation
from agent_core.runtime.dag.minidag import *  # noqa: F403

sys.modules[__name__] = _implementation
