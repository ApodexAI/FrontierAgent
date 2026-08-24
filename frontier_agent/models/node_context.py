"""NodeContext — minimal facade between a DAG node and the framework."""

from __future__ import annotations

import logging
from collections.abc import Callable

from frontier_agent.models.pipeline_spec import NodeDefinition

logger = logging.getLogger(__name__)


class NodeContext:
    """Identity-only context object passed to every wrapped node."""

    def __init__(
        self,
        node_def: NodeDefinition,
        task_id_getter: Callable[[], str],
    ) -> None:
        self._node_def = node_def
        self._task_id_getter = task_id_getter

    @property
    def node_id(self) -> str:
        return self._node_def.node_id

    @property
    def role_id(self) -> str:
        return self._node_def.role_id

    @property
    def task_id(self) -> str:
        return self._task_id_getter()


# Alias for callers that import ``DefaultNodeContext``; ``NodeContext``
# is the single concrete implementation.
DefaultNodeContext = NodeContext
