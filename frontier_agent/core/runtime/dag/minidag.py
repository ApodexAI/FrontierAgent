"""MiniDAG — lightweight DAG execution engine replacing LangGraph."""

from __future__ import annotations

import logging
import typing
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

logger = logging.getLogger(__name__)

# ── Type aliases ────────────────────────────────────────────────────────

NodeFn = Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]
ConditionFn = Callable[[dict[str, Any]], str]

# Sentinel for END
END = "__END__"

# ── Reducer extraction ──────────────────────────────────────────────────


def extract_reducers(state_type: type) -> dict[str, Callable]:
    """Extract reducer functions from Annotated type hints.

    Reads TypedDict annotations like `Annotated[list[str], operator.add]`
    and returns {field_name: operator.add} for use in state merging.
    """
    reducers: dict[str, Callable] = {}
    try:
        hints = typing.get_type_hints(state_type, include_extras=True)
    except Exception:
        return reducers

    for field_name, hint in hints.items():
        if hasattr(hint, "__metadata__"):
            for meta in hint.__metadata__:
                if callable(meta):
                    reducers[field_name] = meta
                    break
    return reducers


# ── MiniDAG topology ────────────────────────────────────────────────────


@dataclass
class MiniDAG:
    """Directed acyclic graph of async node functions.

    Builder API is intentionally compatible with LangGraph's StateGraph
    for drop-in replacement.
    """

    nodes: dict[str, NodeFn] = field(default_factory=dict)
    edges: dict[str, str | None] = field(default_factory=dict)
    conditional_edges: dict[str, tuple[ConditionFn, dict[str, str | None]]] = field(
        default_factory=dict,
    )
    entry_point: str = ""
    reducers: dict[str, Callable] = field(default_factory=dict)

    def add_node(self, name: str, fn: NodeFn) -> None:
        """Register a node function."""
        self.nodes[name] = fn

    def add_edge(self, from_node: str, to_node: str | None) -> None:
        """Add an unconditional edge. to_node=None means END."""
        self.edges[from_node] = to_node

    def add_conditional_edges(
        self,
        from_node: str,
        condition: ConditionFn,
        route_map: dict[str, str | None],
    ) -> None:
        """Add a conditional edge with routing table.

        condition(state) returns a key in route_map.
        route_map values: node name or None (END).
        """
        self.conditional_edges[from_node] = (condition, route_map)

    def set_entry_point(self, name: str) -> None:
        """Set the starting node."""
        self.entry_point = name

    def set_reducers(self, reducers: dict[str, Callable]) -> None:
        """Set field-level reducers for state merging."""
        self.reducers = reducers

    def validate(self) -> list[str]:
        """Structural validation. Returns list of error messages (empty = valid)."""
        errors: list[str] = []

        if not self.entry_point:
            errors.append("No entry point set")
        elif self.entry_point not in self.nodes:
            errors.append(f"Entry point '{self.entry_point}' not found in nodes")

        # Check edges reference existing nodes
        for from_node, to_node in self.edges.items():
            if from_node not in self.nodes:
                errors.append(f"Edge from unknown node '{from_node}'")
            if to_node is not None and to_node not in self.nodes:
                errors.append(f"Edge to unknown node '{to_node}'")

        # Check conditional edges
        for from_node, (_, route_map) in self.conditional_edges.items():
            if from_node not in self.nodes:
                errors.append(f"Conditional edge from unknown node '{from_node}'")
            for target in route_map.values():
                if target is not None and target not in self.nodes:
                    errors.append(f"Conditional route to unknown node '{target}'")

        # Check no node has both static and conditional edges
        for node_name in self.edges:
            if node_name in self.conditional_edges:
                errors.append(
                    f"Node '{node_name}' has both static and conditional edges"
                )

        # Check reachability from entry point
        if self.entry_point and self.entry_point in self.nodes:
            reachable = self._reachable_from(self.entry_point)
            for name in self.nodes:
                if name not in reachable:
                    errors.append(f"Node '{name}' is unreachable from entry point")

        return errors

    def _reachable_from(self, start: str) -> set[str]:
        """BFS reachability from a starting node."""
        from collections import deque
        visited: set[str] = set()
        queue: deque[str] = deque([start])
        while queue:
            node = queue.popleft()
            if node in visited:
                continue
            visited.add(node)
            # Static edges
            if node in self.edges:
                target = self.edges[node]
                if target is not None and target not in visited:
                    queue.append(target)
            # Conditional edges
            if node in self.conditional_edges:
                _, route_map = self.conditional_edges[node]
                for target in route_map.values():
                    if target is not None and target not in visited:
                        queue.append(target)
        return visited

    def compile(self, checkpointer: Any | None = None) -> MiniDAGRunner:
        """Freeze topology and return an executor."""
        errors = self.validate()
        if errors:
            raise ValueError(f"Invalid DAG: {'; '.join(errors)}")
        return MiniDAGRunner(dag=self, checkpointer=checkpointer)


# ── MiniDAGRunner executor ──────────────────────────────────────────────


# Maximum nodes to execute before forced termination (prevents infinite loops)
_MAX_STEPS = 100


class MiniDAGRunner:
    """Executes a frozen MiniDAG topology.

    Drop-in replacement for LangGraph's CompiledStateGraph.
    Supports astream(), aget_state(), aget_state_history().
    """

    def __init__(self, dag: MiniDAG, checkpointer: Any | None = None) -> None:
        self._dag = dag
        self._checkpointer = checkpointer
        # Runtime state (per thread_id)
        self._states: dict[str, dict[str, Any]] = {}

    @property
    def current_state(self) -> dict[str, Any] | None:
        """Access the most recently executed state (for debugging)."""
        if self._states:
            return next(iter(reversed(self._states.values())))
        return None

    async def astream(
        self,
        input_data: dict[str, Any] | None = None,
        *,
        config: dict[str, Any] | None = None,
        stream_mode: str = "updates",
    ) -> AsyncIterator[dict[str, dict[str, Any]]]:
        """Execute DAG nodes sequentially, yielding {node_name: delta} after each.

        Args:
            input_data: Initial state dict (or None to resume from checkpoint).
            config: Must contain {"configurable": {"thread_id": "..."}}.
            stream_mode: Only "updates" is supported (matches LangGraph).

        Yields:
            {node_name: state_delta} after each node completes.
        """
        thread_id = self._get_thread_id(config)

        # Load from checkpoint or initialize
        state, resume_node = await self._load_or_init(thread_id, input_data)
        current_node = resume_node or self._dag.entry_point

        step = 0
        while current_node is not None and step < _MAX_STEPS:
            step += 1
            node_fn = self._dag.nodes[current_node]

            logger.debug("miniDAG [%s] executing node '%s' (step %d)", thread_id, current_node, step)

            # Execute node
            try:
                delta = await node_fn(state)
            except Exception:
                logger.exception("miniDAG [%s] node '%s' failed", thread_id, current_node)
                raise

            if delta is None:
                delta = {}

            # Merge delta into state
            state = self._merge_state(state, delta)

            # Route to next node
            next_node = self._resolve_next(current_node, state)

            # Save checkpoint
            if self._checkpointer:
                await self._checkpointer.save(
                    thread_id=thread_id,
                    node_name=current_node,
                    state=state,
                    next_node=next_node,
                )

            # Cache state for aget_state
            self._states[thread_id] = state

            # Yield in LangGraph's stream format
            yield {current_node: delta}

            current_node = next_node

        if step >= _MAX_STEPS:
            logger.error("miniDAG [%s] hit max steps (%d), forcing termination", thread_id, _MAX_STEPS)

    async def aget_state(
        self, config: dict[str, Any],
    ) -> SimpleNamespace:
        """Get current state snapshot. Compatible with LangGraph's snapshot.values pattern."""
        thread_id = self._get_thread_id(config)

        # Try in-memory cache first
        if thread_id in self._states:
            return SimpleNamespace(values=dict(self._states[thread_id]))

        # Try checkpoint
        if self._checkpointer:
            result = await self._checkpointer.load_latest(thread_id)
            if result:
                state, _, _ = result
                return SimpleNamespace(values=state)

        return SimpleNamespace(values={})

    async def aget_state_history(
        self, config: dict[str, Any],
    ) -> AsyncIterator[SimpleNamespace]:
        """Iterate checkpoint history. Each entry has .values and .metadata."""
        thread_id = self._get_thread_id(config)

        if self._checkpointer:
            history = await self._checkpointer.load_history(thread_id)
            for entry in history:
                yield SimpleNamespace(
                    values=entry.get("state", {}),
                    metadata={
                        "node_name": entry.get("node_name", ""),
                        "checkpoint_id": entry.get("checkpoint_id", ""),
                        "created_at": entry.get("created_at", ""),
                    },
                )

    # ── Internal helpers ────────────────────────────────────────────────

    def _get_thread_id(self, config: dict[str, Any] | None) -> str:
        return (config or {}).get("configurable", {}).get("thread_id", "default")

    async def _load_or_init(
        self,
        thread_id: str,
        input_data: dict[str, Any] | None,
    ) -> tuple[dict[str, Any], str | None]:
        """Load state from checkpoint or initialize from input_data.

        Returns (state, resume_node_or_None).
        """
        # Try checkpoint for resume
        if self._checkpointer:
            result = await self._checkpointer.load_latest(thread_id)
            if result:
                state, _completed_node, next_node = result
                if next_node:
                    logger.info(
                        "miniDAG [%s] resuming from checkpoint at node '%s'",
                        thread_id, next_node,
                    )
                    return state, next_node

        # Fresh start
        return dict(input_data or {}), None

    def _merge_state(
        self,
        current: dict[str, Any],
        delta: dict[str, Any],
    ) -> dict[str, Any]:
        """Merge node output into current state.

        Fields with reducers (e.g. operator.add for lists) use the reducer.
        All other fields use last-write-wins.
        """
        merged = dict(current)
        for key, value in delta.items():
            if key in self._dag.reducers and key in merged:
                try:
                    merged[key] = self._dag.reducers[key](merged[key], value)
                except (TypeError, ValueError) as e:
                    logger.warning("Reducer failed for '%s': %s, using last-write", key, e)
                    merged[key] = value
            else:
                merged[key] = value
        return merged

    def _resolve_next(self, current: str, state: dict[str, Any]) -> str | None:
        """Determine the next node to execute.

        Priority: conditional edges > static edges > END (None).
        """
        # Conditional edges first
        if current in self._dag.conditional_edges:
            condition_fn, route_map = self._dag.conditional_edges[current]
            result = condition_fn(state)
            target = route_map.get(result)
            logger.debug("miniDAG condition '%s' → '%s' → target '%s'", current, result, target)
            return target

        # Static edges
        if current in self._dag.edges:
            return self._dag.edges[current]

        # No edge = END
        return None
