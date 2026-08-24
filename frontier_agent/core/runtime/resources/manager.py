"""Resource manager — tool permissions and LLM routing.

Tool permissions are driven by AgentRegistry (dynamic roles) and per-call
``ToolPermissionContext`` (built from ``NodeExecutionPolicy`` by pipeline
nodes).  The manager itself knows nothing about workflow phases.
"""

from __future__ import annotations

from frontier_agent.core.errors import PermissionDenied
from frontier_agent.core.llm import LLMClient
from frontier_agent.core.runtime.resources.llm import (
    resolve_base_llm_for_role,
    wrap_llm_with_middleware,
)
from frontier_agent.core.runtime.resources.permissions import get_effective_tools_for_role
from frontier_agent.core.runtime.resources.tool_permission import ToolPermissionContext
from frontier_agent.core.tool import Tool


class ResourceManager:
    """Manages tool permissions and LLM routing per agent role."""

    def __init__(
        self,
        llm: LLMClient,
        tools: dict[str, Tool],
    ) -> None:
        self._llm = llm
        self._tools = tools  # name → tool instance
        self._role_llms: dict[str, LLMClient] = {}  # cached per-role LLMs
        # Process-wide allow/deny policy layered under every per-call
        # ``permission_context``. Driven by config injection (Agent Protocol
        # ``Request.config['tools']`` / agent_team profile ``tools:``) so an
        # operator can switch a tool (e.g. ``web_search``) off for a run
        # without editing role definitions. ``None`` = no global restriction.
        self._global_policy: ToolPermissionContext | None = None

    @property
    def llm(self) -> LLMClient:
        return self._llm

    @property
    def global_tool_policy(self) -> ToolPermissionContext | None:
        """The active process-wide tool policy, or ``None`` if unset."""
        return self._global_policy

    def set_global_tool_policy(
        self, policy: ToolPermissionContext | None,
    ) -> None:
        """Install (or clear) the process-wide tool allow/deny policy.

        Callers set this unconditionally per run — passing an empty policy
        or ``None`` clears it — so a prior run's policy never leaks into the
        next on a reused manager.
        """
        if policy is not None and policy.is_empty():
            policy = None
        self._global_policy = policy

    def _effective_context(
        self, permission_context: ToolPermissionContext | None,
    ) -> ToolPermissionContext | None:
        """Layer the global policy under a per-call permission context."""
        if self._global_policy is None:
            return permission_context
        if permission_context is None:
            return self._global_policy
        return self._global_policy.merge(permission_context)

    @property
    def all_tools(self) -> dict[str, Tool]:
        return dict(self._tools)

    def get_tools_for_role(
        self,
        role_id: str,
        *,
        permission_context: ToolPermissionContext | None = None,
    ) -> list[Tool]:
        """Return the tools permitted for a role under the given permission context."""
        allowed = get_effective_tools_for_role(
            role_id=role_id,
            permission_context=self._effective_context(permission_context),
        )
        return [t for name, t in self._tools.items() if name in allowed]

    def get_tool_for_role(
        self,
        role_id: str,
        tool_name: str,
        *,
        permission_context: ToolPermissionContext | None = None,
    ) -> Tool | None:
        """Return a single permitted tool, or None if unavailable for the role."""
        if not self.check_permission(
            role_id,
            tool_name,
            permission_context=permission_context,
        ):
            return None
        return self._tools.get(tool_name)

    def get_tool_names_for_role(
        self,
        role_id: str,
        *,
        permission_context: ToolPermissionContext | None = None,
    ) -> list[str]:
        allowed = get_effective_tools_for_role(
            role_id=role_id,
            permission_context=self._effective_context(permission_context),
        )
        return [name for name in self._tools if name in allowed]

    def check_permission(
        self,
        role_id: str,
        tool_name: str,
        *,
        permission_context: ToolPermissionContext | None = None,
    ) -> bool:
        """Check if a tool is permitted for a role under the given permission context."""
        allowed = get_effective_tools_for_role(
            role_id=role_id,
            permission_context=self._effective_context(permission_context),
        )
        return tool_name in allowed

    def require_permission(
        self,
        role_id: str,
        tool_name: str,
        *,
        permission_context: ToolPermissionContext | None = None,
    ) -> None:
        if not self.check_permission(
            role_id,
            tool_name,
            permission_context=permission_context,
        ):
            raise PermissionDenied(role_id, tool_name)

    def get_llm(self, role_id: str | None = None) -> LLMClient:
        """Get LLM for a role. Uses per-role model if configured, else default.

        If an LLMMiddlewareChain is registered, wraps the returned LLM
        in an LLMProxy so all chat/stream calls pass through the chain.
        This is transparent to callers — pipeline nodes need zero changes.
        """
        base_llm = self._resolve_base_llm(role_id)
        return wrap_llm_with_middleware(
            base_llm,
            role_id=role_id or "default",
        )

    def get_raw_llm(self, role_id: str | None = None) -> LLMClient:
        """Get LLM WITHOUT middleware wrapping. Use for internal operations
        (e.g., summarization) to avoid infinite recursion."""
        return self._resolve_base_llm(role_id)

    def set_role_llm(self, role_id: str, llm: LLMClient) -> None:
        """Register a base LLM (no middleware) under ``role_id``.

        Downstream ``get_llm(role_id)`` calls return this LLM wrapped with
        middleware. Used by workflow nodes that build per-task LLMs from a
        profile (the ``agent_team`` main agent registers ``swarm_main``) so
        peer nodes in the same DAG can pick up the same LLM via
        ``ctx.stream_llm(role="swarm_main")`` instead of falling
        back to the default LLM, which may not be reachable on the
        profile's gateway / routing group.
        """
        self._role_llms[role_id] = llm

    def _resolve_base_llm(self, role_id: str | None) -> LLMClient:
        """Resolve the base LLM for a role (no middleware)."""
        return resolve_base_llm_for_role(
            default_llm=self._llm,
            role_id=role_id,
            cache=self._role_llms,
        )
