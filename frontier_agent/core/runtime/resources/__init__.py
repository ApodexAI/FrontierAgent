"""Resource management — tool permissions and LLM routing per agent role."""

from frontier_agent.core.runtime.resources.llm import (
    resolve_base_llm_for_role,
    wrap_llm_with_middleware,
)
from frontier_agent.core.runtime.resources.manager import ResourceManager
from frontier_agent.core.runtime.resources.permissions import (
    get_allowed_tools_for_role,
    get_effective_tools_for_role,
)
from frontier_agent.core.runtime.resources.tool_permission import (
    ToolPermissionContext,
    from_execution_policy,
)

__all__ = [
    "ResourceManager",
    "ToolPermissionContext",
    "from_execution_policy",
    "get_allowed_tools_for_role",
    "get_effective_tools_for_role",
    "resolve_base_llm_for_role",
    "wrap_llm_with_middleware",
]
