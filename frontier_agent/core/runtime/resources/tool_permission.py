"""Tool permission context for filtering available tools."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ToolPermissionContext:
    """Permission filter for tool names.

    Supports two common patterns:
    - allowlist intersection via ``allow_names``
    - explicit denials via ``deny_names`` / ``deny_prefixes``
    """

    allow_names: frozenset[str] | None = None
    deny_names: frozenset[str] = frozenset()
    deny_prefixes: tuple[str, ...] = ()

    def blocks(self, tool_name: str) -> bool:
        lowered = tool_name.lower()
        return lowered in self.deny_names or any(
            lowered.startswith(prefix) for prefix in self.deny_prefixes
        )

    def allows(self, tool_name: str) -> bool:
        lowered = tool_name.lower()
        if lowered in self.deny_names or any(
            lowered.startswith(p) for p in self.deny_prefixes
        ):
            return False
        if self.allow_names is None:
            return True
        return lowered in self.allow_names

    def filter(self, tool_names: set[str]) -> set[str]:
        return {name for name in tool_names if self.allows(name)}

    def is_empty(self) -> bool:
        """True when this context imposes no restriction at all."""
        return (
            self.allow_names is None
            and not self.deny_names
            and not self.deny_prefixes
        )

    def merge(self, other: ToolPermissionContext) -> ToolPermissionContext:
        """Combine two contexts into a stricter one (fail-closed).

        - ``deny_names`` / ``deny_prefixes``: union — anything either side
          blocks stays blocked.
        - ``allow_names``: when both sides constrain the allowlist, the
          result is their **intersection** (a tool must clear both). When
          only one side constrains it, that one wins; when neither does,
          the result stays unconstrained (``None``).

        Used to layer a per-run / per-request policy under a profile-level
        policy without either silently overriding the other.
        """
        if self.allow_names is None:
            allow = other.allow_names
        elif other.allow_names is None:
            allow = self.allow_names
        else:
            allow = self.allow_names & other.allow_names
        return ToolPermissionContext(
            allow_names=allow,
            deny_names=self.deny_names | other.deny_names,
            deny_prefixes=tuple(
                dict.fromkeys((*self.deny_prefixes, *other.deny_prefixes))
            ),
        )

    @classmethod
    def from_iterables(
        cls,
        *,
        allow_names: set[str] | list[str] | tuple[str, ...] | None = None,
        deny_names: set[str] | list[str] | tuple[str, ...] = (),
        deny_prefixes: tuple[str, ...] | list[str] = (),
    ) -> ToolPermissionContext:
        normalized_allow = None
        if allow_names is not None:
            normalized_allow = frozenset(name.lower() for name in allow_names)
        return cls(
            allow_names=normalized_allow,
            deny_names=frozenset(name.lower() for name in deny_names),
            deny_prefixes=tuple(prefix.lower() for prefix in deny_prefixes),
        )


def from_config_map(mapping: Any | None) -> ToolPermissionContext:
    """Build a ToolPermissionContext from a ``{tool_name: bool}`` config map.

    This is the shape callers inject via the Agent Protocol ``Request.config``
    (``config: {tools: {web_search: false, web_fetch: false}}``) or an agent_team
    profile's ``tools:`` block:

    - ``name: false`` → the tool is denied (added to ``deny_names``).
    - ``name: true``  → the tool is explicitly allowed; if **any** entry is
      ``true`` the result becomes an allowlist (``allow_names``), so only the
      tools flagged ``true`` survive. Pure-``false`` maps stay a denylist and
      leave every other tool untouched.

    Non-bool / non-dict input yields an empty (no-op) context so a malformed
    config can never accidentally open or close tools in surprising ways.
    Only a bare ``True`` / ``False`` counts — a quoted ``"false"`` or ``0`` /
    ``1`` is NOT a bool and is ignored (with a warning), so it can never
    silently fail to disable a tool the operator thought they had switched
    off.
    """
    if not isinstance(mapping, dict) or not mapping:
        return ToolPermissionContext()

    allow: set[str] = set()
    deny: set[str] = set()
    for name, enabled in mapping.items():
        if not isinstance(name, str):
            continue
        if enabled is True:
            allow.add(name)
        elif enabled is False:
            deny.add(name)
        else:
            # Surface — never silently drop — a non-bool value. A quoted
            # ``"false"`` or ``0`` leaves the tool UNCHANGED; without this
            # an operator who wrote ``web_search: "false"`` would believe
            # they disabled web access when they did not.
            logger.warning(
                "Tool policy entry %r=%r ignored: value must be a bare bool "
                "(true/false), got %s — the tool is left UNCHANGED.",
                name, enabled, type(enabled).__name__,
            )
    return ToolPermissionContext.from_iterables(
        allow_names=allow or None,
        deny_names=deny,
    )


def from_execution_policy(policy: Any | None) -> ToolPermissionContext:
    """Build a ToolPermissionContext from a node/workflow execution policy.

    Accepts any object exposing ``allow_tools`` / ``deny_tools`` /
    ``deny_tool_prefixes`` attributes so kernel code can consume policy data
    without importing higher-level pipeline models directly.
    """
    if policy is None:
        return ToolPermissionContext()

    allow_tools = getattr(policy, "allow_tools", None)
    deny_tools = getattr(policy, "deny_tools", ())
    deny_prefixes = getattr(policy, "deny_tool_prefixes", ())
    return ToolPermissionContext.from_iterables(
        allow_names=allow_tools,
        deny_names=deny_tools,
        deny_prefixes=deny_prefixes,
    )
