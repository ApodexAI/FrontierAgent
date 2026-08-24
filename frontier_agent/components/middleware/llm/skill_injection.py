from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from frontier_agent.components.middleware.llm.base import (
    LLMCallContext,
    LLMMiddleware,
)
from frontier_agent.core.messages import Message, system_msg, text_of

if TYPE_CHECKING:
    from frontier_agent.core.protocols import SkillLoader

logger = logging.getLogger(__name__)

RoleFilter = Callable[[str], bool]


class SkillInjectionMiddleware(LLMMiddleware):
    """Progressive skill injection — lightweight metadata only.

    Two-step loading, so unused skills cost almost nothing:
      Step 1: Inject skill names + descriptions + paths into system message (~100 tokens/skill)
      Step 2: LLM calls read_text to load full SKILL.md on demand (only when relevant)

    Whether a role receives injection is an explicit per-role product
    decision — the caller sets ``enable_skills=True`` on the role's
    AgentDefinition. Roles that opt in must also declare ``read_text``
    in ``allowed_tools`` for the LLM to actually load SKILL.md content;
    otherwise the metadata is informational only.

    Roles default to ``enable_skills=False``, so adding a new role never
    silently starts consuming skill metadata.
    """

    # Budget constants (aligned with Claude Code: MAX_LISTING_DESC_CHARS=250, ~1% context)
    _SKILL_DESC_MAX_CHARS = 250
    _SKILL_METADATA_MAX_CHARS = 8000  # ~2k tokens — fits ~20 skills at 250 chars each

    def __init__(
        self,
        skill_loader: SkillLoader | None = None,
        *,
        role_filter: RoleFilter | None = None,
    ) -> None:
        """Construct the middleware.

        ``skill_loader`` is taken via constructor injection so the
        middleware does not import a concrete loader class. If ``None``,
        the middleware falls back to looking the loader up in the
        runtime registry — kept for back-compat with code paths that
        still register a global loader.

        ``role_filter`` decides whether a given ``role_id`` receives
        injection. ``None`` falls back to the built-in fail-closed gate
        keyed on ``AgentDefinition.enable_skills`` — what a
        process-wide chain relies on. Workflow-local chains
        whose composition is already an explicit opt-in (e.g. a
        pre-filtered loader on a one-off chain) can pass
        ``lambda _: True`` to skip the gate without having to flip a
        role flag they don't own. See
        ``workflows/apodex_react_skills/nodes/main_agent.py``.
        """
        self._skill_section: str | None = None
        self._loader: SkillLoader | None = skill_loader
        self._agent_reg: Any | None = None
        self._role_filter: RoleFilter | None = role_filter

    @property
    def name(self) -> str:
        return "skill_injection"

    @classmethod
    def _truncate_description(cls, text: str) -> str:
        """Truncate a skill description to budget, breaking at word boundary.

        Always returns at most _SKILL_DESC_MAX_CHARS characters (including "…").
        """
        if len(text) <= cls._SKILL_DESC_MAX_CHARS:
            return text
        # Reserve 1 char for ellipsis
        limit = cls._SKILL_DESC_MAX_CHARS - 1
        truncated = text[:limit]
        # Break at last space to avoid cutting mid-word
        last_space = truncated.rfind(" ")
        if last_space > limit // 2:
            truncated = truncated[:last_space]
        return truncated + "…"

    def _build_skill_section(self) -> str:
        """Build lightweight skill metadata (cached after first call).

        Budget-aware: per-skill descriptions capped at _SKILL_DESC_MAX_CHARS,
        total metadata capped at _SKILL_METADATA_MAX_CHARS. Excess skills
        are omitted with a count comment (never cut mid-entry).
        """
        if self._skill_section is not None:
            return self._skill_section

        try:
            skill_loader = self._resolve_loader()
            if skill_loader is None:
                # Don't cache: a loader may be registered later (e.g.
                # delayed bootstrap, lazy plugin) and we want the next
                # call to pick it up without an explicit invalidate.
                return ""
            enabled_skills = skill_loader.get_enabled_skills()
            if not enabled_skills:
                self._skill_section = ""
                return ""

            entries_list: list[str] = []
            total_chars = 0
            for s in enabled_skills:
                desc = self._truncate_description(s.description)
                lines = [
                    f'  <skill name="{s.name}" id="{s.skill_id}">',
                    f"    <description>{desc}</description>",
                    f"    <path>plugins/skills/{s.skill_id}/SKILL.md</path>",
                ]
                if s.allowed_tools:
                    tools_str = ", ".join(s.allowed_tools)
                    lines.append(f"    <allowed-tools>{tools_str}</allowed-tools>")
                lines.append("  </skill>")
                entry = "\n".join(lines)
                if total_chars + len(entry) > self._SKILL_METADATA_MAX_CHARS:
                    omitted = len(enabled_skills) - len(entries_list)
                    entries_list.append(
                        f"  <!-- {omitted} more skill(s) omitted due to budget -->"
                    )
                    break
                entries_list.append(entry)
                total_chars += len(entry)

            entries = "\n".join(entries_list)
            self._skill_section = (
                "\n\n## Available Skills\n\n"
                "You have access to **skills** — expert workflows "
                "for specific task types. If a skill closely matches "
                "your current task, consider calling `read_text` on "
                "its `<path>` to load the workflow. Only load a skill "
                "if it is clearly relevant — do NOT load skills for "
                "simple tasks like translation, formatting, or "
                "editing existing files.\n\n"
                "If no skill matches, proceed normally with all "
                "available tools.\n\n"
                f"<available_skills>\n{entries}\n</available_skills>"
            )
        except Exception as e:
            # Transient lookup or filesystem hiccup — log and bail
            # without poisoning the cache; next call retries.
            logger.debug("SkillInjectionMiddleware: skills unavailable: %s", e)
            return ""

        return self._skill_section

    def invalidate_cache(self) -> None:
        """Call after skill toggle/install/reload to refresh the cached section."""
        self._skill_section = None

    def _resolve_loader(self) -> SkillLoader | None:
        """Return the constructor-injected loader, or fall back to the registry.

        The registry fallback exists so legacy callsites (bootstrap that
        registers a global loader, tests that use the registry) keep
        working until everything switches to constructor injection.
        Returns ``None`` if no loader is available — caller treats that
        as "skills feature unavailable".
        """
        if self._loader is not None:
            return self._loader
        try:
            from frontier_agent.components.skills import FileSystemSkillLoader
            from frontier_agent.core.runtime.registries import services as registry
            return registry.get(FileSystemSkillLoader)
        except Exception:
            return None

    def _role_can_load_skills(self, role_id: str) -> bool:
        """Return True iff this role should receive injection.

        Defers to a constructor-injected ``role_filter`` when supplied,
        else falls back to the built-in fail-closed gate keyed on
        ``AgentDefinition.enable_skills`` (unknown roles, registry
        lookup failures, or any other error map to False).
        """
        if self._role_filter is not None:
            return self._role_filter(role_id)
        agent_reg = self._agent_reg
        if agent_reg is None:
            try:
                from frontier_agent.core.runtime.registries import services as registry
                from frontier_agent.core.runtime.registries.agents import (
                    AgentRegistry,
                )
                agent_reg = registry.get(AgentRegistry)
            except Exception:
                return False
            self._agent_reg = agent_reg
        try:
            if not agent_reg.has(role_id):
                return False
            return bool(agent_reg.get(role_id).enable_skills)
        except Exception:
            return False

    async def before_llm(
        self, ctx: LLMCallContext, messages: list[Message]
    ) -> list[Message]:
        # Only inject for roles that can actually call read_text to load skills
        if not self._role_can_load_skills(ctx.role_id):
            return messages

        skill_section = self._build_skill_section()
        if not skill_section:
            return messages

        # Find first system message and append skill metadata
        for i, msg in enumerate(messages):
            if msg.get("role") == "system":
                content = text_of(msg.get("content"))
                if "<available_skills>" in content:
                    return messages  # Already injected (e.g. react_solve did it)
                messages = list(messages)
                messages[i] = system_msg(content + skill_section)
                return messages

        return messages
