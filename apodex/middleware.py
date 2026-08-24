from pathlib import Path
from typing import Any

# Skills live in the harness repo (plugins/skills/), not the user's cwd the
# terminal chdirs into — so the loader needs an explicit absolute path.
_SKILLS_DIR = Path(__file__).resolve().parents[1] / "plugins" / "skills"


def _wrap_skills_llm(llm: Any, skill_ids: list[str]) -> Any:
    """Wrap ``llm`` with skill injection for the given skill ids.

    Reuses the framework's ``SkillInjectionMiddleware`` (the pattern in
    ``workflows/apodex_react_skills``): skill metadata is injected into the
    system message and the model loads full ``SKILL.md`` bodies via ``read_text``
    — so it works regardless of the workspace file-read gate. ``["*"]`` enables
    all discovered skills. Best-effort: returns ``llm`` unchanged on any failure.
    """
    ids = set(skill_ids)
    if not ids:
        return llm
    try:
        import sys

        from frontier_agent.components.middleware.llm import (
            LLMMiddlewareChain,
            SkillInjectionMiddleware,
        )
        from frontier_agent.components.skills import (
            AllowlistSkillLoader,
            FileSystemSkillLoader,
        )
        skills_dir = getattr(sys.modules.get("apodex.session"), "_SKILLS_DIR", _SKILLS_DIR)
        loader = FileSystemSkillLoader(skill_dirs=[skills_dir])
        loader.discover()
        if "*" in ids:
            ids = {s.skill_id for s in loader.list_skills()}
        if not ids:
            return llm
        chain = LLMMiddlewareChain()
        chain.add(SkillInjectionMiddleware(
            skill_loader=AllowlistSkillLoader(loader, frozenset(ids)),
            role_filter=lambda _: True,
        ))
        return chain.wrap_llm(llm, role_id="apodex")
    except Exception:  # pragma: no cover - never let skills wiring break a run
        return llm
