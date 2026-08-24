"""YAML-driven agent profiles for apodex.

A *profile* is everything that makes the terminal behave as one kind of agent
(coding, research, …): the LLM to use, the tools it may call, the skills to
inject, and the system prompt. Profiles are YAML files — model/base_url/tools/
skills live in the YAML; **secrets (API keys) live only in ``.env``**, pulled in
via ``config/providers.yaml`` (``llm.provider:``) or explicit ``${VAR}`` refs.

Discovery: built-in profiles ship next to this file; a user's
``~/.apodex/profiles/<name>.yaml`` overrides a built-in of the same name.

This mirrors how the shipped workflows load their profile YAMLs (see any
``workflows/<name>/profile.py`` and ``config/providers.yaml``); we reuse ``frontier_agent.infra.config._resolve_env_vars`` for ``${VAR}`` expansion
and ``frontier_agent.infra.providers.resolve_provider`` for credential injection.

The public API is ``AgentProfile`` / ``get_profile`` / ``profile_names``.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from apodex.config import ModelConfig, RuntimeConfigStatus, inspect_runtime_config
from frontier_agent.infra.providers import environment_variable_source

_PKG_DIR = Path(__file__).resolve().parent
_USER_DIR = Path(os.path.expanduser("~/.apodex/profiles"))
_TERMINAL_WORKFLOW_MODES = ("react", "agent_team")


@dataclass(frozen=True)
class AgentProfile:
    """A resolved profile: LLM config + prompt + tools + skills + observers.

    ``system_prompt`` / ``tools`` stay callables (bound at load time) so the
    session's per-run call sites — ``profile.system_prompt(cwd)`` /
    ``profile.tools()`` — are unchanged.
    """

    name: str
    description: str
    model_config: ModelConfig                  # active LLM config (default model)
    models: list[str]                          # selectable models (/model); [0] = default
    system_prompt: Callable[[str], str]        # (cwd) -> system prompt
    tools: Callable[[], list[Any]]             # () -> [Tool]
    skills: list[str]                          # skill ids; [] none, ["*"] all
    extra_observers: Callable[[list[str]], list[Any]]  # (tool_names) -> [observer]
    max_turns: int | None = None               # profile default (CLI overrides)
    provider: str = "custom"
    api_key_env: str | None = None
    base_url_env: str | None = None
    model_env: str | None = None
    path: str = ""
    # Optional native workflow selector.  Profiles without one continue to use
    # the lightweight terminal ReAct loop.
    workflow: str | None = None
    workflow_profile: str | None = None

    def runtime_config(
        self, cfg: ModelConfig, *, mode: str | None = None,
    ) -> RuntimeConfigStatus:
        """Return the safe, local preflight view for ``cfg``."""
        return inspect_runtime_config(cfg, profile=self, mode=mode)


def _robustness_observers(tool_names: list[str]) -> list[Any]:
    """Generic, reusable hardening shared by all profiles (best-effort: any
    observer that fails to import is simply skipped).

    - TextRepetitionGuard (hint-only): nudge a model that loops on the same
      output. We keep the DEFAULT (``enable_stop=False``) because this is a
      top-level interactive agent — a false-positive stop would cut a
      legitimate session short.
    - LeakedToolCallRetryObserver: recover tool calls a (weaker) model emits as
      plain text instead of structured tool_calls.
    """
    obs: list[Any] = []
    try:
        from frontier_agent.components.observers.text_repetition_guard import (
            TextRepetitionGuard,
        )
        obs.append(TextRepetitionGuard())  # hint-only (main-agent default)
    except Exception:
        pass
    try:
        from frontier_agent.components.observers.leaked_tool_call_retry import (
            LeakedToolCallRetryObserver,
        )
        obs.append(LeakedToolCallRetryObserver(tool_names=tool_names))
    except Exception:
        pass
    return obs


def _static_prompt(text: str) -> Callable[[str], str]:
    def _p(_cwd: str) -> str:
        return text
    return _p


def _prompt_builder(base: str) -> Callable[[str], str]:
    from apodex.prompts import build_research_prompt, build_system_prompt
    return {
        "coding": build_system_prompt,
        "research": build_research_prompt,
    }.get(base, build_system_prompt)


def _tool_factory(names: list[str]) -> Callable[[], list[Any]]:
    def _factory() -> list[Any]:
        from apodex.agent_tools import terminal_tool_registry
        reg = terminal_tool_registry()
        out: list[Any] = []
        for n in names:
            if n not in reg:
                raise KeyError(
                    f"unknown tool {n!r} in profile; "
                    f"available: {', '.join(sorted(reg))}"
                )
            out.append(reg[n])
        return out
    return _factory


def _model_list(llm: dict[str, Any]) -> list[str]:
    """The selectable models: ``llm.models`` (list) or the single ``llm.model``.
    First entry is the default. Duplicates removed, order preserved."""
    raw = llm.get("models")
    candidates = list(raw) if isinstance(raw, list) else []
    single = llm.get("model")
    if single:
        candidates.append(single)
    out: list[str] = []
    for m in candidates:
        s = str(m).strip()
        if s and s not in out:
            out.append(s)
    return out


def _model_env_source(llm: dict[str, Any]) -> str | None:
    raw = llm.get("models")
    if isinstance(raw, list) and raw:
        return environment_variable_source(raw[0])
    return environment_variable_source(llm.get("model"))


def _resolve_llm(
    llm: dict[str, Any], source_llm: dict[str, Any],
) -> tuple[ModelConfig, list[str], str, str | None, str | None, str | None]:
    """Build the active :class:`ModelConfig` + the selectable model list from a
    profile's (env-expanded) ``llm`` block.

    ``llm.provider`` (a ``config/providers.yaml`` name) injects api_key/base_url;
    explicit ``api_key`` / ``base_url`` (incl. ``${VAR}`` refs) win.
    """
    provider = llm.get("provider")
    provider_name = str(provider) if provider else "custom"
    api_key_env = environment_variable_source(source_llm.get("api_key"))
    base_url_env = environment_variable_source(source_llm.get("base_url"))
    if provider:
        from frontier_agent.infra.providers import provider_metadata, resolve_provider
        creds = resolve_provider(str(provider))
        metadata = provider_metadata(str(provider))
        llm.setdefault("api_key", creds.get("api_key") or "")
        llm.setdefault("base_url", creds.get("base_url"))
        if "api_key" not in source_llm:
            api_key_env = metadata.api_key_env
        if "base_url" not in source_llm:
            base_url_env = metadata.base_url_env
    models = _model_list(llm)
    cfg = ModelConfig(
        model=models[0] if models else "",  # preflight reports an actionable error
        api_key=str(llm.get("api_key") or ""),
        base_url=(str(llm["base_url"]) if llm.get("base_url") else None),
        temperature=float(llm.get("temperature", 0.0)),
        # Absent stays absent rather than becoming a default: sending an explicit
        # top_p/top_k a profile never asked for would change sampling for every
        # existing mode.
        top_p=(float(llm["top_p"]) if llm.get("top_p") is not None else None),
        top_k=(
            int((llm.get("extra_body") or {})["top_k"])
            if (llm.get("extra_body") or {}).get("top_k") is not None
            else None
        ),
        max_tokens=int(llm.get("max_tokens", 8192)),
        context_window=int(llm.get("context_window", 128_000)),
    )
    return (
        cfg,
        models,
        provider_name,
        api_key_env,
        base_url_env,
        _model_env_source(source_llm),
    )


def _profile_path(name: str) -> Path:
    for d in (_USER_DIR, _PKG_DIR):  # user dir wins over built-in
        p = d / f"{name}.yaml"
        if p.is_file():
            return p
    raise KeyError(
        f"unknown mode {name!r}; available: {', '.join(profile_names())}"
    )


def _build(name: str) -> AgentProfile:
    import yaml

    from frontier_agent.infra.config import _resolve_env_vars

    path = _profile_path(name)
    source = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    raw = _resolve_env_vars(source)
    if not isinstance(raw, dict):
        raise ValueError(f"profile {path} must be a YAML mapping")

    agent = dict(raw.get("agent") or {})
    override = str(agent.get("system_prompt") or "").strip()
    system_prompt = (
        _static_prompt(override) if override
        else _prompt_builder(str(agent.get("base_prompt") or "coding"))
    )
    max_turns = agent.get("max_turns")
    model_config, models, provider, api_key_env, base_url_env, model_env = _resolve_llm(
        dict(raw.get("llm") or {}), dict(source.get("llm") or {}),
    )
    return AgentProfile(
        name=str(raw.get("name") or name),
        description=str(raw.get("description") or ""),
        model_config=model_config,
        models=models,
        system_prompt=system_prompt,
        tools=_tool_factory([str(t) for t in (raw.get("tools") or [])]),
        skills=[str(s) for s in (raw.get("skills") or [])],
        extra_observers=_robustness_observers,
        max_turns=int(max_turns) if max_turns is not None else None,
        provider=provider,
        api_key_env=api_key_env,
        base_url_env=base_url_env,
        model_env=model_env,
        path=str(path),
        workflow=(str(raw["workflow"]) if raw.get("workflow") else None),
        workflow_profile=(str(raw["workflow_profile"]) if raw.get("workflow_profile") else None),
    )


_CACHE: dict[str, AgentProfile] = {}


def profile_names() -> list[str]:
    """All discoverable profile names (built-in + user override dir)."""
    names: set[str] = set()
    for d in (_PKG_DIR, _USER_DIR):
        if d.is_dir():
            names.update(p.stem for p in d.glob("*.yaml"))
    return sorted(names)


def terminal_mode_names() -> list[str]:
    """The deliberately small mode surface exposed by the terminal product."""
    available = set(profile_names())
    return [name for name in _TERMINAL_WORKFLOW_MODES if name in available]


def get_profile(name: str) -> AgentProfile:
    """Return the profile for ``name`` (built once, cached). Raises ``KeyError``
    for an unknown name."""
    if name not in _CACHE:
        _CACHE[name] = _build(name)
    return _CACHE[name]


__all__ = ["AgentProfile", "get_profile", "profile_names"]
