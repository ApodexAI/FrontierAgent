"""Tests for the YAML profile loader (apodex.profiles).

Covers: env/provider resolution into a ModelConfig, the user-override dir,
tool-name resolution (and unknown-tool errors), and skill wrapping.
"""

from __future__ import annotations

import logging
import pathlib

import pytest

import apodex.profiles as P
from frontier_agent.infra import providers


@pytest.fixture(autouse=True)
def _fresh_caches():
    """Each test resolves profiles/providers from the current env."""
    P._CACHE.clear()
    P._workflow_tool_names.cache_clear()
    providers._reset_cache()
    yield
    P._CACHE.clear()
    P._workflow_tool_names.cache_clear()
    providers._reset_cache()


def _write_profile(dir_, name: str, body: str):
    dir_.mkdir(parents=True, exist_ok=True)
    (dir_ / f"{name}.yaml").write_text(body, encoding="utf-8")


def test_builtin_coding_profile_resolves_from_env(monkeypatch):
    """The built-in coding profile pulls model/base_url/key from .env via the
    openai provider (+ ${VAR} expansion)."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-123")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://example.test/v1")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-test")

    prof = P.get_profile("coding")
    cfg = prof.model_config
    assert cfg.model == "gpt-test"
    assert cfg.base_url == "https://example.test/v1"
    assert cfg.api_key == "sk-test-123"
    assert prof.provider == "openai"
    assert prof.api_key_env == "OPENAI_API_KEY"
    assert prof.base_url_env == "OPENAI_BASE_URL"
    assert prof.model_env == "OPENAI_MODEL"
    assert prof.path.endswith("apodex/profiles/coding.yaml")

    names = {t.name for t in prof.tools()}
    assert "file_editor_str_replace" in names and "web_search" not in names
    assert "coding" in P.profile_names() and "research" in P.profile_names()


def test_models_list_default_is_first(tmp_path, monkeypatch):
    """`llm.models` is a selectable list; the active model is the first entry."""
    monkeypatch.setattr(P, "_USER_DIR", tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-x")
    _write_profile(tmp_path, "multi", (
        "name: multi\n"
        "llm:\n  provider: openai\n  models: [alpha, beta, gamma]\n"
        "agent:\n  base_prompt: coding\ntools: [bash]\nskills: []\n"
    ))
    prof = P.get_profile("multi")
    assert prof.models == ["alpha", "beta", "gamma"]
    assert prof.model_config.model == "alpha"  # default = first


def test_single_model_key_still_works(tmp_path, monkeypatch):
    """Back-compat: a single `llm.model` becomes a one-element list."""
    monkeypatch.setattr(P, "_USER_DIR", tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-x")
    _write_profile(tmp_path, "single", (
        "name: single\n"
        "llm:\n  provider: openai\n  model: only-model\n"
        "agent:\n  base_prompt: coding\ntools: [bash]\nskills: []\n"
    ))
    prof = P.get_profile("single")
    assert prof.models == ["only-model"]
    assert prof.model_config.model == "only-model"


def test_context_window_default_when_unset(monkeypatch):
    monkeypatch.delenv("OPENAI_CONTEXT_WINDOW", raising=False)
    assert P.get_profile("coding").model_config.context_window == 128_000


def test_native_workflow_modes_are_explicit_and_use_shipped_profiles(monkeypatch):
    """The TUI must not silently emulate a team with its generic ReAct loop."""
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://example.test/v1")
    monkeypatch.setenv("OPENAI_MODEL", "test-model")

    react = P.get_profile("react")
    team = P.get_profile("agent_team")

    assert (react.workflow, react.workflow_profile) == (
        "stateful-react-agent", "tui",
    )
    assert (team.workflow, team.workflow_profile) == (
        "agent_team", "tui",
    )


def test_workflow_modes_expose_the_tools_their_workflow_profile_binds(monkeypatch):
    """``tool_names`` must be the list that runs, not a second copy of it.

    ``react``/``agent_team`` dispatch to a native workflow and never bind a
    top-level ``tools:``, so reading one would both miss the coordinator-only
    tools and report on tools the workflow does not have.
    """
    import yaml

    monkeypatch.setenv("OPENAI_API_KEY", "test")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://example.test/v1")
    monkeypatch.setenv("OPENAI_MODEL", "test-model")

    repo = pathlib.Path(P.__file__).resolve().parents[2]
    for mode in ("react", "agent_team"):
        profile = P.get_profile(mode)
        assert profile.declared_tools == ()          # nothing to drift
        workflow_yaml = (
            repo / "workflows" / profile.workflow.replace("-", "_")
            / "profiles" / f"{profile.workflow_profile}.yaml"
        )
        agent = yaml.safe_load(workflow_yaml.read_text(encoding="utf-8"))["agent"]
        expected = {
            str(t)
            for key in ("agent_tools", "main_agent_tools", "sub_agent_tools")
            for t in (agent.get(key) or [])
        }
        assert set(profile.tool_names) == expected, mode
        assert "web_search" in profile.tool_names, mode

    # Coordinator-only, and present in no apodex profile YAML: it can only have
    # come from the workflow profile.
    assert "create_subagent" in P.get_profile("agent_team").tool_names


def test_reading_workflow_tool_names_stays_quiet(monkeypatch, caplog):
    """The loader's own config warnings belong to the run, not the preflight.

    ``load_swarm_profile`` warns about provider label mismatches and empty
    keys. Dispatch loads the same profile again and logs them where the
    renderer routes them, so repeating them here would print each one twice,
    the second time onto a stderr the TUI is about to cover.
    """
    monkeypatch.setenv("OPENAI_PROVIDER", "local")
    monkeypatch.setenv("OPENAI_API_KEY", "EMPTY")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://model:30000/v1")
    monkeypatch.setenv("OPENAI_MODEL", "local-model")

    with caplog.at_level(logging.WARNING):
        assert "web_search" in P._workflow_tool_names("agent_team", "tui")
    assert caplog.records == []

    # Restored, not left off: the same load logs normally at dispatch.
    P._workflow_tool_names.cache_clear()
    with caplog.at_level(logging.WARNING):
        from workflows.agent_team.profile import load_swarm_profile
        load_swarm_profile("tui")
    assert caplog.records


def test_workflow_tool_names_never_break_startup(monkeypatch):
    """An unreadable workflow profile checks no credentials; it does not raise."""
    P._workflow_tool_names.cache_clear()
    assert P._workflow_tool_names("no-such-workflow", "tui") == ()
    assert P._workflow_tool_names("agent_team", "no-such-profile") == ()


def test_native_workflow_modes_accept_local_openai_compatible_provider(monkeypatch):
    """The GPU Compose path needs no real API key for its local SGLang server."""
    monkeypatch.setenv("OPENAI_PROVIDER", "local")
    monkeypatch.setenv("OPENAI_API_KEY", "EMPTY")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://model:30000/v1")
    monkeypatch.setenv("OPENAI_MODEL", "local-model")
    monkeypatch.setenv("OPENAI_CONTEXT_WINDOW", "32768")
    monkeypatch.setenv("OPENAI_MAX_TOKENS", "8192")

    for mode in ("react", "agent_team"):
        profile = P.get_profile(mode)
        assert profile.provider == "local"
        assert profile.runtime_config(profile.model_config, mode=mode).ok
        assert profile.model_config.api_key == "EMPTY"
        assert profile.model_config.base_url == "http://model:30000/v1"
        assert profile.model_config.model == "local-model"
        assert profile.model_config.context_window == 32768
        assert profile.model_config.max_tokens == 8192


def test_user_dir_overrides_builtin(tmp_path, monkeypatch):
    """A user profile of the same name wins over the built-in one."""
    monkeypatch.setattr(P, "_USER_DIR", tmp_path)  # bound at import → patch it
    monkeypatch.setenv("OPENAI_API_KEY", "sk-x")
    _write_profile(tmp_path, "coding", (
        "name: coding\n"
        "llm:\n  provider: openai\n  model: my-override-model\n"
        "agent:\n  base_prompt: coding\n"
        "tools: [bash, read_file]\n"
        "skills: []\n"
    ))
    prof = P.get_profile("coding")
    assert prof.model_config.model == "my-override-model"
    assert {t.name for t in prof.tools()} == {"bash", "read_file"}


def test_unknown_tool_name_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(P, "_USER_DIR", tmp_path)
    _write_profile(tmp_path, "coding", (
        "name: coding\nllm:\n  provider: openai\n  model: m\n"
        "agent:\n  base_prompt: coding\ntools: [bash, not_a_real_tool]\nskills: []\n"
    ))
    with pytest.raises(KeyError, match="not_a_real_tool"):
        P.get_profile("coding").tools()


def test_explicit_var_creds_without_provider(tmp_path, monkeypatch):
    """A profile can skip `provider:` and reference secrets via ${VAR} directly."""
    monkeypatch.setattr(P, "_USER_DIR", tmp_path)
    monkeypatch.setenv("MY_KEY", "sk-explicit")
    monkeypatch.setenv("MY_URL", "https://custom.test/v1")
    _write_profile(tmp_path, "custom", (
        "name: custom\n"
        "llm:\n  model: some-model\n  api_key: ${MY_KEY}\n  base_url: ${MY_URL}\n"
        "agent:\n  base_prompt: coding\ntools: [bash]\nskills: []\n"
    ))
    profile = P.get_profile("custom")
    cfg = profile.model_config
    assert cfg.api_key == "sk-explicit" and cfg.base_url == "https://custom.test/v1"
    assert profile.provider == "custom"
    assert profile.api_key_env == "MY_KEY"
    assert profile.base_url_env == "MY_URL"


def test_unknown_profile_raises():
    with pytest.raises(KeyError):
        P.get_profile("does-not-exist")


def test_system_prompt_override(tmp_path, monkeypatch):
    monkeypatch.setattr(P, "_USER_DIR", tmp_path)
    _write_profile(tmp_path, "p", (
        "name: p\nllm:\n  provider: openai\n  model: m\n"
        "agent:\n  system_prompt: 'CUSTOM PROMPT'\ntools: [bash]\nskills: []\n"
    ))
    assert P.get_profile("p").system_prompt("/tmp") == "CUSTOM PROMPT"


def test_wrap_skills_llm_noop_and_wrap(tmp_path, monkeypatch):
    from apodex import session as S
    from apodex.session import _wrap_skills_llm

    sentinel = object()
    # No skills requested → returned unchanged.
    assert _wrap_skills_llm(sentinel, []) is sentinel

    # No skills *available* → also unchanged. This is the shipped state: the
    # loader is wired but no SKILL.md files are bundled, so a profile asking
    # for "*" must degrade to a plain LLM rather than fail.
    monkeypatch.setattr(S, "_SKILLS_DIR", tmp_path / "empty")
    assert _wrap_skills_llm(sentinel, ["*"]) is sentinel

    # A discoverable skill → the LLM is wrapped with the injection middleware.
    skill = tmp_path / "skills" / "demo"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text(
        "---\nname: demo\ndescription: a test skill\n---\n\nBody text.\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(S, "_SKILLS_DIR", tmp_path / "skills")
    assert _wrap_skills_llm(sentinel, ["*"]) is not sentinel
