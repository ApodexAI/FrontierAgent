"""Secret-safe BYOK runtime preflight tests."""

from __future__ import annotations

import asyncio
import os
from types import SimpleNamespace

import pytest

from apodex import cli
from apodex.config import (
    ModelConfig,
    format_preflight_errors,
    format_runtime_config_status,
    inspect_runtime_config,
)


def _profile(**overrides):
    values = {
        "name": "coding",
        "path": "/profiles/coding.yaml",
        "provider": "openai",
        "api_key_env": "OPENAI_API_KEY",
        "base_url_env": "OPENAI_BASE_URL",
        "model_env": "OPENAI_MODEL",
        "tool_names": (),
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_valid_status_exposes_hostname_and_no_secret_fragments():
    secret = "sk-demo-secret-9876"
    status = inspect_runtime_config(
        ModelConfig(
            model="gpt-test",
            api_key=secret,
            base_url="https://user:password@example.test/v1?token=hidden",
        ),
        profile=_profile(),
        mode="coding",
        environ={},
    )

    assert status.ok
    assert status.endpoint_host == "example.test"
    rendered = format_runtime_config_status(status)
    assert "OPENAI_API_KEY" in rendered
    assert "configured" in rendered
    for fragment in (secret, "sk-d", "9876", "password", "token=hidden"):
        assert fragment not in rendered
        assert fragment not in repr(status)


def test_remote_missing_key_empty_model_and_malformed_url_are_blocking():
    status = inspect_runtime_config(
        ModelConfig(model="", api_key="${UNRESOLVED}", base_url="ssh://key@host/path"),
        profile=_profile(),
        environ={},
    )

    assert {issue.code for issue in status.errors} == {
        "missing_api_key", "missing_model", "invalid_base_url",
    }
    rendered = format_preflight_errors(status)
    assert "export OPENAI_API_KEY=..." in rendered
    assert "export OPENAI_MODEL=..." in rendered
    assert "export OPENAI_BASE_URL=..." in rendered
    assert "ssh://" not in rendered


def test_local_empty_placeholder_is_allowed():
    status = inspect_runtime_config(
        ModelConfig(model="local-model", api_key="EMPTY", base_url="http://localhost:8000/v1"),
        profile=_profile(
            provider="local", api_key_env="LOCAL_API_KEY", base_url_env="LOCAL_BASE_URL",
        ),
        environ={},
    )

    assert status.ok
    assert status.api_key_configured
    assert status.endpoint_host == "localhost"


_WEB_TOOLS = ("web_search", "web_fetch", "bash", "read_file")


def test_search_credentials_are_checked_when_the_profile_binds_the_web_tools():
    cfg = ModelConfig(model="gpt-test", api_key="secret", base_url="https://api.test/v1")
    web = inspect_runtime_config(
        cfg, profile=_profile(tool_names=_WEB_TOOLS), environ={},
    )
    # Both warn: a coding session that never searches must still start.
    assert web.ok
    assert [issue.code for issue in web.warnings] == [
        "missing_serper_api_key", "missing_jina_api_key",
    ]
    rendered = format_runtime_config_status(web)
    assert "warning: SERPER_API_KEY" in rendered
    assert "warning: JINA_API_KEY" in rendered

    with_search = inspect_runtime_config(
        cfg,
        profile=_profile(tool_names=_WEB_TOOLS),
        environ={"SERPER_API_KEY": "search-secret"},
    )
    assert with_search.ok
    assert [issue.code for issue in with_search.warnings] == ["missing_jina_api_key"]

    no_web_tools = inspect_runtime_config(
        cfg, profile=_profile(tool_names=("bash", "read_file")), environ={},
    )
    assert no_web_tools.ok and not no_web_tools.warnings


def test_every_selectable_terminal_mode_preflights_its_search_credentials():
    """The check used to key on ``research``, a mode the CLI cannot select.

    Both shipped modes bind web_search, so a blank SERPER_API_KEY reached the
    model as an error string inside a tool result instead of failing preflight.
    """
    from apodex.profiles import get_profile, terminal_mode_names

    cfg = ModelConfig(model="gpt-test", api_key="secret", base_url="https://api.test/v1")
    modes = terminal_mode_names()
    assert modes, "the terminal must expose at least one mode"
    for mode in modes:
        profile = get_profile(mode)
        assert "web_search" in profile.tool_names, mode
        status = inspect_runtime_config(cfg, profile=profile, mode=mode, environ={})
        assert "missing_serper_api_key" in [i.code for i in status.warnings], mode
        # Warned about, never refused a startup.
        assert status.ok, mode


def test_cli_fails_before_session_construction_with_actionable_guidance(
    monkeypatch, capsys,
):
    profile = _profile()
    profile.model_config = ModelConfig(
        model="gpt-test", api_key="", base_url="https://api.test/v1",
    )
    profile.max_turns = 50
    profile.runtime_config = lambda cfg, mode=None: inspect_runtime_config(
        cfg, profile=profile, mode=mode, environ={},
    )
    monkeypatch.setattr(cli, "_load_env", lambda: None)
    monkeypatch.setattr(cli, "terminal_mode_names", lambda: ["react", "agent_team"])
    monkeypatch.setattr(cli, "get_profile", lambda _mode: profile)

    def _unexpected_session(**_kwargs):
        raise AssertionError("TerminalSession must not be constructed")

    monkeypatch.setattr(cli, "TerminalSession", _unexpected_session)
    result = asyncio.run(cli._amain(["--no-tui", "--no-sandbox"]))

    assert result == 2
    stderr = capsys.readouterr().err
    assert "OPENAI_API_KEY" in stderr
    assert "export OPENAI_API_KEY=..." in stderr
    assert "frontier-agent --cwd ." in stderr
    assert "login" not in stderr.lower()


def test_mode_switch_rejects_legacy_modes_before_mutating_session(monkeypatch):
    from apodex import session as session_module
    from apodex.session import TerminalSession

    messages: list[str] = []
    terminal = object.__new__(TerminalSession)
    terminal.mode = "coding"
    terminal.r = SimpleNamespace(note=messages.append, error=messages.append)
    monkeypatch.setattr(session_module, "terminal_mode_names", lambda: ["react", "agent_team"])

    assert asyncio.run(terminal._slash("/mode research")) is False
    assert terminal.mode == "coding"
    rendered = "\n".join(messages)
    assert "unknown mode 'research'" in rendered
    assert "react, agent_team" in rendered


def test_resume_rejects_legacy_saved_mode_before_mutating_session(
    tmp_path, monkeypatch,
):
    from apodex import session as session_module
    from apodex.session import TerminalSession

    terminal = object.__new__(TerminalSession)
    terminal.mode = "coding"
    terminal.cwd = str(tmp_path)
    monkeypatch.setattr(session_module, "terminal_mode_names", lambda: ["react", "agent_team"])

    with pytest.raises(ValueError, match="no longer available"):
        terminal.switch_session({"mode": "research", "cwd": str(tmp_path)})
    assert terminal.mode == "coding"


def test_cli_resume_rejects_legacy_saved_mode(tmp_path, monkeypatch):
    from apodex import session as session_module

    monkeypatch.setattr(cli, "_load_env", lambda: None)
    monkeypatch.setattr(cli, "terminal_mode_names", lambda: ["react", "agent_team"])
    monkeypatch.setattr(
        session_module,
        "load_session_state",
        lambda _sid: {
            "session_id": "saved",
            "mode": "research",
            "cwd": str(tmp_path),
            "model": "saved-model",
            "history": [],
        },
    )

    before = os.getcwd()
    result = asyncio.run(cli._amain(["--resume", "saved", "--no-tui", "--no-sandbox"]))

    assert result == 2
    # A rejected resume must not have moved the process. It used to chdir into
    # the saved session's cwd first and validate after, so every later caller
    # in the same process ran from a directory it never asked for — and once
    # pytest removed that tmp_path, anything resolving a relative path broke.
    assert os.getcwd() == before


def test_cli_resume_without_id_lists_sessions_before_runtime_setup(
    tmp_path, monkeypatch, capsys,
):
    monkeypatch.setattr(cli, "_load_env", lambda: pytest.fail("should not load config"))
    seen: dict[str, object] = {}

    def _fake_list(extra_roots=None, workspace=None):
        seen["workspace"] = workspace
        return [{
            "session_id": "20260805-120000-coding-ab12",
            "mode": "coding",
            "cwd": "/work/project",
            "message_count": 7,
            "modified_at": "2026-08-05 12:00",
        }]

    monkeypatch.setattr("apodex.session.list_saved_sessions", _fake_list)

    result = asyncio.run(cli._amain(["--resume", "--cwd", str(tmp_path)]))

    assert result == 0
    # The listing answers before the chdir, so the run tree must be named
    # explicitly rather than inferred from the launch directory.
    assert seen["workspace"] == str(tmp_path)
    output = capsys.readouterr().out
    assert "Saved sessions:" in output
    assert "20260805-120000-coding-ab12" in output
    assert "/work/project" in output
