"""The optional user env file and the CLI's configuration precedence.

Every test drives the real resolver against a temporary HOME and a temporary
launch directory. None of them relies on a checkout ``.env``: the autouse
fixture in ``conftest.py`` already points HOME/XDG at ``tmp_path``.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import ClassVar

import pytest

from apodex import cli
from apodex.userenv import (
    USER_ENV_FILE_VAR,
    EnvResolution,
    apply_user_env,
    load_environment,
    user_env_path,
)

_SECRET = "sk-user-file-secret-7Q9x"
_OTHER_SECRET = "sk-project-secret-Zz41"


def _write(path: Path, text: str, *, mode: int = 0o600) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    path.chmod(mode)
    return path


@pytest.fixture
def user_file(tmp_path) -> Path:
    """The default user env location under the fixture HOME."""
    home = Path(os.environ["HOME"])
    return home / ".config" / "apodex" / "env"


@pytest.fixture
def launch(tmp_path, monkeypatch) -> Path:
    """An empty launch directory with no ``.env`` anywhere above it."""
    launch = tmp_path / "launch"
    launch.mkdir()
    monkeypatch.chdir(launch)
    # Earlier tests may leave publish_model_overrides' variables behind in the
    # real environment; the precedence assertions need a clean slate.
    for name in ("OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_MODEL", "OPENAI_MAX_TOKENS"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.delenv(USER_ENV_FILE_VAR, raising=False)
    return launch


# ── location ──────────────────────────────────────────────────────────────


def test_default_location_follows_xdg_then_home() -> None:
    assert user_env_path({"XDG_CONFIG_HOME": "/x/cfg", "HOME": "/h"}) == Path(
        "/x/cfg/apodex/env",
    )
    assert user_env_path({"HOME": "/h"}) == Path("/h/.config/apodex/env")
    # A blank XDG value means unset, as the spec says, rather than "/apodex/env".
    assert user_env_path({"XDG_CONFIG_HOME": "  ", "HOME": "/h"}) == Path(
        "/h/.config/apodex/env",
    )


def test_explicit_path_variable_wins_over_the_config_directory() -> None:
    env = {USER_ENV_FILE_VAR: "~/elsewhere/agent.env", "HOME": "/h"}
    assert user_env_path(env) == Path("~/elsewhere/agent.env").expanduser()


# ── presence and absence ──────────────────────────────────────────────────


def test_missing_user_file_is_not_an_error(launch) -> None:
    resolution = load_environment()

    assert resolution.user_env_path is None
    assert resolution.applied == ()
    assert resolution.notes == ()
    assert resolution.dotenv_paths == ()


def test_user_file_supplies_values_nothing_else_set(launch, user_file) -> None:
    _write(
        user_file,
        (
            f"OPENAI_API_KEY={_SECRET}\n"
            "OPENAI_BASE_URL=https://user.example/v1\n"
            "OPENAI_MODEL=user-model\n"
            "BLANK_VALUE=\n"
            "# comment\n"
        ),
    )

    resolution = load_environment()

    assert resolution.user_env_path == user_file
    assert set(resolution.applied) == {"OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_MODEL"}
    assert os.environ["OPENAI_API_KEY"] == _SECRET
    assert os.environ["OPENAI_MODEL"] == "user-model"
    # A blank line in the file is "not provided", not an empty credential.
    assert "BLANK_VALUE" not in os.environ
    assert resolution.notes == ()


def test_user_file_is_read_literally_without_interpolation(launch, user_file, monkeypatch) -> None:
    monkeypatch.setenv("SOMEWHERE_ELSE", "https://attacker.example/v1")
    _write(user_file, "OPENAI_BASE_URL=${SOMEWHERE_ELSE}\n")

    load_environment()

    # dotenv would have expanded this against the exported environment; the
    # user file never resolves against whatever happens to be exported.
    assert os.environ["OPENAI_BASE_URL"] == "${SOMEWHERE_ELSE}"


@pytest.mark.skipif(
    os.name != "posix" or os.geteuid() == 0,
    reason="root ignores file modes",
)
def test_unreadable_user_file_is_reported_by_name_and_ignored(launch, user_file) -> None:
    _write(user_file, f"OPENAI_API_KEY={_SECRET}\n", mode=0o000)

    resolution = load_environment()

    assert resolution.user_env_path == user_file
    assert resolution.applied == ()
    assert "OPENAI_API_KEY" not in os.environ
    assert len(resolution.notes) == 1
    assert str(user_file) in resolution.notes[0]
    assert "ignoring it" in resolution.notes[0]
    assert _SECRET not in resolution.notes[0]


# ── precedence ────────────────────────────────────────────────────────────


def test_exported_environment_wins_over_the_user_file(launch, user_file, monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_MODEL", "exported-model")
    _write(user_file, "OPENAI_MODEL=user-model\n")

    resolution = load_environment()

    assert os.environ["OPENAI_MODEL"] == "exported-model"
    assert "OPENAI_MODEL" not in resolution.applied


def test_launch_directory_dotenv_wins_over_the_user_file(launch, user_file) -> None:
    _write(launch / ".env", "OPENAI_MODEL=project-model\n")
    _write(user_file, "OPENAI_MODEL=user-model\nOPENAI_MAX_TOKENS=1234\n")

    resolution = load_environment()

    assert os.environ["OPENAI_MODEL"] == "project-model"
    assert os.environ["OPENAI_MAX_TOKENS"] == "1234"  # only the file had it
    assert resolution.dotenv_paths == (launch / ".env",)
    assert resolution.applied == ("OPENAI_MAX_TOKENS",)


def test_ancestor_dotenv_is_still_discovered(launch, user_file) -> None:
    _write(launch.parent / ".env", "OPENAI_MODEL=ancestor-model\n")
    _write(user_file, "OPENAI_MODEL=user-model\n")

    resolution = load_environment()

    assert os.environ["OPENAI_MODEL"] == "ancestor-model"
    assert [p.resolve() for p in resolution.dotenv_paths] == [
        (launch.parent / ".env").resolve(),
    ]


def test_exported_environment_wins_over_every_file(launch, user_file, monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_MODEL", "exported-model")
    _write(launch / ".env", "OPENAI_MODEL=project-model\n")
    _write(user_file, "OPENAI_MODEL=user-model\n")

    load_environment()

    assert os.environ["OPENAI_MODEL"] == "exported-model"


# ── the credential / endpoint pair guard ──────────────────────────────────


def test_user_key_is_withheld_when_a_project_overrides_only_the_endpoint(
    launch,
    user_file,
    capsys,
) -> None:
    _write(launch / ".env", "OPENAI_BASE_URL=https://other.example/v1\n")
    _write(
        user_file,
        (
            f"OPENAI_API_KEY={_SECRET}\n"
            "OPENAI_BASE_URL=https://user.example/v1\n"
            "OPENAI_MODEL=user-model\n"
        ),
    )

    resolution = load_environment()

    # The key written next to user.example must not travel to other.example.
    assert "OPENAI_API_KEY" not in os.environ
    assert os.environ["OPENAI_BASE_URL"] == "https://other.example/v1"
    assert os.environ["OPENAI_MODEL"] == "user-model"  # unpaired: still applied
    assert resolution.withheld == ("OPENAI_API_KEY",)
    assert len(resolution.notes) == 1
    note = resolution.notes[0]
    assert "OPENAI_API_KEY" in note and "OPENAI_BASE_URL" in note
    assert str(user_file) in note
    assert _SECRET not in note
    assert "other.example" not in note  # the overriding value is not echoed either


def test_same_endpoint_written_differently_does_not_withhold(launch, user_file) -> None:
    _write(launch / ".env", "OPENAI_BASE_URL=https://user.example/v1/\n")
    _write(user_file, (f"OPENAI_API_KEY={_SECRET}\nOPENAI_BASE_URL=https://user.example/v1\n"))

    resolution = load_environment()

    assert os.environ["OPENAI_API_KEY"] == _SECRET
    assert resolution.withheld == ()


def test_user_endpoint_is_withheld_when_a_different_key_is_already_set(
    launch,
    user_file,
    monkeypatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", _OTHER_SECRET)
    _write(user_file, (f"OPENAI_API_KEY={_SECRET}\nOPENAI_BASE_URL=https://user.example/v1\n"))

    resolution = load_environment()

    assert os.environ["OPENAI_API_KEY"] == _OTHER_SECRET
    assert "OPENAI_BASE_URL" not in os.environ
    assert resolution.withheld == ("OPENAI_BASE_URL",)
    assert _SECRET not in resolution.notes[0]
    assert _OTHER_SECRET not in resolution.notes[0]


def test_fully_overridden_pair_needs_no_note(launch, user_file) -> None:
    _write(
        launch / ".env",
        (f"OPENAI_API_KEY={_OTHER_SECRET}\nOPENAI_BASE_URL=https://other.example/v1\n"),
    )
    _write(user_file, (f"OPENAI_API_KEY={_SECRET}\nOPENAI_BASE_URL=https://user.example/v1\n"))

    resolution = load_environment()

    assert os.environ["OPENAI_API_KEY"] == _OTHER_SECRET
    assert resolution.withheld == ()
    assert resolution.notes == ()


def test_a_lone_key_in_the_user_file_is_applied_as_written(launch, user_file) -> None:
    # No endpoint next to it means the user chose "this key, wherever I point
    # the CLI"; the guard only protects pairs that were written as pairs.
    _write(launch / ".env", "OPENAI_BASE_URL=https://other.example/v1\n")
    _write(user_file, f"OPENAI_API_KEY={_SECRET}\n")

    resolution = load_environment()

    assert os.environ["OPENAI_API_KEY"] == _SECRET
    assert resolution.withheld == ()


def test_pair_guard_covers_every_provider_prefix(tmp_path) -> None:
    path = _write(
        tmp_path / "env",
        (
            "ANTHROPIC_API_KEY=a-key\nANTHROPIC_BASE_URL=https://a.example\n"
            "BEDROCK_API_KEY=b-key\nBEDROCK_BASE_URL=https://b.example\n"
        ),
    )
    environ = {"ANTHROPIC_BASE_URL": "https://elsewhere.example"}

    _, applied, withheld, notes, defined = apply_user_env(environ, path=path)

    assert withheld == ("ANTHROPIC_API_KEY",)
    assert set(applied) == {"BEDROCK_API_KEY", "BEDROCK_BASE_URL"}
    assert "ANTHROPIC_API_KEY" not in environ
    assert len(notes) == 1
    assert set(defined) == {
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_BASE_URL",
        "BEDROCK_API_KEY",
        "BEDROCK_BASE_URL",
    }


# ── secrets stay out of every channel ─────────────────────────────────────


@pytest.mark.skipif(os.name != "posix", reason="file modes are POSIX")
def test_world_readable_file_gets_a_permission_note_without_its_contents(
    launch,
    user_file,
) -> None:
    _write(user_file, f"OPENAI_API_KEY={_SECRET}\n", mode=0o644)

    resolution = load_environment()

    assert os.environ["OPENAI_API_KEY"] == _SECRET
    assert any("chmod 600" in note for note in resolution.notes)
    assert all(_SECRET not in note for note in resolution.notes)


def test_resolution_never_carries_a_value(launch, user_file) -> None:
    _write(user_file, f"OPENAI_API_KEY={_SECRET}\nOPENAI_BASE_URL=https://u.example/v1\n")

    resolution = load_environment()

    assert _SECRET not in repr(resolution)
    assert "u.example" not in repr(resolution)


def test_forwarded_names_are_names_only_and_only_when_set(launch, user_file, monkeypatch) -> None:
    _write(user_file, f"OPENAI_API_KEY={_SECRET}\nCUSTOM_PROVIDER_TOKEN=abc\n")
    monkeypatch.setenv("SERPER_API_KEY", "serper-secret")
    monkeypatch.delenv("JINA_API_KEY", raising=False)

    resolution = load_environment()
    names = resolution.forwarded_names()

    assert "OPENAI_API_KEY" in names  # from the file
    assert "CUSTOM_PROVIDER_TOKEN" in names  # file-defined, even if unlisted
    assert "SERPER_API_KEY" in names  # well-known and exported
    assert "JINA_API_KEY" not in names  # well-known but not set
    assert _SECRET not in " ".join(names)


def test_empty_resolution_forwards_only_exported_well_known_names(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_MODEL", "m")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    names = EnvResolution.empty().forwarded_names()

    assert "OPENAI_MODEL" in names
    assert "OPENAI_API_KEY" not in names


# ── the CLI: --cwd, --model, and where .env is looked up ──────────────────


class _RecordingSession:
    """Stands in for TerminalSession once preflight has passed."""

    constructed: ClassVar[list[dict]] = []

    def __init__(self, **kwargs) -> None:
        type(self).constructed.append(kwargs)
        self.session_id = "recorded-session"
        self.history: list = []

    async def run_task(self, task: str) -> None:
        return None


@pytest.fixture
def cli_harness(monkeypatch, launch):
    """A CLI whose session is recorded rather than run, with fresh profile caches."""
    from apodex import profiles
    from frontier_agent.infra import providers

    monkeypatch.setattr(profiles, "_CACHE", {})
    providers._reset_cache()
    _RecordingSession.constructed = []
    monkeypatch.setattr(cli, "TerminalSession", _RecordingSession)
    for name in ("OPENAI_MAX_TOKENS", "SERPER_API_KEY", "JINA_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    yield _RecordingSession
    providers._reset_cache()


def _run_cli(*args: str) -> int:
    return asyncio.run(cli._amain(["--no-tui", "--no-sandbox", *args]))


def test_dotenv_is_resolved_from_the_launch_directory_not_from_cwd(
    cli_harness,
    launch,
    tmp_path,
    capsys,
) -> None:
    target = tmp_path / "target"
    target.mkdir()
    _write(
        launch / ".env",
        (
            f"OPENAI_API_KEY={_SECRET}\nOPENAI_BASE_URL=https://launch.example/v1\n"
            "OPENAI_MODEL=launch-model\n"
        ),
    )
    _write(
        target / ".env",
        (
            f"OPENAI_API_KEY={_OTHER_SECRET}\nOPENAI_BASE_URL=https://target.example/v1\n"
            "OPENAI_MODEL=target-model\n"
        ),
    )

    assert _run_cli("--cwd", str(target), "-p", "task") == 0

    cfg = cli_harness.constructed[0]["cfg"]
    assert cli_harness.constructed[0]["cwd"] == str(target)
    assert cfg.model == "launch-model"
    assert cfg.base_url == "https://launch.example/v1"
    assert cfg.api_key == _SECRET
    out = capsys.readouterr()
    assert _SECRET not in out.out + out.err
    assert _OTHER_SECRET not in out.out + out.err


def test_a_target_dotenv_alone_does_not_configure_the_run(
    cli_harness,
    launch,
    tmp_path,
    capsys,
) -> None:
    target = tmp_path / "target"
    target.mkdir()
    _write(
        target / ".env",
        (
            f"OPENAI_API_KEY={_OTHER_SECRET}\nOPENAI_BASE_URL=https://target.example/v1\n"
            "OPENAI_MODEL=target-model\n"
        ),
    )

    assert _run_cli("--cwd", str(target), "-p", "task") == 2

    assert cli_harness.constructed == []
    err = capsys.readouterr().err
    assert "preflight failed" in err
    assert "OPENAI_API_KEY" in err
    assert _OTHER_SECRET not in err


def test_cli_flags_outrank_every_file_and_the_user_file_fills_the_rest(
    cli_harness,
    launch,
    user_file,
    tmp_path,
    monkeypatch,
) -> None:
    target = tmp_path / "target"
    target.mkdir()
    _write(
        user_file,
        (
            f"OPENAI_API_KEY={_SECRET}\nOPENAI_BASE_URL=https://user.example/v1\n"
            "OPENAI_MODEL=user-model\n"
        ),
    )
    _write(launch / ".env", "OPENAI_MODEL=project-model\n")

    assert _run_cli("--cwd", str(target), "--model", "flag-model", "-p", "task") == 0

    cfg = cli_harness.constructed[0]["cfg"]
    assert cfg.model == "flag-model"  # explicit option
    assert cfg.api_key == _SECRET  # user file default
    assert cfg.base_url == "https://user.example/v1"
    # The workflow reads the model from the environment; the flag reached it.
    assert os.environ["OPENAI_MODEL"] == "flag-model"


def test_withheld_key_explains_the_preflight_failure_once(
    cli_harness,
    launch,
    user_file,
    tmp_path,
    capsys,
) -> None:
    _write(
        launch / ".env", ("OPENAI_BASE_URL=https://other.example/v1\nOPENAI_MODEL=project-model\n")
    )
    _write(user_file, (f"OPENAI_API_KEY={_SECRET}\nOPENAI_BASE_URL=https://user.example/v1\n"))

    assert _run_cli("-p", "task") == 2

    err = capsys.readouterr().err
    assert err.count("was not applied") == 1  # printed once, not per channel
    assert err.index("was not applied") < err.index("preflight failed")
    assert _SECRET not in err
    assert cli_harness.constructed == []
