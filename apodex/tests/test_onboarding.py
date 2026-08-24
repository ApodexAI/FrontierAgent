"""First-run environment detection and TUI coverage."""

from __future__ import annotations

from types import SimpleNamespace

from textual.widgets import Static

from apodex.onboarding import (
    NVIDIA_SGLANG,
    THIRD_PARTY,
    choice_guidance,
    detect_onboarding_environment,
    format_probe,
)
from apodex.tui.onboarding import OnboardingApp


def _status(*, ok: bool = True) -> SimpleNamespace:
    return SimpleNamespace(
        ok=ok,
        provider="openai",
        model="apodex-test",
        api_key_configured=ok,
    )


def test_detector_reports_env_and_complete_nvidia_container_stack(
    tmp_path, monkeypatch,
) -> None:
    secret = "sk-must-never-be-rendered"
    (tmp_path / ".env").write_text("OPENAI_API_KEY=hidden\n", encoding="utf-8")
    monkeypatch.setattr(
        "apodex.onboarding.shutil.which", lambda command: f"/usr/bin/{command}",
    )

    def runner(command):
        if command[0] == "nvidia-smi":
            return True, "NVIDIA H100 80GB HBM3, 81559 MiB"
        if command[1:2] == ("info",):
            return True, '{"io.containerd.runc.v2":{},"nvidia":{}}'
        return True, "2.39.1"

    probe = detect_onboarding_environment(
        cwd=str(tmp_path),
        runtime_status=_status(),
        environ={"OPENAI_API_KEY": secret},
        runner=runner,
    )

    assert probe.env_path == str(tmp_path / ".env")
    assert probe.configured_key_vars == ("OPENAI_API_KEY",)
    assert probe.nvidia_container_ready
    rendered = format_probe(probe)
    assert "NVIDIA H100" in rendered
    assert "NVIDIA containers: ready" in rendered
    assert secret not in rendered
    assert secret not in repr(probe)


def test_detector_explains_missing_nvidia_container_components(
    tmp_path, monkeypatch,
) -> None:
    monkeypatch.setattr("apodex.onboarding.shutil.which", lambda _command: None)
    probe = detect_onboarding_environment(
        cwd=str(tmp_path),
        runtime_status=_status(ok=False),
        environ={},
        runner=lambda _command: (False, ""),
    )

    assert not probe.nvidia_container_ready
    assert probe.missing_nvidia_components == (
        "NVIDIA driver / nvidia-smi",
        "Docker CLI",
        "Docker Compose v2",
    )
    guidance = choice_guidance(NVIDIA_SGLANG, probe)
    assert "Missing:" in guidance
    assert "./docker/run-sglang.sh" in guidance


def test_agent_container_recognizes_its_configured_local_model_service(
    tmp_path, monkeypatch,
) -> None:
    monkeypatch.setattr("apodex.onboarding.shutil.which", lambda _command: None)
    local_status = SimpleNamespace(
        ok=True,
        provider="local",
        model="local-model",
        api_key_configured=True,
    )
    probe = detect_onboarding_environment(
        cwd=str(tmp_path),
        runtime_status=local_status,
        environ={
            "APODEX_IN_CONTAINER": "1",
            "APODEX_LOCAL_RUNTIME": "sglang",
        },
        runner=lambda _command: (False, ""),
    )

    assert probe.managed_local_service
    assert probe.managed_local_runtime == "sglang"
    assert probe.suggested_choice == NVIDIA_SGLANG
    assert probe.nvidia_container_ready
    assert "Connected to the configured local model service" in choice_guidance(
        NVIDIA_SGLANG, probe,
    )
    assert "connected to local model service" in format_probe(probe)


async def test_tui_onboarding_shows_probe_and_returns_selected_route() -> None:
    probe = detect_onboarding_environment(
        cwd="/tmp",
        runtime_status=_status(ok=False),
        environ={},
        runner=lambda _command: (False, ""),
    )
    app = OnboardingApp(probe)

    async with app.run_test(size=(110, 34)) as pilot:
        probe_text = app.query_one("#onboarding-probe", Static).render().plain
        assert ".env:" in probe_text
        assert "NVIDIA containers:" in probe_text

        await pilot.press("down", "enter", "enter")
        await pilot.pause()

    assert app.return_value == THIRD_PARTY
