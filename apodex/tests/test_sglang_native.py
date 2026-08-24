from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _module() -> dict:
    return runpy.run_path(str(ROOT / "scripts/run-sglang-native.py"))


def test_native_dotenv_parser_never_evaluates_shell_syntax(tmp_path) -> None:
    marker = tmp_path / "must-not-exist"
    env_file = tmp_path / "profile.env"
    env_file.write_text(
        "# profile\n"
        "SGLANG_MODEL_ID='Qwen/example'\n"
        f"SGLANG_EXTRA_ARGS=$(touch {marker})\n",
        encoding="utf-8",
    )

    values = _module()["load_dotenv"](env_file)

    assert values["SGLANG_MODEL_ID"] == "Qwen/example"
    assert values["SGLANG_EXTRA_ARGS"] == f"$(touch {marker})"
    assert not marker.exists()


def test_native_environment_maps_public_profile_port_to_loopback(
    tmp_path, monkeypatch
) -> None:
    for key in (
        "SGLANG_PORT",
        "SGLANG_NATIVE_HOST",
        "SGLANG_DOWNLOAD_DIR",
        "HF_HOME",
    ):
        monkeypatch.delenv(key, raising=False)
    cache = tmp_path / "huggingface"

    environ = _module()["runtime_environment"](
        {
            "SGLANG_PORT": "31000",
            "SGLANG_DOWNLOAD_DIR": str(cache),
        }
    )

    assert environ["SGLANG_SERVER_HOST"] == "127.0.0.1"
    assert environ["SGLANG_SERVER_PORT"] == "31000"
    assert environ["SGLANG_BASE_URL"] == "http://127.0.0.1:31000"
    assert environ["SGLANG_DOWNLOAD_DIR"] == str(cache)


@pytest.mark.parametrize(
    ("native_host", "expected"),
    (
        # A NIC-specific bind does not answer on loopback, so polling
        # 127.0.0.1 would burn the whole startup timeout on a healthy server.
        ("10.0.0.5", "http://10.0.0.5:30000"),
        # A wildcard bind does answer on loopback, and loopback is the safer
        # address to dial.
        ("0.0.0.0", "http://127.0.0.1:30000"),
        ("::", "http://127.0.0.1:30000"),
        ("::1", "http://[::1]:30000"),
    ),
)
def test_native_health_url_follows_the_configured_listener(
    native_host, expected, monkeypatch
) -> None:
    for key in ("SGLANG_PORT", "SGLANG_NATIVE_HOST", "SGLANG_DOWNLOAD_DIR"):
        monkeypatch.delenv(key, raising=False)

    environ = _module()["runtime_environment"]({"SGLANG_NATIVE_HOST": native_host})

    assert environ["SGLANG_BASE_URL"] == expected


def test_native_liveness_falls_back_to_signals_without_procfs() -> None:
    """On macOS `/proc` does not exist; a live server must not read as exited.

    Reporting one as gone makes `up` claim it failed while the detached child
    keeps the port and GPU, and `down` then refuses to reclaim them.
    """
    module = _module()

    class _NoProcfs:
        def __init__(self, *_args: object) -> None:
            pass

        def read_bytes(self) -> bytes:
            raise FileNotFoundError("no procfs on this platform")

    # run_path returns a copy of the namespace, so reach the live globals.
    module["is_managed_process"].__globals__["Path"] = _NoProcfs

    assert module["is_managed_process"](os.getpid()) is True
    assert module["is_managed_process"](2**22 - 1) is False


def test_native_import_probe_timeout_is_inconclusive_not_a_failure(tmp_path) -> None:
    """A cold torch/CUDA import must neither crash nor read as 'not installed'."""
    module = _module()
    assert module["IMPORT_PROBE_TIMEOUT"] >= 120
    module["_module_available"].__globals__["IMPORT_PROBE_TIMEOUT"] = 1
    (tmp_path / "slowmod.py").write_text(
        "import time\ntime.sleep(30)\n", encoding="utf-8"
    )

    result = module["_module_available"](
        sys.executable, "slowmod", {"PYTHONPATH": str(tmp_path)}
    )

    assert result is None


def test_native_launcher_exposes_the_same_lifecycle_as_docker() -> None:
    module = _module()

    for command in ("doctor", "up", "smoke", "tui", "status", "logs", "down"):
        args = module["parse_args"]([command])
        assert args.command == command


@pytest.mark.parametrize(
    ("key", "value"),
    (("SGLANG_PORT", "70000"), ("SGLANG_STARTUP_TIMEOUT", "never")),
)
def test_native_environment_rejects_invalid_runtime_values(
    key, value, monkeypatch
) -> None:
    monkeypatch.delenv(key, raising=False)

    with pytest.raises(ValueError):
        _module()["runtime_environment"]({key: value})
