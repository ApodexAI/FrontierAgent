from __future__ import annotations

import importlib.util
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run-sglang-native.py"


def _load_native_launcher():
    spec = importlib.util.spec_from_file_location("frontier_sglang_native", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_driver_550_accepts_last_cuda12_qwen35_release() -> None:
    launcher = _load_native_launcher()

    compatible, message = launcher.native_sglang_compatibility(
        "550.127.08",
        "0.5.10.post1",
        "Qwen/Qwen3.5-example",
    )

    assert compatible
    assert "CUDA 12" in message


def test_driver_550_rejects_cuda13_sglang() -> None:
    launcher = _load_native_launcher()

    compatible, message = launcher.native_sglang_compatibility(
        "550.127.08",
        "0.5.11",
        "Qwen/Qwen3.5-example",
    )

    assert not compatible
    assert "driver 580+" in message


def test_qwen35_rejects_sglang_before_model_support() -> None:
    launcher = _load_native_launcher()

    compatible, message = launcher.native_sglang_compatibility(
        "550.127.08",
        "0.5.9",
        "Qwen/Qwen3.5-example",
    )

    assert not compatible
    assert "starts in the SGLang 0.5.10" in message


def test_driver_580_accepts_cuda13_sglang() -> None:
    launcher = _load_native_launcher()

    compatible, message = launcher.native_sglang_compatibility(
        "580.95.05",
        "0.5.17",
        "Qwen/Qwen3.5-example",
    )

    assert compatible
    assert "CUDA 13" in message


def test_bootstrap_selects_latest_reviewed_track_for_driver() -> None:
    launcher = _load_native_launcher()

    assert launcher.recommended_native_track("579.99")["id"] == "cu12"
    assert launcher.recommended_native_track("580.1")["id"] == "cu13"


def test_native_tracks_record_their_lazy_jit_requirements() -> None:
    launcher = _load_native_launcher()

    cu12 = launcher.native_track_for_sglang("0.5.10.post1")
    cu13 = launcher.native_track_for_sglang("0.5.17")

    assert cu12["jit_toolkit_minimum"] == "12.9"
    assert cu13["jit_toolkit_minimum"] == "13.0"
    assert cu12["support_tier"] == "compatibility_hint"
    assert cu13["support_tier"] == "official_default"
    assert "curand.h" in cu12["required_jit_headers"]
    assert launcher.native_track_for_sglang("0.5.9") is None


def _fake_nvcc(tmp_path: Path, version: str) -> Path:
    toolkit = tmp_path / f"cuda-{version}"
    nvcc = toolkit / "bin" / "nvcc"
    nvcc.parent.mkdir(parents=True)
    nvcc.write_text(
        f"#!/bin/sh\necho 'Cuda compilation tools, release {version}, V{version}.0'\n",
        encoding="utf-8",
    )
    nvcc.chmod(0o755)
    return toolkit


def _fake_cuda_probe_python(tmp_path: Path, payload: str) -> Path:
    python = tmp_path / "cuda-probe-python"
    python.write_text(f"#!/bin/sh\necho '{payload}'\n", encoding="utf-8")
    python.chmod(0o755)
    return python


def test_cuda_runtime_probe_reports_the_real_device_and_wheel(tmp_path: Path) -> None:
    launcher = _load_native_launcher()
    python = _fake_cuda_probe_python(
        tmp_path,
        '{"ok":true,"torch":"2.9.1+cu128","torch_cuda":"12.8",'
        '"device":"NVIDIA GeForce RTX 5090","capability":[12,0],"mean":16.0}',
    )

    ok, message = launcher._cuda_runtime_probe(str(python), {"PATH": ""})

    assert ok
    assert "RTX 5090" in message
    assert "sm_120" in message
    assert "2.9.1+cu128" in message


def test_cuda_runtime_probe_surfaces_initialization_errors(tmp_path: Path) -> None:
    launcher = _load_native_launcher()
    python = _fake_cuda_probe_python(
        tmp_path,
        '{"ok":false,"error":"RuntimeError: CUDA error 804"}',
    )

    ok, message = launcher._cuda_runtime_probe(str(python), {"PATH": ""})

    assert ok is False
    assert "error 804" in message


def test_cuda_toolkit_probe_accepts_matching_nvcc_and_headers(tmp_path: Path) -> None:
    launcher = _load_native_launcher()
    toolkit = _fake_nvcc(tmp_path, "12.9")
    (toolkit / "include").mkdir()
    (toolkit / "include" / "curand.h").write_text("", encoding="utf-8")
    track = launcher.native_track_for_sglang("0.5.10.post1")

    ok, message = launcher._cuda_toolkit_probe({"CUDA_HOME": str(toolkit), "PATH": ""}, track)

    assert ok
    assert "CUDA JIT toolkit 12.9" in message


def test_cuda_toolkit_probe_rejects_wrong_family_or_missing_header(
    tmp_path: Path,
) -> None:
    launcher = _load_native_launcher()
    cu12 = launcher.native_track_for_sglang("0.5.10.post1")
    cuda13 = _fake_nvcc(tmp_path, "13.0")

    ok, message = launcher._cuda_toolkit_probe({"CUDA_HOME": str(cuda13), "PATH": ""}, cu12)
    assert not ok
    assert "CUDA 12 family" in message

    cuda12 = _fake_nvcc(tmp_path, "12.9")
    ok, message = launcher._cuda_toolkit_probe({"CUDA_HOME": str(cuda12), "PATH": ""}, cu12)
    assert not ok
    assert "curand.h" in message


def test_bootstrap_rejects_driver_below_native_floor() -> None:
    launcher = _load_native_launcher()

    try:
        launcher.recommended_native_track("524.99")
    except ValueError as exc:
        assert "driver 525+" in str(exc)
    else:
        raise AssertionError("unsupported driver unexpectedly selected a track")


def test_language_only_rejection_covers_the_whole_0_5_10_line() -> None:
    """The bootstrap script and the doctor must agree on every patch pin.

    The launcher used to gate its ``--language-only`` strip on ``version ==
    "0.5.10.post1"`` while the doctor gated on the release line, so bumping
    ``recommended_sglang`` to any other 0.5.10.x left the flag in place and made
    the doctor fail the run instead.
    """
    launcher = _load_native_launcher()

    assert launcher.rejects_language_only("0.5.10")
    assert launcher.rejects_language_only("0.5.10.post1")
    assert launcher.rejects_language_only("0.5.10.post2")
    assert not launcher.rejects_language_only("0.5.9")
    assert not launcher.rejects_language_only("0.5.11")
    assert not launcher.rejects_language_only("0.5.17")
    # An unparseable version must not be claimed as broken.
    assert not launcher.rejects_language_only("not-a-version")


def test_sanitize_extra_args_only_rewrites_when_it_must() -> None:
    launcher = _load_native_launcher()
    extra = "--max-running-requests 1 --language-only"

    assert launcher.sanitize_extra_args("0.5.10.post2", extra) == ("--max-running-requests 1")
    # Releases that honour the flag keep it, byte for byte.
    assert launcher.sanitize_extra_args("0.5.17", extra) == extra
    # Nothing to drop => returned unchanged, so callers can use inequality as
    # "this was rewritten" without tripping on requoting.
    assert launcher.sanitize_extra_args("0.5.10.post1", '--foo "a b"') == '--foo "a b"'
    # A malformed string is the doctor's to report; raising here would abort the
    # bootstrap script before it can.
    assert launcher.sanitize_extra_args("0.5.10.post1", '--foo "x') == '--foo "x'


def test_recommended_track_pin_is_sanitized_by_the_same_rule() -> None:
    """Every driver the matrix supports yields a usable 5090-profile arg string."""
    launcher = _load_native_launcher()
    profile_extra = "--max-running-requests 1 --language-only"

    for driver in ("525", "550.127.08", "579.99", "580.1", "585"):
        pin = launcher.recommended_native_track(driver)["recommended_sglang"]
        sanitized = launcher.sanitize_extra_args(pin, profile_extra)
        assert not (launcher.rejects_language_only(pin) and "--language-only" in sanitized), (
            f"driver {driver} pins {pin} but keeps --language-only"
        )


def test_runtime_environment_adds_selected_python_bin_to_path(tmp_path: Path) -> None:
    launcher = _load_native_launcher()
    python = tmp_path / "venv" / "bin" / "python"

    environ = launcher.runtime_environment(
        {
            "SGLANG_PYTHON": str(python),
            "SGLANG_MODEL_ID": "private-org/qwen35-checkpoint",
        }
    )

    assert environ["PATH"].split(os.pathsep, 1)[0] == str(python.parent)
