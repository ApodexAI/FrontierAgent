from __future__ import annotations

import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ENTRYPOINT = ROOT / "docker" / "gpu-entrypoint.sh"


def _run(mode: str, **overrides: str) -> subprocess.CompletedProcess[str]:
    environ = {"PATH": os.environ.get("PATH", "")}
    environ.update(overrides)
    return subprocess.run(
        ["bash", str(ENTRYPOINT), mode],
        cwd=ROOT,
        env=environ,
        capture_output=True,
        text=True,
        check=False,
    )


def test_tui_without_a_model_prints_actionable_profile_help() -> None:
    result = _run("tui")

    assert result.returncode == 2
    assert "No SGLang model source is configured" in result.stderr
    assert "FRONTIER_AGENT_GPU_PROFILE=qwen35-gptq-cu12" in result.stderr
    assert "SGLANG_LOCAL_MODEL_PATH=/models/checkpoint" in result.stderr


def test_runtime_profile_does_not_select_a_test_repository() -> None:
    result = _run("tui", FRONTIER_AGENT_GPU_PROFILE="qwen35-gptq-cu12")

    assert result.returncode == 2
    assert "No SGLang model source is configured" in result.stderr
    assert "does not select a model repository" in result.stderr


def test_cu13_profile_does_not_select_a_test_repository() -> None:
    result = _run("tui", FRONTIER_AGENT_GPU_PROFILE="qwen35-gptq-cu13")

    assert result.returncode == 2
    assert "No SGLang model source is configured" in result.stderr
    assert "does not select a model repository" in result.stderr


def test_unknown_profile_lists_both_tracks() -> None:
    result = _run("tui", FRONTIER_AGENT_GPU_PROFILE="bogus")

    assert result.returncode == 2
    assert "available profiles: qwen35-gptq-cu12 qwen35-gptq-cu13" in result.stderr


def _stub_interpreter(tmp_path: Path, reported_version: str) -> str:
    """An ``SGLANG_PYTHON`` that reports *reported_version* for any argument.

    The profile/track rules are decided by the SGLang build actually present,
    so the tests must supply that version instead of depending on whatever is
    importable on the machine running the suite.
    """
    stub = tmp_path / "python3-stub"
    stub.write_text(f"#!/bin/sh\necho {reported_version}\n", encoding="utf-8")
    stub.chmod(0o755)
    return str(stub)


def test_cu13_profile_pins_bfloat16_and_language_only(tmp_path: Path) -> None:
    """The two settings the RTX 5090 bring-up had to discover the hard way.

    ``moe_wna16`` is the only MoE GPTQ path that loads, and the dtype must
    match the checkpoint's declared ``bfloat16`` or the fused kernels fail to
    compile. ``--language-only`` keeps the unused vision encoder out of VRAM.
    """
    result = _run(
        "doctor",
        FRONTIER_AGENT_GPU_PROFILE="qwen35-gptq-cu13",
        SGLANG_PYTHON=_stub_interpreter(tmp_path, "0.5.17"),
    )

    assert "SGLANG_DTYPE=bfloat16" in result.stdout
    assert "SGLANG_QUANTIZATION=moe_wna16" in result.stdout
    assert "SGLANG_EXTRA_ARGS=--max-running-requests 1 --language-only" in result.stdout


def test_cu12_profile_never_passes_language_only(tmp_path: Path) -> None:
    result = _run(
        "doctor",
        FRONTIER_AGENT_GPU_PROFILE="qwen35-gptq-cu12",
        SGLANG_PYTHON=_stub_interpreter(tmp_path, "0.5.10.post1"),
    )

    assert "SGLANG_DTYPE=auto" in result.stdout
    assert "SGLANG_QUANTIZATION=moe_wna16" in result.stdout
    assert "SGLANG_EXTRA_ARGS=--max-running-requests 1" in result.stdout
    assert "--language-only" not in result.stdout


def test_profile_settings_stay_overridable(tmp_path: Path) -> None:
    result = _run(
        "doctor",
        FRONTIER_AGENT_GPU_PROFILE="qwen35-gptq-cu13",
        SGLANG_PYTHON=_stub_interpreter(tmp_path, "0.5.17"),
        SGLANG_CONTEXT_LENGTH="131072",
    )

    assert "SGLANG_CONTEXT_LENGTH=131072" in result.stdout


def test_cu13_profile_refuses_to_start_on_a_cu12_image(tmp_path: Path) -> None:
    """The profile name is operator input; the running build is the authority.

    Both tags and both profiles are published, so the mismatch is reachable by
    a single wrong environment variable. Caught here it names the cause; left
    to SGLang it surfaces as a demand for ``--encoder-urls``.
    """
    result = _run(
        "server",
        FRONTIER_AGENT_GPU_PROFILE="qwen35-gptq-cu13",
        SGLANG_MODEL_ID="example/model",
        SGLANG_PYTHON=_stub_interpreter(tmp_path, "0.5.10.post1"),
    )

    assert result.returncode == 2
    assert "--language-only" in result.stderr
    assert "qwen35-gptq-cu12" in result.stderr


def test_cu12_profile_refuses_to_start_on_a_cu13_image(tmp_path: Path) -> None:
    result = _run(
        "server",
        FRONTIER_AGENT_GPU_PROFILE="qwen35-gptq-cu12",
        SGLANG_MODEL_ID="example/model",
        SGLANG_PYTHON=_stub_interpreter(tmp_path, "0.5.17"),
    )

    assert result.returncode == 2
    assert "0.5.17" in result.stderr
    assert "qwen35-gptq-cu13" in result.stderr


def test_matching_track_starts_without_a_profile_complaint(tmp_path: Path) -> None:
    """An unreadable version must not block startup either (see below)."""
    for version in ("0.5.17", "not-a-version"):
        result = _run(
            "tui",
            FRONTIER_AGENT_GPU_PROFILE="qwen35-gptq-cu13",
            SGLANG_PYTHON=_stub_interpreter(tmp_path, version),
        )

        # Stops at the model source, having cleared the track check.
        assert result.returncode == 2, version
        assert "No SGLang model source is configured" in result.stderr, version
        assert "encoder disaggregation" not in result.stderr, version


def test_help_lists_gpu_image_modes() -> None:
    result = _run("help")

    assert result.returncode == 0
    assert "server|tui|agent|doctor|shell" in result.stdout
