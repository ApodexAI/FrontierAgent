from __future__ import annotations

import runpy
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import cast

import pytest

ROOT = Path(__file__).resolve().parents[2]
BuildCommand = Callable[[Mapping[str, str]], list[str]]
build_command = cast(
    BuildCommand,
    runpy.run_path(str(ROOT / "docker/sglang_entrypoint.py"))["build_command"],
)


def _option(command: list[str], name: str) -> str:
    return command[command.index(name) + 1]


def test_hugging_face_model_id_uses_the_persistent_download_cache() -> None:
    command = build_command({
        "SGLANG_MODEL_ID": "apodex/example",
        "SGLANG_SERVED_MODEL_NAME": "local-model",
    })

    assert _option(command, "--model-path") == "apodex/example"
    assert _option(command, "--served-model-name") == "local-model"
    assert _option(command, "--download-dir") == "/root/.cache/huggingface"


def test_native_runtime_can_select_python_listener_and_download_directory() -> None:
    command = build_command(
        {
            "SGLANG_MODEL_ID": "apodex/example",
            "SGLANG_SERVER_HOST": "127.0.0.1",
            "SGLANG_SERVER_PORT": "31000",
            "SGLANG_DOWNLOAD_DIR": "/models/huggingface",
        },
        python_executable="/opt/sglang/bin/python",
    )

    assert command[0] == "/opt/sglang/bin/python"
    assert _option(command, "--host") == "127.0.0.1"
    assert _option(command, "--port") == "31000"
    assert _option(command, "--download-dir") == "/models/huggingface"


def test_local_checkpoint_overrides_model_id_and_is_not_downloaded(tmp_path) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()

    command = build_command({
        "SGLANG_MODEL_ID": "apodex/fallback",
        "SGLANG_LOCAL_MODEL_PATH": str(checkpoint),
        "SGLANG_TP_SIZE": "2",
        "SGLANG_EXTRA_ARGS": "--max-running-requests 2",
    })

    assert _option(command, "--model-path") == str(checkpoint)
    assert _option(command, "--tp-size") == "2"
    assert "--download-dir" not in command
    assert command[-2:] == ["--max-running-requests", "2"]


def test_existing_optional_server_settings_keep_their_argument_boundaries() -> None:
    command = build_command({
        "SGLANG_MODEL_ID": "apodex/example",
        "SGLANG_CONTEXT_LENGTH": "65536",
        "SGLANG_TOOL_CALL_PARSER": "qwen3_coder",
        "SGLANG_REASONING_PARSER": "qwen3",
        "SGLANG_DTYPE": "bfloat16",
        "SGLANG_QUANTIZATION": "fp8",
        "SGLANG_TRUST_REMOTE_CODE": "true",
        "SGLANG_EXTRA_ARGS": '--chat-template "/templates/my model.jinja"',
    })

    assert _option(command, "--context-length") == "65536"
    assert _option(command, "--tool-call-parser") == "qwen3_coder"
    assert _option(command, "--reasoning-parser") == "qwen3"
    assert _option(command, "--dtype") == "bfloat16"
    assert _option(command, "--quantization") == "fp8"
    assert "--trust-remote-code" in command
    assert command[-2:] == ["--chat-template", "/templates/my model.jinja"]


def test_local_checkpoint_must_be_an_existing_directory(tmp_path) -> None:
    missing = tmp_path / "missing"

    with pytest.raises(SystemExit, match="existing checkpoint directory"):
        build_command({"SGLANG_LOCAL_MODEL_PATH": str(missing)})


def test_a_model_source_is_required() -> None:
    with pytest.raises(SystemExit, match="set SGLANG_MODEL_ID"):
        build_command({})
