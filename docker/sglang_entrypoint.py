#!/usr/bin/env python3
"""Translate Compose environment variables into an SGLang server command."""

from __future__ import annotations

import os
import shlex
from collections.abc import Mapping
from pathlib import Path


def _value(
    name: str,
    default: str = "",
    *,
    environ: Mapping[str, str] = os.environ,
) -> str:
    return environ.get(name, default).strip()


def _enabled(name: str, *, environ: Mapping[str, str] = os.environ) -> bool:
    return _value(name, environ=environ).lower() in {"1", "true", "yes", "on"}


def build_command(
    environ: Mapping[str, str] = os.environ,
    *,
    python_executable: str = "python3",
) -> list[str]:
    """Build the SGLang argv from a Hugging Face ID or local checkpoint."""
    local_model = _value("SGLANG_LOCAL_MODEL_PATH", environ=environ)
    model_id = _value("SGLANG_MODEL_ID", environ=environ)

    if local_model:
        local_path = Path(local_model)
        if not local_path.is_dir():
            raise SystemExit(
                "SGLANG_LOCAL_MODEL_PATH must be an existing checkpoint directory: "
                f"{local_model}"
            )
        model = str(local_path)
    elif model_id:
        model = model_id
    else:
        raise SystemExit(
            "set SGLANG_MODEL_ID or mount a checkpoint with "
            "SGLANG_LOCAL_MODEL_PATH"
        )

    command = [
        python_executable,
        "-m",
        "sglang.launch_server",
        "--model-path",
        model,
        "--served-model-name",
        _value("SGLANG_SERVED_MODEL_NAME", model, environ=environ),
        "--host",
        _value("SGLANG_SERVER_HOST", "0.0.0.0", environ=environ),
        "--port",
        _value("SGLANG_SERVER_PORT", "30000", environ=environ),
        "--tp-size",
        _value("SGLANG_TP_SIZE", "1", environ=environ),
        "--mem-fraction-static",
        _value("SGLANG_MEM_FRACTION_STATIC", "0.88", environ=environ),
    ]
    if not local_model:
        command.extend(
            (
                "--download-dir",
                _value(
                    "SGLANG_DOWNLOAD_DIR",
                    "/root/.cache/huggingface",
                    environ=environ,
                ),
            )
        )

    optional_arguments = {
        "SGLANG_CONTEXT_LENGTH": "--context-length",
        "SGLANG_TOOL_CALL_PARSER": "--tool-call-parser",
        "SGLANG_REASONING_PARSER": "--reasoning-parser",
        # A checkpoint whose packaged chat template is not the one to serve with
        # needs an explicit override — Apodex ships a thinking-enabled Jinja
        # template alongside the weights. In Docker the path must be visible
        # INSIDE the container, so pass a path under the mounted model
        # directory rather than an arbitrary host path.
        "SGLANG_CHAT_TEMPLATE": "--chat-template",
        "SGLANG_DTYPE": "--dtype",
        "SGLANG_QUANTIZATION": "--quantization",
    }
    for environment_name, argument_name in optional_arguments.items():
        if value := _value(environment_name, environ=environ):
            command.extend((argument_name, value))

    if _enabled("SGLANG_TRUST_REMOTE_CODE", environ=environ):
        command.append("--trust-remote-code")

    if extra := _value("SGLANG_EXTRA_ARGS", environ=environ):
        command.extend(shlex.split(extra))

    return command


def main() -> None:
    command = build_command(
        python_executable=_value("SGLANG_PYTHON", "python3"),
    )
    print("Starting SGLang:", shlex.join(command), flush=True)
    os.execvp(command[0], command)


if __name__ == "__main__":
    main()
