from __future__ import annotations

import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
IMAGE = "ghcr.io/apodexai/frontieragent"
_COMPOSE_FALLBACK = re.compile(r"\$\{[A-Z_]+:-(?P<default>[^}]*)\}")


def _yaml(name: str) -> dict:
    return yaml.safe_load((ROOT / name).read_text(encoding="utf-8"))


def _fallback(interpolation: str) -> str:
    """The value Compose substitutes when the env file omits the variable."""
    match = _COMPOSE_FALLBACK.fullmatch(interpolation)
    assert match is not None, f"expected ${{VAR:-default}}, got {interpolation!r}"
    return match["default"]


def _dotenv(name: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in (ROOT / name).read_text(encoding="utf-8").splitlines():
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            values[key] = value
    return values


def test_default_compose_pulls_release_image_and_preserves_cli_state() -> None:
    compose = _yaml("compose.yaml")
    agent = compose["services"]["agent"]

    assert "build" not in agent
    assert IMAGE in agent["image"]
    assert agent["pull_policy"] == "always"
    assert agent["environment"]["APODEX_IN_CONTAINER"] == "1"
    assert agent["environment"]["SANDBOX_BACKEND"] == "container"
    assert "security_opt" not in agent
    assert ".:/project" in agent["volumes"]
    assert agent["working_dir"] == "/project"
    assert "./.apodex/runs:/apodex-runs" in agent["volumes"]
    assert "frontier-agent-inputs:/inputs:ro" in agent["volumes"]
    assert "frontier-agent-state:/root/.apodex" in agent["volumes"]
    assert agent["environment"]["APODEX_HOST_UID"] == "${APODEX_HOST_UID:-}"
    assert agent["environment"]["APODEX_HOST_GID"] == "${APODEX_HOST_GID:-}"
    assert agent["environment"]["APODEX_HOST_RUNS_ROOT"] == (
        "${APODEX_HOST_RUNS_ROOT:-.apodex/runs}"
    )
    assert agent["environment"]["APODEX_HOST_OUTPUTS_ROOT"] == (
        "${APODEX_HOST_RUNS_ROOT:-.apodex/runs}"
    )
    assert agent["environment"]["APODEX_SESSION_WORKSPACES_ROOT"] == (
        "/apodex-runs"
    )
    assert agent["environment"]["APODEX_WORKSPACE_LINK"] == "/workspace"


def test_development_compose_is_the_only_compose_file_that_builds() -> None:
    compose = _yaml("compose.yaml")
    development = _yaml("compose.dev.yaml")

    assert all("build" not in service for service in compose["services"].values())
    assert development["services"]["agent"]["build"]["context"] == "."
    assert development["services"]["eval"]["build"]["context"] == "."
    assert development["services"]["agent"]["pull_policy"] == "build"


def test_sglang_compose_mounts_an_optional_local_checkpoint_read_only() -> None:
    compose = _yaml("compose.sglang.yaml")
    model = compose["services"]["model"]

    assert model["environment"]["SGLANG_MODEL_ID"] == "${SGLANG_MODEL_ID:-}"
    assert model["environment"]["SGLANG_LOCAL_MODEL_PATH"] == (
        "${SGLANG_LOCAL_MODEL_PATH:+/opt/frontier-agent/local-model}"
    )
    assert model["environment"]["SGLANG_CHAT_TEMPLATE"] == (
        "${SGLANG_CHAT_TEMPLATE:-}"
    )
    local_mount = next(
        mount
        for mount in model["volumes"]
        if isinstance(mount, dict) and mount.get("target") == "/opt/frontier-agent/local-model"
    )
    assert local_mount == {
        "type": "bind",
        "source": "${SGLANG_LOCAL_MODEL_PATH:-./docker/empty-model}",
        "target": "/opt/frontier-agent/local-model",
        "read_only": True,
    }
    assert "./docker/smoke_sglang.py:/opt/frontier-agent/smoke_sglang.py:ro" in (model["volumes"])
    device = model["deploy"]["resources"]["reservations"]["devices"][0]
    assert device["count"] == "${SGLANG_GPU_COUNT:-1}"
    assert model["restart"] == "${SGLANG_RESTART_POLICY:-no}"
    assert model["logging"]["driver"] == "local"


def test_sglang_compose_fallback_budgets_fit_the_fallback_context() -> None:
    """A minimal env file that names only a model must still get a sane budget.

    Compose substitutes these defaults itself, so they are never seen by the
    launcher doctors: a mismatch here hands the agent a budget the doctors
    would have rejected while still reporting PASS.
    """
    agent = _yaml("compose.sglang.yaml")["services"]["agent"]["environment"]
    context = int(_fallback(agent["OPENAI_CONTEXT_WINDOW"]))
    inputs = int(_fallback(agent["OPENAI_MAX_INPUT_TOKENS"]))
    outputs = int(_fallback(agent["OPENAI_MAX_TOKENS"]))

    assert inputs + outputs <= context
    assert inputs * 10 > context * 8

    smoke = _dotenv(".env.sglang.example")
    assert (context, inputs, outputs) == (
        int(smoke["SGLANG_CONTEXT_LENGTH"]),
        int(smoke["SGLANG_MAX_INPUT_TOKENS"]),
        int(smoke["SGLANG_MAX_OUTPUT_TOKENS"]),
    )


def test_launcher_scripts_avoid_bash4_only_expansions() -> None:
    """macOS still ships bash 3.2, where ${var,,} is a fatal bad substitution.

    `bash -n` cannot catch this because it does not expand parameters, and CI
    runs bash 5 — so the failure only ever reaches a macOS operator.
    """
    # Positional parameters count too: ${1,,} was the actual regression.
    case_modification = re.compile(r"\$\{[A-Za-z_0-9]+(,,|\^\^)")
    for name in (
        "docker/run-sglang.sh",
        "docker/run.sh",
        "docker/sglang-doctor.sh",
        "docker/entrypoint.sh",
    ):
        text = (ROOT / name).read_text(encoding="utf-8")
        assert not case_modification.search(text), name


def test_smoke_profile_has_safe_context_and_explicit_scope() -> None:
    env = _dotenv(".env.sglang.example")

    assert env["SGLANG_PROFILE"] == "smoke-0.8b"
    assert env["SGLANG_MODEL_ID"] == "apodex/Apodex-1.0-0.8B-SFT"
    assert env["SGLANG_IMAGE"].endswith("-runtime")
    assert int(env["SGLANG_MAX_INPUT_TOKENS"]) + int(env["SGLANG_MAX_OUTPUT_TOKENS"]) <= int(
        env["SGLANG_CONTEXT_LENGTH"]
    )
    assert int(env["SGLANG_MAX_INPUT_TOKENS"]) * 10 > (int(env["SGLANG_CONTEXT_LENGTH"]) * 8)
    assert env["SGLANG_GPU_COUNT"] == "1"
    assert env["SGLANG_RESTART_POLICY"] == "no"


#: Every 35B template points at the published production checkpoint. A local
#: export still wins when set, which is why ``SGLANG_LOCAL_MODEL_PATH`` stays
#: empty in the committed examples rather than being pre-filled with a path
#: that exists on nobody else's disk.
_PRODUCTION_MODEL_ID = "apodex/Apodex-1.1-mini"


def test_35b_profiles_select_a_checkpoint_and_fit_context() -> None:
    generic_profiles = (
        "config/sglang/35b-4090.env.example",
        "config/sglang/35b-multigpu.env.example",
    )
    for name in generic_profiles:
        env = _dotenv(name)
        assert env["SGLANG_MODEL_ID"] == _PRODUCTION_MODEL_ID
        assert env["SGLANG_LOCAL_MODEL_PATH"] == ""
        # Left to checkpoint metadata on these two: unlike the 5090 template,
        # neither pins a quantization backend, so they load whatever format the
        # repository ships.
        assert env["SGLANG_QUANTIZATION"] == ""
        assert int(env["SGLANG_MAX_INPUT_TOKENS"]) + int(env["SGLANG_MAX_OUTPUT_TOKENS"]) <= int(
            env["SGLANG_CONTEXT_LENGTH"]
        )
        assert int(env["SGLANG_MAX_INPUT_TOKENS"]) * 10 > (int(env["SGLANG_CONTEXT_LENGTH"]) * 8)

    env = _dotenv("config/sglang/35b-5090.env.example")
    assert env["SGLANG_MODEL_ID"] == _PRODUCTION_MODEL_ID
    assert env["SGLANG_LOCAL_MODEL_PATH"] == ""
    assert env["SGLANG_QUANTIZATION"] == "moe_wna16"
    assert env["SGLANG_TOOL_CALL_PARSER"] == "qwen3_coder"
    assert env["SGLANG_REASONING_PARSER"] == "qwen3"
    assert "--language-only" in env["SGLANG_EXTRA_ARGS"]
    assert int(env["SGLANG_MAX_INPUT_TOKENS"]) + int(env["SGLANG_MAX_OUTPUT_TOKENS"]) <= int(
        env["SGLANG_CONTEXT_LENGTH"]
    )
    assert int(env["SGLANG_MAX_INPUT_TOKENS"]) * 10 > (
        int(env["SGLANG_CONTEXT_LENGTH"]) * 8
    )


def test_optional_network_override_requires_an_explicit_subnet() -> None:
    network = _yaml("compose.network.yaml")["networks"]["default"]
    subnet = network["ipam"]["config"][0]["subnet"]

    assert subnet.startswith("${APODEX_DOCKER_SUBNET:?")


def test_gpu_launcher_exposes_diagnostic_lifecycle_commands() -> None:
    launcher = (ROOT / "docker/run-sglang.sh").read_text(encoding="utf-8")
    native_launcher = (ROOT / "scripts/run-sglang-native.py").read_text(encoding="utf-8")
    for command in ("doctor", "up", "smoke", "tui", "status", "logs", "down"):
        assert f"  {command}" in launcher
        assert f'"{command}"' in native_launcher
    assert "SGLANG_BUILD_AGENT" in launcher
    assert "APODEX_DOCKER_SUBNET" in launcher
    assert "APODEX_HOST_UID" in launcher
    assert "SGLANG_NATIVE_HOST" in native_launcher
    assert "SGLANG_PYTHON" in native_launcher


def test_clone_to_run_helpers_cover_remote_and_native_gpu_linux() -> None:
    remote = (ROOT / "scripts/run-linux.sh").read_text(encoding="utf-8")
    gpu = (ROOT / "scripts/run-linux-gpu.sh").read_text(encoding="utf-8")

    assert "OPENAI_BASE_URL" in remote
    assert "This helper does not install or run a local model server" in remote
    assert "recommended_native_track" in gpu
    assert 'sglang==$sglang_version' in gpu
    assert ".venv-sglang" in gpu
    assert "install_system_deps" in gpu
    assert "never install or upgrade anything" in gpu


def test_linux_launchers_survive_bash_before_4_4() -> None:
    """No bare empty-array expansion under ``set -u``.

    bash before 4.4 (CentOS/RHEL 7 ship 4.2 -- a distro ``install_libnuma``
    explicitly supports) treats an EMPTY array as unset under ``set -u`` and
    aborts. ``runtime_args`` / ``forwarded`` / ``privilege`` are all empty on
    ordinary paths, so a bare ``"${a[@]}"`` killed the launcher after the
    install had already run. ``${a[@]+"${a[@]}"}`` is safe on every version.
    """
    for name in ("scripts/run-linux.sh", "scripts/run-linux-gpu.sh"):
        script = (ROOT / name).read_text(encoding="utf-8")
        for array in ("runtime_args", "forwarded", "privilege"):
            bare = f'"${{{array}[@]}}"'
            guarded = f'${{{array}[@]+"${{{array}[@]}}"}}'
            offending = [
                line for line in script.splitlines()
                if bare in line and guarded not in line
            ]
            assert not offending, f"{name}: unguarded {bare} in {offending}"


def test_gpu_launcher_defers_to_operator_configuration() -> None:
    """Env-over-file precedence, the same order ``runtime_environment`` uses.

    Both settings are read as "environment if set, else the dotenv file". Reading
    only the shell environment made the AutoDL cache default override the
    ``SGLANG_DOWNLOAD_DIR`` / ``HF_HOME`` / ``HF_HUB_CACHE`` configured in
    ``.env.sglang``, and made the ``--language-only`` strip replace an exported
    ``SGLANG_EXTRA_ARGS`` with the profile's value.
    """
    gpu = (ROOT / "scripts/run-linux-gpu.sh").read_text(encoding="utf-8")

    for key in ("SGLANG_DOWNLOAD_DIR", "HF_HOME", "HF_HUB_CACHE", "SGLANG_EXTRA_ARGS"):
        assert f"${{{key}-$(dotenv_value {key})}}" in gpu, key
    # The release-line rule lives in one place, shared with the doctor.
    assert "sanitize_extra_args" in gpu
    assert "0.5.10.post1" not in gpu


def test_container_entrypoint_can_align_tool_files_with_host_identity() -> None:
    entrypoint = (ROOT / "docker/entrypoint.sh").read_text(encoding="utf-8")
    sandbox = (ROOT / "plugins/tools/_sandbox.py").read_text(encoding="utf-8")

    assert "align_tool_identity" in entrypoint
    assert "APODEX_HOST_UID" in entrypoint
    assert "APODEX_TOOL_HOST_IDENTITY" in entrypoint
    assert "os.chown(target, identity.uid, identity.gid)" in sandbox
    # The mounts include the host checkout itself, so the harness may only add
    # permission bits; replacing the mode outlives the container.
    assert "target.chmod(0o770)" not in sandbox
    assert "target.chmod(current | 0o777)" in sandbox


def test_eval_service_keeps_the_identity_aligning_entrypoint() -> None:
    eval_service = _yaml("compose.yaml")["services"]["eval"]

    assert eval_service["entrypoint"][0] == "/app/docker/entrypoint.sh"
    assert eval_service["entrypoint"][1:] == [
        "python",
        "-m",
        "benchmarks.public.runner.run_subprocess",
    ]


def test_publish_workflow_uses_canonical_image_and_runtime_smoke() -> None:
    workflow = (ROOT / ".github/workflows/docker-publish.yml").read_text(encoding="utf-8")

    assert f"images: {IMAGE}" in workflow
    assert "type=sha,format=long" in workflow
    assert "docker logout ghcr.io" in workflow
    assert workflow.index('docker pull "$image"') < workflow.index("docker logout ghcr.io")
    assert "for attempt in 1 2 3 4 5 6" in workflow
    assert "Published image is not anonymously pullable" in workflow
    assert 'docker run --rm "$image" --version' in workflow
    assert "python tools/import_smoke.py --stage 2" in workflow


def test_user_docs_use_the_published_registry_name() -> None:
    # The container recipes live in docs/install/docker.md; the README links to it
    # rather than repeating them.
    paths = [ROOT / "docs/install/docker.md", ROOT / "compose.yaml"]

    assert all(IMAGE in path.read_text(encoding="utf-8") for path in paths)
    assert "docs/install/docker.md" in (ROOT / "README.md").read_text(encoding="utf-8")

    # The hyphenated spelling is not the published name and must appear nowhere.
    for path in [*paths, ROOT / "README.md"]:
        assert "ghcr.io/apodexai/frontier-agent" not in path.read_text(encoding="utf-8")


def test_gpu_docs_link_to_the_official_nvidia_toolkit_installation() -> None:
    guide = ROOT / "docs/install/linux-nvidia.md"
    assert guide.is_file()
    text = guide.read_text(encoding="utf-8")

    assert "docs.nvidia.com/datacenter/cloud-native/container-toolkit/" in text
    assert "install-guide.html" in text
    assert "sudo nvidia-ctk runtime configure --runtime=docker" in text
    assert "sudo systemctl restart docker" in text
    assert "35b-4090.env.example" in text
    assert "35b-5090.env.example" in text
    assert "docker login ghcr.io" in text
    assert "infrastructure" in text.lower()
    assert "capability" in text.lower()

    # The README is an overview, so what it must carry is the route to this guide.
    assert "docs/install/linux-nvidia.md" in (ROOT / "README.md").read_text(encoding="utf-8")
