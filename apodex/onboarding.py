"""First-run deployment guidance and local capability detection.

The onboarding flow deliberately does not validate credentials over the
network: doing so could create billable traffic or leak a key to a mistyped
endpoint.  It checks the resolved profile locally and reports which pieces of
an NVIDIA container stack are available.  Both line mode and the Textual
front-end consume the same immutable :class:`OnboardingProbe`.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

APODEX_API = "apodex_api"
THIRD_PARTY = "third_party"
NVIDIA_SGLANG = "nvidia_sglang"
NVIDIA_TRANSFORMERS = "nvidia_transformers"

DEPLOYMENT_CHOICES = (
    APODEX_API,
    THIRD_PARTY,
    NVIDIA_SGLANG,
    NVIDIA_TRANSFORMERS,
)

DEPLOYMENT_LABELS = {
    APODEX_API: "Apodex API platform",
    THIRD_PARTY: "Third-party API key (BYOK)",
    NVIDIA_SGLANG: "Local NVIDIA container · SGLang",
    NVIDIA_TRANSFORMERS: "Local NVIDIA container · Transformers",
}


@dataclass(frozen=True)
class OnboardingProbe:
    """Secret-free facts shown by either onboarding UI."""

    env_path: str | None
    configured_key_vars: tuple[str, ...]
    active_config_ok: bool
    active_config_summary: str
    nvidia_smi: bool
    gpu_summary: str
    docker: bool
    docker_daemon: bool
    docker_compose: bool
    nvidia_runtime: bool
    managed_local_service: bool = False
    managed_local_runtime: str = ""

    @property
    def nvidia_container_ready(self) -> bool:
        return self.managed_local_service or (
            self.nvidia_smi
            and self.docker
            and self.docker_daemon
            and self.docker_compose
            and self.nvidia_runtime
        )

    @property
    def missing_nvidia_components(self) -> tuple[str, ...]:
        if self.managed_local_service:
            return ()
        missing: list[str] = []
        if not self.nvidia_smi:
            missing.append("NVIDIA driver / nvidia-smi")
        if not self.docker:
            missing.append("Docker CLI")
        elif not self.docker_daemon:
            missing.append("reachable Docker daemon")
        if not self.docker_compose:
            missing.append("Docker Compose v2")
        if self.docker_daemon and not self.nvidia_runtime:
            missing.append("NVIDIA Container Toolkit runtime")
        return tuple(missing)

    @property
    def suggested_choice(self) -> str:
        if self.managed_local_runtime == "sglang":
            return NVIDIA_SGLANG
        if self.managed_local_runtime == "transformers":
            return NVIDIA_TRANSFORMERS
        return APODEX_API


def _run_probe(command: Sequence[str], *, timeout: float = 3.0) -> tuple[bool, str]:
    try:
        result = subprocess.run(
            list(command),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return False, ""
    output = (result.stdout or result.stderr or "").strip()
    return result.returncode == 0, output


def _find_env_file(cwd: str) -> str | None:
    """Mirror CLI dotenv discovery without importing or exposing its values."""
    current = Path(cwd).resolve()
    for directory in (current, *current.parents):
        candidate = directory / ".env"
        if candidate.is_file():
            return str(candidate)
    return None


_KEY_VARS = (
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "BEDROCK_API_KEY",
    "DEEPSEEK_API_KEY",
    "DASHSCOPE_API_KEY",
    "OPENROUTER_API_KEY",
    "LOCAL_API_KEY",
)


def detect_onboarding_environment(
    *,
    cwd: str,
    runtime_status: object,
    environ: Mapping[str, str] | None = None,
    env_path: str | None = None,
    runner: Callable[[Sequence[str]], tuple[bool, str]] | None = None,
) -> OnboardingProbe:
    """Detect locally usable credentials and NVIDIA-container prerequisites.

    ``runner`` is injectable to keep tests deterministic.  A configured key is
    reported only by variable name; its value never enters this object.
    """
    env = os.environ if environ is None else environ
    run = runner or _run_probe
    configured = tuple(name for name in _KEY_VARS if (env.get(name) or "").strip())

    nvidia_smi = shutil.which("nvidia-smi") is not None
    gpu_summary = ""
    if nvidia_smi:
        nvidia_smi, gpu_output = run((
            "nvidia-smi",
            "--query-gpu=name,memory.total",
            "--format=csv,noheader",
        ))
        if nvidia_smi:
            gpu_summary = "; ".join(
                line.strip() for line in gpu_output.splitlines() if line.strip()
            )

    docker = shutil.which("docker") is not None
    docker_daemon = docker_compose = nvidia_runtime = False
    if docker:
        docker_daemon, runtime_output = run((
            "docker", "info", "--format", "{{json .Runtimes}}",
        ))
        if docker_daemon:
            nvidia_runtime = "nvidia" in runtime_output.lower()
        docker_compose, _ = run(("docker", "compose", "version", "--short"))

    active_ok = bool(getattr(runtime_status, "ok", False))
    provider = str(getattr(runtime_status, "provider", "custom") or "custom")
    model = str(getattr(runtime_status, "model", "") or "missing")
    key_ready = bool(getattr(runtime_status, "api_key_configured", False))
    active_summary = (
        f"provider={provider}, model={model}, API key="
        f"{'configured' if key_ready else 'missing'}"
    )
    local_runtime = (env.get("APODEX_LOCAL_RUNTIME") or "").strip().lower()
    if local_runtime not in {"sglang", "transformers"}:
        local_runtime = ""
    managed_local_service = (
        env.get("APODEX_IN_CONTAINER", "").strip() == "1"
        and provider == "local"
        and active_ok
    )
    return OnboardingProbe(
        env_path=env_path or _find_env_file(cwd),
        configured_key_vars=configured,
        active_config_ok=active_ok,
        active_config_summary=active_summary,
        nvidia_smi=nvidia_smi,
        gpu_summary=gpu_summary,
        docker=docker,
        docker_daemon=docker_daemon,
        docker_compose=docker_compose,
        nvidia_runtime=nvidia_runtime,
        managed_local_service=managed_local_service,
        managed_local_runtime=local_runtime if managed_local_service else "",
    )


def choice_guidance(choice: str, probe: OnboardingProbe) -> str:
    """Return concise next steps for a deployment choice."""
    if choice == APODEX_API:
        ready = "Current .env is ready." if probe.active_config_ok else (
            "Add the key, OpenAI-compatible base URL, and model from "
            "https://platform.apodex.ai to .env."
        )
        return (
            f"{ready}\nRequired variables: OPENAI_API_KEY, OPENAI_BASE_URL, "
            "OPENAI_MODEL. Then run: frontier-agent"
        )
    if choice == THIRD_PARTY:
        ready = "Current .env is ready." if probe.active_config_ok else (
            "Point the generic OpenAI-compatible settings at your provider."
        )
        return (
            f"{ready}\nSet OPENAI_API_KEY, OPENAI_BASE_URL, and OPENAI_MODEL "
            "in .env. Then run: frontier-agent"
        )
    if choice == NVIDIA_SGLANG:
        readiness = _local_readiness(probe)
        return (
            f"{readiness}\nCopy .env.sglang.example to .env.sglang, choose the "
            "model and GPU settings, then run: ./docker/run-sglang.sh"
        )
    if choice == NVIDIA_TRANSFORMERS:
        readiness = _local_readiness(probe)
        return (
            f"{readiness}\nTransformers Serve is best for evaluation or moderate "
            "load; SGLang is recommended for production. Copy "
            ".env.transformers.example to .env.transformers, choose a model, "
            "then run: ./docker/run-transformers.sh"
        )
    raise ValueError(f"unknown deployment choice: {choice}")


def _local_readiness(probe: OnboardingProbe) -> str:
    if probe.managed_local_service:
        return "Connected to the configured local model service."
    if probe.nvidia_container_ready:
        return "NVIDIA container stack detected."
    return "Missing: " + ", ".join(probe.missing_nvidia_components)


def format_probe(probe: OnboardingProbe) -> str:
    env = probe.env_path or "not found"
    keys = ", ".join(probe.configured_key_vars) or "none"
    if probe.managed_local_service:
        gpu = "managed by the connected model service"
        container = "connected to local model service"
    else:
        gpu = probe.gpu_summary or ("available" if probe.nvidia_smi else "not detected")
        container = (
            "ready" if probe.nvidia_container_ready
            else "missing " + ", ".join(probe.missing_nvidia_components)
        )
    return (
        f".env: {env}\n"
        f"configured key variables: {keys}\n"
        f"active profile: {probe.active_config_summary}\n"
        f"NVIDIA GPU: {gpu}\n"
        f"NVIDIA containers: {container}"
    )


def run_line_onboarding(probe: OnboardingProbe) -> str | None:
    """Run the first-use selector in an interactive line-mode terminal."""
    print("\nWelcome to FrontierAgent — first-time setup\n")
    print(format_probe(probe))
    print("\nChoose how FrontierAgent should reach a model:")
    for index, choice in enumerate(DEPLOYMENT_CHOICES, start=1):
        recommended = " (recommended)" if choice == probe.suggested_choice else ""
        print(f"  {index}. {DEPLOYMENT_LABELS[choice]}{recommended}")
    default_index = DEPLOYMENT_CHOICES.index(probe.suggested_choice) + 1
    while True:
        try:
            raw = input(
                f"Selection [1-4, Enter for {default_index}, q to quit]: "
            ).strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            return None
        if raw in {"q", "quit", "exit"}:
            return None
        if not raw:
            raw = str(default_index)
        if raw.isdigit() and 1 <= int(raw) <= len(DEPLOYMENT_CHOICES):
            choice = DEPLOYMENT_CHOICES[int(raw) - 1]
            print("\n" + choice_guidance(choice, probe) + "\n")
            return choice
        print("Enter 1, 2, 3, 4, or q.")


__all__ = [
    "APODEX_API",
    "DEPLOYMENT_CHOICES",
    "DEPLOYMENT_LABELS",
    "NVIDIA_SGLANG",
    "NVIDIA_TRANSFORMERS",
    "THIRD_PARTY",
    "OnboardingProbe",
    "choice_guidance",
    "detect_onboarding_environment",
    "format_probe",
    "run_line_onboarding",
]
