#!/usr/bin/env python3
"""Run SGLang and FrontierAgent directly on a Linux GPU environment.

This path is for managed GPU containers such as AutoDL, or Linux hosts where
SGLang is already installed natively. It deliberately does not install or nest
Docker. The Docker and native launchers share the same ``.env.sglang`` profiles
and SGLang command builder.
"""

from __future__ import annotations

import argparse
import ctypes.util
import importlib.util
import json
import os
import re
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Mapping
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ENV_FILE = REPO_ROOT / ".env.sglang"
DEFAULT_STATE_DIR = REPO_ROOT / ".apodex" / "sglang-native"
COMPATIBILITY_FILE = REPO_ROOT / "config" / "sglang" / "compatibility.json"
_DOTENV_KEY = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
IMPORT_PROBE_TIMEOUT = 180
NVIDIA_SMI_TIMEOUT = 30
PACKAGE_VERSION_TIMEOUT = 30
CUDA_PROBE_TIMEOUT = 60


def load_dotenv(path: Path) -> dict[str, str]:
    """Read the simple, non-executable dotenv format used by our profiles."""
    values: dict[str, str] = {}
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        if "=" not in line:
            raise ValueError(f"{path}:{line_number}: expected KEY=VALUE")
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not _DOTENV_KEY.fullmatch(key):
            raise ValueError(f"{path}:{line_number}: invalid environment key {key!r}")
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        values[key] = value
    return values


def runtime_environment(config: Mapping[str, str]) -> dict[str, str]:
    """Build native server settings without exposing SGLang on a public NIC."""
    environ = dict(config)
    environ.update(os.environ)
    port = environ.get("SGLANG_PORT", "30000").strip() or "30000"
    if not port.isdigit() or not 1 <= int(port) <= 65535:
        raise ValueError("SGLANG_PORT must be an integer between 1 and 65535")
    startup_timeout = environ.get("SGLANG_STARTUP_TIMEOUT", "3600").strip() or "3600"
    if not startup_timeout.isdigit() or int(startup_timeout) < 1:
        raise ValueError("SGLANG_STARTUP_TIMEOUT must be a positive integer")
    environ["SGLANG_STARTUP_TIMEOUT"] = startup_timeout
    environ["SGLANG_SERVER_HOST"] = (
        environ.get("SGLANG_NATIVE_HOST", "127.0.0.1").strip() or "127.0.0.1"
    )
    environ["SGLANG_SERVER_PORT"] = port
    if not environ.get("SGLANG_DOWNLOAD_DIR", "").strip():
        hf_home = environ.get("HF_HOME", "").strip()
        environ["SGLANG_DOWNLOAD_DIR"] = hf_home or str(Path.home() / ".cache" / "huggingface")
    for key in ("SGLANG_DOWNLOAD_DIR", "SGLANG_LOCAL_MODEL_PATH"):
        raw = environ.get(key, "").strip()
        if raw:
            path = Path(raw).expanduser()
            environ[key] = str(path if path.is_absolute() else REPO_ROOT / path)
    # SGLang JIT compilation launches helpers such as ``ninja`` by name. A
    # native virtual environment can supply those console scripts even though
    # the launcher itself invokes its Python by absolute path, so propagate the
    # interpreter's bin directory to all server children.
    python_bin = str(Path(select_sglang_python(environ)).expanduser().parent)
    current_path = environ.get("PATH", "")
    environ["PATH"] = os.pathsep.join(part for part in (python_bin, current_path) if part)
    environ["SGLANG_BASE_URL"] = f"http://{client_host(environ)}:{port}"
    return environ


def client_host(environ: Mapping[str, str]) -> str:
    """Return the address a client must dial to reach the configured listener.

    A wildcard bind is reachable over loopback, but a specific NIC is not: with
    ``SGLANG_NATIVE_HOST=10.0.0.5`` the server never answers on 127.0.0.1, so a
    hardcoded loopback URL would make the health probe poll until the startup
    timeout against an already-healthy server.
    """
    host = environ["SGLANG_SERVER_HOST"]
    if host in {"0.0.0.0", "::", "*"}:
        return "127.0.0.1"
    if ":" in host:
        return f"[{host}]"
    return host


def _load_command_builder() -> Any:
    path = REPO_ROOT / "docker" / "sglang_entrypoint.py"
    spec = importlib.util.spec_from_file_location("frontier_sglang_entrypoint", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load SGLang command builder from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.build_command


def select_sglang_python(environ: Mapping[str, str]) -> str:
    configured = environ.get("SGLANG_PYTHON", "").strip()
    if configured:
        path = Path(configured).expanduser()
        return str(path if path.is_absolute() else REPO_ROOT / path)
    local = REPO_ROOT / ".venv-sglang" / "bin" / "python"
    if local.is_file():
        return str(local)
    return shutil.which("python3") or sys.executable


def build_server_command(environ: Mapping[str, str]) -> list[str]:
    builder = _load_command_builder()
    return builder(
        environ,
        python_executable=select_sglang_python(environ),
    )


def state_dir(environ: Mapping[str, str]) -> Path:
    configured = environ.get("SGLANG_NATIVE_STATE_DIR", "").strip()
    if not configured:
        return DEFAULT_STATE_DIR
    path = Path(configured).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def _pid_path(environ: Mapping[str, str]) -> Path:
    return state_dir(environ) / "server.pid"


def _log_path(environ: Mapping[str, str]) -> Path:
    return state_dir(environ) / "server.log"


def read_pid(environ: Mapping[str, str]) -> int | None:
    try:
        pid = int(_pid_path(environ).read_text(encoding="utf-8").strip())
    except (FileNotFoundError, ValueError, OSError):
        return None
    return pid if pid > 1 else None


def _pid_is_alive(pid: int) -> bool:
    """Signal-0 liveness, for hosts that do not expose procfs."""
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, PermissionError):
        # PermissionError means the PID now belongs to another user, so it is
        # neither ours to trust nor ours to signal.
        return False
    return True


def is_managed_process(pid: int) -> bool:
    """Avoid signalling a recycled PID that no longer belongs to SGLang."""
    try:
        command = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ")
    except OSError:
        # Without procfs (macOS, minimal images) fall back to plain liveness.
        # Reporting a live server as gone would make `up` claim it exited while
        # the detached child keeps the port and GPU, and `down` would then
        # refuse to reclaim them.
        return _pid_is_alive(pid)
    return b"sglang.launch_server" in command


def health_url(environ: Mapping[str, str]) -> str:
    return f"{environ['SGLANG_BASE_URL']}/health"


def is_healthy(environ: Mapping[str, str]) -> bool:
    try:
        with urllib.request.urlopen(health_url(environ), timeout=2) as response:
            return response.status == 200
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


def _positive_int(value: str) -> bool:
    return value.isdigit() and int(value) > 0


def _module_available(python: str, module: str, environ: Mapping[str, str]) -> bool | None:
    """Import *module* out of process; ``None`` when the probe outlived its budget.

    Importing sglang pulls in torch and the CUDA runtime, which on a cold page
    cache routinely takes minutes on a fresh managed GPU instance. A timeout
    there says nothing about whether the package is installed, so it must not
    abort the doctor or be reported as a missing dependency.
    """
    try:
        probe = subprocess.run(
            [python, "-c", f"import {module}"],
            capture_output=True,
            text=True,
            timeout=IMPORT_PROBE_TIMEOUT,
            env=dict(environ),
            check=False,
        )
    except subprocess.TimeoutExpired:
        return None
    return probe.returncode == 0


def _package_version(python: str, package: str, environ: Mapping[str, str]) -> str | None:
    """Read package metadata from the selected interpreter without importing CUDA."""
    probe = subprocess.run(
        [
            python,
            "-c",
            (f"import importlib.metadata; print(importlib.metadata.version({package!r}))"),
        ],
        capture_output=True,
        text=True,
        timeout=PACKAGE_VERSION_TIMEOUT,
        env=dict(environ),
        check=False,
    )
    if probe.returncode != 0:
        return None
    return probe.stdout.strip() or None


def _cuda_runtime_probe(python: str, environ: Mapping[str, str]) -> tuple[bool | None, str]:
    """Initialize CUDA and run one tiny BF16 matmul in the SGLang runtime.

    Package imports alone do not initialize the driver. A managed container can
    therefore pass the package checks while an incompatible forward-compat
    ``libcuda`` still makes the first real CUDA call fail with error 804 or 35.
    ``None`` means the interpreter did not return a probe result we can trust.
    """
    source = """
import json
try:
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("torch.cuda.is_available() is false")
    device = torch.cuda.get_device_properties(0)
    left = torch.ones((16, 16), device="cuda", dtype=torch.bfloat16)
    result = left @ left
    torch.cuda.synchronize()
    print(json.dumps({
        "ok": True,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "device": device.name,
        "capability": list(torch.cuda.get_device_capability(0)),
        "mean": float(result.float().mean().item()),
    }))
except Exception as exc:
    print(json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}"}))
"""
    try:
        probe = subprocess.run(
            [python, "-c", source],
            capture_output=True,
            text=True,
            timeout=CUDA_PROBE_TIMEOUT,
            env=dict(environ),
            check=False,
        )
    except subprocess.TimeoutExpired:
        return None, f"CUDA runtime probe exceeded {CUDA_PROBE_TIMEOUT}s"
    try:
        payload = json.loads(probe.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError):
        detail = probe.stderr.strip() or probe.stdout.strip() or "no output"
        return None, f"CUDA runtime probe returned an unreadable result: {detail[:300]}"
    if not payload.get("ok"):
        return False, str(payload.get("error", "unknown CUDA initialization error"))
    capability = "".join(str(part) for part in payload["capability"])
    return (
        True,
        f"CUDA BF16 matmul works on {payload['device']} (sm_{capability}, "
        f"torch {payload['torch']}, CUDA {payload['torch_cuda']})",
    )


def _version_triplet(version: str) -> tuple[int, int, int] | None:
    match = re.match(r"^(\d+)\.(\d+)\.(\d+)", version)
    if not match:
        return None
    major, minor, patch = match.groups()
    return int(major), int(minor), int(patch)


#: The 0.5.10 release line reads ``--language-only`` as a request for encoder
#: disaggregation, so a standalone server started with it never becomes ready.
#: A half-open version range, not an equality on one pin: the bootstrap script
#: and :func:`doctor` must agree for every patch release the compatibility
#: matrix may recommend (``0.5.10``, ``0.5.10.post1``, ``0.5.10.post2``, ...).
_LANGUAGE_ONLY_BROKEN_RANGE = ((0, 5, 10), (0, 5, 11))


def rejects_language_only(sglang_version: str) -> bool:
    """True when *sglang_version* cannot serve standalone with ``--language-only``."""
    parsed = _version_triplet(sglang_version)
    if parsed is None:
        return False
    minimum, maximum_exclusive = _LANGUAGE_ONLY_BROKEN_RANGE
    return minimum <= parsed < maximum_exclusive


def sanitize_extra_args(sglang_version: str, extra_args: str) -> str:
    """Drop ``sglang.launch_server`` flags *sglang_version* cannot honour.

    Bootstrap scripts call this so the configuration they hand the server
    already satisfies the same rule :func:`doctor` enforces, instead of failing
    the doctor and stopping with nothing started.

    Returns *extra_args* byte-identical when there is nothing to drop, so a
    caller can use inequality as "this was rewritten" without re-quoting noise.
    A malformed string is returned unchanged too: reporting it is the doctor's
    job, and raising here would abort the launcher before it can.
    """
    if not rejects_language_only(sglang_version):
        return extra_args
    try:
        parsed = shlex.split(extra_args)
    except ValueError:
        return extra_args
    if "--language-only" not in parsed:
        return extra_args
    return shlex.join(arg for arg in parsed if arg != "--language-only")


def _compatibility_config() -> dict:
    """Load the checked-in native/container compatibility source of truth."""
    return json.loads(COMPATIBILITY_FILE.read_text(encoding="utf-8"))


def native_track_for_sglang(sglang_version: str) -> dict | None:
    """Return the configured CUDA track containing *sglang_version*."""
    parsed = _version_triplet(sglang_version)
    if parsed is None:
        return None
    for track in _compatibility_config()["native"]["tracks"]:
        minimum = _version_triplet(track["sglang_minimum"])
        maximum = _version_triplet(track.get("sglang_maximum_exclusive", ""))
        if minimum is None or parsed < minimum:
            continue
        if maximum is not None and parsed >= maximum:
            continue
        return track
    return None


def _cuda_toolkit_probe(
    environ: Mapping[str, str], track: Mapping[str, object]
) -> tuple[bool, str]:
    """Verify the nvcc family and development headers used by lazy GPU JITs."""
    configured_home = environ.get("CUDA_HOME", "").strip() or environ.get("CUDA_PATH", "").strip()
    cuda_home = Path(configured_home).expanduser() if configured_home else None
    nvcc = cuda_home / "bin" / "nvcc" if cuda_home else None
    if nvcc is not None and not nvcc.is_file():
        return False, f"CUDA_HOME/CUDA_PATH points to {cuda_home}, but {nvcc} is missing"
    if nvcc is None:
        resolved = shutil.which("nvcc", path=environ.get("PATH"))
        if resolved:
            nvcc = Path(resolved)
        else:
            default_nvcc = Path("/usr/local/cuda/bin/nvcc")
            nvcc = default_nvcc if default_nvcc.is_file() else None
    if nvcc is None:
        return (
            False,
            f"the {track['id']} track needs CUDA {track['jit_toolkit_minimum']}+ "
            "nvcc for lazy SGLang/FlashInfer kernels; set CUDA_HOME to a matching toolkit",
        )

    try:
        probe = subprocess.run(
            [str(nvcc), "--version"],
            capture_output=True,
            text=True,
            timeout=30,
            env=dict(environ),
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, f"cannot run {nvcc}: {exc}"
    match = re.search(r"release\s+(\d+)\.(\d+)", probe.stdout + probe.stderr)
    if probe.returncode != 0 or match is None:
        return False, f"cannot determine the CUDA toolkit version from {nvcc}"
    toolkit = tuple(int(part) for part in match.groups())
    minimum = tuple(int(part) for part in str(track["jit_toolkit_minimum"]).split("."))
    expected_major = int(str(track["cuda_major"]))
    if toolkit[0] != expected_major or toolkit < minimum:
        return (
            False,
            f"{nvcc} is CUDA {toolkit[0]}.{toolkit[1]}, but the {track['id']} "
            f"track requires CUDA {track['jit_toolkit_minimum']}+ within the "
            f"CUDA {expected_major} family",
        )

    toolkit_root = cuda_home or nvcc.parent.parent
    required_headers = track.get("required_jit_headers", [])
    if not isinstance(required_headers, list):
        return False, f"the {track['id']} track has invalid required_jit_headers metadata"
    missing = [
        str(header)
        for header in required_headers
        if not (toolkit_root / "include" / str(header)).is_file()
    ]
    if missing:
        return (
            False,
            f"CUDA {toolkit[0]}.{toolkit[1]} at {toolkit_root} is missing JIT "
            f"development header(s): {', '.join(missing)}",
        )
    return (
        True,
        f"CUDA JIT toolkit {toolkit[0]}.{toolkit[1]} at {toolkit_root} matches "
        f"the {track['id']} track",
    )


def recommended_native_track(driver_version: str) -> dict:
    """Select the newest checked-in native track supported by a host driver.

    The host driver, rather than ``nvcc`` or the CUDA label printed by
    ``nvidia-smi``, determines which bundled CUDA userspace a wheel may use.
    Returning the reviewed track itself lets bootstrap scripts install the
    exact same pin that the doctor later validates.
    """
    try:
        driver_major = int(driver_version.split(".", 1)[0])
    except ValueError as exc:
        raise ValueError(f"cannot parse NVIDIA driver version {driver_version!r}") from exc

    config = _compatibility_config()["native"]
    minimum_driver = int(config["minimum_driver_major"])
    if driver_major < minimum_driver:
        raise ValueError(f"native GPU serving requires NVIDIA driver {minimum_driver}+")

    compatible = [
        track for track in config["tracks"] if driver_major >= int(track["driver_minimum_major"])
    ]
    if not compatible:
        raise ValueError(f"no native SGLang track supports driver {driver_version}")
    return max(compatible, key=lambda track: int(track["driver_minimum_major"]))


def native_sglang_compatibility(
    driver_version: str, sglang_version: str, model_id: str
) -> tuple[bool, str]:
    """Check the verified native CUDA/SGLang boundary for this profile.

    Rules live in config/sglang/compatibility.json so the doctor, release
    tooling, and installation documentation can share one reviewed mapping.
    """
    try:
        driver_major = int(driver_version.split(".", 1)[0])
    except ValueError:
        return False, f"cannot parse NVIDIA driver version {driver_version!r}"
    parsed_sglang = _version_triplet(sglang_version)
    if parsed_sglang is None:
        return False, f"cannot parse SGLang version {sglang_version!r}"

    config = _compatibility_config()["native"]
    minimum_driver = int(config["minimum_driver_major"])
    if driver_major < minimum_driver:
        return False, f"native GPU serving requires NVIDIA driver {minimum_driver}+"

    for requirement in config["model_minimum_sglang"]:
        if model_id.startswith(requirement["model_prefix"]):
            minimum = _version_triplet(requirement["minimum_version"])
            if minimum is not None and parsed_sglang < minimum:
                return (
                    False,
                    f"{requirement['model_prefix']} support starts in the SGLang "
                    f"{requirement['minimum_version']} release line",
                )

    selected_track = native_track_for_sglang(sglang_version)
    if selected_track is None:
        return False, f"SGLang {sglang_version} is outside the configured native tracks"

    required_driver = int(selected_track["driver_minimum_major"])
    if driver_major < required_driver:
        fallback = config["tracks"][0]
        return (
            False,
            f"the configured {selected_track['id']} native track requires NVIDIA "
            f"driver {required_driver}+; use SGLang "
            f"{fallback['recommended_sglang']} on driver "
            f"{minimum_driver}-{required_driver - 1}",
        )
    return (
        True,
        f"SGLang {sglang_version} (CUDA {selected_track['cuda_major']}) "
        f"matches driver {driver_version}",
    )


def doctor(environ: Mapping[str, str], env_file: Path) -> int:
    failures = 0
    warnings = 0

    def passed(message: str) -> None:
        print(f"PASS  {message}")

    def failed(message: str) -> None:
        nonlocal failures
        failures += 1
        print(f"FAIL  {message}")

    def warned(message: str) -> None:
        nonlocal warnings
        warnings += 1
        print(f"WARN  {message}")

    print("FrontierAgent native NVIDIA doctor")
    print(f"Config: {env_file}\n")

    permissions = env_file.stat().st_mode & 0o777
    if permissions in {0o400, 0o600}:
        passed(f"configuration permissions are {permissions:o}")
    else:
        warned(f"configuration permissions are {permissions:o}; use chmod 600")

    if sys.platform.startswith("linux"):
        passed(f"Linux environment ({os.uname().machine})")
    else:
        failed("native SGLang currently requires Linux")

    if ctypes.util.find_library("numa"):
        passed("the libnuma runtime required by SGLang kernels is available")
    else:
        failed("libnuma is missing; install libnuma1 (Debian/Ubuntu) before startup")

    nvidia_smi = shutil.which("nvidia-smi")
    gpu_lines: list[str] = []
    driver_versions: list[str] = []
    if not nvidia_smi:
        failed("nvidia-smi is not installed")
    else:
        try:
            probe = subprocess.run(
                [
                    nvidia_smi,
                    "--query-gpu=index,name,memory.total,driver_version",
                    "--format=csv,noheader",
                ],
                capture_output=True,
                text=True,
                timeout=NVIDIA_SMI_TIMEOUT,
                check=False,
            )
        except subprocess.TimeoutExpired:
            failed(
                f"nvidia-smi did not answer within {NVIDIA_SMI_TIMEOUT}s; the "
                "driver or a GPU is likely wedged"
            )
        else:
            if probe.returncode == 0 and probe.stdout.strip():
                gpu_lines = probe.stdout.strip().splitlines()
                driver_versions = [line.rsplit(",", 1)[-1].strip() for line in gpu_lines]
                passed(f"NVIDIA driver reports {len(gpu_lines)} GPU(s)")
                for line in gpu_lines:
                    print(f"      {line}")
            else:
                failed("nvidia-smi could not enumerate a GPU")

    python = select_sglang_python(environ)
    installed_sglang: str | None = None
    if Path(python).is_file() or shutil.which(python):
        passed(f"SGLang Python is {python}")
        installed_sglang = _package_version(python, "sglang", environ)
        importable = _module_available(python, "sglang", environ)
        if importable:
            version_suffix = f" ({installed_sglang})" if installed_sglang else ""
            passed(f"the sglang Python package is importable{version_suffix}")
        elif importable is None:
            warned(
                f"importing sglang exceeded {IMPORT_PROBE_TIMEOUT}s; this is "
                "usually a cold model/CUDA cache, so rerun the doctor to confirm"
            )
        else:
            failed(
                "sglang is not importable; select an AutoDL SGLang image or "
                "install it in a dedicated environment and set SGLANG_PYTHON"
            )
    else:
        failed(f"SGLang Python does not exist: {python}")

    expected_sglang = environ.get("SGLANG_EXPECTED_VERSION", "").strip()
    if expected_sglang:
        if installed_sglang == expected_sglang:
            passed(f"SGLang matches SGLANG_EXPECTED_VERSION={expected_sglang}")
        elif installed_sglang:
            failed(
                f"installed SGLang {installed_sglang} does not match "
                f"SGLANG_EXPECTED_VERSION={expected_sglang}"
            )
        else:
            failed(f"could not verify SGLANG_EXPECTED_VERSION={expected_sglang}")

    if driver_versions and installed_sglang:
        oldest_driver = min(driver_versions, key=lambda value: int(value.split(".", 1)[0]))
        compatible, message = native_sglang_compatibility(
            oldest_driver,
            installed_sglang,
            environ.get("SGLANG_MODEL_ID", ""),
        )
        (passed if compatible else failed)(message)

        runtime_ok, runtime_message = _cuda_runtime_probe(python, environ)
        if runtime_ok is None:
            warned(runtime_message)
        elif runtime_ok:
            passed(runtime_message)
        else:
            failed(f"CUDA runtime is unusable: {runtime_message}")

        selected_track = native_track_for_sglang(installed_sglang)
        if selected_track is not None:
            toolkit_ok, toolkit_message = _cuda_toolkit_probe(environ, selected_track)
            (passed if toolkit_ok else failed)(toolkit_message)

    if installed_sglang:
        extra_args = environ.get("SGLANG_EXTRA_ARGS", "")
        try:
            parsed_extra_args = shlex.split(extra_args)
        except ValueError as exc:
            failed(f"SGLANG_EXTRA_ARGS is malformed: {exc}")
        else:
            if rejects_language_only(installed_sglang) and "--language-only" in parsed_extra_args:
                failed(
                    "SGLang 0.5.10 standalone mode must omit --language-only; "
                    "that release interprets it as encoder disaggregation"
                )

    tp_size = environ.get("SGLANG_TP_SIZE", "1").strip() or "1"
    if not _positive_int(tp_size):
        failed("SGLANG_TP_SIZE must be a positive integer")
    elif gpu_lines and int(tp_size) > len(gpu_lines):
        failed(f"SGLANG_TP_SIZE={tp_size} exceeds detected GPU count")
    else:
        passed(f"tensor parallel size is {tp_size}")

    context = environ.get("SGLANG_CONTEXT_LENGTH", "32768").strip()
    inputs = environ.get("SGLANG_MAX_INPUT_TOKENS", "27000").strip()
    outputs = environ.get("SGLANG_MAX_OUTPUT_TOKENS", "4096").strip()
    if all(_positive_int(value) for value in (context, inputs, outputs)):
        if int(inputs) + int(outputs) <= int(context):
            passed(f"token budgets fit context ({inputs} + {outputs} <= {context})")
        else:
            failed(f"token budgets exceed context ({inputs} + {outputs} > {context})")
        if int(inputs) * 10 > int(context) * 8:
            passed("input budget remains above the 80% compaction threshold")
        else:
            failed("input budget must remain above 80% of the context")
    else:
        failed("context and token budgets must be positive integers")

    try:
        command = build_server_command(environ)
        passed(f"model source is configured as {command[command.index('--model-path') + 1]}")
    except (RuntimeError, SystemExit, ValueError) as exc:
        failed(str(exc))

    download_dir = Path(environ["SGLANG_DOWNLOAD_DIR"]).expanduser()
    disk_probe = download_dir
    while not disk_probe.exists() and disk_probe != disk_probe.parent:
        disk_probe = disk_probe.parent
    free_gb = shutil.disk_usage(disk_probe).free // (1024**3)
    minimum_raw = environ.get("SGLANG_MIN_FREE_GB", "60") or "60"
    minimum = int(minimum_raw) if _positive_int(minimum_raw) else 60
    if minimum_raw != str(minimum):
        warned("SGLANG_MIN_FREE_GB is invalid; using 60 GiB")
    if free_gb >= minimum:
        passed(f"{free_gb} GiB free on the model-cache filesystem")
    else:
        warned(
            f"only {free_gb} GiB free on {disk_probe}; first setup recommends "
            f"at least {minimum} GiB"
        )

    host = environ["SGLANG_SERVER_HOST"]
    if host in {"127.0.0.1", "localhost", "::1"}:
        passed("native server is bound to loopback")
    else:
        warned(f"SGLANG_NATIVE_HOST={host} exposes an unauthenticated model endpoint")

    port = int(environ["SGLANG_SERVER_PORT"])
    family = socket.AF_INET6 if ":" in host else socket.AF_INET
    with socket.socket(family) as probe_socket:
        try:
            # Probe the address SGLang will actually bind, not loopback: a
            # busy NIC-specific port looks free on 127.0.0.1 and vice versa.
            probe_socket.bind((host, port))
        except socket.gaierror:
            failed(f"SGLANG_NATIVE_HOST={host} does not resolve")
        except OSError:
            if is_healthy(environ):
                warned(f"port {port} is already serving a healthy SGLang endpoint")
            else:
                failed(f"port {port} is occupied by another process")
        else:
            passed(f"port {port} is available")

    print(f"\nSummary: {failures} failure(s), {warnings} warning(s)")
    return 1 if failures else 0


def _tail_log(environ: Mapping[str, str], lines: int = 80) -> None:
    path = _log_path(environ)
    if not path.is_file():
        return
    content = path.read_text(encoding="utf-8", errors="replace").splitlines()
    for line in content[-lines:]:
        print(line, file=sys.stderr)


def up(environ: Mapping[str, str]) -> int:
    pid = read_pid(environ)
    if pid and is_managed_process(pid):
        if is_healthy(environ):
            print(f"SGLang is already healthy (pid {pid}).")
            return 0
        print(f"SGLang process {pid} is still starting.")
    elif is_healthy(environ):
        print("A healthy SGLang endpoint is already listening; leaving it unmanaged.")
        return 0
    else:
        command = build_server_command(environ)
        runtime = state_dir(environ)
        runtime.mkdir(parents=True, exist_ok=True)
        Path(environ["SGLANG_DOWNLOAD_DIR"]).expanduser().mkdir(parents=True, exist_ok=True)
        log_file = _log_path(environ).open("a", encoding="utf-8")
        print(f"Starting native SGLang; logs: {_log_path(environ)}")
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            env=dict(environ),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        log_file.close()
        _pid_path(environ).write_text(f"{process.pid}\n", encoding="utf-8")
        pid = process.pid

    timeout = int(environ["SGLANG_STARTUP_TIMEOUT"])
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if is_healthy(environ):
            print(f"SGLang is healthy at {environ['SGLANG_BASE_URL']} (pid {pid}).")
            return 0
        if not is_managed_process(pid):
            print("SGLang exited before becoming healthy; recent logs:", file=sys.stderr)
            _tail_log(environ)
            return 1
        time.sleep(2)
    print(f"SGLang did not become healthy within {timeout}s.", file=sys.stderr)
    _tail_log(environ)
    return 1


def down(environ: Mapping[str, str]) -> int:
    pid = read_pid(environ)
    if not pid or not is_managed_process(pid):
        print("No managed native SGLang process is running.")
        return 0
    print(f"Stopping native SGLang (pid {pid})...")
    os.killpg(pid, signal.SIGTERM)
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline and is_managed_process(pid):
        time.sleep(0.5)
    if is_managed_process(pid):
        print("SGLang did not stop gracefully; sending SIGKILL.", file=sys.stderr)
        os.killpg(pid, signal.SIGKILL)
    _pid_path(environ).unlink(missing_ok=True)
    return 0


def status(environ: Mapping[str, str]) -> int:
    pid = read_pid(environ)
    managed = bool(pid and is_managed_process(pid))
    healthy = is_healthy(environ)
    print(f"managed_process: {'running' if managed else 'stopped'}")
    print(f"endpoint: {'healthy' if healthy else 'unavailable'}")
    print(f"url: {environ['SGLANG_BASE_URL']}")
    print(f"log: {_log_path(environ)}")
    return 0 if healthy else 1


def smoke(environ: Mapping[str, str]) -> int:
    if up(environ) != 0:
        return 1
    smoke_script = REPO_ROOT / "docker" / "smoke_sglang.py"
    return subprocess.run(
        [sys.executable, str(smoke_script)],
        cwd=REPO_ROOT,
        env=dict(environ),
        check=False,
    ).returncode


def tui(environ: Mapping[str, str], forwarded: list[str]) -> int:
    if up(environ) != 0:
        return 1
    agent_env = dict(environ)
    agent_env.update(
        {
            "OPENAI_PROVIDER": "local",
            "OPENAI_API_KEY": "EMPTY",
            "OPENAI_BASE_URL": f"{environ['SGLANG_BASE_URL']}/v1",
            "OPENAI_MODEL": environ.get("SGLANG_SERVED_MODEL_NAME", "local-model"),
            "OPENAI_CONTEXT_WINDOW": environ.get("SGLANG_CONTEXT_LENGTH", "32768"),
            "OPENAI_MAX_INPUT_TOKENS": environ.get("SGLANG_MAX_INPUT_TOKENS", "27000"),
            "OPENAI_MAX_TOKENS": environ.get("SGLANG_MAX_OUTPUT_TOKENS", "4096"),
        }
    )
    command = [
        "uv",
        "run",
        "frontier-agent",
        "--native",
        "--mode",
        "react",
        *forwarded,
    ]
    try:
        return subprocess.run(command, cwd=REPO_ROOT, env=agent_env, check=False).returncode
    except FileNotFoundError:
        print("uv is required to start FrontierAgent; install uv and rerun.", file=sys.stderr)
        return 1


def follow_logs(environ: Mapping[str, str]) -> int:
    path = _log_path(environ)
    if not path.exists():
        print(f"No native SGLang log exists yet: {path}", file=sys.stderr)
        return 1
    try:
        return subprocess.run(["tail", "-n", "100", "-F", str(path)], check=False).returncode
    except KeyboardInterrupt:
        return 130


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run SGLang natively on Linux or inside a managed GPU container."
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=Path(os.environ.get("SGLANG_ENV_FILE", DEFAULT_ENV_FILE)),
    )
    parser.add_argument(
        "command",
        nargs="?",
        default="tui",
        choices=("doctor", "up", "smoke", "tui", "status", "logs", "down"),
    )
    parser.add_argument("args", nargs=argparse.REMAINDER)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    env_file = args.env_file.expanduser().resolve()
    if not env_file.is_file():
        print(
            f"SGLang config not found: {env_file}\n"
            "Copy a config/sglang profile to .env.sglang first.",
            file=sys.stderr,
        )
        return 2
    try:
        config = load_dotenv(env_file)
        environ = runtime_environment(config)
    except (OSError, ValueError) as exc:
        print(f"Invalid SGLang configuration: {exc}", file=sys.stderr)
        return 2

    if args.command == "doctor":
        return doctor(environ, env_file)
    if args.command == "up":
        return up(environ)
    if args.command == "smoke":
        return smoke(environ)
    if args.command == "tui":
        return tui(environ, args.args)
    if args.command == "status":
        return status(environ)
    if args.command == "logs":
        return follow_logs(environ)
    return down(environ)


if __name__ == "__main__":
    raise SystemExit(main())
