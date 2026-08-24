#!/usr/bin/env bash
# Bootstrap native SGLang and FrontierAgent on a Linux NVIDIA GPU machine.

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
env_file="${SGLANG_ENV_FILE:-$repo_root/.env.sglang}"
profile="smoke"
command="tui"
python_version="${FRONTIER_AGENT_PYTHON_VERSION:-3.12}"
install_system_deps=0
setup_only=0
declare -a forwarded=()

usage() {
  printf '%s\n' \
    "Usage: ./scripts/run-linux-gpu.sh [OPTIONS] [COMMAND] [-- TUI_ARGS...]" \
    "" \
    "Install a driver-compatible SGLang in an isolated virtual environment," \
    "then diagnose or run FrontierAgent against the local NVIDIA GPU endpoint." \
    "" \
    "Commands (default: tui):" \
    "  doctor  up  smoke  tui  status  logs  down" \
    "" \
    "Options may appear before or after the command. Everything after -- is" \
    "forwarded to the FrontierAgent TUI unchanged." \
    "" \
    "Options:" \
    "  --profile smoke|4090|5090|multigpu|PATH" \
    "                           Template used only when .env.sglang is absent" \
    "  --env-file PATH          SGLang configuration file" \
    "  --python VERSION         Managed Python version (default: 3.12)" \
    "  --install-system-deps    Install libnuma with the detected OS package manager" \
    "  --setup-only             Install and run doctor without starting a model" \
    "  -h, --help               Show this help" \
    "" \
    "Safety:" \
    "  This script never installs or replaces an NVIDIA driver or CUDA toolkit." \
    "  It manages only .venv-sglang and FrontierAgent's .venv in this checkout."
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --profile)
      [ "$#" -ge 2 ] || { printf 'error: --profile requires a value\n' >&2; exit 2; }
      profile="$2"
      shift 2
      ;;
    --env-file)
      [ "$#" -ge 2 ] || { printf 'error: --env-file requires a path\n' >&2; exit 2; }
      env_file="$2"
      shift 2
      ;;
    --python)
      [ "$#" -ge 2 ] || { printf 'error: --python requires a version\n' >&2; exit 2; }
      python_version="$2"
      shift 2
      ;;
    --install-system-deps)
      install_system_deps=1
      shift
      ;;
    --setup-only)
      # Does not rewrite ``command``: the doctor runs before every command
      # anyway, and the setup_only gate below is what stops short of starting a
      # model. Clobbering it would discard an explicit command for no gain and
      # make the ordering of the two words matter.
      setup_only=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    doctor|up|smoke|tui|status|logs|down)
      # Keep parsing our own options after the command instead of treating the
      # remainder as forwarded args. Absorbing them silently meant
      # ``run-linux-gpu.sh up --setup-only`` passed --setup-only through to
      # run-sglang-native.py (which drops it into argparse's REMAINDER and
      # ignores it) and started a model anyway. Only ``--`` begins forwarding.
      command="$1"
      shift
      ;;
    --)
      shift
      forwarded=("$@")
      break
      ;;
    *)
      printf '%s\n\n' \
        "error: unknown option or command: $1" \
        "Arguments for the FrontierAgent TUI go after --, for example:" \
        "  ./scripts/run-linux-gpu.sh tui -- --cwd /path/to/project" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [ "$(uname -s)" != "Linux" ]; then
  printf 'error: native SGLang requires Linux and an NVIDIA GPU\n' >&2
  exit 1
fi

if [ -L "$env_file" ]; then
  printf 'error: refusing to trust a symlinked configuration: %s\n' "$env_file" >&2
  exit 1
fi

# Inspection and shutdown must still work if the GPU is unhealthy or the host
# is offline. These paths never install or upgrade anything.
case "$command" in
  status|logs|down)
    if [ ! -f "$env_file" ]; then
      printf 'error: SGLang configuration does not exist: %s\n' "$env_file" >&2
      exit 1
    fi
    lifecycle_python=""
    for candidate in \
      "$repo_root/.venv/bin/python" \
      "$repo_root/.venv-sglang/bin/python" \
      "$(command -v python3 || true)"; do
      if [ -n "$candidate" ] && [ -x "$candidate" ]; then
        lifecycle_python="$candidate"
        break
      fi
    done
    [ -n "$lifecycle_python" ] || {
      printf 'error: python3 is required for %s\n' "$command" >&2
      exit 1
    }
    # ``${a[@]+"${a[@]}"}``: bash before 4.4 (CentOS/RHEL 7 ship 4.2) treats an
    # empty array as unset under ``set -u`` and aborts on a bare "${a[@]}".
    exec "$lifecycle_python" "$repo_root/scripts/run-sglang-native.py" \
      --env-file "$env_file" "$command" ${forwarded[@]+"${forwarded[@]}"}
    ;;
esac

case "$(uname -m)" in
  x86_64|amd64) ;;
  *)
    printf '%s\n' \
      "error: automatic SGLang wheel installation is currently certified only on Linux x86_64." \
      "Set up SGLang manually for $(uname -m), then use scripts/run-sglang-native.py." >&2
    exit 1
    ;;
esac

command -v nvidia-smi >/dev/null 2>&1 || {
  printf '%s\n' \
    'error: nvidia-smi is unavailable.' \
    'Install/select a provider image with a working NVIDIA driver before rerunning.' >&2
  exit 1
}

gpu_inventory="$(nvidia-smi \
  --query-gpu=index,name,memory.total,driver_version \
  --format=csv,noheader,nounits 2>/dev/null || true)"
if [ -z "$gpu_inventory" ]; then
  printf 'error: nvidia-smi could not enumerate a usable GPU\n' >&2
  exit 1
fi
driver_version="$(printf '%s\n' "$gpu_inventory" | awk -F, 'NR == 1 {gsub(/[[:space:]]/, "", $4); print $4}')"
if [ -z "$driver_version" ]; then
  printf 'error: could not read the NVIDIA driver version\n' >&2
  exit 1
fi

printf 'Detected NVIDIA GPU inventory:\n%s\n' "$gpu_inventory"

find_uv() {
  if command -v uv >/dev/null 2>&1; then
    command -v uv
    return 0
  fi
  for candidate in "${HOME}/.local/bin/uv" "${HOME}/.cargo/bin/uv"; do
    if [ -x "$candidate" ]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done
  return 1
}

uv_bin="$(find_uv || true)"
if [ -z "$uv_bin" ]; then
  if ! command -v curl >/dev/null 2>&1 && ! command -v wget >/dev/null 2>&1; then
    printf '%s\n' \
      'error: uv is not installed and neither curl nor wget is available.' \
      'Install uv from https://docs.astral.sh/uv/ and rerun.' >&2
    exit 1
  fi
  printf 'uv is not installed; downloading the official installer from astral.sh...\n'
  uv_installer="$(mktemp "${TMPDIR:-/tmp}/frontier-agent-uv.XXXXXX")"
  trap 'rm -f "${uv_installer:-}"' EXIT
  if command -v curl >/dev/null 2>&1; then
    curl --proto '=https' --tlsv1.2 -LsSf https://astral.sh/uv/install.sh -o "$uv_installer"
  else
    wget -q https://astral.sh/uv/install.sh -O "$uv_installer"
  fi
  sh "$uv_installer"
  uv_bin="$(find_uv || true)"
  if [ -z "$uv_bin" ]; then
    printf 'error: uv installation finished but the executable was not found\n' >&2
    exit 1
  fi
fi
export PATH="$(dirname "$uv_bin"):$PATH"

printf 'Ensuring managed Python %s is available...\n' "$python_version"
"$uv_bin" python install "$python_version"
bootstrap_python="$("$uv_bin" python find "$python_version")"

track_fields="$("$bootstrap_python" - "$repo_root/scripts/run-sglang-native.py" "$driver_version" <<'PY'
import importlib.util
import sys

path, driver = sys.argv[1:]
spec = importlib.util.spec_from_file_location("frontier_sglang_native", path)
if spec is None or spec.loader is None:
    raise SystemExit(f"cannot load {path}")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
try:
    track = module.recommended_native_track(driver)
except ValueError as exc:
    raise SystemExit(str(exc)) from exc
print(track["id"], track["cuda_major"], track["recommended_sglang"])
PY
)"
read -r track_id cuda_major sglang_version <<< "$track_fields"

if [ -z "${track_id:-}" ] || [ -z "${sglang_version:-}" ]; then
  printf 'error: compatibility matrix did not select an SGLang runtime\n' >&2
  exit 1
fi
printf 'Selected reviewed runtime: %s (CUDA %s userspace), SGLang %s for driver %s\n' \
  "$track_id" "$cuda_major" "$sglang_version" "$driver_version"

have_libnuma() {
  "$bootstrap_python" - <<'PY'
import ctypes.util
raise SystemExit(0 if ctypes.util.find_library("numa") else 1)
PY
}

install_libnuma() {
  if [ ! -r /etc/os-release ]; then
    printf 'error: cannot identify the Linux distribution to install libnuma\n' >&2
    return 1
  fi
  # os-release is a distribution-owned KEY=VALUE file, not user configuration.
  # shellcheck disable=SC1091
  . /etc/os-release
  distro="${ID:-unknown} ${ID_LIKE:-}"
  case "$distro" in
    *debian*|*ubuntu*)
      package_command=(apt-get update)
      package_install=(apt-get install -y libnuma1)
      ;;
    *fedora*|*rhel*|*centos*|*rocky*|*almalinux*)
      if command -v dnf >/dev/null 2>&1; then
        package_command=(dnf makecache)
        package_install=(dnf install -y numactl-libs)
      else
        package_command=(yum makecache)
        package_install=(yum install -y numactl-libs)
      fi
      ;;
    *arch*)
      package_command=(pacman -Sy)
      package_install=(pacman -S --noconfirm numactl)
      ;;
    *suse*)
      package_command=(zypper --non-interactive refresh)
      package_install=(zypper --non-interactive install libnuma1)
      ;;
    *)
      printf 'error: unsupported distribution for automatic libnuma installation: %s\n' "$distro" >&2
      return 1
      ;;
  esac

  privilege=()
  if [ "$(id -u)" -ne 0 ]; then
    command -v sudo >/dev/null 2>&1 || {
      printf 'error: installing libnuma requires root or sudo\n' >&2
      return 1
    }
    privilege=(sudo)
  fi
  # ``privilege`` is empty when already root; see the set -u note above.
  ${privilege[@]+"${privilege[@]}"} "${package_command[@]}"
  ${privilege[@]+"${privilege[@]}"} "${package_install[@]}"
}

if ! have_libnuma; then
  if [ "$install_system_deps" -eq 1 ]; then
    printf 'libnuma is missing; installing the distribution package...\n'
    install_libnuma
    have_libnuma || { printf 'error: libnuma is still unavailable after installation\n' >&2; exit 1; }
  else
    printf '%s\n' \
      'error: the libnuma runtime required by SGLang is missing.' \
      'Rerun with --install-system-deps, or install libnuma with your OS package manager.' >&2
    exit 1
  fi
fi

case "$profile" in
  smoke) profile_path="$repo_root/.env.sglang.example" ;;
  4090) profile_path="$repo_root/config/sglang/35b-4090.env.example" ;;
  5090) profile_path="$repo_root/config/sglang/35b-5090.env.example" ;;
  multigpu) profile_path="$repo_root/config/sglang/35b-multigpu.env.example" ;;
  *)
    profile_path="$profile"
    case "$profile_path" in /*) ;; *) profile_path="$PWD/$profile_path" ;; esac
    ;;
esac

if [ ! -f "$env_file" ]; then
  [ -f "$profile_path" ] || { printf 'error: profile does not exist: %s\n' "$profile_path" >&2; exit 1; }
  umask 077
  mkdir -p "$(dirname "$env_file")"
  cp "$profile_path" "$env_file"
  chmod 600 "$env_file"
  printf 'Created %s from %s\n' "$env_file" "$profile_path"
else
  printf 'Preserving existing SGLang configuration: %s\n' "$env_file"
fi

dotenv_value() {
  "$bootstrap_python" - "$repo_root/scripts/run-sglang-native.py" "$env_file" "$1" <<'PY'
import importlib.util
import pathlib
import sys

launcher_path, env_path, key = sys.argv[1:]
spec = importlib.util.spec_from_file_location("frontier_sglang_native", launcher_path)
if spec is None or spec.loader is None:
    raise SystemExit(f"cannot load {launcher_path}")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
print(module.load_dotenv(pathlib.Path(env_path)).get(key, ""))
PY
}

sanitize_extra_args() {
  "$bootstrap_python" - "$repo_root/scripts/run-sglang-native.py" "$1" "$2" <<'PY'
import importlib.util
import sys

launcher_path, version, extra_args = sys.argv[1:]
spec = importlib.util.spec_from_file_location("frontier_sglang_native", launcher_path)
if spec is None or spec.loader is None:
    raise SystemExit(f"cannot load {launcher_path}")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
sys.stdout.write(module.sanitize_extra_args(version, extra_args))
PY
}

configured_model_id="${SGLANG_MODEL_ID:-$(dotenv_value SGLANG_MODEL_ID)}"
configured_model_path="${SGLANG_LOCAL_MODEL_PATH:-$(dotenv_value SGLANG_LOCAL_MODEL_PATH)}"
if [ -z "$configured_model_id" ] && [ -z "$configured_model_path" ]; then
  printf '%s\n' \
    "error: $env_file does not select a model checkpoint." \
    'Set SGLANG_MODEL_ID or SGLANG_LOCAL_MODEL_PATH, then rerun; SGLang was not downloaded.' >&2
  exit 1
fi

# Read an explicitly configured interpreter without evaluating the dotenv file.
configured_python="$(dotenv_value SGLANG_PYTHON)"

if [ -n "${SGLANG_PYTHON:-}" ]; then
  configured_python="$SGLANG_PYTHON"
fi

if [ -n "$configured_python" ]; then
  case "$configured_python" in
    /*) sglang_python="$configured_python" ;;
    *) sglang_python="$repo_root/$configured_python" ;;
  esac
  [ -x "$sglang_python" ] || {
    printf 'error: configured SGLANG_PYTHON is not executable: %s\n' "$sglang_python" >&2
    exit 1
  }
  printf 'Using explicitly configured SGLang environment: %s\n' "$sglang_python"
else
  sglang_venv="$repo_root/.venv-sglang"
  sglang_python="$sglang_venv/bin/python"
  free_repo_kb="$(df -Pk "$repo_root" | awk 'NR == 2 {print $4}')"
  if [ -n "$free_repo_kb" ] && [ "$free_repo_kb" -lt 15728640 ]; then
    printf 'warning: less than 15 GiB is free for the SGLang virtual environment\n' >&2
  fi
  if [ ! -x "$sglang_python" ]; then
    printf 'Creating isolated SGLang environment: %s\n' "$sglang_venv"
    "$uv_bin" venv --python "$python_version" "$sglang_venv"
  fi
  installed_version="$("$sglang_python" - <<'PY' 2>/dev/null || true
import importlib.metadata
try:
    print(importlib.metadata.version("sglang"))
except importlib.metadata.PackageNotFoundError:
    pass
PY
)"
  if [ "$installed_version" != "$sglang_version" ]; then
    printf 'Installing SGLang %s into the isolated environment (this may download several GB)...\n' \
      "$sglang_version"
    # Every SGLang release in the compatibility matrix depends on a prerelease
    # flash-attn-4; the only stable version on PyPI is an unrelated 0.0.1
    # placeholder. uv before 0.12.0 would not satisfy that pin at all and failed
    # the install outright with "No solution found". 0.12.0 admits such a
    # transitive prerelease on its own, but this helper reuses whatever uv is
    # already on PATH and only downloads a fresh one when none exists, so an
    # image that ships an older uv still hits it.
    #
    # Do not drop these two arguments after testing on a current uv, where they
    # look like a no-op: they are what keeps the older-uv path working. Both
    # tracks resolve identically with and without them on 0.12.0+.
    #
    # uv's explicit mode only honours top-level requirements, hence naming
    # flash-attn-4 here; the floor is low enough to cover both tracks, and
    # sglang's own stricter constraint wins the intersection. explicit rather
    # than allow, because allow opens prereleases everywhere and pulls in a beta
    # pydantic plus ten other unrelated packages.
    "$uv_bin" pip install --python "$sglang_python" --upgrade \
      --prerelease=explicit "sglang==$sglang_version" "flash-attn-4>=4.0.0b4"
  else
    printf 'SGLang %s is already installed in %s\n' "$installed_version" "$sglang_venv"
  fi
  export SGLANG_EXPECTED_VERSION="$sglang_version"

  # SGLang 0.5.10 gives --language-only encoder-disaggregation semantics and
  # cannot start a standalone Qwen3.5 server with that newer-profile flag.
  #
  # Start from the value the server will actually see: runtime_environment()
  # layers os.environ over the dotenv file, so an exported SGLANG_EXTRA_ARGS
  # wins. Reading only the file and then exporting the result would replace the
  # operator's explicit setting with the profile's.
  #
  # Which flags a release rejects is decided by sanitize_extra_args() in
  # run-sglang-native.py, the same helper the doctor's check uses, so a new
  # recommended patch pin cannot make the two disagree.
  configured_extra="${SGLANG_EXTRA_ARGS-$(dotenv_value SGLANG_EXTRA_ARGS)}"
  sanitized_extra="$(sanitize_extra_args "$sglang_version" "$configured_extra")"
  if [ "$sanitized_extra" != "$configured_extra" ]; then
    export SGLANG_EXTRA_ARGS="$sanitized_extra"
    printf 'Adjusted runtime args for SGLang %s: %s -> %s\n' \
      "$sglang_version" "$configured_extra" "$sanitized_extra"
  fi
fi
export SGLANG_PYTHON="$sglang_python"

# The AutoDL default must yield to a cache the operator already configured.
# runtime_environment() layers os.environ over the dotenv file, so exporting
# these after reading only the shell environment would override the paths set
# in $env_file: the checkpoint would be re-downloaded into a second directory
# (tens of GB) and an HF_HUB_OFFLINE=1 setup pointed at the configured cache
# would stop finding its model. ``${VAR-...}`` rather than ``${VAR:-...}``
# because an exported-but-empty value replaces the file value on the Python
# side too, so it also means "no cache configured".
#
# Read inside the AutoDL guard: each dotenv_value spawns an interpreter that
# loads the launcher module, and no other host needs the answer.
if [ -d /root/autodl-tmp ] && [ -w /root/autodl-tmp ]; then
  effective_download_dir="${SGLANG_DOWNLOAD_DIR-$(dotenv_value SGLANG_DOWNLOAD_DIR)}"
  effective_hf_home="${HF_HOME-$(dotenv_value HF_HOME)}"
  effective_hf_hub_cache="${HF_HUB_CACHE-$(dotenv_value HF_HUB_CACHE)}"
  if [ -z "$effective_download_dir$effective_hf_home$effective_hf_hub_cache" ]; then
    export HF_HOME=/root/autodl-tmp/huggingface
    export HF_HUB_CACHE="$HF_HOME/hub"
    export SGLANG_DOWNLOAD_DIR="$HF_HUB_CACHE"
    printf 'Detected AutoDL persistent storage; model cache: %s\n' "$SGLANG_DOWNLOAD_DIR"
  else
    printf '%s\n' \
      "Detected AutoDL persistent storage, but $env_file already configures a" \
      "model cache; keeping it (SGLANG_DOWNLOAD_DIR=${effective_download_dir:-unset}," \
      "HF_HOME=${effective_hf_home:-unset}, HF_HUB_CACHE=${effective_hf_hub_cache:-unset})."
  fi
fi

printf 'Installing FrontierAgent and Python %s dependencies...\n' "$python_version"
cd "$repo_root"
"$uv_bin" sync --inexact --python "$python_version"

if [ "$setup_only" -eq 1 ]; then
  printf 'Running native GPU doctor (setup only; no model will be started)...\n'
else
  printf 'Running native GPU doctor before %s...\n' "$command"
fi
if ! "$bootstrap_python" "$repo_root/scripts/run-sglang-native.py" --env-file "$env_file" doctor; then
  printf 'error: GPU doctor failed; no model process was started\n' >&2
  exit 1
fi

if [ "$setup_only" -eq 1 ] || [ "$command" = "doctor" ]; then
  printf 'GPU setup complete. Start with: ./scripts/run-linux-gpu.sh tui\n'
  exit 0
fi

exec "$bootstrap_python" "$repo_root/scripts/run-sglang-native.py" \
  --env-file "$env_file" "$command" ${forwarded[@]+"${forwarded[@]}"}
