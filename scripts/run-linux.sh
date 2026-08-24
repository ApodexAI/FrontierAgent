#!/usr/bin/env bash
# Bootstrap and run FrontierAgent on Linux with a remote OpenAI-compatible endpoint.

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mode="${FRONTIER_AGENT_MODE:-react}"
workspace="${FRONTIER_AGENT_WORKSPACE:-$repo_root}"
runtime="${FRONTIER_AGENT_RUNTIME:-auto}"
setup_only=0
declare -a forwarded=()

usage() {
  printf '%s\n' \
    "Usage: ./scripts/run-linux.sh [OPTIONS] [-- FRONTIER_AGENT_ARGS...]" \
    "" \
    "Prepare a cloned FrontierAgent checkout and connect it to a remote LLM." \
    "This helper does not install or run a local model server." \
    "" \
    "Options:" \
    "  --mode react|agent_team  Workflow to start (default: react)" \
    "  --cwd PATH               Workspace the agent may operate on" \
    "  --native                 Use the workspace-local runtime (default)" \
    "  --bwrap                  Require bubblewrap isolation" \
    "  --docker                 Require a reachable Docker daemon" \
    "  --setup-only             Install/configure without starting the TUI" \
    "  -h, --help               Show this help" \
    "" \
    "Environment equivalents:" \
    "  FRONTIER_AGENT_MODE, FRONTIER_AGENT_WORKSPACE, FRONTIER_AGENT_RUNTIME" \
    "  OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_MODEL"
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --mode)
      [ "$#" -ge 2 ] || { printf 'error: --mode requires a value\n' >&2; exit 2; }
      mode="$2"
      shift 2
      ;;
    --cwd)
      [ "$#" -ge 2 ] || { printf 'error: --cwd requires a path\n' >&2; exit 2; }
      workspace="$2"
      shift 2
      ;;
    --native)
      runtime="native"
      shift
      ;;
    --bwrap)
      runtime="bwrap"
      shift
      ;;
    --docker)
      runtime="docker"
      shift
      ;;
    --setup-only)
      setup_only=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      forwarded=("$@")
      break
      ;;
    *)
      printf 'error: unknown option: %s (put FrontierAgent options after --)\n\n' "$1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

case "$mode" in
  react|agent_team) ;;
  *) printf 'error: --mode must be react or agent_team\n' >&2; exit 2 ;;
esac

case "$runtime" in
  auto|native|bwrap|docker) ;;
  *) printf 'error: FRONTIER_AGENT_RUNTIME must be auto, native, bwrap, or docker\n' >&2; exit 2 ;;
esac

if [ "$(uname -s)" != "Linux" ]; then
  printf 'error: this helper requires Linux; macOS users should run ./scripts/run-macos.sh\n' >&2
  exit 1
fi

if [ ! -d "$workspace" ]; then
  printf 'error: workspace directory does not exist: %s\n' "$workspace" >&2
  exit 1
fi
workspace="$(cd "$workspace" && pwd)"

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

env_value() {
  awk -v key="$1" '
    {
      line = $0
      sub(/^[[:space:]]*export[[:space:]]+/, "", line)
      if (index(line, key "=") == 1) {
        value = substr(line, length(key) + 2)
        sub(/\r$/, "", value)
        quote = substr(value, 1, 1)
        if (length(value) >= 2 && (quote == "\"" || quote == "\047") \
            && substr(value, length(value), 1) == quote) {
          value = substr(value, 2, length(value) - 2)
        }
        result = value
      }
    }
    END { print result }
  ' "$repo_root/.env"
}

validate_env_value() {
  case "$2" in
    *$'\n'*|*$'\r'*)
      printf 'error: %s must be a single-line value\n' "$1" >&2
      exit 1
      ;;
  esac
}

if [ -L "$repo_root/.env" ]; then
  printf 'error: refusing to write or trust a symlinked configuration: %s/.env\n' "$repo_root" >&2
  exit 1
fi

if [ ! -f "$repo_root/.env" ]; then
  api_key="${OPENAI_API_KEY:-}"
  base_url="${OPENAI_BASE_URL:-}"
  model_name="${OPENAI_MODEL:-}"

  if [ -z "$api_key" ] || [ -z "$base_url" ] || [ -z "$model_name" ]; then
    if [ ! -t 0 ]; then
      printf '%s\n' \
        'error: .env is missing and setup is not interactive.' \
        'Set OPENAI_API_KEY, OPENAI_BASE_URL, and OPENAI_MODEL, then rerun.' >&2
      exit 1
    fi
    printf '%s\n' \
      'First-run setup needs an OpenAI-compatible model endpoint.' \
      'The API key is hidden while typing; use EMPTY for an unauthenticated endpoint.'
    if [ -z "$api_key" ]; then
      IFS= read -r -s -p 'OPENAI_API_KEY: ' api_key
      printf '\n'
    fi
    if [ -z "$base_url" ]; then
      IFS= read -r -p 'OPENAI_BASE_URL (normally ending in /v1): ' base_url
    fi
    if [ -z "$model_name" ]; then
      IFS= read -r -p 'OPENAI_MODEL: ' model_name
    fi
  fi

  if [ -z "$api_key" ] || [ -z "$base_url" ] || [ -z "$model_name" ]; then
    printf 'error: OPENAI_API_KEY, OPENAI_BASE_URL, and OPENAI_MODEL cannot be empty\n' >&2
    exit 1
  fi
  validate_env_value OPENAI_API_KEY "$api_key"
  validate_env_value OPENAI_BASE_URL "$base_url"
  validate_env_value OPENAI_MODEL "$model_name"
  case "$base_url" in
    http://*|https://*) ;;
    *) printf 'error: OPENAI_BASE_URL must start with http:// or https://\n' >&2; exit 1 ;;
  esac

  umask 077
  {
    printf '# Generated by scripts/run-linux.sh; do not commit this file.\n'
    # Copy the template verbatim and substitute the three answers in place,
    # rather than printing them and appending a range of the template: keying
    # that range off a comment's wording ('# Optional: web search') meant
    # rewording the comment silently dropped every optional setting.
    #
    # The values reach awk through its environment, not argv or -v: a command
    # line is world-readable in ps output and one of these is an API key.
    OPENAI_API_KEY="$api_key" \
    OPENAI_BASE_URL="$base_url" \
    OPENAI_MODEL="$model_name" \
    awk '
      BEGIN { split("OPENAI_API_KEY OPENAI_BASE_URL OPENAI_MODEL", required, " ") }
      {
        line = $0
        sub(/^[[:space:]]*export[[:space:]]+/, "", line)
        for (slot in required) {
          key = required[slot]
          if (index(line, key "=") == 1) {
            print key "=" ENVIRON[key]
            next
          }
        }
        print
      }
    ' "$repo_root/.env.example"
  } > "$repo_root/.env"
  chmod 600 "$repo_root/.env"
  # A template that stops carrying one of the required keys would otherwise
  # yield a .env that silently omits it.
  for required_key in OPENAI_API_KEY OPENAI_BASE_URL OPENAI_MODEL; do
    if [ -z "$(env_value "$required_key")" ]; then
      printf 'error: generated .env lacks %s; %s no longer defines it\n' \
        "$required_key" "$repo_root/.env.example" >&2
      exit 1
    fi
  done
  printf 'Created %s/.env\n' "$repo_root"
else
  missing=""
  for required_key in OPENAI_API_KEY OPENAI_BASE_URL OPENAI_MODEL; do
    if [ -z "$(env_value "$required_key")" ]; then
      missing="${missing} ${required_key}"
    fi
  done
  if [ -n "$missing" ]; then
    printf 'error: .env has empty required values:%s\n' "$missing" >&2
    printf 'Edit %s/.env and rerun this script.\n' "$repo_root" >&2
    exit 1
  fi
  case "$(env_value OPENAI_BASE_URL)" in
    http://*|https://*) ;;
    *) printf 'error: OPENAI_BASE_URL in .env must start with http:// or https://\n' >&2; exit 1 ;;
  esac
fi

sync_args=(sync --inexact --python 3.12)
case "$runtime" in
  bwrap) sync_args+=(--extra sandbox) ;;
esac

printf 'Installing FrontierAgent and Python 3.12 dependencies...\n'
cd "$repo_root"
"$uv_bin" "${sync_args[@]}"

runtime_label="auto"
runtime_args=()
case "$runtime" in
  auto)
    ;;
  native)
    runtime_label="native"
    runtime_args=(--native)
    ;;
  bwrap)
    command -v bwrap >/dev/null 2>&1 || {
      printf '%s\n' \
        'error: bubblewrap was requested but bwrap is not installed.' \
        'On Debian/Ubuntu: sudo apt-get install bubblewrap' >&2
      exit 1
    }
    runtime_label="bwrap"
    runtime_args=(--bwrap)
    ;;
  docker)
    command -v docker >/dev/null 2>&1 || {
      printf 'error: Docker was requested but its CLI is not installed\n' >&2
      exit 1
    }
    docker info >/dev/null 2>&1 || {
      printf 'error: Docker was requested but its daemon is unreachable\n' >&2
      exit 1
    }
    runtime_label="docker"
    runtime_args=(--docker)
    ;;
esac

if [ "$setup_only" -eq 1 ]; then
  printf 'Setup complete. Start with: ./scripts/run-linux.sh --mode %s --cwd %q\n' \
    "$mode" "$workspace"
  exit 0
fi

printf 'Starting FrontierAgent (%s, runtime=%s) in workspace: %s\n' "$mode" "$runtime_label" "$workspace"
# ``${a[@]+"${a[@]}"}`` rather than a bare ``"${a[@]}"``: bash before 4.4
# (CentOS/RHEL 7 ship 4.2) treats an EMPTY array as unset under ``set -u`` and
# aborts. Both arrays are empty on the default path -- no runtime flag, no
# forwarded args -- so the plain form fails after the install has already run.
exec "$uv_bin" run frontier-agent \
  ${runtime_args[@]+"${runtime_args[@]}"} \
  --mode "$mode" \
  --cwd "$workspace" \
  ${forwarded[@]+"${forwarded[@]}"}
