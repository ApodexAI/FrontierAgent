#!/usr/bin/env bash
# Bootstrap and run FrontierAgent after cloning the repository on macOS.

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mode="${FRONTIER_AGENT_MODE:-react}"
workspace="${FRONTIER_AGENT_WORKSPACE:-$repo_root}"
runtime="${FRONTIER_AGENT_RUNTIME:-auto}"
setup_only=0

usage() {
  printf '%s\n' \
    "Usage: ./scripts/run-macos.sh [OPTIONS] [-- FRONTIER_AGENT_ARGS...]" \
    "" \
    "Prepare a cloned FrontierAgent checkout and start the macOS TUI." \
    "" \
    "Options:" \
    "  --mode react|agent_team  Workflow to start (default: react)" \
    "  --cwd PATH               Workspace the agent may operate on" \
    "  --native                 Require the native macOS runtime" \
    "  --docker                 Require Docker Desktop" \
    "  --setup-only             Install/configure without starting the TUI" \
    "  -h, --help               Show this help" \
    "" \
    "Environment equivalents:" \
    "  FRONTIER_AGENT_MODE, FRONTIER_AGENT_WORKSPACE, FRONTIER_AGENT_RUNTIME"
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
    --docker)
      runtime="docker"
      shift
      ;;
    --setup-only)
      setup_only=1
      shift
      ;;
    --)
      shift
      break
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      printf 'error: unknown option: %s\n\n' "$1" >&2
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
  auto|native|docker) ;;
  *) printf 'error: FRONTIER_AGENT_RUNTIME must be auto, native, or docker\n' >&2; exit 2 ;;
esac

if [ "$(uname -s)" != "Darwin" ]; then
  printf 'error: this helper is for macOS; Linux users should run ./scripts/run-linux.sh\n' >&2
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
  command -v curl >/dev/null 2>&1 || {
    printf 'error: curl is required to install uv\n' >&2
    exit 1
  }
  printf 'uv is not installed; downloading the official installer from astral.sh...\n'
  uv_installer="$(mktemp "${TMPDIR:-/tmp}/frontier-agent-uv.XXXXXX")"
  trap 'rm -f "${uv_installer:-}"' EXIT
  curl --proto '=https' --tlsv1.2 -LsSf https://astral.sh/uv/install.sh -o "$uv_installer"
  sh "$uv_installer"
  uv_bin="$(find_uv || true)"
  if [ -z "$uv_bin" ]; then
    printf 'error: uv installation finished but the uv executable was not found\n' >&2
    exit 1
  fi
fi

env_value() {
  awk -v key="$1" '
    {
      line = $0
      sub(/^[[:space:]]*export[[:space:]]+/, "", line)
    }
    index(line, key "=") == 1 {
      value = substr(line, length(key) + 2)
      sub(/\r$/, "", value)
      quote = substr(value, 1, 1)
      if (length(value) >= 2 && (quote == "\"" || quote == "\047") \
          && substr(value, length(value), 1) == quote) {
        value = substr(value, 2, length(value) - 2)
      }
      result = value
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
      '首次运行需要配置一个 OpenAI-compatible 模型端点。' \
      '密钥输入不会显示；本地或 SSH 转发的 SGLang 可填写 EMPTY。'
    if [ -z "$api_key" ]; then
      IFS= read -r -s -p 'OPENAI_API_KEY: ' api_key
      printf '\n'
    fi
    if [ -z "$base_url" ]; then
      IFS= read -r -p 'OPENAI_BASE_URL（通常以 /v1 结尾）: ' base_url
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
    printf '# Generated by scripts/run-macos.sh; do not commit this file.\n'
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

printf 'Installing FrontierAgent and Python 3.12 dependencies...\n'
cd "$repo_root"
# A launcher must not uninstall dev/document extras already present in a user's
# checkout. --inexact adds required packages while preserving those additions.
"$uv_bin" sync --inexact --python 3.12

if [ "$runtime" = "docker" ]; then
  command -v docker >/dev/null 2>&1 || {
    printf 'error: Docker CLI not found; install and start Docker Desktop\n' >&2
    exit 1
  }
  docker info >/dev/null 2>&1 || {
    printf 'error: Docker Desktop is not running or its daemon is unreachable\n' >&2
    exit 1
  }
fi

if [ "$setup_only" -eq 1 ]; then
  printf 'Setup complete. Start with: ./scripts/run-macos.sh --mode %s --cwd %q\n' \
    "$mode" "$workspace"
  exit 0
fi

case "$runtime" in
  native)
    printf 'Starting FrontierAgent (%s) in workspace: %s\n' "$mode" "$workspace"
    exec "$uv_bin" run frontier-agent --native --mode "$mode" --cwd "$workspace" "$@"
    ;;
  docker)
    printf 'Starting FrontierAgent (%s) in workspace: %s\n' "$mode" "$workspace"
    exec "$uv_bin" run frontier-agent --docker --mode "$mode" --cwd "$workspace" "$@"
    ;;
  auto)
    printf 'Starting FrontierAgent (%s) in workspace: %s\n' "$mode" "$workspace"
    exec "$uv_bin" run frontier-agent --mode "$mode" --cwd "$workspace" "$@"
    ;;
esac
