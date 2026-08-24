#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
env_file="${SGLANG_ENV_FILE:-$repo_root/.env.sglang}"
compose_files=(-f "$repo_root/compose.yaml" -f "$repo_root/compose.sglang.yaml")

env_value() {
  local name="$1"
  # Mirrors `docker compose --env-file`, including its unquoting: a value the
  # operator wrote as KEY="1" must be read as 1, or the flags below silently
  # disagree with the Compose interpolation of the same file.
  awk -v key="$name" '
    index($0, key "=") == 1 {
      value = substr($0, length(key) + 2)
      sub(/\r$/, "", value)
      quote = substr(value, 1, 1)
      if (length(value) >= 2 && (quote == "\"" || quote == "\047") \
          && substr(value, length(value), 1) == quote) {
        value = substr(value, 2, length(value) - 2)
      }
      print value
    }
  ' "$env_file" | tail -n 1
}

# Lowercased with tr rather than bash's case-modification expansion: bash 3.2
# (the /bin/bash macOS still ships) treats that as a fatal bad substitution,
# killing this script before the doctor can explain that this path needs Linux.
enabled() {
  case "$(printf '%s' "${1:-}" | tr '[:upper:]' '[:lower:]')" in
    1|true|yes|on) return 0 ;;
    *) return 1 ;;
  esac
}

usage() {
  cat <<'EOF'
Usage: ./docker/run-sglang.sh COMMAND [ARGS...]

Commands:
  doctor [quick]  Validate the Linux, Docker, NVIDIA, disk, and model config
  up              Start the model and wait for its health check
  smoke           Start the model and verify API + structured tool calls
  tui [ARGS...]    Start the ReAct TUI (default command; forwards CLI args)
  status           Show Compose service state
  logs             Follow model logs
  down             Stop services; keep the Hugging Face cache volume

For a private GHCR image, run `docker login ghcr.io`. Set
SGLANG_BUILD_AGENT=1 in the env file to build the agent from this checkout.
EOF
}

if [ ! -f "$env_file" ]; then
  echo "error: SGLang config not found: $env_file" >&2
  echo "copy .env.sglang.example to .env.sglang and select a model source first" >&2
  exit 2
fi

# Files created by the unprivileged tool process should map back to the invoking
# host user instead of an image-specific system uid. The container entrypoint
# validates these values before changing the tool identity.
export APODEX_HOST_UID="$(id -u)"
export APODEX_HOST_GID="$(id -g)"
export APODEX_LOCAL_UTC_OFFSET="$(date +%z)"
export APODEX_HOST_RUNS_ROOT="$repo_root/.apodex/runs"
mkdir -p "$repo_root/.apodex/runs"

if enabled "$(env_value SGLANG_BUILD_AGENT)"; then
  compose_files+=(-f "$repo_root/compose.dev.yaml")
fi
if [ -n "$(env_value APODEX_DOCKER_SUBNET)" ]; then
  compose_files+=(-f "$repo_root/compose.network.yaml")
fi

compose=(docker compose --env-file "$env_file" "${compose_files[@]}")

start_model() {
  echo "Starting the SGLang model service (the first image/model download can take a while)..." >&2
  if ! "${compose[@]}" up -d --wait \
    --wait-timeout "${SGLANG_STARTUP_TIMEOUT:-3600}" model; then
    echo "SGLang failed to become healthy; recent model logs:" >&2
    "${compose[@]}" logs --tail 100 model >&2 || true
    exit 1
  fi
}

command="${1:-tui}"
if [ "$#" -gt 0 ]; then
  shift
fi

case "$command" in
  doctor)
    exec env SGLANG_ENV_FILE="$env_file" "$repo_root/docker/sglang-doctor.sh" "${1:-full}"
    ;;
  up)
    start_model
    ;;
  smoke)
    start_model
    exec "${compose[@]}" exec -T model python3 /opt/frontier-agent/smoke_sglang.py
    ;;
  tui)
    start_model
    echo "The model remains running after the TUI exits. Use ./docker/run-sglang.sh down to stop it." >&2
    exec "${compose[@]}" run --rm agent --mode react "$@"
    ;;
  status)
    exec "${compose[@]}" ps
    ;;
  logs)
    exec "${compose[@]}" logs -f model
    ;;
  down)
    exec "${compose[@]}" down
    ;;
  -h|--help|help)
    usage
    ;;
  *)
    # Backwards compatibility: historical usage passed FrontierAgent CLI flags
    # directly, for example `run-sglang.sh --mode agent_team`.
    start_model
    exec "${compose[@]}" run --rm agent "$command" "$@"
    ;;
esac
