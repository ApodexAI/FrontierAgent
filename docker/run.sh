#!/usr/bin/env bash
set -euo pipefail

# Convenience runner script using docker compose
# Usage:
#   ./docker/run.sh                        # Run interactive agent CLI
#   ./docker/run.sh -p "explain main.py"   # Run agent prompt
#   ./docker/run.sh eval --limit 1         # Run benchmark evaluation

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export APODEX_HOST_UID="$(id -u)"
export APODEX_HOST_GID="$(id -g)"
export APODEX_LOCAL_UTC_OFFSET="$(date +%z)"
export APODEX_HOST_RUNS_ROOT="$repo_root/.apodex/runs"
mkdir -p "$repo_root/.apodex/runs"

if [ "${1:-}" = "eval" ]; then
  shift
  # `docker compose run SERVICE ARGS...` replaces the service command, so
  # include the required benchmark defaults before forwarding overrides.
  exec docker compose run --rm eval \
    --benchmark browsecomp \
    --out /app/results/smoke \
    "$@"
else
  exec docker compose run --rm agent "$@"
fi
