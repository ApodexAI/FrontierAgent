#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
env_file="${TRANSFORMERS_ENV_FILE:-$repo_root/.env.transformers}"
compose_files=(-f "$repo_root/compose.yaml" -f "$repo_root/compose.transformers.yaml")

if [ ! -f "$env_file" ]; then
  echo "error: Transformers config not found: $env_file" >&2
  echo "copy .env.transformers.example to .env.transformers first" >&2
  exit 2
fi

compose=(docker compose --env-file "$env_file" "${compose_files[@]}")

echo "Starting Transformers Serve (the first image build/model download can take a while)..." >&2
if ! "${compose[@]}" up -d --wait \
  --wait-timeout "${TRANSFORMERS_STARTUP_TIMEOUT:-3600}" model; then
  echo "Transformers Serve failed to become healthy; recent model logs:" >&2
  "${compose[@]}" logs --tail 100 model >&2 || true
  exit 1
fi

exec "${compose[@]}" run --rm agent "$@"
