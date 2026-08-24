#!/bin/bash
set -euo pipefail

BENCHMARK="${1:-browsecomp}"
PROFILE="${2:-benchmark}"
# Drop the two positionals we consumed so "$@" carries only extra runner flags.
shift $(( $# < 2 ? $# : 2 ))

echo "Running benchmark evaluation with agent_team pipeline..."
uv run python -m benchmarks.public.runner.run_subprocess \
  --benchmark "${BENCHMARK}" \
  --pipeline agent_team \
  --profile "${PROFILE}" \
  "$@"
