#!/usr/bin/env bash
set -euo pipefail

run_dir="./results/$(date +%F)_frontier_search"

# The bundled scorer tree holds this benchmark's ground truth (standard_answer
# .json plus reference answers in each extract.py). Collecting answers from a
# checkout where the agent can read those files invalidates the score. See
# docs/eval-frontier-search.md for the isolation options.
if [[ -d benchmarks/frontier_search_bench/eval ]] && \
   [[ "${FRONTIER_SEARCH_ALLOW_UNISOLATED:-}" != "1" ]]; then
  cat >&2 <<'EOF'
ERROR: benchmarks/frontier_search_bench/eval/ is readable from this checkout.

  It contains the ground truth for all 41 queries, so an agent with file tools
  can read the answers and the resulting score is not comparable.

  This wrapper cannot attest that a later Docker or bubblewrap boundary denies
  access to that tree. Either run the collection/export commands from
  docs/eval-frontier-search.md directly inside a verified sandbox, or run this
  wrapper from a checkout/image that excludes
  benchmarks/frontier_search_bench/eval/.

  For local development only, re-run with:
      FRONTIER_SEARCH_ALLOW_UNISOLATED=1 ./scripts/evaluate-frontier-search.sh
EOF
  exit 1
fi

if [[ "${FRONTIER_SEARCH_ALLOW_UNISOLATED:-}" == "1" ]]; then
  echo "WARNING: collecting without an OS-level read boundary — development only." >&2
fi

uv run python -m benchmarks.public.runner.run_subprocess \
  --benchmark frontier_search \
  --pipeline stateful-react-agent \
  --profile default \
  --runs 1 \
  --concurrency 10 \
  --no-shuffle \
  --out "$run_dir"

uv run python -m benchmarks.public.runner.export_frontier_search \
  "$run_dir" \
  --out "$run_dir/frontier_agent.json"

echo "Answers exported to $run_dir/frontier_agent.json"
echo "Run the official scorer with:"
echo "uv run python -m benchmarks.public.runner.score_frontier_search --models frontier_agent=$run_dir/frontier_agent.json --out $run_dir/official_scores"
