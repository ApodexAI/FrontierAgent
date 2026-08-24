uv run python -m benchmarks.public.runner.run_subprocess \
  --benchmark frontier_science_research \
  --pipeline stateful-react-agent \
  --profile default \
  --runs 1 \
  --concurrency 50 \
  --out ./results/$(date +%F)_frontier_science_research


# uv run python -m benchmarks.public.runner.check_progress ./results/$(date +%F)_frontier_science_research
