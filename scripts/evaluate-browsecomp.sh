uv run python -m benchmarks.public.runner.run_subprocess \
  --benchmark browsecomp \
  --pipeline stateful-react-agent \
  --profile keep5 \
  --runs 1 \
  --concurrency 50 \
  --out ./results/$(date +%F)_browsecomp


# uv run python -m benchmarks.public.runner.check_progress ./results/$(date +%F)_browsecomp
