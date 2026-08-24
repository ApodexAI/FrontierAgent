uv run python -m benchmarks.public.runner.run_subprocess \
  --benchmark deepsearchqa \
  --pipeline stateful-react-agent \
  --profile default \
  --runs 1 \
  --concurrency 50 \
  --out ./results/$(date +%F)_deepsearchqa


# uv run python -m benchmarks.public.runner.check_progress ./results/$(date +%F)_deepsearchqa
