uv run python -m benchmarks.public.runner.run_subprocess \
  --benchmark superchem_text \
  --pipeline stateful-react-agent \
  --profile default \
  --runs 1 \
  --concurrency 50 \
  --out ./results/$(date +%F)_superchem_text


# uv run python -m benchmarks.public.runner.check_progress ./results/$(date +%F)_superchem_text
