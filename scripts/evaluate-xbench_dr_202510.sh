uv run python -m benchmarks.public.runner.run_subprocess \
  --benchmark xbench_dr_202510 \
  --pipeline stateful-react-agent \
  --profile default \
  --runs 1 \
  --concurrency 50 \
  --out ./results/$(date +%F)_xbench_dr_202510


# uv run python -m benchmarks.public.runner.check_progress ./results/$(date +%F)_xbench_dr_202510
