# Public benchmark integrations

This package contains FrontierAgent's evaluation runtime and integrations for
public benchmarks, including BrowseComp, HLE, OfficeQA, GDPval, and
FrontierScience. It owns the registry, dataset adapters, judges, subprocess
runner, download/standardization scripts, and gitignored local artifacts.

The independently maintained FrontierSearchBench source is intentionally kept
outside this package at [`../frontier_search_bench/`](../frontier_search_bench/).
The adapter in `families/frontier_search.py` lets the shared runner collect its
answers without mixing FrontierAgent-specific runtime code into that source
tree.

Run a one-question smoke evaluation after installing the eval dependencies and
placing the selected dataset under `datasets/`:

```bash
uv run python -m benchmarks.public.runner.run_subprocess \
  --benchmark browsecomp --pipeline stateful-react-agent --profile default \
  --limit 1 --concurrency 1 --out ./results/smoke
```

See the [benchmark registry](../README.md) and
[evaluation guide](../../docs/eval.md) for dataset setup, all benchmark keys,
FrontierSearchBench scoring, and extension instructions.
