# Upstream provenance

This directory is imported from
[`ApodexAI/frontier-search-bench`](https://github.com/ApodexAI/frontier-search-bench)
with Git subtree.

- Imported upstream commit: `0a5323b9823f8ee05486bbae11ca96999e5d5af9`
- FrontierAgent prefix: `benchmarks/frontier_search_bench`
- Import mode: squashed subtree

This directory documents itself as if answer collection happened elsewhere,
because upstream has no collector. In FrontierAgent it does: the adapter in
`benchmarks/public/families/frontier_search.py` feeds the shared subprocess
runner, and [`docs/eval-frontier-search.md`](../../docs/eval-frontier-search.md)
owns the collect / export / score workflow. Read that instead of inferring a FrontierAgent
workflow from the upstream README.

Keep FrontierAgent adapters outside this directory when possible. Pull a future
upstream release from the repository root with:

```bash
git subtree pull \
  --prefix=benchmarks/frontier_search_bench \
  https://github.com/ApodexAI/frontier-search-bench.git main --squash
```

The official scorers contain ground truth. Do not expose this directory to the
agent process during a scored collection run; see
`docs/eval-frontier-search.md` for the isolation requirement.
