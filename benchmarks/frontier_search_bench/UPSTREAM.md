# Maintenance notes

This directory is maintained directly in FrontierAgent at
`benchmarks/frontier_search_bench`. It contains the canonical query set and
official scorers; it is not synchronized from a separate public repository.

Keep FrontierAgent-specific collection adapters outside this directory when
possible. The adapter in
[`../public/families/frontier_search.py`](../public/families/frontier_search.py)
feeds the shared subprocess runner, while
[`../../docs/eval-frontier-search.md`](../../docs/eval-frontier-search.md) owns
the collect/export/score workflow.

The official scorers contain ground truth. Do not expose this directory to the
agent process during a scored collection run; see the
[FrontierSearchBench evaluation guide](../../docs/eval-frontier-search.md) for
the isolation requirement.
