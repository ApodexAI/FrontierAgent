# Frozen baselines

`results/` is gitignored and the repo lives on a FUSE mount that has already
handed out stale file handles. A baseline that exists only there is one bad
`rm -rf` from being unrecoverable — and it is unrecoverable by definition, since
the point of freezing is that the pre-refresh kernel no longer exists.

So the comparison-essential part is committed here: `results.json` (40 KB) plus
the provenance block. Full trajectories (32 MB) stay in `results/` only.

    python tools/compare_golden.py tools/golden results/after

`compare_golden.py` reads a directory containing `results.json`; these files are
renamed, so point it at `results/golden_react_base_<date>` while that exists, or
copy one back under the expected name.
