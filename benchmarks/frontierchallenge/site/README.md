# FrontierChallenge website

This directory contains the dependency-free static website for the benchmark.
It lives in FrontierAgent alongside FrontierChallenge's task documentation and
leaderboard links so they can evolve together.

Published preview: <https://urban-chainsaw-mnormyp.pages.github.io/>

## Preview locally

From `benchmarks/frontierchallenge/`, run:

```sh
python3 -m http.server 8000 --directory site
```

Then visit <http://localhost:8000>. Stop the server with `Ctrl-C`.

## Build the Worker bundle

```sh
python3 site/build_site.py
```

The generated Worker module under `site/dist/` is intentionally ignored by Git
and recreated for each bundle. The separate GitHub Pages deployment is defined
in the repository-level
[`frontierchallenge-pages.yml`](../../../.github/workflows/frontierchallenge-pages.yml)
workflow.

## Paper figures

The SVG files in `site/assets/` are publication figures copied from the paper's
`figs/` directory. Refresh these copies whenever the corresponding paper figure
is regenerated so the website and manuscript continue to show the same data.
