# FrontierChallenge website

This directory contains the dependency-free static website for the benchmark.
It lives in the public FrontierChallenge repository so website content, task
documentation, and leaderboard links can evolve together.

Published preview: <https://urban-chainsaw-mnormyp.pages.github.io/>

## Preview locally

From the repository root, run:

```sh
python3 -m http.server 8000 --directory site
```

Then visit <http://localhost:8000>. Stop the server with `Ctrl-C`.

## Build for hosting

```sh
python3 site/build_site.py
```

The generated `site/dist/` directory is intentionally ignored by Git and is
recreated for each hosting build. GitHub Pages deployment is defined in
`.github/workflows/pages.yml`.

## Paper figures

The SVG files in `site/assets/` are publication figures copied from the paper's
`figs/` directory. Refresh these copies whenever the corresponding paper figure
is regenerated so the website and manuscript continue to show the same data.
