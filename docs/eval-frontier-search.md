# FrontierSearchBench evaluation

[Documentation index](README.md) ·
[Evaluation operator guide](eval.md) ·
[Benchmark registry](../benchmarks/README.md)

FrontierSearchBench is the one benchmark in this repository whose operating model
differs from the other fourteen: it is scored by an **external cross-query
scorer** that ships with its own ground truth, so collection and scoring are two
separate commands, and a comparable score requires an OS-level read boundary
around the checkout. That is why it has its own page — [`eval.md`](eval.md) owns
everything the benchmarks share (installation, judge preflight, datasets, the
runner's options, results), and this page owns only what is specific here.

The canonical benchmark implementation lives at
[`benchmarks/frontier_search_bench/`](../benchmarks/frontier_search_bench/).
FrontierAgent's collection adapter is deliberately kept outside that directory,
in
[`benchmarks/public/families/frontier_search.py`](../benchmarks/public/families/frontier_search.py),
so the query/scorer implementation remains separate from the shared runtime.
See its [maintenance notes](../benchmarks/frontier_search_bench/UPSTREAM.md) for
the ownership and isolation boundaries. The benchmark implementation consumes
unified JSON answer files; inside FrontierAgent, the shared subprocess runner
collects and exports those answers as described below.

No dataset download is needed. The 41 queries and all 41 official scorers are
bundled with FrontierAgent.

## Evaluation isolation requirement

> The bundled scorer source contains ground truth (`standard_answer.json` plus
> reference answers inside each `extract.py`). A benchmark agent that can read
> the parent checkout can inspect those answers and invalidate the run. Either
> invoke the collection runner inside a verified Docker/bubblewrap boundary that
> denies access to `benchmarks/frontier_search_bench/eval/`, or collect from a
> checkout/image that omits that directory. Native execution without an OS-level
> read boundary is suitable for development only, not for a comparable score.

`scripts/evaluate-frontier-search.sh` uses a deliberately conservative check: it
refuses to start whenever that directory is readable because a shell wrapper
cannot attest that a later Docker/bubblewrap boundary is configured correctly.
Use the runner and exporter commands below directly inside a verified sandbox,
or use the convenience script from a checkout/image that omits `eval/`.
`FRONTIER_SEARCH_ALLOW_UNISOLATED=1` overrides the refusal for local development
and prints a warning into the run's own output; it is not evidence of isolation.

## Credentials

The wrapper maps `JUDGE_API_KEY` and `JUDGE_BASE_URL` onto the scorer's
`OPENROUTER_API_KEY` and `OPENROUTER_BASE_URL`, always as a pair from one
provider — set `OPENROUTER_*` directly and the `JUDGE_*` values are left alone.

`JUDGE_MODEL` does **not** reach that scorer. It pins its own slate
(`anthropic/claude-sonnet-4`, `openai/gpt-5`, `google/gemini-2.5-pro`,
`anthropic/claude-opus-4.6`), so the endpoint must serve those route names. An
incompatible endpoint fails during the scoring phase, after answer collection has
already completed. `--only 1 --dry-run` validates the answer file and scoring
plan without contacting the endpoint. A live compatibility check currently
requires scoring at least one query with `--only 1` (see [Score](#3-score)).

## 1. Collect

The collection runner records `judge_method: external_pending` and writes `null`
for `reward`, `is_correct`, and aggregate accuracy until the official scorer runs.
That is deliberate: reporting pending answers as wrong would understate the run.

Collect all 41 answers in canonical order:

```bash
uv run python -m benchmarks.public.runner.run_subprocess \
  --benchmark frontier_search \
  --pipeline stateful-react-agent \
  --profile default \
  --no-shuffle --concurrency 10 \
  --out ./results/frontier_search
```

`--no-shuffle` keeps canonical query order, which makes a partial run readable
against the query list. Completed `result.json` files are resumable, so an
interrupted collection continues where it stopped.

## 2. Export

Export the run into the benchmark's unified JSON contract. For a multi-run
evaluation, export each `run_<n>` separately as one model/run input.

```bash
uv run python -m benchmarks.public.runner.export_frontier_search \
  ./results/frontier_search \
  --out ./results/frontier_search/frontier_agent.json
```

An incomplete run exits 3 without a usable submission: the official ranking
averages `total_rate` over the queries a model actually answered, so exporting 12
of 41 would report a headline score for the 12 easiest. Finish the run, or pass
`--allow-partial` when you knowingly want the partial number.

## 3. Score

Run all 41 official scorers. The wrapper reuses `JUDGE_API_KEY` and
`JUDGE_BASE_URL`, executes a temporary copy of the benchmark implementation,
and preserves aggregate plus per-query artifacts under the requested result
directory.

```bash
uv run python -m benchmarks.public.runner.score_frontier_search \
  --models frontier_agent=./results/frontier_search/frontier_agent.json \
  --out ./results/frontier_search/official_scores
```

| Option | Use |
|---|---|
| `--models name=path name2=path2` | Compare multiple platforms or runs in one matrix |
| `--only 1,22,41` | Score a subset of queries |
| `--dry-run` | Validate inputs and print the scoring plan; does not contact the endpoint |
| `--force-rerun` | Ignore cached per-query scores |

Re-running against the same `--out` reuses the per-query scores already in
`<out>/per_query`, so a scorer run interrupted at query 30 does not re-pay for
queries 1–29.

## Convenience script

`./scripts/evaluate-frontier-search.sh` performs collection and export in one
step (into `./results/<date>_frontier_search`), applies the conservative refusal
whenever `eval/` is readable, and prints the scoring command to run next. It does
not score for you — scoring costs money and pins its own model slate, so it stays
an explicit step.

## Scoring internals

The bundled scorer documentation is the reference for what the scorers do:
[`eval/verifiable/README.md`](../benchmarks/frontier_search_bench/eval/verifiable/README.md)
covers the CLI, the three-stage pipeline, normalization, and the meaning of
coverage. The unified JSON answer contract it consumes — and which
`export_frontier_search` produces — is documented in the benchmark's
[`README.md`](../benchmarks/frontier_search_bench/README.md).
