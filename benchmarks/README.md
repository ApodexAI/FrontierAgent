# Benchmarks

[Documentation index](../docs/README.md) ·
[Evaluation operator guide](../docs/eval.md)

The benchmark tree has three deliberately separate areas:

- [`public/`](public/) contains FrontierAgent's integrations for public
  benchmarks such as GDPval, HLE, OfficeQA, BrowseComp, and FrontierScience,
  together with their registry, runner, judges, dataset tooling, and local
  result directories.
- [`frontier_search_bench/`](frontier_search_bench/) contains
  [FrontierSearchBench](frontier_search_bench/): 41 verifiable deep-search
  queries, their official scorers, and batch evaluation tooling. Answer
  collection is integrated through the adapter under [`public/`](public/).
- [`frontierchallenge/`](frontierchallenge/) contains
  [FrontierChallenge](frontierchallenge/)'s Harbor runtime, 97-task registry,
  taxonomy, image recipes, documentation, and Hugging Face dataset setup flow.

The public evaluation harness runs one question per Python subprocess. Runs are
independently reproducible, resumable, and protected from a single hung task by
a process-level timeout. Use the
[evaluation guide](../docs/eval.md) for installation, credentials, judge
preflight, datasets, and result interpretation. This page owns the benchmark
registry and extension contract.

## Supported benchmarks

Each dataset registers its default workflow and scoring implementation. Passing
`--pipeline` overrides the default.

| Key | Family | Default pipeline | Scoring |
|---|---|---|---|
| `browsecomp` | BrowseComp | `stateful-react-agent` | LLM judge |
| `browsecomp_zh` | BrowseComp | `stateful-react-agent` | Chinese LLM judge |
| `xbench_dr_202510` | xbench-DR | `stateful-react-agent` | LLM judge |
| `hle_text` | HLE | `stateful-react-agent` | Pinned LLM judge |
| `superchem_text` | MCQ (HLE schema) | `stateful-react-agent` | LLM judge |
| `frontier_science_research` | FrontierScience | `stateful-react-agent` | LLM judge |
| `frontier_science_olympiad` | FrontierScience | `stateful-react-agent` | LLM judge |
| `deepsearchqa` | DeepSearchQA | `stateful-react-agent` | LLM judge |
| `widesearch` | WideSearch | `stateful-react-agent` | Structural F1 |
| `frontier_search` | FrontierSearchBench | `stateful-react-agent` | Bundled post-collection batch scorer |
| `officeqa` | OfficeQA | `stateful-react-agent` | Official deterministic reward |
| `officeqa_full` | OfficeQA-Full | `stateful-react-agent` | Official deterministic reward |
| `gdpval` | GDPval | `stateful-react-agent` | Deterministic deliverable validation |
| `apex` | APEX | `stateful-react-agent` | LLM rubric score |
| `onemillion_bench` | OneMillion-Bench | `agent_team` | Weighted LLM rubric score |

The runtime registry in `benchmarks.public.core.registry.REGISTRY` is authoritative.

## Run

```bash
# Smoke test
uv run python -m benchmarks.public.runner.run_subprocess \
  --benchmark browsecomp --pipeline stateful-react-agent --profile default \
  --limit 1 --concurrency 1 --out ./results/smoke

# Five resumable runs, each with 30 workers
uv run python -m benchmarks.public.runner.run_subprocess \
  --benchmark browsecomp --pipeline stateful-react-agent --profile default \
  --runs 5 --concurrency 30 --out ./results/browsecomp
```

Omit `--pipeline` to use the dataset default. Inspect progress and aggregate
accuracy with:

```bash
uv run python -m benchmarks.public.runner.check_progress ./results/browsecomp
```

`frontier_search` is collected first and scored after all answers are exported;
see [FrontierSearchBench evaluation](../docs/eval-frontier-search.md) for its
three-step workflow and its isolation requirement. Collection results
intentionally contain `null` accuracy rather than misreporting pending answers as
wrong.

## File benchmark datasets

Dataset files live outside Git. The
[evaluation guide](../docs/eval.md#file-benchmarks) owns the download commands,
the gated-dataset requirements, and `OFFICEQA_DOC_MODE`; this table records the
layout each dataset must end up with under the datasets root.

| Local directory | Standardized data | Additional inputs |
|---|---|---|
| `OfficeQA/` | `standardized_data.jsonl`, `standardized_full.jsonl` | Parsed Treasury corpus; optional PDFs |
| `GDPval/` | `standardized_data.jsonl` | `reference_files/` |
| `APEX/` | `standardized_data.jsonl` | World snapshots and task overlays |
| `OneMillion-Bench/` | `standardized_data.jsonl` | Domain rubrics and economic values |

File tasks receive read-only inputs under `/inputs`, work in `/workspace`, and
persist requested deliverables under `/outputs`. GDPval's OSS scorer validates
deliverable presence and readable structure; it does not run the source
project's agentic pairwise quality comparison against human work.

## Repository layout

```text
benchmarks/
├── public/                    public benchmark integrations and runtime
│   ├── core/                  registry, task schema, and adapters
│   ├── families/              dataset configurations
│   ├── judges/                deterministic and LLM-based scoring
│   ├── runner/                subprocess runner and progress reporting
│   ├── harbor_agent/          benchmark agent invoked by the runner
│   ├── scripts/               dataset download and standardization
│   ├── datasets/              local source data (gitignored)
│   └── results/               local run artifacts (gitignored)
├── frontier_search_bench/     FrontierSearchBench queries and official scorers
└── frontierchallenge/         FrontierChallenge scientific-workflow runtime
```

## Add a benchmark

1. Put `standardized_data.jsonl` and optional attachments in
   `benchmarks/public/datasets/<DatasetName>/`, or declare a bundled JSON source.
2. Export `CONFIGS: list[tuple[str, DatasetConfig]]` from a module under
   `benchmarks/public/families/`; family modules are auto-discovered.
3. Reuse/register an inline judge, or set `scoring_mode="external"` and provide
   an exporter plus batch-scoring entry point.

Standardized inline-scored rows require `task_id`, `task_question`,
`ground_truth`, and `answer_type`. Bundled external-scored datasets may select
JSON-list input, omit ground truth from collection tasks, and provide a default
answer type through `DatasetConfig`.
