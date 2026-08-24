# Evaluation

[Documentation index](../docs/README.md) ·
[Benchmark registry and extension reference](../benchmarks/README.md) ·
[FrontierSearchBench evaluation](eval-frontier-search.md)

The evaluation layer under `benchmarks/public/` converts public datasets into
isolated tasks, runs a selected workflow once per question subprocess, collects
text or file outputs, and dispatches the benchmark-specific judge. The standalone
FrontierSearchBench source and official scorers remain next to it under
`benchmarks/frontier_search_bench/`. The evaluation layer consumes the framework;
the framework never imports it.

## Install and configure

```bash
uv sync --python 3.12 --extra eval --extra sandbox --extra document-readers
cp .env.example .env
```

Agent credentials use `OPENAI_*`. Judge credentials use `JUDGE_API_KEY` and
`JUDGE_BASE_URL`; judge models are pinned by benchmark code. Search and fetch tools
use `SERPER_API_KEY` and `JINA_API_KEY`.

FrontierSearchBench does not use `JUDGE_MODEL`: its external scorer pins its own
model slate and takes the `JUDGE_*` credentials under different names. See
[FrontierSearchBench evaluation](eval-frontier-search.md#credentials).

Two optional settings pay for themselves on any real run:

- `SUMMARY_LLM_BASE_URL` + `SUMMARY_LLM_MODEL_NAME` let `web_fetch` condense a
  fetched page instead of dropping the raw page into context. Unset, extraction
  falls back to `OPENAI_BASE_URL` / `OPENAI_MODEL`, so it still works — it just
  bills whole scraped pages to your primary model. A small fast model is the
  right choice here; this is called once per fetch.
- `READDOC_VISION_URL` / `_MODEL` / `_KEY` enable image reading. Unset, an image
  yields a placeholder note rather than content, so benchmarks with image
  questions (OfficeQA) lose those points silently.

The runner sets `JUDGE_SESSION` once per run and every worker subprocess
inherits it, so the whole batch shares one upstream session and the identical
grading prompt hits the gateway's prompt cache. Gateways that don't know the
header ignore it.

### The judge model, and why a run may refuse to start

Each benchmark pins the model its official grader used — that pin is what makes
a score comparable with published results. Several pins are OpenRouter-style
names, which a plain OpenAI key and most gateways cannot route:

| Benchmark | Pinned judge model | Typically needs `JUDGE_MODEL`? |
|---|---|---|
| `browsecomp` | `gpt-4.1-2025-04-14` | no |
| `hle_text`, `superchem_text` | `o3-mini-2025-01-31` | no |
| `apex`, `onemillion_bench` | `gpt-4.1-2025-04-14` (global default) | no |
| `officeqa`, `officeqa_full`, `gdpval` | none — deterministic scorers | n/a |
| `browsecomp_zh` | `gpt-4o` | often |
| `deepsearchqa` | `google/gemini-2.5-flash` | **yes** |
| `frontier_science_research`, `frontier_science_olympiad` | `openai/gpt-5` | **yes** |
| `widesearch` | `openai/gpt-4.1` | **yes** |
| `xbench_dr_202510` | `google/gemini-2.0-flash-001` | **yes** |

`JUDGE_MODEL` overrides every pin. The run logs a warning when it does, because
a score graded by a substitute model is no longer comparable with the official
grader — that belongs in the run's own output, not in someone's memory.

Every run preflights the judge's model and **exits 2 before running a single
question** if it is unreachable, printing the model name and the gateway's own
error. Without that check an unreachable judge is silent: it swallows the
transport error, returns `NOT_ATTEMPTED` for every question, and the run
completes reporting 0% accuracy — indistinguishable from a model that answered
everything wrong, after however long the run took.

## Datasets

Nothing under `benchmarks/public/datasets/` is tracked by Git, so most benchmarks need
local data before they can run. FrontierSearchBench is the exception: its 41
queries and official scorers are bundled under `benchmarks/frontier_search_bench/`.
Other text benchmarks come from one archive; file benchmarks are downloaded per
dataset by [the script below](#file-benchmarks).

```bash
wget https://huggingface.co/datasets/apodex/Deep-Research-Benchmarks/resolve/main/deep_research_benchmarks_260607.zip
unzip -P 'apodex*()_2026' deep_research_benchmarks_260607.zip
mkdir -p benchmarks/public/datasets
mv benchmarks/datasets/* benchmarks/public/datasets/ && rmdir benchmarks/datasets
rm deep_research_benchmarks_260607.zip
```

Keep the password single-quoted: it contains shell metacharacters, and an
unquoted `*()` fails or expands against the working directory.

The `mv` is not optional. The published archive still carries the pre-rename
`benchmarks/datasets/<DatasetName>/` prefix, so extracting it at the repository
root writes to a path the runner no longer reads — every text benchmark would
then fail to find its data. If `FRONTIER_AGENT_DATASETS_DIR` is set, move the
extracted directories there instead of into `benchmarks/public/datasets/`.

The archive covers BrowseComp, BrowseComp-ZH, xbench-DR, SuperChem,
FrontierScience, DeepSearchQA, and WideSearch. HLE is the one exception — it is
not redistributed here. Accept the license on `cais/hle`, then place its
standardized JSONL at
`benchmarks/public/datasets/HLE-text/standardized_data.jsonl` yourself.

Each dataset unpacks to `benchmarks/public/datasets/<DatasetName>/standardized_data.jsonl`,
where the directory is the `key` its `DatasetConfig` declares in
`benchmarks.public.core.registry.REGISTRY`. Set `FRONTIER_AGENT_DATASETS_DIR` to move
that root out of the checkout — worth doing once the file corpora are real,
since OfficeQA and GDPval are several GB each.

## Run

```bash
uv run python -m benchmarks.public.runner.run_subprocess \
  --benchmark browsecomp --pipeline stateful-react-agent --profile default \
  --limit 1 --concurrency 1 --out ./results/smoke
```

Important options include `--runs`, `--limit`, `--offset`, `--no-shuffle`,
`--answer-type`, `--category`, and `--fs-mode`. If `--pipeline` is omitted, the
dataset's registered default is used. Completed `result.json` files are resumable.
Pipeline IDs are exact registry keys: use `stateful-react-agent` for the single-agent
workflow and `agent_team` or `agent_team_report` for team workflows. The former
`react_base` pipeline has been removed and consolidated into `stateful-react-agent`.

### FrontierSearchBench

FrontierSearchBench is scored by an external cross-query scorer that ships with
its own ground truth, so it deviates from every other benchmark here in three
ways: collection and scoring are separate commands, the collection runner reports
`null` accuracy (`judge_method: external_pending`) rather than misreporting
pending answers as wrong, and a comparable score requires an OS-level read
boundary around `benchmarks/frontier_search_bench/eval/`.

[FrontierSearchBench evaluation](eval-frontier-search.md) owns that workflow —
the isolation requirement, the collect / export / score commands, the scorer's
credentials and options, and `scripts/evaluate-frontier-search.sh`.

### Open-book or closed-book

Each benchmark declares whether it is answerable only from what it provides, and
the runner applies that by default — so `--benchmark officeqa` unbinds
`web_search` / `web_fetch` / `download_file` without you asking, and
`--benchmark browsecomp` keeps them. The policy is logged at the top of every
run, and it reaches the workflow's tool list rather than being advisory.

| Benchmark | Default | Why |
|---|---|---|
| `officeqa`, `officeqa_full` | closed-book | The answer is derived from the Treasury corpus mounted at `/inputs`. |
| `apex` | closed-book | Its own prompt says to use only the provided files. |
| `gdpval` | open-book | Real-world deliverables; the task does not restrict sources. |
| `browsecomp`, `frontier_search`, `onemillion_bench`, … | open-book | Web research is the task. |

`--no-web` and `--web` override it. Reach for `--web` on a corpus benchmark only
knowing the result is no longer comparable with closed-book reports of it — the
gap between the two is itself worth measuring, since it tells you how much of a
score came from the corpus and how much from the open web.

## File benchmarks

These are not in the text archive from [Datasets](#datasets); download their
public source data separately:

```bash
uv run python benchmarks/public/scripts/download_datasets.py officeqa gdpval onemillion apex
```

OfficeQA and APEX are gated and require accepting their Hugging Face terms plus
`HF_TOKEN`. `OFFICEQA_DOC_MODE=parsed` mounts the parsed Treasury corpus;
`OFFICEQA_DOC_MODE=raw` requires downloading PDFs with `--raw-pdfs`.

| Benchmark | Default pipeline | Inputs and outputs | Scoring |
|---|---|---|---|
| OfficeQA / Full | `stateful-react-agent` | Treasury corpus at `/inputs` | Official deterministic numeric/text reward |
| GDPval | `stateful-react-agent` | Reference inputs at `/inputs`; deliverables from `/outputs` | Deterministic artifact structure validation |
| APEX | `stateful-react-agent` | Safely extracted world plus task overlay | LLM rubric score |
| OneMillion-Bench | `agent_team` | Text task, no file mounts | Weighted LLM rubric score |

GDPval intentionally excludes the original benchmark's agentic pairwise grader. The
score here validates that requested deliverables exist and are structurally readable;
it is not a quality comparison against the human reference deliverable.

OneMillion-Bench scores the weighted fraction of checklist items an answer hits,
against the `ONEMILLION_PASS` threshold (default `0.5`). Changing it changes
reported accuracy, so leave it alone unless you mean to report a different bar.

## Task filesystem contract

The runner translates standardized row metadata into sandbox metadata:

- `_dataset_root`: dataset location for resolving relative source files;
- `_sandbox_mounts`: read-only inputs mounted under `/inputs`;
- `_sys_prompt_addendum`: benchmark-specific filesystem instructions;
- `_collect_outputs`: collect persistent files from `/outputs` for grading.

Each question has its own `/workspace`. World archives are checked for path traversal
before extraction. Output collection never treats arbitrary workspace files as final
deliverables.

## Results and progress

Single runs write `tasks/`, `trials/`, `results.json`, and `summary.txt`. Multi-run
evaluation writes one `run_<n>/` directory per seed. Inspect progress with:

```bash
uv run python -m benchmarks.public.runner.check_progress ./results/run
```

See [benchmarks/README.md](../benchmarks/README.md) for the registry table and
dataset layout used when adding a benchmark.
