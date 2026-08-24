# FrontierSearchBench

Apodex's benchmark for deep search / deep research. The goal is to compare AI platforms side by side on challenging tasks that are long-horizon, retrieval-heavy, and verifiable.

The repository holds two things:

1. **Verifiable benchmark queries** (`queries/verifiable.json`) — 41 evaluation inputs whose answers are unique or programmatically checkable (numbers, entities, enumerable sets, …).
2. **Evaluation scripts** (`eval/verifiable/`) — auto-scoring of model answers (41/41 implemented).

## Pipeline and data contract

```
queries/verifiable.json  ──►  (external collection: platform answers → unified JSON)  ──►  eval/verifiable/
```

Answer collection does not happen in this repository; the eval scripts only consume the **unified JSON** interface below.

**Query file** (`queries/verifiable.json`, 41 items) — each entry looks like:

```json
{ "id": 1, "query": "I want to invest in Singapore real estate. Please help me find, among the condominiums that obtained TOP (Temporary Occupation Permit) in 2019, the three projects with the highest resale return rate in 2025, and give the specific return rate for each." }
```

Note: **ground truth is NOT stored in the query file** — it lives in each `eval/verifiable/scorers/query_NN/auto_scorer.py`.

**Model answer file** (one per platform/model per batch):

```json
[
  { "id": 1,  "query": "...", "response": "<model free-text answer>" },
  { "id": 22, "query": "...", "report_content": "<final report>", "response": "..." }
]
```

The eval scripts read `report_content` and fall back to `response`. This is the only interface contract between external collection and the evaluation here.

**Ready-to-fill template**: [`queries/answer_template.json`](queries/answer_template.json) contains all 41 entries with the exact `query` text and empty `report_content` / `response` — fill in `report_content` (or `response`) per entry and the file is a valid eval input. Field rules enforced by `run_all.py`:

- top level is a JSON **list**; **one file per model/platform**;
- `id`: int, one of the canonical 1–41 (missing/non-int entries are skipped, duplicates keep the first, unknown ids are ignored);
- `query`: must match `queries/verifiable.json` **verbatim** — a mismatch aborts the run; a missing `query` only warns (alignment falls back to `id`);
- `report_content` preferred, `response` fallback; extra fields are tolerated.

> The template is **generated verbatim from `queries/verifiable.json`**. If the question bank ever changes, regenerate it (a stale template fails closed — `run_all.py` aborts on query mismatch):
>
> ```bash
> python3 -c 'import json,pathlib;p=pathlib.Path;src=json.loads(p("queries/verifiable.json").read_text(encoding="utf-8"));p("queries/answer_template.json").write_text(json.dumps([{"id":e["id"],"query":e["query"],"report_content":"","response":""} for e in src],ensure_ascii=False,indent=2)+"\n",encoding="utf-8")'
> ```

## Repository layout

```
frontier-search-bench/
├── queries/
│   └── verifiable.json           # 41 items
└── eval/
    └── verifiable/               # auto-scoring for verifiable queries (41/41 implemented)
        ├── run_all.py            # cross-query aggregation entry (N unified JSONs → model × query matrix + ranking)
        └── scorers/              # 41 query_NN/ dirs + shared pipeline/
```

## Environment setup

Requires **Python 3.10+**.

```bash
pip install -r eval/verifiable/requirements.txt
cp eval/verifiable/.env.example eval/verifiable/.env   # fill in OPENROUTER_API_KEY
```

See [`eval/verifiable/README.md`](eval/verifiable/README.md) for details.

## Quickstart

Prerequisite: unified JSON answer files prepared per the data contract above.

**1. Score a single query** (full instructions in [`eval/verifiable/README.md`](eval/verifiable/README.md)):

```bash
cd eval/verifiable
python scorers/query_22/auto_scorer.py \
    --models claude=/path/to/claude_unified.json \
    --output-dir scorers/query_22/auto_scores
```

> Always run `python <path>/auto_scorer.py` directly — **do not use `python -m`**. Some queries rely on `from extract import …`, which needs Python to put the script's own directory on `sys.path`, and that only happens when the script is invoked directly. See [`eval/verifiable/README.md`](eval/verifiable/README.md).

**2. Cross-query aggregation** (run all 41 queries in one go and produce the model × query matrix + overall ranking; alternative to single-query scoring):

```bash
# still inside eval/verifiable/
python run_all.py \
    --models claude=/path/to/claude_unified.json \
             gpt=/path/to/gpt_unified.json
# artifacts land in all_results/: matrix.csv, ranking.md, coverage.json, logs/
```

> Entry point, CLI, pre-flight checks, normalization and the meaning of coverage are documented in the "batch evaluation (cross-query)" section of [`eval/verifiable/README.md`](eval/verifiable/README.md).

## Status

- `queries/verifiable.json`: 41 items in place.
- `eval/verifiable/`: auto-scoring implemented for all 41 queries — see the checklist in [`eval/verifiable/README.md`](eval/verifiable/README.md). Top-level `run_all.py` provides the cross-query aggregation entry.

## Sub-document index

- [`eval/verifiable/README.md`](eval/verifiable/README.md) — full usage of the eval scripts, CLI, three-stage pipeline details, maintenance workflow.
