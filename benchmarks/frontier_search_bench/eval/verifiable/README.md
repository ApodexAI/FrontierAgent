# Verifiable Eval — auto-scoring scripts

Programmatic scoring for the 41 queries in `queries/verifiable.json` at the
FrontierSearchBench directory root.

## Directory layout

```
eval/verifiable/
├── README.md
├── requirements.txt
├── .env.example
├── run_all.py                          # cross-query aggregation entry (see "Batch evaluation (cross-query)" below)
└── scorers/
    ├── pipeline/                       # shared extract-align framework (37 of the 41 queries use extraction; 15 of those go on to alignment)
    │   ├── __init__.py
    │   ├── extraction_pipeline.py      # Stage 1: LLM-extract structured claims from a model's answer
    │   └── alignment.py                # Stage 2: align claims to ground-truth DIMS
    └── query_NN/                       # one subdirectory per query, directory number = query id
        ├── extract.py                  # extraction spec: ENTITIES + PROMPT_HINTS + VALUE_SCHEMA
        ├── auto_scorer.py              # GT, scoring rules, entry point; imports constants from extract.py
        └── scoring_framework.md        # (some queries) human-readable scoring rules
```

Some queries inline the `extract.py` constants directly in `auto_scorer.py`, so a few `scorers/query_NN/` directories contain only `auto_scorer.py` (Q05/Q15/Q31/Q38).

## Current coverage

Auto-scoring is implemented for all 41 queries:

| Query | One-line summary |
|---:|---|
| Q01 | Top-3 resale-return Singapore condos among 2019-TOP projects |
| Q02 | State-level New Areas (国家级新区): GDP growth-rate gap turning negative |
| Q03 | SpaceX Starship flight-test count + per-flight progress/failure points |
| Q04 | Global INES level-4+ nuclear accidents, 1945–2026 |
| Q05 | Project-985 universities' CS admission cutoffs in Henan |
| Q06 | Nobel-winning papers once rejected by journals |
| Q07 | IMO-participating country with the most Fields Medalists |
| Q08 | James's second-highest-scoring game vs. the Cavaliers during Kobe's career |
| Q09 | Chinese athletes with ≥3 Olympic medals but no gold |
| Q10 | 2016–2025 Chinese-language films, three-condition filter |
| Q11 | Papers rejected by the big-three venues with citations > 10000 |
| Q12 | Nobel literature laureates expelled/exiled by their home country |
| Q13 | Largest single-day drop along the supply chain of a tech stock that fell 5% on earnings |
| Q14 | Heads of state holding multiple citizenships while in office |
| Q15 | Neva Masquerade WCF Best Cat catteries |
| Q16 | Nobel wait-time statistics, 2000–2025 |
| Q17 | 2025–2026 AI paper releases → >5% abnormal stock moves within 48h |
| Q18 | Architect riddle → skyscraper's current name |
| Q19 | Pre-1935 unbuilt competition schemes Kunio Maekawa worked on at Le Corbusier's office |
| Q20 | Earworm song identification |
| Q21 | Years a Nobel prize was accepted by proxy (do/in letter clue) |
| Q22 | First drugs approved under the fast-track designation |
| Q23 | 2025 Megatron-MoE issues in the slime repository |
| Q24 | Other works by the female journalist in the documentary 《等等》 |
| Q25 | Profit split across the Starbucks latte supply chain |
| Q26 | A-share stock identification |
| Q27 | Geometry of cutting a 540° Möbius strip 0.5 cm from the edge |
| Q28 | NASA exoplanets + JWST water-bearing planets |
| Q29 | CUDA kernel optimizations FlashAttention v1→v3 borrowed from non-attention papers |
| Q30 | Alan Yuille's academic genealogy (traced to the 18th century) |
| Q31 | Geometry/physics of a pencil lead drilled at an angle through a wax block |
| Q32 | Mean character length of GT answers in DOCBENCH (Zou et al., 2025) |
| Q33 | US counties on the due-south path from Denver to the South Pole (north→south order) |
| Q34 | Peninsula hotels with a To Summer (观夏) store within 1 km |
| Q35 | Farthest-apart Haidilao stores per country |
| Q36 | Coordinates of a Greek roadside GeoGuessr photo |
| Q37 | Amazon rainforest net loss 2015–2024 + top-3 hotspots |
| Q38 | Shortest 7-city TSP route |
| Q39 | Filtering livable capitals worldwide + mean distance to the equator |
| Q40 | Route 66 elevation profile (highest/lowest/7 state lines) |
| Q41 | Bearing reckoning from the Empire State Building + visible waters |

## Install

Requires Python 3.10+.

```bash
pip install -r requirements.txt
```

## Configure

Put OpenRouter (or any OpenAI-protocol-compatible LLM gateway) credentials in `.env`:

```bash
cp .env.example .env
# edit .env and fill in OPENROUTER_API_KEY=sk-...
```

Place `.env` at `eval/verifiable/.env`, the location consistently supported by
the batch runner and all query scorers. Alternatively, export
`OPENROUTER_API_KEY` and `OPENROUTER_BASE_URL` in the environment.

The code reads two environment variables: `OPENROUTER_API_KEY` (required) and `OPENROUTER_BASE_URL` (optional, default `https://openrouter.ai/api/v1`). Models are called through the OpenAI SDK.

## Score a single query

```bash
cd eval/verifiable
python scorers/query_22/auto_scorer.py \
    --models claude=/path/to/claude_responses.json \
             gpt=/path/to/gpt_responses.json \
    --output-dir scorers/query_22/auto_scores
```

> Run `python <path>/auto_scorer.py` directly (do **not** use `python -m`) — some queries load their local spec module via `from extract import …`, which relies on Python automatically putting the script's own directory on `sys.path`, and that only happens when the script is invoked directly.

**Input JSON format** (one file per model; covering all 41 queries is recommended, but a missing id is not fatal — the scorer looks up the matching entry by id and treats the model as "didn't answer this query" when absent):

```json
[
  {"id": 1,  "query": "...", "response": "<model free-text answer>"},
  {"id": 22, "query": "...", "report_content": "<final report>", "response": "..."},
  ...
]
```

`id` aligns with the repository's `queries/verifiable.json`. `scorers/pipeline/extraction_pipeline.py::load_models_input` takes `report_content` and falls back to `response`; plain-text files are also accepted (the whole file is treated as the response).

> A ready-to-fill skeleton is available at [`queries/answer_template.json`](../../queries/answer_template.json) (all 41 entries with verbatim `query` text and empty answer fields — fill them in and the file is a valid input; `run_all.py` checks `query` against the bank verbatim and aborts on mismatch).

**Output**: lands in `--output-dir` (for most queries the default is `scorers/query_NN/auto_scores/`). One subdirectory per model `<model>/`, containing `extraction.json`, `alignment.json` (only for the 15 alignment queries), and `score.json` (per-model single-query score); the top level of the directory additionally gets `ranking_report.md` (human-readable ranking) and `scores.json` (cross-model aggregate; 30 queries write it, 11 queries only write per-model `<model>/score.json` without the top-level file — see "Batch evaluation (cross-query)" below).

### CLI is not fully uniform

The vast majority of queries accept `--models name=path ...` + `--output-dir`. A few queries differ slightly because of unusual GT structure or a simpler pipeline — check `--help` before running:

| Query | Note |
|---|---|
| Q05, Q15, Q31, Q38 | accept `--query-id`, `--result-json` (can skip Stage 1 and score directly) |
| Q16, Q20, Q21, Q23, Q24, Q25, Q26, Q27, Q35, Q36 | `--output-dir` is `required` (other queries default to `scorers/query_NN/auto_scores/`) |
| Q03, Q29, Q40 | skip Stage 2 alignment; match GT directly by keywords |
| Q17, Q19 | expose `--judge-model` for the Tier-4 LLM-judge fallback (default claude-sonnet-4) |
| most queries | expose `--primary-model`/`--secondary-model`/`--judge-model` etc. to override the pipeline's default LLMs |

If an `argparse` option is `required=True` and no value is given, the script fails fast and names the missing argument.

## Batch evaluation (cross-query)

`run_all.py` is the orchestrator — it runs the single-query flow above across all 41 queries and aggregates, which is the convenient way to compare several models' overall performance in one go.

```bash
cd eval/verifiable
python run_all.py \
    --models claude=/path/to/claude_unified.json \
             gpt=/path/to/gpt_unified.json
```

**Main flags** (full list via `python run_all.py --help`):

| flag | effect |
|---|---|
| `--models name=path ...` | (required) one unified JSON per model |
| `--out PATH` | aggregate output directory (default `eval/verifiable/all_results/`) |
| `--only 22,33` | run only the given queries (comma-separated, optional Q prefix) |
| `--force` | allow overwriting an existing non-empty `--out` directory |
| `--force-rerun` | rerun a query even if its `scores.json` already contains all target models |
| `--allow-query-mismatch` | tolerate id↔query text mismatches (default: abort) |
| `--dry-run` | pre-flight + print the plan only, without calling any scorer |
| `--per-query-timeout SEC` | per-query subprocess timeout (default 3600, 0 = unlimited) |

**Pre-flight checks** (run immediately at startup, no API spend):

- Compare every model JSON's `(id, query)` against `queries/verifiable.json`, with NFKC normalization + zero-width-character stripping + whitespace folding. A text mismatch aborts by default; `--allow-query-mismatch` is needed to continue.
- Print each model's 41-query coverage matrix (✓ / ✗). If a model is missing an id, that model **only** shows `N/A` on that query — other models are unaffected and the query is not skipped as a whole.

**Output** (lands in the `--out` directory):

- `matrix.csv` — 41 rows × N columns; each cell is `total_rate` (0–1 normalized score) / `N/A` (model missing that id) / `failed` (scorer error).
- `ranking.md` — overall model ranking (descending mean `total_rate` over answered queries) + a coverage column (X/41); the header records each model JSON's sha256 and the hash of the canonical 41-query set, for input traceability.
- `coverage.json` — full pre-flight details and per-query run status, machine-friendly.
- `logs/query_NN.log` — complete stdout/stderr of every query's scorer.

**Idempotency**: a query is skipped entirely when its `scorers/query_NN/auto_scores/scores.json` already contains all `--models`; if any model is missing, the whole query reruns. The 30 queries that write `scores.json` overwrite it wholesale, so pass all the models you care about in one invocation to avoid clobbering each other.

**Per-model synthesis**: the other 11 queries (Q16/Q20/Q21/Q23/Q24/Q25/Q26/Q27/Q35/Q36/Q40) historically only write per-model `<model>/score.json` and no top-level `scores.json`. When the orchestrator cannot read a top-level `scores.json`, it automatically scans these per-model files and synthesizes an equivalent dict so these 11 queries also enter the ranking. Q20 is special: scoring_framework v2.2 explicitly says "each song counts independently, no cap", so the query has no true max_score; to let it participate in the cross-query mean, the scorer writes a **normalization reference baseline** into the per-model score.json — `max_score = whitelist size = 4` — and fills `total_rate = min(score/4, 1.0)` accordingly. This is used for aggregation only and does not change the query's raw score semantics.

## Pipeline overview

37 queries share the four-stage framework in `scorers/pipeline/` (15 of them additionally go through Stage 2 alignment):

1. **Extract** (`scorers/pipeline/extraction_pipeline.py`): use an LLM to extract the model's free-text answer into structured claims as defined by the query's `VALUE_SCHEMA` in `extract.py`.
2. **Align** (`scorers/pipeline/alignment.py::align_claims`): use an LLM judge to align each claim to the baseline dimensions (`DIMS`/`BASELINE_*`), emitting a `canonical_id` (when a known dimension is hit) or `null` (when unalignable).
3. **Null verify** (optional; `scorers/pipeline/alignment.py::export_null_claims_for_review` + `apply_null_resolutions`): export the `canonical_id=null` entries from Stage 2 to an external web-search agent for verification; the verdicts (`baseline_add` / `hallucination` / `unresolved`) are written to `null_resolutions.json` and merged back into the alignment result by `apply_null_resolutions`.
4. **Score** (per-query `auto_scorer.py`): aggregate the score according to the query's scoring matrix (see `scoring_framework.md` or the docstring at the top of `auto_scorer.py`).

Some queries (e.g. Q03 SpaceX Starship, Q29 FlashAttention, Q40 Route 66) skip Stage 2 and compare against GT by keyword matching directly.

## Maintenance

- Adding a query: create `query_NN/` under `scorers/`, following the structure of `scorers/query_22/` (`extract.py`+`auto_scorer.py`) or `scorers/query_20/` (plus `scoring_framework.md`).
- The GT freeze time is recorded in the docstring at the top of `auto_scorer.py`; GT changes should come with a version bump.
- Schema self-check: after writing a new scorer or finishing a `run_all.py` round, run `python lint_scores_schemas.py` to verify every query's `scores.json` parses correctly through `extract_total_rates` — it runs 10 synthetic cases and scans the existing `scorers/query_*/auto_scores/scores.json`. The silent score-dropping bugs historically seen in Q03/Q28/Q39 (`total` / list-form) are all caught by it.
