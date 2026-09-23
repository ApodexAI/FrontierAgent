# Running the benchmark

[Quickstart](quickstart.md) runs one task. This page covers complete runs,
subsets, and resuming.

## Open and full tracks

The open track contains the 81 tasks that use the redistributable image:

```bash
./scripts/setup.sh --track open
./scripts/run_eval.sh --agent claude-code --model <model>
```

The full track adds 16 tasks that execute ORCA. Prepare the licensed local
runtime first:

```bash
./scripts/build_orca_runtime.sh --orca-root /path/to/orca-6.0.1
./scripts/setup.sh --track full
./scripts/run_eval.sh --track full --agent claude-code --model <model>
```

Setup writes verified local paths under `.frontierchallenge/`. The runner
validates the GitHub/solve/reference registries again before staging anything.
Track membership follows each task's declared execution environment, not
software names in its instruction or supplied files. Thus
`task_098_orca_claisen_thermochemistry`, which only reads precomputed ORCA
output, remains an open-track task.

## Runtime

Docker with the pinned shared image is the only supported public evaluation
path. `run_eval.sh` fails before Harbor starts if the daemon or Compose v2 is
missing. See [Docker](providers/docker.md).

## Subsets and concurrency

```bash
# one task
./scripts/run_eval.sh --agent codex --model <model> \
  --include-task-name task_011_cell_migration_wound_healing

# repeatable glob filters
--include-task-name 'task_2*' --exclude-task-name task_216_n2_multireference_curve_nevpt2

# trials and agent phases in flight
--n-concurrent 8 --n-concurrent-agents 4
```

Use disjoint include lists and distinct job names to shard across machines.
`summarize_results.py` accepts one job directory at a time; it does not merge
shards. A shard's report is partial, not a separate full-benchmark result.
Do not average shard Pass Rates as though they were full runs.

Concurrency must fit both machine resources and model-provider rate limits.
Start with one task, then increase gradually.

## Timeouts and staging

Each task declares its own agent timeout. To raise shorter timeouts without
lowering longer ones:

```bash
--min-agent-timeout-sec 86400 \
--stage-dir /data/frontierchallenge/stage-24h
```

Use a separate stage directory for each concurrent run that changes timeouts.
Verifier timeouts use `--verifier-timeout-multiplier` (default 40).

Legacy staging caches containing Hugging Face symlinks are automatically
rebuilt once. Only the effective include/exclude selection is staged, checked
for ORCA, unsealed, and passed to Harbor; leftover stage directories are ignored.

## Resume and results

Reusing a job name resumes completed work when the requested task set matches:

```bash
./scripts/run_eval.sh --agent claude-code --model <model> --job-name <same-name>
```

Results are written under `results/harbor/<job>/`. Read the aggregate with:

```bash
python3 scripts/summarize_results.py results/harbor/<job>
cat results/harbor/<job>/summary.json
```

Official Pass Rate requires completed `task_score == 1.0`; `passed` means the
same thing in rewards and summaries. The default denominator is 97, including missing tasks.
See [Scoring](scoring.md) for partial scores and explicitly labeled subsets.
Automatic summaries use the runner's current selection. If a job directory
contains trials from an older selection, use repeatable `--task-id <id>` with
the standalone summarizer, or start a fresh job directory.

Before a long run, verify one task reaches `evaluation_complete = 1`, confirm
the selected backend in the startup banner, and confirm the local ORCA runtime
before selecting the full track.
