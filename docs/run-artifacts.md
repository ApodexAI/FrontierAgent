# Run artifacts and timestamps

Interactive runs keep their user-visible state and deliverables together below
the directory passed with `--cwd`:

```text
<cwd>/.apodex/
├── runs/
│   └── <session-id>/
│       ├── session.json       # resumable conversation checkpoint
│       ├── trace.jsonl        # ordered LLM and tool event trace
│       ├── engine.log         # warnings and failure diagnostics
│       ├── trajectories/      # workflow and sub-agent trajectories
│       ├── workspace/         # run-private clones, drafts, and scratch files
│       └── outputs/           # persistent user deliverables
└── runtime/
    └── native/                     # native-only caches and temporary state
        ├── cache/
        ├── dependencies/
        ├── home/
        ├── inputs/
        └── tmp/
```

`runs/` has the same host layout in native, Docker CLI, and Docker Compose
mode. The active `workspace/` and `outputs/` directories are exposed through
the stable `/workspace` and `/outputs` paths. The directory passed with
`--cwd` remains the project/coding root; it is mounted at `/project` in Docker
and is no longer used as scratch space. Benchmark and Harbor runs retain their
own result-tree layout because their harness owns artifact collection.

## Session IDs and time zones

New session IDs use the launching system's local wall clock and include its
numeric UTC offset:

```text
20260812-153045+0800-react-ab12
```

The offset makes a directory name understandable without silently losing the
instant it represents. Native mode reads the system time zone directly. The
Docker launchers pass the host's current offset into the container, so Docker
and native runs created at the same time use the same clock.

Persisted `created_at` metadata and event timestamps remain canonical UTC. This
keeps machine processing, ordering, and cross-time-zone comparison unambiguous;
only human-facing directory names and session-list timestamps use local time.

For unattended Compose invocations that bypass `docker/run.sh`, export the host
offset first. Compose falls back to `+0000` when it is unavailable and displays
the host run root relative to the project:

```bash
export APODEX_LOCAL_UTC_OFFSET="$(date +%z)"
docker compose run --rm agent
```

The helper launchers set `APODEX_HOST_RUNS_ROOT` to an absolute host path. This
value is only shown to the user: it is the path the TUI prints as `Host:` and
the one a resumed follow-up hands to the model. A direct Compose run instead
shows `.apodex/runs`, relative to the repository where Compose was invoked.

## Finding and resuming a run

Use `frontier-agent --resume` or `/sessions` rather than manually scanning the
tree. `/log` prints the active trace location. For direct inspection:

```bash
run_dir="/path/to/project/.apodex/runs/<session-id>"
tail -f "$run_dir/engine.log"
python -m json.tool "$run_dir/session.json"
```

Older checkpoints under `~/.apodex/sessions` and the previous native runtime
tree are still discovered and can be resumed. They are not renamed or moved;
after an old session is resumed, subsequent checkpoints use the unified run
directory under the selected `--cwd`.
