#!/usr/bin/env bash
# Freeze the react_base regression baseline BEFORE the kernel refresh lands.
#
# react_base's published Apodex-1.0 numbers are the thing the port can silently
# break: the refresh replaces llm_client / agent_loop / loop_types underneath it.
# This run is the only comparison point, and it stops being obtainable the
# moment Lane A touches those files.
#
# --no-shuffle is required, not cosmetic: the post-refresh re-run must select
# the same questions, and shuffling is seeded per run index.
#
#   ./tools/freeze_golden.sh            # 20 questions, the baseline
#   ./tools/freeze_golden.sh 3 4        # smaller/faster smoke first
#
# Compare later with:
#   uv run python tools/compare_golden.py <golden_dir> <new_dir>
set -euo pipefail

LIMIT="${1:-20}"
CONCURRENCY="${2:-10}"
OUT="${GOLDEN_OUT:-./results/golden_react_base_$(date +%F)}"

cd "$(dirname "$0")/.."

if [[ ! -f .env ]]; then
  echo "error: no .env — cp .env.example .env and fill OPENAI_* / SERPER_API_KEY /" >&2
  echo "       JINA_API_KEY / JUDGE_* before freezing the baseline." >&2
  exit 1
fi
if [[ ! -f benchmarks/public/datasets/BrowseComp/standardized_data.jsonl ]]; then
  echo "error: browsecomp dataset missing at" >&2
  echo "       benchmarks/public/datasets/BrowseComp/standardized_data.jsonl" >&2
  exit 1
fi

# One cheap LLM call before spending $LIMIT workers to learn one config fact.
# Skip with PREFLIGHT=0 only if you have already run it in this shell.
if [[ "${PREFLIGHT:-1}" == "1" ]]; then
  uv run python tools/preflight.py --pipeline react_base --profile keep5 || {
    echo "preflight failed — fix the config above before freezing the baseline" >&2
    exit 1
  }
  echo
fi

# Set the judge session here, not inside the runner, so the provenance block
# below records the same value the run actually used.
export JUDGE_SESSION="${JUDGE_SESSION:-judge-golden-$(date +%s)}"

# The repo lives on a FUSE network mount (fuse.aliyun-alinas-efc), where N
# concurrent workers importing the tree can read a stale view of files edited
# minutes earlier — three workers in the first attempt at this baseline died on
# an ImportError for a name that had already been fixed. Dropping stale
# bytecode removes the part of that we control.
find frontier_agent plugins workflows benchmarks -name __pycache__ -type d \
  -exec rm -rf {} + 2>/dev/null || true

echo "freezing react_base golden: limit=$LIMIT concurrency=$CONCURRENCY -> $OUT"
uv run python -m benchmarks.public.runner.run_subprocess \
  --benchmark browsecomp \
  --pipeline react_base \
  --profile keep5 \
  --runs 1 \
  --no-shuffle \
  --limit "$LIMIT" \
  --concurrency "$CONCURRENCY" \
  --out "$OUT"

# Record what produced these numbers. Without this the comparison is unfalsifiable:
# a model or endpoint change would look identical to a kernel regression.
{
  echo "frozen_at: $(date -Iseconds)"
  echo "git_commit: $(git rev-parse HEAD)"
  echo "git_branch: $(git rev-parse --abbrev-ref HEAD)"
  echo "benchmark: browsecomp  pipeline: react_base  profile: keep5"
  echo "limit: $LIMIT  shuffle: off"
  # Read through config, not the shell: .env is not exported into this shell,
  # and a baseline whose model is recorded as "<from .env>" is unfalsifiable.
  # Endpoints are recorded as a fingerprint only, never verbatim or partially.
  # Provenance needs them to be *distinguishable* (so an endpoint change cannot
  # be mistaken for a kernel regression), nothing more, and this file is checked
  # in. An earlier version kept the last three domain labels alongside the hash;
  # for a three-label host that is the entire host name, so the redaction was a
  # no-op and internal endpoints landed in the repo verbatim.
  uv run python -c "
import hashlib, os
from frontier_agent.infra.config import get_config


def tag(url: str) -> str:
    host = (url or '').split('//')[-1].split('/')[0]
    if not host:
        return '<unset>'
    return f\"#{hashlib.sha256(host.encode()).hexdigest()[:12]}\"


c = get_config()
print('agent_model:', c.openai_model)
print('agent_endpoint:', tag(c.openai_base_url))
print('judge_endpoint:', tag(os.environ.get('JUDGE_BASE_URL') or ''))
print('judge_session:', os.environ.get('JUDGE_SESSION', '<unset>'))
"
} > "$OUT/GOLDEN.txt"

echo
echo "summary:"; cat "$OUT/summary.txt" 2>/dev/null || true

# A baseline containing crashed workers is unusable: those questions score 0 for
# an environment reason, and a later run that crashes on a *different* question
# is indistinguishable from a real regression. Refuse to call it frozen.
# Crashed trials also get their result.json deleted. run_subprocess.py resumes
# from any existing result.json, so without this a re-run into the same
# directory replays the cached crashes in milliseconds and looks like the fix
# did nothing — while successful questions still resume, which is the point.
CRASHED=$(uv run python - "$OUT" <<'PY'
import json, sys
from pathlib import Path
out = Path(sys.argv[1])
d = json.loads((out / "results.json").read_text())
rows = d if isinstance(d, list) else d.get("results", [])
bad = [r for r in rows if r.get("judge_method") in ("worker_crash", "subprocess_timeout")
       or (r.get("error") and not r.get("predicted_answer"))]
print(len(bad))
for r in bad:
    qid = r["question_id"]
    print(f"  q{qid}: {r.get('judge_method')} {r.get('error')}", file=sys.stderr)
    (out / "trials" / str(qid) / "result.json").unlink(missing_ok=True)
PY
)
echo
if [[ "${CRASHED:-0}" != "0" ]]; then
  echo "NOT FROZEN — $CRASHED question(s) failed for environment reasons (see above)." >&2
  echo "Their result.json was cleared, so re-running retries exactly those:" >&2
  echo "  GOLDEN_OUT=$OUT $0 $LIMIT $CONCURRENCY" >&2
  echo "If it keeps happening, lower concurrency — the FUSE mount is the suspect." >&2
  exit 1
fi
echo "baseline FROZEN at $OUT (0 crashed workers)"
cat "$OUT/GOLDEN.txt"
