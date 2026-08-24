#!/usr/bin/env bash
# Live A/B for the tool-result preview shape, on a deliberately small context.
#
# Both arms run the SAME questions, seed and profile; the only difference is
# TOOL_RESULT_TRUNCATION. The context window and the per-tool cap are shrunk on
# purpose so overflow and compaction happen within a handful of turns instead of
# an hour in — the point is to exercise the truncate → spill → recover path
# often, not to produce a headline benchmark number.
#
#   scripts/run-truncation-ab.sh                    # defaults below
#   BENCH=widesearch LIMIT=12 scripts/run-truncation-ab.sh
#   ARM_VAR=TOOL_EXEC_RESULT_MAX_CHARS ARMS="8000 50000" \
#     OPENAI_CONTEXT_WINDOW=262144 TOOL_RESULT_MAX_CHARS=150000 \
#     scripts/run-truncation-ab.sh
#
# Read the result with:
#   uv run python scripts/truncation_metrics.py \
#       results/<stamp>_trunc-ab/head results/<stamp>_trunc-ab/middle \
#       --labels head,middle
set -euo pipefail

# The arm variable. Defaults to the preview shape; set ARM_VAR/ARMS to A/B
# something else with the same harness (e.g. the per-tool cap:
# ARM_VAR=TOOL_EXEC_RESULT_MAX_CHARS ARMS="8000 50000"). Read the window
# interaction in docs/tool-result-truncation-ab.md before changing a cap: a cap
# is only meaningful relative to the context window it competes with.
ARM_VAR="${ARM_VAR:-TOOL_RESULT_TRUNCATION}"
ARMS="${ARMS:-head middle}"

BENCH="${BENCH:-deepsearchqa}"          # long, tool-heavy runs
PIPELINE="${PIPELINE:-stateful-react-agent}"
PROFILE="${PROFILE:-benchmark}"         # reads OPENAI_CONTEXT_WINDOW + COMPACTION_SPILL
LIMIT="${LIMIT:-8}"
CONCURRENCY="${CONCURRENCY:-4}"
SEED="${SEED:-42}"
OUT_ROOT="${OUT_ROOT:-./results/$(date +%F_%H%M)_trunc-ab}"

# ── The knobs that make the experiment cheap ────────────────────────────
# 32K window: the tiered policy's 80% trigger / 60% relief geometry is unchanged,
# it just arrives ~8x sooner. 2K per-tool cap: nearly every bash/grep result
# overflows, so the preview shape is on the critical path of every turn.
export OPENAI_CONTEXT_WINDOW="${OPENAI_CONTEXT_WINDOW:-32768}"
export OPENAI_MAX_INPUT_TOKENS="${OPENAI_MAX_INPUT_TOKENS:-28672}"
export TOOL_EXEC_RESULT_MAX_CHARS="${TOOL_EXEC_RESULT_MAX_CHARS:-2000}"
# web_fetch / web_search / read_file set max_result_chars=0, so on a search
# benchmark the per-tool cap above never fires and the arms would be identical.
# This is the cap that governs them; 8K makes nearly every fetched page overflow.
export TOOL_RESULT_MAX_CHARS="${TOOL_RESULT_MAX_CHARS:-8000}"
# Recovery reads are the metric that separates the arms; they need a store.
export COMPACTION_SPILL="${COMPACTION_SPILL:-true}"

if [ "${ARM_VAR}" = "TOOL_EXEC_RESULT_MAX_CHARS" ]; then
  for ARM in ${ARMS}; do
    if [ "${ARM}" -gt "${TOOL_RESULT_MAX_CHARS}" ]; then
      echo "refusing to run: arm ${ARM_VAR}=${ARM} exceeds the global cap" >&2
      echo "TOOL_RESULT_MAX_CHARS=${TOOL_RESULT_MAX_CHARS}, which the loop applies" >&2
      echo "to every result — the large arm would be clamped and measure nothing." >&2
      echo "Raise TOOL_RESULT_MAX_CHARS (150000 is the production default)." >&2
      exit 2
    fi
  done
fi

for ARM in ${ARMS}; do
  echo "── arm: ${ARM_VAR}=${ARM} ─────────────────────────────────────────"
  env "${ARM_VAR}=${ARM}" uv run python -m benchmarks.runner.run_subprocess \
    --benchmark "${BENCH}" \
    --pipeline "${PIPELINE}" \
    --profile "${PROFILE}" \
    --runs 1 \
    --limit "${LIMIT}" \
    --seed "${SEED}" \
    --concurrency "${CONCURRENCY}" \
    --out "${OUT_ROOT}/${ARM}"
done

LABELS="$(echo "${ARMS}" | tr -s ' ' ',')"
DIRS=""
for ARM in ${ARMS}; do DIRS="${DIRS} ${OUT_ROOT}/${ARM}"; done

# shellcheck disable=SC2086  # DIRS/ARMS are deliberately word-split lists
uv run python scripts/truncation_metrics.py ${DIRS} \
  --labels "${LABELS}" \
  --json "${OUT_ROOT}/metrics.json"
