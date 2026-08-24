#!/usr/bin/env bash
set -euo pipefail

sglang_python="${SGLANG_PYTHON:-/usr/bin/python3}"
server_port="${SGLANG_SERVER_PORT:-${SGLANG_PORT:-30000}}"

#: Variables a runtime profile may set, in the order doctor reports them.
profile_variables="SGLANG_SERVED_MODEL_NAME SGLANG_TOOL_CALL_PARSER \
SGLANG_REASONING_PARSER SGLANG_TP_SIZE SGLANG_CONTEXT_LENGTH \
SGLANG_MAX_INPUT_TOKENS SGLANG_MAX_OUTPUT_TOKENS SGLANG_MEM_FRACTION_STATIC \
SGLANG_DTYPE SGLANG_QUANTIZATION SGLANG_TRUST_REMOTE_CODE SGLANG_EXTRA_ARGS"

installed_sglang_version() {
  "$sglang_python" -c 'import sglang; print(sglang.__version__)' 2>/dev/null || true
}

# "0.5.10.post1" -> "0 5 10". Empty when the version cannot be parsed, which
# every caller must read as "unknown", never as "old".
sglang_release_line() {
  printf '%s\n' "${1:-}" |
    sed -n 's/^\([0-9][0-9]*\)\.\([0-9][0-9]*\)\.\([0-9][0-9]*\).*/\1 \2 \3/p'
}

# True for the 0.5.10 release line, which reads --language-only as a request
# for encoder disaggregation. Mirrors rejects_language_only() in
# scripts/run-sglang-native.py; the half-open range covers every 0.5.10 patch.
sglang_line_rejects_language_only() {
  set -- $1
  [ "$1" -eq 0 ] && [ "$2" -eq 5 ] && [ "$3" -eq 10 ]
}

# Echoes an explanation when the selected profile cannot run against the
# SGLang build actually present in this image, and stays silent otherwise.
# An unreadable version is silent too: refusing to start on a version we could
# not parse would be worse than letting SGLang report the real error.
profile_track_conflict() {
  profile="$1"
  version="$(installed_sglang_version)"
  line="$(sglang_release_line "$version")"
  [ -n "$line" ] || return 0

  case "$profile" in
    qwen35-gptq-cu12)
      if ! sglang_line_rejects_language_only "$line"; then
        echo "profile ${profile} targets the SGLang 0.5.10 line, but this image ships ${version}."
        echo "Use FRONTIER_AGENT_GPU_PROFILE=qwen35-gptq-cu13 with a cu13 image."
      fi
      ;;
    qwen35-gptq-cu13)
      if sglang_line_rejects_language_only "$line"; then
        echo "profile ${profile} passes --language-only, which SGLang ${version} reads as"
        echo "encoder disaggregation and which then demands separate --encoder-urls."
        echo "Use FRONTIER_AGENT_GPU_PROFILE=qwen35-gptq-cu12 with this cu12 image,"
        echo "or run the cu13 image (SGLang 0.5.17) instead."
      fi
      ;;
  esac
}

require_matching_track() {
  [ -n "${FRONTIER_AGENT_GPU_PROFILE:-}" ] || return 0
  conflict="$(profile_track_conflict "${FRONTIER_AGENT_GPU_PROFILE}")"
  [ -n "$conflict" ] || return 0
  printf '%s\n' "$conflict" >&2
  exit 2
}

report_profile_settings() {
  [ -n "${FRONTIER_AGENT_GPU_PROFILE:-}" ] || return 0
  echo "effective GPU profile: ${FRONTIER_AGENT_GPU_PROFILE}"
  for name in $profile_variables; do
    eval "value=\${${name}:-}"
    echo "  ${name}=${value}"
  done
}

apply_gpu_profile() {
  case "${FRONTIER_AGENT_GPU_PROFILE:-}" in
    "")
      ;;
    qwen35-gptq-cu12)
      export SGLANG_SERVED_MODEL_NAME="${SGLANG_SERVED_MODEL_NAME:-local-model}"
      export SGLANG_TOOL_CALL_PARSER="${SGLANG_TOOL_CALL_PARSER:-qwen3_coder}"
      export SGLANG_REASONING_PARSER="${SGLANG_REASONING_PARSER:-qwen3}"
      export SGLANG_TP_SIZE="${SGLANG_TP_SIZE:-1}"
      export SGLANG_CONTEXT_LENGTH="${SGLANG_CONTEXT_LENGTH:-32768}"
      export SGLANG_MAX_INPUT_TOKENS="${SGLANG_MAX_INPUT_TOKENS:-27000}"
      export SGLANG_MAX_OUTPUT_TOKENS="${SGLANG_MAX_OUTPUT_TOKENS:-4096}"
      export SGLANG_MEM_FRACTION_STATIC="${SGLANG_MEM_FRACTION_STATIC:-0.85}"
      export SGLANG_DTYPE="${SGLANG_DTYPE:-auto}"
      export SGLANG_QUANTIZATION="${SGLANG_QUANTIZATION:-moe_wna16}"
      export SGLANG_TRUST_REMOTE_CODE="${SGLANG_TRUST_REMOTE_CODE:-0}"
      if [ -z "${SGLANG_EXTRA_ARGS:-}" ]; then
        # SGLang 0.5.10 must not use --language-only: in that release the flag
        # selects encoder disaggregation and requires separate encoder URLs.
        export SGLANG_EXTRA_ARGS="--max-running-requests 1"
      fi
      ;;
    qwen35-gptq-cu13)
      export SGLANG_SERVED_MODEL_NAME="${SGLANG_SERVED_MODEL_NAME:-local-model}"
      export SGLANG_TOOL_CALL_PARSER="${SGLANG_TOOL_CALL_PARSER:-qwen3_coder}"
      export SGLANG_REASONING_PARSER="${SGLANG_REASONING_PARSER:-qwen3}"
      export SGLANG_TP_SIZE="${SGLANG_TP_SIZE:-1}"
      export SGLANG_CONTEXT_LENGTH="${SGLANG_CONTEXT_LENGTH:-32768}"
      export SGLANG_MAX_INPUT_TOKENS="${SGLANG_MAX_INPUT_TOKENS:-27000}"
      export SGLANG_MAX_OUTPUT_TOKENS="${SGLANG_MAX_OUTPUT_TOKENS:-4096}"
      export SGLANG_MEM_FRACTION_STATIC="${SGLANG_MEM_FRACTION_STATIC:-0.85}"
      # Verified on RTX 5090 with SGLang 0.5.17: the checkpoint's config.json
      # declares bfloat16, and the MoE GPTQ weights only load through the
      # moe_wna16 path. Plain gptq rejects MoE and gptq_marlin hits dtype
      # mismatches in the fused Marlin MoE kernel.
      export SGLANG_DTYPE="${SGLANG_DTYPE:-bfloat16}"
      export SGLANG_QUANTIZATION="${SGLANG_QUANTIZATION:-moe_wna16}"
      export SGLANG_TRUST_REMOTE_CODE="${SGLANG_TRUST_REMOTE_CODE:-0}"
      if [ -z "${SGLANG_EXTRA_ARGS:-}" ]; then
        # --language-only skips the vision encoder, freeing VRAM on single-card
        # 32 GB hosts. Verified on 0.5.17; the flag stopped meaning encoder
        # disaggregation after the 0.5.10 line, but 0.5.11-0.5.16 are untested
        # here, so require_matching_track checks the running build rather than
        # trusting the profile name.
        export SGLANG_EXTRA_ARGS="--max-running-requests 1 --language-only"
      fi
      ;;
    *)
      echo "unknown FRONTIER_AGENT_GPU_PROFILE=${FRONTIER_AGENT_GPU_PROFILE}" >&2
      echo "available profiles: qwen35-gptq-cu12 qwen35-gptq-cu13" >&2
      exit 2
      ;;
  esac
}

require_model_source() {
  if [ -n "${SGLANG_MODEL_ID:-}" ] || [ -n "${SGLANG_LOCAL_MODEL_PATH:-}" ]; then
    return
  fi
  cat >&2 <<'EOF'
No SGLang model source is configured.

Choose one of these runtime options:
  --env SGLANG_MODEL_ID=<hugging-face-model-id>
  --volume /host/checkpoint:/models/checkpoint:ro \
    --env SGLANG_LOCAL_MODEL_PATH=/models/checkpoint

Each optional runtime profile configures 32K context, moe_wna16, qwen3_coder,
and one running request, and does not select a model repository. Model
weights are not included here:
  FRONTIER_AGENT_GPU_PROFILE=qwen35-gptq-cu12  (SGLang 0.5.10, CUDA 12)
  FRONTIER_AGENT_GPU_PROFILE=qwen35-gptq-cu13  (SGLang 0.5.17, CUDA 13)
EOF
  exit 2
}

run_server() {
  export SGLANG_SERVER_PORT="$server_port"
  exec "$sglang_python" /app/docker/sglang_entrypoint.py
}

run_tui() {
  export SGLANG_SERVER_HOST="${SGLANG_SERVER_HOST:-0.0.0.0}"
  export SGLANG_SERVER_PORT="$server_port"
  "$sglang_python" /app/docker/sglang_entrypoint.py &
  server_pid=$!

  cleanup() {
    if kill -0 "$server_pid" 2>/dev/null; then
      kill -TERM "$server_pid" 2>/dev/null || true
      wait "$server_pid" 2>/dev/null || true
    fi
  }
  trap cleanup EXIT INT TERM

  "$sglang_python" - "$server_port" <<'PY'
import sys
import time
import urllib.request

port = sys.argv[1]
url = f"http://127.0.0.1:{port}/health"
for _ in range(360):
    try:
        with urllib.request.urlopen(url, timeout=5) as response:
            if response.status == 200:
                raise SystemExit(0)
    except Exception:
        time.sleep(5)
raise SystemExit(f"SGLang did not become healthy at {url}")
PY

  export OPENAI_PROVIDER=local
  export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"
  export OPENAI_BASE_URL="http://127.0.0.1:${server_port}/v1"
  export OPENAI_MODEL="${OPENAI_MODEL:-${SGLANG_SERVED_MODEL_NAME:-local-model}}"
  export OPENAI_CONTEXT_WINDOW="${OPENAI_CONTEXT_WINDOW:-${SGLANG_CONTEXT_LENGTH:-32768}}"
  export OPENAI_MAX_INPUT_TOKENS="${OPENAI_MAX_INPUT_TOKENS:-${SGLANG_MAX_INPUT_TOKENS:-27000}}"
  export OPENAI_MAX_TOKENS="${OPENAI_MAX_TOKENS:-${SGLANG_MAX_OUTPUT_TOKENS:-4096}}"
  /opt/venv/bin/frontier-agent --mode "${FRONTIER_AGENT_MODE:-react}" "$@"
}

case "${1:-server}" in
  server)
    shift || true
    apply_gpu_profile
    require_matching_track
    require_model_source
    run_server "$@"
    ;;
  tui)
    shift
    apply_gpu_profile
    require_matching_track
    require_model_source
    run_tui "$@"
    ;;
  agent|frontier-agent)
    shift
    exec /opt/venv/bin/frontier-agent "$@"
    ;;
  doctor)
    # A diagnostic reports every check, so no single failure may abort the run.
    # The exit status still summarises them for scripted callers.
    doctor_failed=0
    apply_gpu_profile
    report_profile_settings
    if profile_conflict="$(profile_track_conflict "${FRONTIER_AGENT_GPU_PROFILE:-}")" &&
      [ -n "$profile_conflict" ]; then
      printf '%s\n' "$profile_conflict" >&2
      doctor_failed=1
    fi
    if sglang_version="$(installed_sglang_version)" && [ -n "$sglang_version" ]; then
      echo "sglang ${sglang_version}"
    else
      echo "sglang is not importable from ${sglang_python}" >&2
      doctor_failed=1
    fi
    /opt/venv/bin/frontier-agent --version || doctor_failed=1
    if command -v nvidia-smi >/dev/null 2>&1; then
      nvidia-smi -L || doctor_failed=1
    else
      echo "nvidia-smi is not available; run the image with an NVIDIA GPU runtime"
    fi
    echo "bundled GPU profiles: qwen35-gptq-cu12 qwen35-gptq-cu13"
    exit "$doctor_failed"
    ;;
  help|--help|-h)
    cat <<'EOF'
Usage: IMAGE [server|tui|agent|doctor|shell]

Set FRONTIER_AGENT_GPU_PROFILE=qwen35-gptq-cu12 (CUDA 12 / SGLang 0.5.10) or
qwen35-gptq-cu13 (CUDA 13 / SGLang 0.5.17) for a bundled Qwen runtime
profile. Always configure SGLANG_MODEL_ID/SGLANG_LOCAL_MODEL_PATH explicitly;
the image does not embed a model repository or weights.
EOF
    ;;
  shell)
    shift
    exec /bin/bash "$@"
    ;;
  *)
    exec "$@"
    ;;
esac
