#!/usr/bin/env bash
set -u

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
env_file="${SGLANG_ENV_FILE:-$repo_root/.env.sglang}"
mode="${1:-full}"
failures=0
warnings=0

pass() { printf 'PASS  %s\n' "$*"; }
warn() { printf 'WARN  %s\n' "$*"; warnings=$((warnings + 1)); }
fail() { printf 'FAIL  %s\n' "$*"; failures=$((failures + 1)); }

env_value() {
  local name="$1"
  [ -f "$env_file" ] || return 0
  # Unquote like `docker compose --env-file` does, so a path written as
  # KEY="/data/model" is checked as the directory the operator meant.
  awk -v key="$name" '
    index($0, key "=") == 1 {
      value = substr($0, length(key) + 2)
      sub(/\r$/, "", value)
      quote = substr(value, 1, 1)
      if (length(value) >= 2 && (quote == "\"" || quote == "\047") \
          && substr(value, length(value), 1) == quote) {
        value = substr(value, 2, length(value) - 2)
      }
      print value
    }
  ' "$env_file" | tail -n 1
}

positive_integer() {
  [[ "$1" =~ ^[1-9][0-9]*$ ]]
}

printf 'FrontierAgent local NVIDIA doctor\n'
printf 'Config: %s\n\n' "$env_file"

if [ "$(uname -s)" = "Linux" ]; then
  pass "Linux host ($(uname -m))"
else
  fail "the local NVIDIA Compose path currently requires Linux"
fi

for command in docker nvidia-smi awk sed stat df; do
  if command -v "$command" >/dev/null 2>&1; then
    pass "$command is installed"
  else
    fail "$command is not installed"
  fi
done

if [ ! -f "$env_file" ]; then
  fail "configuration is missing; copy .env.sglang.example to .env.sglang"
else
  permissions="$(stat -c '%a' "$env_file" 2>/dev/null || true)"
  case "$permissions" in
    600|400) pass "configuration permissions are $permissions" ;;
    *) warn "configuration permissions are ${permissions:-unknown}; use chmod 600 $env_file" ;;
  esac
fi

if docker info >/dev/null 2>&1; then
  pass "Docker daemon is reachable"
else
  fail "Docker daemon is not reachable (check the service and socket permissions)"
fi

if docker compose version >/dev/null 2>&1; then
  pass "Docker Compose is available: $(docker compose version --short 2>/dev/null || true)"
else
  fail "Docker Compose plugin is unavailable"
fi

gpu_count="$(
  nvidia-smi --query-gpu=index --format=csv,noheader,nounits 2>/dev/null \
    | awk 'NF {count++} END {print count + 0}' \
    || true
)"
if positive_integer "$gpu_count"; then
  pass "NVIDIA driver reports $gpu_count GPU(s)"
  nvidia-smi --query-gpu=index,name,memory.total,driver_version \
    --format=csv,noheader 2>/dev/null | sed 's/^/      /'
else
  fail "nvidia-smi could not enumerate a GPU"
  gpu_count=0
fi

tp_size="$(env_value SGLANG_TP_SIZE)"
tp_size="${tp_size:-1}"
gpu_reservation="$(env_value SGLANG_GPU_COUNT)"
gpu_reservation="${gpu_reservation:-1}"
if ! positive_integer "$tp_size"; then
  fail "SGLANG_TP_SIZE must be a positive integer"
elif [ "$gpu_count" -gt 0 ] && [ "$tp_size" -gt "$gpu_count" ]; then
  fail "SGLANG_TP_SIZE=$tp_size exceeds the $gpu_count detected GPU(s)"
else
  pass "tensor parallel size is $tp_size"
fi
if ! positive_integer "$gpu_reservation"; then
  fail "SGLANG_GPU_COUNT must be a positive integer"
elif [ "$gpu_count" -gt 0 ] && [ "$gpu_reservation" -gt "$gpu_count" ]; then
  fail "SGLANG_GPU_COUNT=$gpu_reservation exceeds detected GPU count $gpu_count"
else
  pass "Compose will reserve $gpu_reservation GPU(s)"
fi

context="$(env_value SGLANG_CONTEXT_LENGTH)"
input_tokens="$(env_value SGLANG_MAX_INPUT_TOKENS)"
output_tokens="$(env_value SGLANG_MAX_OUTPUT_TOKENS)"
context="${context:-32768}"
input_tokens="${input_tokens:-27000}"
output_tokens="${output_tokens:-4096}"
if positive_integer "$context" && positive_integer "$input_tokens" && positive_integer "$output_tokens"; then
  if [ $((input_tokens + output_tokens)) -le "$context" ]; then
    pass "token budgets fit the context ($input_tokens + $output_tokens <= $context)"
  else
    fail "token budgets exceed context ($input_tokens + $output_tokens > $context)"
  fi
  if [ $((input_tokens * 10)) -gt $((context * 8)) ]; then
    pass "input budget remains above the 80% compaction threshold"
  else
    fail "input budget must remain above 80% of the context for compaction"
  fi
else
  fail "context and token budgets must be positive integers"
fi

model_id="$(env_value SGLANG_MODEL_ID)"
local_model="$(env_value SGLANG_LOCAL_MODEL_PATH)"
if [ -n "$local_model" ]; then
  if [ -d "$local_model" ] || [ -d "$repo_root/$local_model" ]; then
    pass "local model checkpoint exists"
  else
    fail "SGLANG_LOCAL_MODEL_PATH does not point to a directory"
  fi
elif [ -n "$model_id" ]; then
  pass "Hugging Face model source is configured"
else
  fail "set SGLANG_MODEL_ID or SGLANG_LOCAL_MODEL_PATH"
fi

free_kb="$(df -Pk "$repo_root" 2>/dev/null | awk 'NR == 2 {print $4}')"
min_free_gb="${SGLANG_MIN_FREE_GB:-60}"
if [[ "$free_kb" =~ ^[0-9]+$ ]]; then
  free_gb=$((free_kb / 1024 / 1024))
  if [ "$free_gb" -ge "$min_free_gb" ]; then
    pass "$free_gb GiB free disk space (minimum $min_free_gb GiB)"
  else
    warn "only $free_gb GiB free; first-time SGLang setup should have at least $min_free_gb GiB"
  fi
fi

subnet="$(env_value APODEX_DOCKER_SUBNET)"
if [ -n "$subnet" ]; then
  warn "custom Docker subnet requested: $subnet; verify it does not overlap host/VPN routes"
elif command -v ip >/dev/null 2>&1 && ip route 2>/dev/null | awk '$1 == "0.0.0.0/1" || $1 == "128.0.0.0/1" {found=1} END {exit !found}'; then
  warn "a VPN owns both half-default routes; set APODEX_DOCKER_SUBNET if Compose cannot allocate a network"
else
  pass "no custom Docker subnet requested"
fi

port="$(env_value SGLANG_PORT)"
port="${port:-30000}"
if ! command -v ss >/dev/null 2>&1; then
  # Without iproute2 there is nothing to conclude; claiming the port is free
  # would send the operator into a bind failure during `up`.
  warn "ss is not installed (iproute2); cannot check whether host port $port is free"
elif ss -ltn 2>/dev/null | awk -v port=":$port" '$4 ~ port "$" {found=1} END {exit !found}'; then
  warn "host port $port is already listening (this is expected if the model service is running)"
else
  pass "host port $port is available"
fi

if [ -d "$repo_root/.apodex/runs" ]; then
  owner="$(stat -c '%U:%G' "$repo_root/.apodex/runs" 2>/dev/null || true)"
  mode_bits="$(stat -c '%a' "$repo_root/.apodex/runs" 2>/dev/null || true)"
  if [ "$owner" = "$(id -un):$(id -gn)" ] && [[ "$mode_bits" != *7 ]]; then
    pass "host output directory is owned by $owner (mode $mode_bits)"
  else
    warn "host output directory is owned by $owner (mode $mode_bits); new runs align tool UID/GID, but existing ownership may need repair"
  fi
fi

if [ "$mode" != "quick" ] && docker info >/dev/null 2>&1; then
  smoke_image="${NVIDIA_SMOKE_IMAGE:-nvidia/cuda:12.9.0-base-ubuntu22.04}"
  printf '\nRunning container GPU passthrough test (may pull %s)...\n' "$smoke_image"
  if docker run --rm --gpus all "$smoke_image" nvidia-smi -L >/dev/null 2>&1; then
    pass "NVIDIA GPU is visible inside a Docker container"
  else
    fail "Docker GPU passthrough failed; configure NVIDIA Container Toolkit and restart Docker"
  fi
else
  warn "container GPU passthrough test skipped in quick mode"
fi

printf '\nSummary: %d failure(s), %d warning(s)\n' "$failures" "$warnings"
if [ "$failures" -ne 0 ]; then
  exit 1
fi
