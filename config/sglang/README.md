# Configure SGLang for FrontierAgent

The files in this directory are starting points for running an
OpenAI-compatible SGLang server. They are not universal hardware presets: model
architecture, checkpoint format, SGLang version, driver, available VRAM, context
length, and concurrency all affect whether a configuration fits.

Use `.env.sglang.example` at the repository root for the small infrastructure
smoke test. Use a file in this directory only after the smoke test passes:

| Template | Intended starting point | Important caveat |
|---|---|---|
| `35b-4090.env.example` | One 24 GB RTX 4090 | Needs a 4-bit export; measure its limits |
| `35b-5090.env.example` | One 32 GB RTX 5090 | Values certified against a GPTQ-Int4 MoE export, which is also the only format that fits this card |
| `35b-multigpu.env.example` | Two GPUs with tensor parallelism | Needs a 4-bit export; validate peer access and per-card VRAM |

All three name `apodex/Apodex-1.1-mini` in `SGLANG_MODEL_ID`, and that repository
publishes **FP16** weights — roughly 70 GB for a 35B checkpoint, before any KV
cache. None of these consumer-GPU templates can load that as-is: quantize first
and point `SGLANG_LOCAL_MODEL_PATH` at the export (it takes precedence over the
repository id), or substitute a 4-bit repository. The model id is there to say
*which* model to serve, not to promise it fits.

## Quick start

For Docker on a Linux NVIDIA host:

```bash
cp config/sglang/35b-5090.env.example .env.sglang
chmod 600 .env.sglang
$EDITOR .env.sglang

./docker/run-sglang.sh doctor
./docker/run-sglang.sh up
./docker/run-sglang.sh smoke
./docker/run-sglang.sh tui
```

For a provider-owned GPU container without nested Docker, use the same env file
with `scripts/run-sglang-native.py`; follow the
[native SGLang guide](../../docs/install/linux-nvidia-native.md).

Do not commit `.env.sglang`. It may contain `HF_TOKEN` and web-tool credentials.
The example files intentionally contain no secrets.

## How the env file is used

The env file configures two different layers:

1. `docker/sglang_entrypoint.py` translates the model settings into
   `python -m sglang.launch_server` arguments.
2. Compose or the native launcher translates the client token limits into
   `OPENAI_*` variables consumed by the FrontierAgent workflow profile.

That distinction matters: `SGLANG_MAX_INPUT_TOKENS` and
`SGLANG_MAX_OUTPUT_TOKENS` limit FrontierAgent requests; they are not SGLang
server flags. `SGLANG_CONTEXT_LENGTH`, by contrast, becomes SGLang's
`--context-length` and is also advertised to FrontierAgent.

## Variable reference

### Model and API identity

| Variable | Meaning |
|---|---|
| `SGLANG_PROFILE` | Human-readable label for recording the chosen configuration; the launcher does not currently consume it or select settings from it |
| `SGLANG_IMAGE` | Docker image containing the pinned SGLang runtime; Docker path only |
| `SGLANG_MODEL_ID` | Hugging Face repository ID passed as `--model-path` |
| `SGLANG_LOCAL_MODEL_PATH` | Existing checkpoint directory; when set, it takes precedence over `SGLANG_MODEL_ID` and is mounted read-only in Docker |
| `SGLANG_SERVED_MODEL_NAME` | Stable name returned by `/v1/models` and sent by FrontierAgent, independent of the checkpoint path |
| `HF_TOKEN` | Token for gated/private Hugging Face repositories; leave empty for public models |

A local checkpoint must contain the model config, tokenizer, index, and every
referenced weight shard. Review a model repository before setting
`SGLANG_TRUST_REMOTE_CODE=1`.

### GPU, context, and request budgets

| Variable | Meaning |
|---|---|
| `SGLANG_TP_SIZE` | SGLang tensor-parallel size (`--tp-size`); normally the number of GPUs sharding one model |
| `SGLANG_GPU_COUNT` | Number of GPUs reserved by Docker Compose; it is not passed to SGLang |
| `SGLANG_CONTEXT_LENGTH` | Maximum server context (`--context-length`), including input and generated tokens |
| `SGLANG_MAX_INPUT_TOKENS` | FrontierAgent input guard, exported as `OPENAI_MAX_INPUT_TOKENS` |
| `SGLANG_MAX_OUTPUT_TOKENS` | Maximum generation per agent call, exported as `OPENAI_MAX_TOKENS` |
| `SGLANG_MEM_FRACTION_STATIC` | Fraction of GPU memory reserved by SGLang for weights and the KV-cache pool |
| `SGLANG_EXTRA_ARGS` | Additional `sglang.launch_server` arguments, parsed as a shell-style argument string |

Keep these invariants true:

```text
SGLANG_MAX_INPUT_TOKENS + SGLANG_MAX_OUTPUT_TOKENS
    <= SGLANG_CONTEXT_LENGTH

SGLANG_MAX_INPUT_TOKENS
    > 0.8 * SGLANG_CONTEXT_LENGTH
```

The second invariant lets FrontierAgent's tiered compaction trigger at 80% of
the workflow context before its hard input guard aborts the request. The doctor
checks both invariants.

Longer context and more simultaneous requests require more KV-cache memory. On
the first successful boot, keep `--max-running-requests 1`; raise context or
concurrency one at a time and record peak VRAM and latency. If startup or long
prefill runs out of memory, first lower `SGLANG_MEM_FRACTION_STATIC` slightly,
reduce `SGLANG_CONTEXT_LENGTH`, or add a smaller `--chunked-prefill-size` through
`SGLANG_EXTRA_ARGS`. A lower static fraction leaves more runtime headroom but
also shrinks the KV-cache pool.

For multi-GPU runs, `SGLANG_TP_SIZE` must not exceed `SGLANG_GPU_COUNT`. Tensor
parallelism also depends on GPU peer access and interconnect performance; two
cards do not automatically provide twice the usable context or throughput.

### Model-specific parsing and precision

| Variable | Meaning |
|---|---|
| `SGLANG_TOOL_CALL_PARSER` | Parser matching the model's emitted tool-call format (`--tool-call-parser`) |
| `SGLANG_REASONING_PARSER` | Parser matching the model's reasoning format (`--reasoning-parser`) |
| `SGLANG_CHAT_TEMPLATE` | Override the checkpoint's packaged chat template (`--chat-template`). Leave empty to use the packaged one. In Docker the path must resolve **inside** the container, so point it at a file under the mounted model directory — a host path such as `/mnt/…/templates/x.jinja` is not visible there |
| `SGLANG_DTYPE` | Weight/activation dtype (`--dtype`); `auto` follows checkpoint metadata |
| `SGLANG_QUANTIZATION` | Explicit SGLang quantization backend (`--quantization`) when the checkpoint requires one |
| `SGLANG_TRUST_REMOTE_CODE` | Adds `--trust-remote-code` when set to `1`, `true`, `yes`, or `on` |

Parser names are model-family specific. A server can be healthy and return text
while structured tool calls still fail. Use `./docker/run-sglang.sh smoke` after
changing the model, chat template, tool parser, or reasoning parser.

The native doctor also enforces a per-model minimum SGLang version from
[`compatibility.json`](compatibility.json), matched by **prefix** against
`SGLANG_MODEL_ID`. Apodex checkpoints are Qwen3.5-architecture and inherit the
same floor, but `apodex/…` does not match `Qwen/Qwen3.5`, so the vendor path
carries its own entry. Serving these weights from a differently-named repository
means adding a prefix there too, or the version floor silently does not apply.

Prefer checkpoints that are already quantized. Do not infer the backend only
from a filename: read the checkpoint's model card and compare it with the SGLang
version's supported quantization methods. Leaving `SGLANG_QUANTIZATION` empty
lets SGLang inspect checkpoint metadata; set it only when the format or model
guidance requires an explicit backend.

### Docker and native launcher controls

| Variable | Meaning |
|---|---|
| `SGLANG_PORT` | Loopback host port in Docker and listening port in native mode; default `30000` |
| `SGLANG_RESTART_POLICY` | Docker Compose restart policy; `no` avoids unexpected GPU use after reboot |
| `APODEX_DOCKER_SUBNET` | Optional explicit Compose subnet for VPN/corporate-network conflicts |
| `SGLANG_BUILD_AGENT` | Build FrontierAgent from this checkout instead of pulling its release image; does not rebuild SGLang |
| `SGLANG_PYTHON` | Native-mode Python executable in the environment where `import sglang` succeeds |
| `SGLANG_EXPECTED_VERSION` | Optional native doctor assertion; it does not install or change SGLang |
| `SGLANG_DOWNLOAD_DIR` | Native SGLang/Hugging Face download directory, ideally on persistent storage |
| `SGLANG_NATIVE_STATE_DIR` | Native launcher's PID/log state directory |
| `SGLANG_NATIVE_HOST` | Native bind address; keep `127.0.0.1` unless an unauthenticated endpoint must be exposed deliberately |
| `HF_HOME`, `HF_HUB_CACHE`, `HF_HUB_OFFLINE` | Hugging Face cache placement and offline behavior for native mode |

`SERPER_*` and `JINA_*` configure FrontierAgent web tools, not SGLang. Local
inference and the TUI do not require those keys, but research tools may.

## A safe tuning sequence

1. Run the root smoke profile to verify the driver, Docker GPU passthrough,
   SGLang API, tool parsing, and FrontierAgent connection.
2. Select the exact checkpoint and confirm that its architecture, quantization,
   and parser names are supported by the pinned `SGLANG_IMAGE` version.
3. Start at one running request and a conservative context window.
4. Run `doctor`, then `up`, inspect `logs`, and run `smoke`.
5. Exercise representative long prompts while watching `nvidia-smi`.
6. Increase context, then concurrency, one variable at a time. Record the image
   digest, checkpoint revision, GPU model/count, peak VRAM, and verified limits.

Do not describe a candidate template as supported hardware until that exact
combination has passed representative workloads on the physical GPU.

## Upstream SGLang references

- [Server arguments](https://docs.sglang.io/docs/advanced_features/server_arguments)
- [Quantization](https://docs.sglang.io/docs/advanced_features/quantization)
- [Tool parser](https://docs.sglang.io/docs/advanced_features/tool_parser)
- [Reasoning parser](https://docs.sglang.io/docs/advanced_features/separate_reasoning)
- [Supported models](https://docs.sglang.io/docs/supported-models)

The installed version is authoritative. Run the following with the same Python
environment or container image used to serve the model:

```bash
python -m sglang.launch_server --help
```

For end-to-end setup and troubleshooting, see
[Linux NVIDIA + Docker](../../docs/install/linux-nvidia.md),
[Linux NVIDIA native](../../docs/install/linux-nvidia-native.md), and the
[GPU compatibility matrix](../../docs/install/gpu-compatibility.md).
