# GPU and SGLang compatibility

Treat runtime compatibility and model capacity as separate gates:

- the host driver must support the selected CUDA userspace track;
- the SGLang release must support the model and quantization;
- the GPU architecture must have compatible kernels;
- total VRAM must hold weights, runtime state, KV cache, and CUDA graphs.

## Runtime tracks

The machine-readable source of truth is
[`config/sglang/compatibility.json`](../../config/sglang/compatibility.json).
The native launcher reads that file during `doctor`.

| Track | Conservative native driver rule | SGLang pin | Intended use |
|---|---|---|---|
| `cu13` | NVIDIA driver 580+ | `0.5.17` | **official recommended default**; verified on RTX 5090 32 GB |
| `cu12` | NVIDIA driver 525+ | `0.5.10.post1` | compatibility hint only; best-effort native fallback observed on RTX 5090 32 GB with driver 570 |

These are tested release tracks, not a claim that every later SGLang release
uses only one CUDA family. SGLang `0.5.11` moved its default build to CUDA 13,
while newer releases may also publish explicitly named CUDA 12.9 images. Advance
a track only after checking the upstream image metadata and running the model
smoke suite.

The support policy is intentionally asymmetric: new and supported deployments
select `cu13`. The `cu12` track is retained only as a compatibility hint for
hosts whose driver is below 580; it is not a co-equal supported default. Its
SGLang `0.5.10` build cannot use
`--language-only`, so the vision encoder stays resident. The native cu12 path
has nevertheless passed the 32 GB single-card chain on an RTX 5090 with driver
570, a CUDA 12.9 JIT toolkit, and the stock one-request profile. A driver of 580
or newer can still run `cu12` — newer drivers keep supporting older CUDA
runtimes — but there is normally no reason to choose it there.

The container entrypoint enforces this pairing rather than trusting the profile
name: `FRONTIER_AGENT_GPU_PROFILE` is checked against the SGLang version
actually installed in the image, and a mismatch fails at startup with the
reason instead of surfacing later as a request for `--encoder-urls`.

For containers, the image's `NVIDIA_REQUIRE_CUDA` metadata and NVIDIA runtime
perform the final driver check. For native Python installation, the doctor uses
the conservative mapping above because the selected wheels share the provider's
host driver directly.

## Model and GPU status

| Model/profile | GPU and VRAM | Runtime | Status |
|---|---|---|---|
| 0.8B infrastructure smoke | RTX 5060, about 8 GB | Docker SGLang | verified plumbing |
| Qwen3.5-35B-A3B GPTQ Int4, 32K, one request | one NVIDIA L20X, about 140 GB, driver 550.127.08 | native SGLang `0.5.10.post1` | health, model listing, and structured tool call passed on 2026-08-11 |
| Qwen3.5-35B-A3B GPTQ Int4, 32K, one request | RTX 5090 32 GB, driver 570.195.03 | native SGLang `0.5.10.post1` (cu12), CUDA 12.9 JIT toolkit | health, model listing, structured tool call, non-greedy sampling, and an end-to-end agent file-tool task passed on 2026-08-15; Docker image not certified |
| Qwen3.5-35B-A3B GPTQ Int4, 32K, one request | RTX 5090 32 GB, driver 595.71.05 | native SGLang `0.5.17` (cu13) | health, model listing, structured tool call, and TUI startup passed on 2026-08-16; Docker image not yet certified |
| 35B 4-bit candidate | RTX 4090 24 GB | Docker/native | not yet certified |
| 35B multi-GPU candidate | two matched NVIDIA GPUs | Docker/native with matching TP | not yet certified |

MoE active parameters reduce compute per generated token, but all quantized
weight shards still need storage and normally need to be resident. A model that
loads at 8K context may still run out of memory at 32K or under concurrent
requests.

## What to record when certifying a machine

- exact GPU model/count and compute capability;
- driver version and container image digest or native package versions;
- exact checkpoint revision and quantization;
- startup time, disk usage, idle/load VRAM, and context length;
- tensor parallel size, concurrency, CUDA graph, and KV-cache settings;
- `/health`, `/v1/models`, structured tool-call smoke, non-greedy sampling,
  and at least one end-to-end agent tool execution.

Do not infer 4090 results from 5090 or L20X results merely because the nominal
VRAM is sufficient; architectures and available quantization kernels differ.

Return to the [installation chooser](README.md).
