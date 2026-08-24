# Native SGLang on Linux GPU environments

Use this path when Linux and the NVIDIA GPU are already available inside the
current environment, but there is no Docker daemon. The main example is an
ordinary AutoDL container instance. AutoDL documents that these instances are
themselves Docker containers and do not support running Docker inside; its
bare-metal offering is a different product:
<https://www.autodl.com/docs/env/>.

If the provider lets you select a custom OCI image while creating the instance,
use the [GPU cloud image guide](gpu-platforms.md) instead. If you control a
Docker daemon with NVIDIA GPU passthrough, use the
[Docker GPU guide](linux-nvidia.md). The [installation chooser](README.md)
explains the distinction.

FrontierAgent and SGLang are separate native processes connected through an
OpenAI-compatible loopback endpoint:

```text
provider container or Linux environment
├── SGLang                 http://127.0.0.1:30000/v1
└── FrontierAgent native   calls the local endpoint
```

This does not create another OS sandbox. On a managed service, the provider's
outer container remains the host boundary; approved FrontierAgent commands can
access files available to the current user inside that instance.

## Prerequisites

- Linux with `nvidia-smi` able to enumerate the assigned GPU;
- a Python environment with a Qwen3.5-compatible SGLang installation;
- the system `libnuma` runtime (`libnuma1` on Debian/Ubuntu), required by
  prebuilt SGLang GPU kernels;
- Python 3.12 and `uv` for FrontierAgent;
- enough persistent disk for the SGLang environment, image-independent runtime
  dependencies, and model weights.

Prefer a provider image that already includes a compatible SGLang/CUDA/PyTorch
stack. If SGLang must be installed, follow its current native installation guide
for the CUDA version exposed by the provider instead of replacing the provider's
driver: <https://docs.sglang.io/docs/get-started/install>. Keep SGLang in a
separate environment from FrontierAgent and point `SGLANG_PYTHON` at that
environment's Python executable.

## Automated quick start

For a Linux x86_64 host or provider container where `nvidia-smi` already works,
the repository helper automates the conservative path:

```bash
# Add --install-system-deps if the image does not already provide libnuma.
./scripts/run-linux-gpu.sh --install-system-deps --setup-only
./scripts/run-linux-gpu.sh smoke
./scripts/run-linux-gpu.sh tui -- --cwd /path/to/project
```

On first use it:

1. inventories the GPU and reads the host driver from `nvidia-smi`;
2. selects the newest compatible reviewed track from
   `config/sglang/compatibility.json`;
3. installs the exact SGLang pin into `.venv-sglang`, separate from
   FrontierAgent's `.venv`;
4. creates `.env.sglang` from the runnable 0.8B smoke profile when absent;
5. runs the native doctor before starting a model.

The script does not install or replace the NVIDIA driver or a system CUDA
toolkit. SGLang/PyTorch wheels supply CUDA userspace; the host driver determines
which reviewed track is safe. `nvcc --version` is therefore not used to choose
the wheel. System packages are modified only when
`--install-system-deps` is explicit.

An existing `.env.sglang` is preserved. Select a candidate only while creating
a new file, for example `--profile 5090`; 35B candidates still require an
explicit checkpoint and physical-GPU certification. If a profile sets
`SGLANG_PYTHON`, the helper treats that environment as operator-managed and
does not install packages into it. Lifecycle-only commands (`status`, `logs`,
and `down`) never run installers, so recovery remains available offline.

### Select SGLang from the NVIDIA driver

Do not install an unpinned latest SGLang before checking the host driver. The
Python wheels carry their own CUDA/PyTorch user-space stack, while the kernel
driver still comes from the host. For the Qwen3.5 profile, use these
conservative native tracks. They are also stored in
[`config/sglang/compatibility.json`](../../config/sglang/compatibility.json),
which the native doctor reads:

| NVIDIA Linux driver | CUDA wheel family | SGLang pin | Qwen3.5 status |
|---|---|---|---|
| below 525 | unsupported | none | upgrade the host/provider image |
| 580 or newer | CUDA 13.x | `0.5.17` | **official recommended default**; verified on a 32 GB RTX 5090 |
| 525-579 | CUDA 12.x | `0.5.10.post1` | compatibility hint only; best-effort native fallback observed on a 140 GB L20X and a 32 GB RTX 5090 with driver 570 |

Choose `0.5.17` whenever the driver is 580 or newer; this cu13 path is the
official recommendation. The `0.5.10.post1` pin remains only as a compatibility
hint for hosts stuck below that boundary, not as a production-default promise:
it cannot use
`--language-only`, so the unused vision encoder stays resident. The native
cu12 path has passed a 32 GB RTX 5090 chain test when paired with a CUDA 12.9
JIT toolkit; this does not certify the combined cu12 Docker image.

This boundary follows NVIDIA's
[CUDA minor-version compatibility range](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html):
CUDA 12.x supports drivers 525 through 579, while CUDA 13.x requires driver 580
or newer. The [SGLang 0.5.10 release](https://github.com/sgl-project/sglang/releases/tag/v0.5.10)
added Qwen3.5 and its
[patched PyPI package](https://pypi.org/project/sglang/0.5.10.post1/) uses CUDA
12.9/PyTorch 2.9.1. SGLang 0.5.11 moved its default packages and images to CUDA
13/PyTorch 2.11; later releases may additionally publish explicitly named
CUDA 12 builds. Recheck the exact wheel/image metadata when advancing either
track instead of inferring CUDA solely from the SGLang version.

For example, a driver 595 container should use:

```bash
nvidia-smi --query-gpu=driver_version --format=csv,noheader
python3 -m venv .venv-sglang
.venv-sglang/bin/python -m pip install --upgrade pip 'sglang==0.5.17'

# Debian/Ubuntu minimal containers may omit this SGLang kernel dependency.
apt-get update
apt-get install -y libnuma1
```

Install SGLang with `pip`, not `uv`, unless the host has `uv` 0.12.0 or newer.
Every SGLang release in the compatibility matrix depends on a prerelease
`flash-attn-4` — 0.5.17 requires `flash-attn-4>=4.0.0b18`, and the only
non-prerelease version on PyPI is an unrelated 0.0.1 placeholder. `pip` accepts
that because the specifier itself names a prerelease, but `uv` before 0.12.0
refuses transitive prereleases outright and fails with `No solution found`. On
an older `uv`, name the package at the top level so its explicit mode applies:

```bash
uv pip install --python .venv-sglang/bin/python \
  --prerelease=explicit 'sglang==0.5.17' 'flash-attn-4>=4.0.0b4'
```

`./scripts/run-linux-gpu.sh` already does this for the environment it manages.

Record the selection in `.env.sglang`. The native doctor verifies both the
exact pin and the CUDA 12/13 driver boundary:

```dotenv
SGLANG_PYTHON=.venv-sglang/bin/python
SGLANG_EXPECTED_VERSION=0.5.17
SGLANG_EXTRA_ARGS=--max-running-requests 1 --language-only
```

A container on driver 550 installs `sglang==0.5.10.post1` instead, and must
drop `--language-only`:

```dotenv
SGLANG_EXPECTED_VERSION=0.5.10.post1
SGLANG_EXTRA_ARGS=--max-running-requests 1
```

SGLang and FlashInfer compile some kernels lazily. The machine-readable track
therefore records a matching JIT toolkit as well as the wheel family: cu12
needs CUDA 12.9 or newer within the CUDA 12 family, while cu13 needs CUDA 13.
The sampling path also needs the `curand.h` development header. The doctor
checks the effective `CUDA_HOME`/`nvcc`, initializes CUDA with the selected
Python, runs a small BF16 matrix multiplication, and verifies these headers.
The bootstrap deliberately does not install or replace a CUDA toolkit.

On a Debian/Ubuntu provider container using the cu12 RTX 5090 path, the
corresponding packages are commonly:

```bash
apt-get update
apt-get install -y \
  cuda-nvcc-12-9 cuda-cudart-dev-12-9 cuda-crt-12-9 libcurand-dev-12-9

export CUDA_HOME=/usr/local/cuda-12.9
export CUDA_PATH="$CUDA_HOME"
export PATH="$CUDA_HOME/bin:$PATH"
```

Package names and toolkit locations are image-specific. Prefer a provider or
custom image whose driver, CUDA userspace, and JIT toolkit already agree over
repairing a running container.

`SGLANG_EXPECTED_VERSION` is an assertion, not an installer. Leave it empty
when using a provider-managed environment whose exact compatible patch version
you intentionally do not control; the doctor still checks the CUDA-family
boundary for versions it can identify.

The `0.5.10.post1` profile must also omit `--language-only`. In that release the
flag enables encoder disaggregation and requires separate `--encoder-urls`;
the standalone Qwen3.5 server should load the complete checkpoint instead.
`--language-only` in the repository's newer 0.5.17 profile has different
deployment expectations and must not be copied blindly across this boundary.

## Configure an RTX 5090 chain test

Two native chains have been verified on a 32 GB RTX 5090: driver 595.71.05 with
SGLang `0.5.17` (cu13) on 2026-08-16, and driver 570.195.03 with SGLang
`0.5.10.post1` plus a CUDA 12.9 JIT toolkit (cu12) on 2026-08-15. The cu12 run
also passed non-greedy sampling and an end-to-end agent file-tool task. The
combined GPU Docker images have not been certified on those cards yet.

```bash
git clone https://github.com/ApodexAI/FrontierAgent.git
cd FrontierAgent

uv sync --python 3.12 --extra dev
cp config/sglang/35b-5090.env.example .env.sglang
chmod 600 .env.sglang
```

Edit `.env.sglang` for the provider environment:

```dotenv
# Python from the environment where `import sglang` succeeds.
SGLANG_PYTHON=/path/to/sglang-environment/bin/python

# Pin selected from the driver matrix above; doctor checks the installed value.
# Use 0.5.17 on driver 580+, or 0.5.10.post1 on driver 525-579.
SGLANG_EXPECTED_VERSION=0.5.17

# Use the provider's persistent data disk, not a small container system disk.
# AutoDL convention:
SGLANG_DOWNLOAD_DIR=/root/autodl-tmp/huggingface

# Keep the unauthenticated API private to this instance.
SGLANG_NATIVE_HOST=127.0.0.1
```

Align `HF_HUB_CACHE` and `SGLANG_DOWNLOAD_DIR` so tokenizer, processor,
configuration, and weight files do not split across the system disk and data
disk. `HF_HOME` is their parent; Hugging Face normally appends `/hub` to it.
After one successful online download, an anonymous deployment can avoid Hub
metadata rate limits on restart:

```dotenv
HF_HOME=/root/autodl-tmp/huggingface
HF_HUB_CACHE=/root/autodl-tmp/huggingface/hub
SGLANG_DOWNLOAD_DIR=/root/autodl-tmp/huggingface/hub
HF_HUB_OFFLINE=1
```

Do not enable offline mode before the full snapshot has been downloaded.

The supplied 5090 profile configures `moe_wna16`, `bfloat16`, the
`qwen3_coder` tool parser, the `qwen3` reasoning parser, text-only loading,
32K context, and one running request. The quantization and dtype are pinned
rather than left at `auto` because the alternatives fail late, during kernel
compilation, without naming the cause: `gptq` rejects the MoE structure
outright, `gptq_marlin` reads fp16 scales its kernel then refuses against bf16
activations, and `float16` triggers a Triton branch-type assertion. The
profile deliberately leaves the model source blank: set the final
published `SGLANG_MODEL_ID`, or use `SGLANG_LOCAL_MODEL_PATH` for a checkpoint.
Treat higher context and concurrency as later capability tests.

## Diagnose, start, and verify

```bash
./scripts/run-sglang-native.py doctor
./scripts/run-sglang-native.py up
./scripts/run-sglang-native.py smoke
```

The native doctor checks the Linux environment, direct GPU visibility, a real
CUDA BF16 operation, the selected SGLang Python, matching `nvcc` and JIT
headers, model source, tensor parallel size, token budgets, cache-disk space,
loopback binding, and port availability. It does not require or probe Docker.

The server runs in its own process group. State is stored under:

```text
.apodex/sglang-native/server.pid
.apodex/sglang-native/server.log
```

Useful lifecycle commands:

```bash
./scripts/run-sglang-native.py status
./scripts/run-sglang-native.py logs
./scripts/run-sglang-native.py down
```

The launcher validates that a saved PID still belongs to
`sglang.launch_server` before sending a signal. It will use a healthy endpoint
that was started separately, but it will not claim ownership of or stop that
external process.

## Run FrontierAgent

```bash
./scripts/run-sglang-native.py tui
```

This starts SGLang if necessary and runs the ReAct workflow using FrontierAgent's
Linux native runtime. The launcher derives the following settings from the same
`.env.sglang` file, so a second `.env` does not need to duplicate them:

```dotenv
OPENAI_BASE_URL=http://127.0.0.1:30000/v1
OPENAI_MODEL=local-model
OPENAI_CONTEXT_WINDOW=32768
OPENAI_MAX_INPUT_TOKENS=27000
OPENAI_MAX_TOKENS=4096
```

Arguments after `tui` are forwarded to FrontierAgent. For example:

```bash
./scripts/run-sglang-native.py tui --cwd /root/autodl-tmp/project
```

## Troubleshooting

| Symptom | Likely cause | Action |
|---|---|---|
| `docker: command not found` | expected on an ordinary managed container | use this native launcher, not `docker/run-sglang.sh` |
| `sglang is not importable` | wrong Python environment selected | set `SGLANG_PYTHON` to the Python where `import sglang` succeeds |
| selected native track requires driver 580+ | driver is in the conservative CUDA 12 range | install `sglang==0.5.10.post1` or upgrade the host driver |
| installed version does not match `SGLANG_EXPECTED_VERSION` | the selected environment drifted | reinstall the configured pin or deliberately update the config and compatibility matrix |
| `requires at least one encoder urls` | `--language-only` was copied to SGLang 0.5.10 | remove `--language-only` for a standalone full-model server |
| `GPTQ Method does not support MoE` | `SGLANG_QUANTIZATION=gptq` against a MoE checkpoint | set `SGLANG_QUANTIZATION=moe_wna16` |
| `moe_wna16_marlin_gemm assumes hidden_states.dtype == w1_scale.dtype` | `gptq_marlin` reads fp16 scales but the activations are bf16 | set `SGLANG_QUANTIZATION=moe_wna16`, not `gptq_marlin` |
| `Mismatched type for col0 between then block` | `SGLANG_DTYPE` disagrees with the checkpoint's declared dtype | set `SGLANG_DTYPE=bfloat16` for the Qwen3.5 GPTQ Int4 checkpoint |
| `libtorchcodec` or `libavutil.so.57` fails to load | FFmpeg runtime is absent | ignore under `--language-only`; install the FFmpeg development libraries only if multimodal input is needed |
| CUDA/PyTorch symbol error | provider CUDA stack and SGLang wheels disagree | select a compatible provider image or follow SGLang's native install matrix |
| CUDA error 804 (`forward compatibility was attempted`) | a container forward-compat `libcuda` shadows the host driver on unsupported hardware | prefer a driver-compatible image; otherwise inspect the loaded `libcuda.so.1` and restore the provider's host-driver library path |
| CUDA error 35 during graph capture or JIT | the wheel track and effective `nvcc`/CUDA runtime use different CUDA families | set `CUDA_HOME`, `CUDA_PATH`, and `PATH` to the toolkit matching the selected track, then restart |
| `SM 12.x requires CUDA >= 12.9` | an RTX 5090 cu12 JIT selected CUDA 12.8 or older | install/select a CUDA 12.9 toolkit; do not switch the JIT compiler to CUDA 13 on a driver below 580 |
| `curand.h: No such file or directory` | the first non-greedy request reached a sampling JIT without CUDA development headers | install the matching `libcurand-dev` package and rerun smoke |
| `libnuma.so.1: cannot open shared object file` | minimal container omits the NUMA runtime | install `libnuma1` on Debian/Ubuntu (equivalent package on other distributions) |
| `No solution found` while installing SGLang | `uv` before 0.12.0 refuses SGLang's prerelease `flash-attn-4` dependency | install with `pip`, upgrade `uv`, or add `--prerelease=explicit` with `flash-attn-4` named at the top level |
| model download fills `/` | cache is on the container system disk | set `SGLANG_DOWNLOAD_DIR` to the persistent data disk |
| Hugging Face returns HTTP 429 during restart | anonymous metadata requests are rate-limited | set `HF_TOKEN`, or after a complete download align `HF_HUB_CACHE`/`SGLANG_DOWNLOAD_DIR` and set `HF_HUB_OFFLINE=1` |
| port 30000 is occupied | prior or unrelated server is running | run `status`; stop the known process or select another `SGLANG_PORT` |
| startup OOM | weights, GDN state, KV pool, or CUDA graphs exceed VRAM | reduce context/concurrency; if necessary add `--disable-cuda-graph` |
| smoke health passes but tool call fails | parser/model integration problem | inspect `logs`; verify `moe_wna16`, `qwen3_coder`, and the checkpoint revision |

After correcting `CUDA_HOME` or the effective compiler, remove only the stale
SGLang JIT caches created by the wrong toolkit (commonly the `tvm-ffi` and
`flashinfer` cache directories) before retrying. Do not copy a hard-coded
`LD_LIBRARY_PATH` from another provider: the host-driver library location is
image-specific.

Do not install NVIDIA Container Toolkit inside an ordinary provider container.
That toolkit configures a Docker daemon on a GPU host; this native path already
receives the GPU directly from the provider.
