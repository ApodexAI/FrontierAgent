# Docker SGLang on Linux with an NVIDIA GPU

This guide is the canonical clean-machine path for a Linux bare-metal host or
VM where Docker and NVIDIA Container Toolkit are available. It runs SGLang and
FrontierAgent as separate Compose services and deliberately distinguishes
infrastructure smoke from production model support.

If you are unsure whether the provider gives you a VM, nested Docker, or a
custom-image field, return to the [installation chooser](README.md) or the
[GPU platform guide](gpu-platforms.md). Check the
[GPU compatibility matrix](gpu-compatibility.md) before selecting a 35B
profile.

If the Linux environment is already a provider-owned container and cannot run
Docker inside it, use [Native SGLang on Linux GPU environments](linux-nvidia-native.md)
instead.

## What “working” means

The local GPU path has separate gates:

1. **Host:** the NVIDIA driver can enumerate the GPU.
2. **Container:** Docker can pass that GPU into a CUDA container.
3. **Server:** SGLang becomes healthy and exposes an OpenAI-compatible API.
4. **Parser:** the model returns a structured `tool_calls` object.
5. **Product:** FrontierAgent's TUI reaches the model and renders tool approval.
6. **Capability:** the production model repeatedly chooses the correct tools and
   produces correct answers on a published evaluation set.

The supplied 0.8B profile is expected to pass gates 1–5. It is intentionally a
small integration fixture and is not evidence that gate 6 passes.

## Hardware and storage

| Profile | Purpose | GPU expectation | Status |
|---|---|---|---|
| `.env.sglang.example` | 0.8B infrastructure smoke | one NVIDIA GPU with about 8 GB VRAM | verified on RTX 5060 |
| `config/sglang/35b-4090.env.example` | 35B single-card candidate | RTX 4090 24 GB, 4-bit checkpoint | must be certified |
| `config/sglang/35b-5090.env.example` | Qwen3.5-35B-A3B GPTQ Int4 chain test | RTX 5090 32 GB | must be certified |
| `config/sglang/35b-multigpu.env.example` | 35B two-card candidate | two matched NVIDIA GPUs | must be certified |

A dense or MoE 35B model contains roughly 70 GB of BF16 weights or 35 GB of
FP8 weights before KV cache, CUDA graphs, allocator overhead, and multimodal
state. A single 4090/5090 therefore requires an INT4/NVFP4/AWQ/GPTQ-style
checkpoint. MoE active parameters reduce compute per token, not necessarily the
weight memory that must be resident.

Reserve at least 60 GB of free disk for first-time setup. The full SGLang image
can be tens of gigabytes; the configured `-runtime` image removes development
tooling. Model weights are cached in the `huggingface-cache` Docker volume.

## 1. Install the host prerequisites

Supported release claims must name exact tested Linux, driver, Docker, Toolkit,
SGLang, checkpoint, and GPU versions. Do not use a distro-agnostic convenience
script to install or replace an NVIDIA driver.

1. Install the NVIDIA driver using your Linux distribution's package manager.
2. Reboot if the installer requests it, then verify:

   ```bash
   nvidia-smi
   ```

3. Install Docker Engine and the Compose plugin from Docker's official
   repository for your distribution:
   <https://docs.docker.com/engine/install/>
4. Install NVIDIA Container Toolkit using NVIDIA's current distribution-specific
   instructions:
   <https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html>
5. Configure Docker and restart it:

   ```bash
   sudo nvidia-ctk runtime configure --runtime=docker
   sudo systemctl restart docker
   ```

The host needs the NVIDIA driver, but not a separate full CUDA Toolkit. CUDA
userspace libraries arrive in the model container.

Docker normally requires `sudo`. Adding an account to the `docker` group grants
root-equivalent access; understand that boundary before following Docker's
non-root post-install steps. NVIDIA documents a separate configuration for
rootless Docker.

## 2. Clone and configure

```bash
git clone https://github.com/ApodexAI/FrontierAgent.git
cd FrontierAgent

# Integration smoke
cp .env.sglang.example .env.sglang
chmod 600 .env.sglang
```

For a 35B checkpoint, copy the candidate matching the target host and fill in
`SGLANG_MODEL_ID` or `SGLANG_LOCAL_MODEL_PATH`. The candidate deliberately
leaves both blank so a compatibility-test repository cannot become a public
runtime default:

See the [SGLang configuration reference](../../config/sglang/README.md) before
changing context, quantization, memory, parser, or concurrency settings.

```bash
cp config/sglang/35b-5090.env.example .env.sglang
chmod 600 .env.sglang
$EDITOR .env.sglang
```

The 5090 candidate follows Qwen's SGLang guidance for `moe_wna16`, the
`qwen3_coder` tool parser, and the `qwen3` reasoning parser. It also uses
SGLang's `--language-only` mode because FrontierAgent does not send image input.
The first pass is intentionally limited to a 32K context and one running
request. Qwen documents a native 262K context and recommends at least 128K for
full thinking quality, but that is a separate VRAM/capability certification
step after the single-card chain works.

On a rented GPU host, inspect `docker info --format '{{.DockerRootDir}}'` and
free space on that filesystem before downloading. The default Hugging Face
cache is a Docker volume, so available space in the repository's filesystem is
not sufficient if Docker uses a smaller system disk.

A local checkpoint path takes precedence over the Hugging Face ID and is mounted
read-only. It must contain the complete configuration, tokenizer, index, and all
referenced weight shards. Enable `SGLANG_TRUST_REMOTE_CODE` only after reviewing
the model repository.

During private development, authenticate before starting the TUI:

```bash
docker login ghcr.io
```

Alternatively, set `SGLANG_BUILD_AGENT=1` in `.env.sglang` to build FrontierAgent
from the checkout. This affects only the agent image; SGLang still comes from
its pinned upstream image.

## 3. Diagnose before downloading

```bash
# Includes an actual CUDA-container passthrough test.
./docker/run-sglang.sh doctor

# Omits the container pull/run when doing a fast configuration check.
./docker/run-sglang.sh doctor quick
```

The doctor does not print tokens. It validates Docker access, Compose, the host
driver, visible GPU count, TP size, token-budget invariants, model source, disk,
port, VPN warning signs, output ownership, and container GPU passthrough.

For a clean RTX 5090 Docker-host chain test, use this exact order:

```bash
cp config/sglang/35b-5090.env.example .env.sglang
chmod 600 .env.sglang

# Build the current branch locally if its release image is not yet published.
sed -i 's/^SGLANG_BUILD_AGENT=0$/SGLANG_BUILD_AGENT=1/' .env.sglang

./docker/run-sglang.sh doctor
./docker/run-sglang.sh up
./docker/run-sglang.sh smoke
./docker/run-sglang.sh tui
```

Keep `./docker/run-sglang.sh logs` open in a second shell during the first
download and warmup. Record `nvidia-smi`, `docker version`, `docker compose
version`, and the SGLang image digest before changing context or memory knobs.

## 4. Start and smoke-test

```bash
./docker/run-sglang.sh up
./docker/run-sglang.sh smoke
```

The smoke command checks `/health`, `/v1/models`, and a structured calculator
tool call. Its final note explicitly says that parser success is not general
agent correctness.

The first pull can take a long time. In another terminal:

```bash
./docker/run-sglang.sh logs
./docker/run-sglang.sh status
```

## 5. Run the TUI

```bash
./docker/run-sglang.sh tui

# Backwards-compatible forms remain valid:
./docker/run-sglang.sh --mode react
./docker/run-sglang.sh --mode agent_team
```

Use ReAct for the first functional test. Agent Team can multiply concurrent
model calls and must not be a single-card certification workload until explicit
concurrency limits have been measured.

Leaving the TUI does not stop the model. This makes follow-up sessions fast and
is reported by the launcher. Stop it explicitly:

```bash
./docker/run-sglang.sh down
```

Downloaded weights remain cached. The model does not automatically restart
after a host reboot unless `SGLANG_RESTART_POLICY=unless-stopped` is explicitly
selected.

## 6. VPN and network conflicts

Some full-tunnel VPNs reserve Docker's candidate address pools. A typical error
is:

```text
all predefined address pools have been fully subnetted
```

Choose a non-overlapping private `/24` after inspecting `ip route` and Docker's
existing networks, then set it in `.env.sglang`:

```dotenv
APODEX_DOCKER_SUBNET=172.29.250.0/24
```

The launcher includes `compose.network.yaml` only when this value is non-empty.
It never rewrites `/etc/docker/daemon.json`. On managed machines, ask the network
administrator for an approved subnet instead of guessing.

## Troubleshooting

| Symptom | Likely cause | Action |
|---|---|---|
| `nvidia-smi` fails | driver/Secure Boot/kernel module issue | repair the host driver before Docker |
| Docker socket permission denied | daemon stopped or account lacks access | start Docker; use `sudo`, rootless Docker, or review docker-group risk |
| `could not select device driver ... gpu` | Toolkit/runtime not configured | run `nvidia-ctk runtime configure`, restart Docker, rerun doctor |
| CUDA container cannot see a GPU | driver/runtime incompatibility | compare host driver with the selected SGLang CUDA generation |
| SGLang OOM during warmup | weights, KV pool, or CUDA graphs exceed VRAM | use the certified quantized checkpoint/profile; reduce context/concurrency |
| `token budgets exceed context` | invalid env values | ensure input + output is at most context and input stays above 80% |
| GHCR returns `unauthorized` | release image is private | `docker login ghcr.io` or set `SGLANG_BUILD_AGENT=1` |
| port 30000 is occupied | another server or prior model is running | inspect `run-sglang.sh status` or select another `SGLANG_PORT` |
| Docker cannot allocate a network | VPN/corporate CIDR overlap | set a reviewed `APODEX_DOCKER_SUBNET` |
| outputs are owned by `nobody` | older image used its internal tool UID | use the helper launcher; repair existing files once with administrator approval |
| health passes but wrong tool is chosen | model capability, prompt, or excessive tool surface | treat infrastructure as passed; run capability evaluation separately |

## Release certification matrix

Before marking a 35B profile supported, publish measurements for each GPU:

- exact checkpoint revision and quantization;
- GPU model/count, driver, Toolkit, Docker, and SGLang image digest;
- idle/load VRAM, startup time, and disk download size;
- maximum context, output reserve, and safe concurrency;
- CUDA graph and KV-cache settings;
- structured-tool-call pass rate;
- read-only TUI task success rate and representative agent evaluation score.

Do not infer RTX 4090 results from RTX 5090 results: memory capacity, architecture,
and available quantization kernels differ.
