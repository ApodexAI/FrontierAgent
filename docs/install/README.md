# Choose an installation path

[Documentation index](../../docs/README.md)

Start here. The correct path depends on where the model runs and on what the
platform lets you control; it does not depend only on the operating-system
name.

If you already have an OpenAI-compatible endpoint and only want to open the TUI
on macOS or Linux, use the copy-and-run [English quickstart](tui-endpoint-quickstart.md)
or [中文教程](tui-endpoint-quickstart.zh-CN.md). It does not deploy a model or
require Docker.

## Three questions

1. **Do you need to run the model on a local NVIDIA GPU?**
   - No: use the [macOS](macos.md) or [ordinary Linux](linux.md) guide and
     configure a hosted OpenAI-compatible endpoint.
   - Yes: continue to question 2.
2. **Can you select a custom OCI image when creating the GPU instance?**
   - Yes: use the [GPU cloud and custom-image guide](gpu-platforms.md). The
     platform pulls the image before the container starts; nested Docker is not
     required.
   - No: continue to question 3.
3. **Can this Linux environment run `docker info`, and can Docker pass through
   the GPU?**
   - Yes: use [Docker SGLang on a Linux NVIDIA host](linux-nvidia.md).
   - No: use [Native SGLang in an existing GPU environment](linux-nvidia-native.md).

## Environment chooser

| Your environment | FrontierAgent | Model runtime | Guide |
|---|---|---|---|
| macOS laptop or desktop | native, optionally Docker | hosted/remote endpoint | [macOS](macos.md) |
| Linux laptop, server, or CI without a local model | `scripts/run-linux.sh` (native, bubblewrap, or Docker) | hosted/remote endpoint | [Linux](linux.md) |
| Any host with Docker and no local Python environment | published agent container | hosted/remote endpoint | [Docker and Compose](docker.md) |
| Linux bare metal or VM with an NVIDIA GPU and Docker daemon | native or agent container | SGLang container | [Linux NVIDIA + Docker](linux-nvidia.md) |
| RunPod-style service that accepts your image at instance creation | inside the provider container | prebuilt FrontierAgent GPU image | [GPU cloud images](gpu-platforms.md) |
| Existing x86_64 Linux GPU environment without nested Docker | `scripts/run-linux-gpu.sh` | isolated native SGLang process | [Linux NVIDIA native](linux-nvidia-native.md) |
| Windows | WSL2, treated as Linux | hosted or WSL2-reachable endpoint | [Linux](linux.md#windows-and-wsl2) |

Chinese-speaking macOS users can follow the detailed
[macOS 中文安装与一键启动指南](macos.zh-CN.md).

## GPU driver / CUDA / SGLang tracks

Local-model serving only supports two reviewed tracks. Select by the host
driver shown in `nvidia-smi` — not by the CUDA version printed inside any
container:

| Track | NVIDIA driver | SGLang pin | Status |
|---|---|---|---|
| `cu13` (**official default**) | 580 or newer | `0.5.17` | recommended and verified on RTX 5090 32 GB |
| `cu12` (compatibility hint) | 525 or newer | `0.5.10.post1` | best-effort native fallback observed on RTX 5090 32 GB with driver 570; cannot use `--language-only` |

New and supported deployments use `cu13`. The `cu12` row is a compatibility
hint for otherwise blocked older-driver environments, not a second recommended
configuration or a production-default promise.

A driver of 580 or newer also runs the `cu12` track, because a newer driver
still supports older CUDA runtimes. The 580 boundary decides which track a host
*can* use, not which one it must: pick `cu12` only when the driver is below
580. The RTX 5090 itself does not require the cu13 track: a 32 GB card on
driver 570 has also passed the native cu12 chain test when paired with a CUDA
12.9 JIT toolkit.

Picking the wrong track fails late — during model load, with CUDA or Triton
errors that do not name the real cause. Check the
[GPU compatibility matrix](gpu-compatibility.md) first.

Before choosing a 35B profile, check the [GPU, driver, VRAM, and SGLang
compatibility matrix](gpu-compatibility.md). A compatible CUDA runtime does not
guarantee that a model fits in VRAM. The
[SGLang configuration reference](../../config/sglang/README.md) explains the
`.env.sglang` templates, each variable, and a safe tuning sequence.

## What each layer is responsible for

```text
GPU host or cloud platform
├── NVIDIA driver and assigned GPU
├── optional container runtime / platform image loader
├── SGLang CUDA and PyTorch userspace
├── model weights and Hugging Face cache
└── FrontierAgent, calling SGLang through an OpenAI-compatible URL
```

The host supplies the driver and GPU. A GPU image supplies CUDA userspace,
PyTorch, SGLang, and FrontierAgent, but never the host driver. Model weights and
tokens should be mounted or injected at runtime rather than baked into a public
image.

## Safe first test

Whichever GPU route you choose, validate in this order:

1. `nvidia-smi` can enumerate the assigned GPU.
2. The selected SGLang/CUDA track matches the driver.
3. The small infrastructure profile becomes healthy.
4. The intended checkpoint fits and returns a structured tool call.
5. FrontierAgent reaches the endpoint.

Health and parser success prove the deployment plumbing. They do not by
themselves prove model quality or safe production concurrency.
