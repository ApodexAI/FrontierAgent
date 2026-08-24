# GPU cloud platforms and custom images

There are three different deployment shapes on rented GPU platforms. Check the
platform feature before installing anything inside an instance.

| Platform capability | Correct path | Typical examples |
|---|---|---|
| choose a public/private OCI image while creating the instance | custom GPU image | RunPod Pod templates, Alibaba Cloud PAI EAS |
| provision a Linux VM and control its Docker daemon | Docker Compose | AWS EC2, Alibaba Cloud ECS, bare-metal hosts |
| receive an already-running provider container with no nested Docker | native processes | ordinary AutoDL instances or RunPod Pods started from a provider image |

RunPod documents public and private registry images in custom Pod templates:
<https://docs.runpod.io/pods/templates/manage-templates>. Alibaba Cloud PAI EAS
supports custom images and recommends ACR for production pulls:
<https://help.aliyun.com/en/pai/deploy-a-model-service-by-using-a-custom-image>.
AutoDL currently documents that external custom-image import is unavailable:
<https://www.autodl.com/docs/image/>.

## Build the combined GPU image

The GPU Dockerfile derives from an official SGLang runtime and installs
FrontierAgent into a separate Python environment, leaving SGLang's CUDA/PyTorch
stack untouched. Build the recommended CUDA 13 track (NVIDIA driver 580+):

```bash
docker build \
  --file docker/sglang-gpu.Dockerfile \
  --build-arg SGLANG_BASE_IMAGE=lmsysorg/sglang:v0.5.17-runtime \
  --build-arg SGLANG_VERSION=0.5.17 \
  --build-arg CUDA_TRACK=cu13 \
  --build-arg FRONTIER_AGENT_VERSION=dev \
  --tag frontieragent-gpu:cu13 \
  .
```

Hosts whose driver is below 580 can still build the legacy CUDA 12 track:

```bash
docker build \
  --file docker/sglang-gpu.Dockerfile \
  --build-arg SGLANG_BASE_IMAGE=lmsysorg/sglang:v0.5.10.post1-runtime \
  --build-arg SGLANG_VERSION=0.5.10.post1 \
  --build-arg CUDA_TRACK=cu12 \
  --build-arg FRONTIER_AGENT_VERSION=dev \
  --tag frontieragent-gpu:cu12 \
  .
```

The `cu13` image/profile is the official recommended default. The `cu12`
profile is only a compatibility hint for an otherwise blocked older-driver
environment; it cannot use `--language-only`, keeps the vision encoder
resident, and its combined Docker image is not certified. A native-process
cu12 recovery was observed on a 32 GB RTX 5090 with driver 570, but that result
does not promote cu12 to a supported production default.

The checked-in machine-readable pins and upstream digests live in
[`config/sglang/compatibility.json`](../../config/sglang/compatibility.json).
Release builds should use both the tag and reviewed digest from that file.

## Start the Qwen runtime profile and TUI

The image does not silently choose a model because an accidental 35B download
is expensive and the final repository may differ from compatibility-test
repositories. Supply the published model ID explicitly, then select the
runtime profile matching the image's CUDA track:

```bash
export HF_TOKEN=your-token-if-needed
export SGLANG_MODEL_ID=your-organization/your-published-model

docker run --rm -it --gpus all \
  --shm-size 32g \
  --ipc=host \
  --volume "$PWD:/workspace" \
  --volume huggingface-cache:/root/.cache/huggingface \
  --env HF_TOKEN \
  --env SGLANG_MODEL_ID \
  --env FRONTIER_AGENT_GPU_PROFILE=qwen35-gptq-cu13 \
  frontieragent-gpu:cu13 tui --cwd /workspace
```

Both profiles select 32K context, `moe_wna16`, the `qwen3_coder` tool parser,
and one running request. `qwen35-gptq-cu13` additionally pins
`SGLANG_DTYPE=bfloat16` to match the checkpoint's `config.json` and passes
`--language-only`, which frees the vision encoder's VRAM; both were verified on
SGLang 0.5.17. `qwen35-gptq-cu12` omits `--language-only` because SGLang 0.5.10
reads it as encoder disaggregation. Neither profile selects a Hugging Face
repository.

The entrypoint refuses to start when the selected profile does not match the
SGLang build in the image, so a `cu13` profile on a `cu12` image fails
immediately and says why. Run `doctor` to print the settings a profile
resolved to:

```bash
docker run --rm --gpus all \
  --env FRONTIER_AGENT_GPU_PROFILE=qwen35-gptq-cu13 \
  frontieragent-gpu:cu13 doctor
```

Every setting can still be overridden with its corresponding `SGLANG_*`
environment variable. `SGLANG_EXTRA_ARGS` is replaced as a whole rather than
appended to, so an override must repeat the flags it still wants — dropping
`--language-only` on a 32 GB card loads the vision encoder and can exhaust
VRAM during model load.

If the model is already present in a host Hugging Face cache, mount the cache
root at the image's standard cache location. Export `SGLANG_MODEL_ID` with the
repository that populated that cache; Hugging Face resolves the current
snapshot from `refs/main`, so no revision hash needs to be copied into the
command:

```bash
docker run --rm -it --gpus all \
  --shm-size 32g \
  --ipc=host \
  --volume "$PWD:/workspace" \
  --volume /path/to/huggingface-cache:/root/.cache/huggingface \
  --env HF_HUB_CACHE=/root/.cache/huggingface \
  --env SGLANG_DOWNLOAD_DIR=/root/.cache/huggingface \
  --env HF_HUB_OFFLINE=1 \
  --env SGLANG_MODEL_ID \
  --env FRONTIER_AGENT_GPU_PROFILE=qwen35-gptq-cu13 \
  frontieragent-gpu:cu13 tui --cwd /workspace
```

Mount the complete cache root, not only its `snapshots/<revision>` suffix: the
cache also needs the repository's `refs` and `blobs` directories. If an
ordinary checkpoint directory is mounted instead, bind that exact directory
to a short container path and set the container path, for example:

```bash
--volume "$model_snapshot:/models/checkpoint:ro" \
--env SGLANG_LOCAL_MODEL_PATH=/models/checkpoint
```

Do not split either argument across shell lines.

## Run on a Docker GPU host

Do not put a Hugging Face token or model weights in the image:

```bash
docker run --rm --gpus all \
  --shm-size 32g \
  --ipc=host \
  --publish 127.0.0.1:30000:30000 \
  --volume huggingface-cache:/root/.cache/huggingface \
  --env HF_TOKEN \
  --env SGLANG_MODEL_ID=your-organization/your-published-model \
  --env SGLANG_SERVED_MODEL_NAME=local-model \
  --env SGLANG_CONTEXT_LENGTH=32768 \
  --env SGLANG_QUANTIZATION=moe_wna16 \
  --env SGLANG_TOOL_CALL_PARSER=qwen3_coder \
  --env SGLANG_REASONING_PARSER=qwen3 \
  --env 'SGLANG_EXTRA_ARGS=--max-running-requests 1 --language-only' \
  frontieragent-gpu:cu13 server
```

For SGLang `0.5.10.post1` (the legacy `cu12` track), do not add
`--language-only`; in that release it selects encoder disaggregation and
requires separate encoder URLs.

On the CUDA 13 track (SGLang 0.5.17), prefer the bundled
`FRONTIER_AGENT_GPU_PROFILE=qwen35-gptq-cu13` over hand-writing these
variables: it pins `SGLANG_DTYPE=bfloat16` and enables `--language-only`, both
verified against the Qwen3.5 35B-A3B GPTQ Int4 checkpoint, and it is checked
against the image's SGLang version at startup. A hand-written variable set
gets no such check.

## Configure a platform template

For a single-container GPU service, set:

| Template field | Value |
|---|---|
| image | the exact versioned GPU image tag, not `latest` |
| GPU type/count | chosen from the compatibility and VRAM matrix |
| container port | `30000` |
| shared memory | at least `32 GiB` where configurable |
| persistent volume | mount at `/root/.cache/huggingface` or set `HF_HOME` |
| startup command | leave the image default, or select `server` |
| secrets | inject `HF_TOKEN`; never include it in the template or image |

The default mode starts the OpenAI-compatible SGLang server. Other image modes
are `doctor`, `tui`, `agent`, `shell`, and `server`. A platform may override
the image command; preserve the image entrypoint so NVIDIA initialization still
runs.

## Use the same image across cloud operating models

The portable unit is the GPU image, but the way an operator enters the running
container differs by platform:

| Platform shape | Common examples | Main container process | Interactive TUI access |
|---|---|---|---|
| GPU Pod with a terminal inside the container | RunPod Pods and similar GPU rental products | `server` | SSH or web terminal, then run FrontierAgent directly |
| GPU VM where SSH enters the host | AWS EC2, Alibaba Cloud ECS, bare metal | `server` in a detached Docker container | `docker exec -it` into that container |
| container orchestrator | Kubernetes, GPU-enabled AWS ECS | `server` in a Pod or task | `kubectl exec -it` or the platform's exec facility |
| managed inference service | Alibaba Cloud PAI EAS and similar products | `server` behind the platform gateway | normally API-only; do not depend on an interactive TUI |
| serverless inference | RunPod Serverless, SageMaker endpoints | platform-specific request handler | not supported by this generic long-running image |

Use this process layout whenever the platform offers an interactive shell:

```text
cloud platform or Docker
└── FrontierAgent GPU container
    ├── PID 1: SGLang server on port 30000
    └── SSH/exec terminal: FrontierAgent TUI connected to 127.0.0.1:30000
```

Do not make the TUI the container's startup command on a cloud service. Cloud
containers start without a terminal attached to PID 1, so the TUI would have
no reliable interactive session. Start the server as PID 1 and launch the TUI
from an SSH, web-terminal, or exec session instead.

## RunPod Pod template

Use a GPU **Pod**, not a Serverless endpoint. In a custom template, start with:

This section assumes the Pod pulls the reviewed FrontierAgent image. If the Pod
instead starts from a general CUDA/provider image and you install the repository
from its terminal, treat it as an existing GPU container and follow the
[native SGLang guide](linux-nvidia-native.md). In that shape, the provider image
can expose a CUDA userspace or forward-compat library newer than the host
driver; select the track from the driver version and let native doctor verify a
real CUDA operation and the effective JIT compiler before downloading weights.

| RunPod field | Value | Notes |
|---|---|---|
| Container Image | `<your-registry>/frontieragent-gpu:cu13` | push the image built above to a registry RunPod can pull from, then pin an immutable tag or digest (not a mutable `latest`/`test` tag) |
| Container Disk | `50 GB` or more | leave headroom for the approximately 34 GB unpacked image |
| Volume Disk or Network Volume | `60-100 GB` | size for the checkpoint, Hugging Face cache, and workspace |
| Volume Mount Path | `/workspace` | keep models and user data on persistent storage |
| Docker Entrypoint | empty | preserve the image's NVIDIA and FrontierAgent entrypoints |
| Docker Start Command | empty or `server` | the image default is `server` |
| HTTP Ports | empty for terminal-only use | add `30000/http` only when an external API is required |

RunPod documents custom images, environment variables, storage, commands, and
ports in its template guide:
<https://docs.runpod.io/pods/templates/manage-templates>. The `/workspace`
volume survives Pod restarts; use a Network Volume when the data must survive
deleting the Pod:
<https://docs.runpod.io/pods/storage/types>.

Configure the model and persistent cache through template environment
variables. The image deliberately has no default model repository:

```dotenv
FRONTIER_AGENT_GPU_PROFILE=qwen35-gptq-cu13
SGLANG_MODEL_ID=your-organization/your-published-model

HF_HOME=/workspace/huggingface
HF_HUB_CACHE=/workspace/huggingface/hub
SGLANG_DOWNLOAD_DIR=/workspace/huggingface/hub

OPENAI_PROVIDER=local
OPENAI_BASE_URL=http://127.0.0.1:30000/v1
OPENAI_MODEL=local-model
OPENAI_API_KEY=EMPTY
```

For an official deployment, choose a Pod whose driver supports the `cu13` image
and `qwen35-gptq-cu13` profile. If an existing host is unavoidably below driver
580, the `cu12` tag and `qwen35-gptq-cu12` profile remain a best-effort
compatibility hint rather than the recommended default.

For a private or gated repository, inject `HF_TOKEN` with a RunPod Secret
rather than placing it in a public template. RunPod supports secret references
in environment values:
<https://docs.runpod.io/pods/templates/environment-variables>.

Before the production repository is published, upload an exported checkpoint
to persistent storage and replace `SGLANG_MODEL_ID` with its in-container path:

```dotenv
SGLANG_LOCAL_MODEL_PATH=/workspace/models/your-checkpoint
```

Do not set `HF_HUB_OFFLINE=1` until every required checkpoint and tokenizer
file is present. After deployment, watch the Pod logs and verify the local
server from the RunPod web terminal or SSH session:

```bash
curl --fail http://127.0.0.1:30000/health
curl --fail http://127.0.0.1:30000/v1/models
```

Then start the interactive client in that terminal:

```bash
/opt/venv/bin/frontier-agent \
  --native \
  --mode react \
  --cwd /workspace
```

`--native` makes tool execution use the current trusted GPU container instead
of expecting a nested Docker daemon. RunPod provides a web terminal and basic
SSH access for Pods:
<https://docs.runpod.io/pods/connect-to-a-pod>.

If port `30000/http` is exposed, RunPod assigns an HTTPS proxy URL of the form
`https://POD_ID-30000.proxy.runpod.net`. That endpoint is public. SGLang in the
CUDA 12 image supports `--api-key`, but a production deployment should still
place an authenticated, rate-limited gateway in front of the model server.
RunPod also documents a proxy timeout for long HTTP requests, which is another
reason not to treat the raw proxy as a production inference gateway:
<https://docs.runpod.io/pods/configuration/expose-ports>.

## GPU VMs: AWS EC2, Alibaba Cloud ECS, and bare metal

On a VM, SSH enters the host rather than the GPU container. Install Docker and
the NVIDIA Container Toolkit on the host, then start the image as a detached
service:

```bash
docker run -d \
  --name frontieragent \
  --restart unless-stopped \
  --gpus all \
  --shm-size 32g \
  --ipc=host \
  --volume /data/frontier:/workspace \
  --volume /data/huggingface:/root/.cache/huggingface \
  --env SGLANG_MODEL_ID=your-organization/your-published-model \
  --env FRONTIER_AGENT_GPU_PROFILE=qwen35-gptq-cu13 \
  frontieragent-gpu:cu13 \
  server
```

Prefer upgrading/selecting a driver 580+ host and keeping the official cu13 tag
and profile. The cu12 tag/profile is only a best-effort hint when an existing
host cannot cross that driver boundary.

After the health check passes, attach an interactive TUI to the same container:

```bash
docker exec -it \
  --env OPENAI_PROVIDER=local \
  --env OPENAI_BASE_URL=http://127.0.0.1:30000/v1 \
  --env OPENAI_API_KEY=EMPTY \
  --env OPENAI_MODEL=local-model \
  frontieragent \
  /opt/venv/bin/frontier-agent --native --mode react --cwd /workspace
```

AWS also supports GPU allocation in ECS task definitions and provides an
ECS-optimized GPU AMI with NVIDIA drivers and the GPU container runtime:
<https://docs.aws.amazon.com/AmazonECS/latest/developerguide/ecs-gpu.html>.

## Kubernetes and other container orchestrators

Use the image as a long-running Deployment or StatefulSet, inject model tokens
from a Secret, and mount the checkpoint/cache from a PersistentVolumeClaim.
Keep port `30000` cluster-internal unless an authenticated gateway fronts it.
For an operator TUI, exec into the already-running model Pod:

```bash
kubectl exec -it deployment/frontieragent -- \
  /opt/venv/bin/frontier-agent --native --mode react --cwd /workspace
```

The Pod environment must include the same `OPENAI_PROVIDER`, `OPENAI_BASE_URL`,
`OPENAI_MODEL`, and model-source variables shown in the RunPod example.

## Managed inference services

Managed inference products usually accept an image, startup command,
environment variables, storage mounts, a health check, and one serving port.
Configure `server` and port `30000`; consume the service through its API rather
than expecting SSH or a TUI.

Alibaba Cloud PAI EAS supports custom images, commands, environment variables,
and OSS/NAS storage mounts. It reserves ports `8080` and `9090`, so keep the
image on its default `30000` port. Alibaba recommends ACR in the same region for
production pulls and separating the model files from the runtime image:
<https://help.aliyun.com/en/pai/deploy-a-model-service-by-using-a-custom-image>.

## Serverless is a separate image contract

Selecting a public image does not by itself make that image serverless-ready.
Serverless platforms invoke a request handler and control worker startup,
shutdown, health, and request serialization. The current GPU image runs a
long-lived OpenAI-compatible SGLang server and does not implement a RunPod
Serverless handler, SageMaker `serve`/`/ping`/`/invocations`, or another
provider-specific adapter.

Publish a separate serverless image only after implementing and testing the
target platform's handler contract. Do not point a serverless endpoint at this
Pod/VM image and assume the platform will translate requests automatically.

Managed products can impose an additional serving contract. For example,
SageMaker endpoints invoke an image with `serve` and expect `/ping` and
`/invocations`; the generic SGLang image needs a SageMaker adapter before it can
be advertised as SageMaker-compatible:
<https://docs.aws.amazon.com/sagemaker/latest/dg/your-algorithms-inference-code.html>.

Return to the [installation chooser](README.md).
