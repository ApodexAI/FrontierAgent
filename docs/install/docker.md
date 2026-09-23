# Run FrontierAgent in Docker

The public Docker workflow builds FrontierAgent from this checkout for your
host architecture (`linux/amd64` or `linux/arm64`). It requires Docker and
network access to download build dependencies, but no local Python environment.
The organization's GHCR package is private. The default `compose.yaml` builds
`frontieragent:local` and reuses it with `pull_policy: never`.

This page covers the CPU agent container. For a **local NVIDIA model server**,
the GPU belongs to a separate SGLang container or process — use
[Docker SGLang on a Linux NVIDIA host](linux-nvidia.md) or
[Native SGLang without nested Docker](linux-nvidia-native.md) instead.

## Build and run with Compose

`compose.yaml` marks `.env` as optional, which requires Docker Compose 2.24 or
newer; older versions reject the file outright.

```bash
git clone https://github.com/ApodexAI/FrontierAgent.git
cd FrontierAgent
cp .env.example .env
docker compose build

# Interactive CLI
docker compose run --rm agent

# One-shot agent command
docker compose run --rm agent -p "explain pyproject.toml"

# Default benchmark evaluation (BrowseComp, one task)
docker compose run --rm eval
```

Compose writes session records and deliverables to `.apodex/runs/<session-id>/`.
Its named state volume is retained for legacy sessions. Attached inputs are
copied into a separate volume that tools can only read. See
[run artifacts and timestamps](../run-artifacts.md) for the on-disk layout.

After building, the convenience helper reuses that local image:

```bash
./docker/run.sh -p "analyze repository structure"
./docker/run.sh eval --limit 5
```

## Pin a release or another image

Users with access to the private GHCR package can explicitly log in and pull
an image. Set the same `FRONTIER_AGENT_IMAGE` when running Compose, which then
uses the downloaded image without pulling or building it. Do not run
`docker compose build` with this override: that would replace the local tag.

```bash
docker login ghcr.io
docker pull ghcr.io/apodexai/frontieragent:latest
FRONTIER_AGENT_IMAGE=ghcr.io/apodexai/frontieragent:latest \
  docker compose run --rm agent -p "explain pyproject.toml"
```

## Direct `docker run`

First run `docker compose build`. Compose is the supported path; this is the equivalent for environments that
cannot use it. The environment variables and mounts are not optional — they are
what tells the runtime it is inside a container and where the three sandbox
roots live.

```bash
docker run --rm -it \
  --env-file .env \
  -e APODEX_IN_CONTAINER=1 \
  -e SANDBOX_BACKEND=container \
  -e FRONTIER_AGENT_WORKSPACE_DIR=/workspace \
  -e APODEX_RUNS_ROOT=/apodex-runs \
  -e APODEX_RUNS_ROOT_PINNED=1 \
  -e APODEX_HOST_RUNS_ROOT="$(pwd)/.apodex/runs" \
  -e APODEX_OUTPUTS_LINK=/outputs \
  -e APODEX_INPUT_STAGING_ROOT=/apodex-inputs \
  -e FRONTIER_AGENT_INPUTS_ROOT=/inputs \
  -v "$(pwd):/workspace" \
  -v "$(pwd)/.apodex/runs:/apodex-runs" \
  -v frontier-agent-inputs:/apodex-inputs \
  -v frontier-agent-inputs:/inputs:ro \
  -v frontier-agent-state:/root/.apodex \
  -v frontier-agent-config:/root/.config/apodex \
  -w /workspace \
  frontieragent:local \
  -p "explain main workflow"
```

## Cloud server (AWS EC2 / Aliyun ECS)

For a terminal deployment accessed over SSH:

1. Provision an EC2 or ECS Linux instance with Docker and the Compose plugin.
2. Clone this repository and create `.env` from `.env.example`.
3. Build and launch the local container:

```bash
git clone https://github.com/ApodexAI/FrontierAgent.git
cd FrontierAgent
cp .env.example .env
# Edit .env, then:
docker compose build
docker compose run --rm agent
```

The container itself is disposable; Compose persists sessions, configuration,
attachments, and deliverables in volumes or the checked-out workspace. Pull the
source updates and rebuild with `docker compose build` to upgrade. This is an interactive SSH/TUI deployment, not a
long-running HTTP service.

## Build from the current checkout

The default build already uses the checkout. For development, the override
forces a rebuild on each launch:

```bash
cp .env.example .env
docker compose -f compose.yaml -f compose.dev.yaml build
docker compose -f compose.yaml -f compose.dev.yaml run --rm agent --version
docker compose -f compose.yaml -f compose.dev.yaml run --rm eval \
  --benchmark browsecomp --limit 1 --out /app/results/smoke
```

`docker build -t apodex:local .` builds the same image under the name that the
macOS `--docker` path expects. See [Contributing](../../CONTRIBUTING.md) for the
rest of the development loop.

Return to the [installation chooser](README.md).
