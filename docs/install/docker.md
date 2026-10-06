# Run FrontierAgent in Docker

FrontierAgent publishes pre-built `linux/amd64` and `linux/arm64` images to the
GitHub Container Registry. Using them requires no local Python environment and
no system dependencies beyond Docker itself. The default `compose.yaml` uses
that published image.

That package is **private**, so an anonymous pull fails with `unauthorized`.
You need one of:

- a GitHub account or token already authorized for the package, in which case
  run `docker login ghcr.io` first; or
- a local build of this checkout, which is what the commands below do and
  needs no registry access.

Because `compose.yaml` sets `pull_policy: always`, the build path keeps the
`compose.dev.yaml` override on every command — its `pull_policy: build` keeps
the local image in use instead of retrying the registry.

This page covers the CPU agent container. For a **local NVIDIA model server**,
the GPU belongs to a separate SGLang container or process — use
[Docker SGLang on a Linux NVIDIA host](linux-nvidia.md) or
[Native SGLang without nested Docker](linux-nvidia-native.md) instead.

## One-click Compose run

`compose.yaml` marks `.env` as optional, which requires Docker Compose 2.24 or
newer; older versions reject the file outright.

```bash
git clone https://github.com/ApodexAI/FrontierAgent.git
cd FrontierAgent
cp .env.example .env

# With registry access, the published image needs no build:
docker compose run --rm agent
```

Without it, build from this checkout and keep the override on every command:

```bash
git clone https://github.com/ApodexAI/FrontierAgent.git
cd FrontierAgent
cp .env.example .env
docker compose -f compose.yaml -f compose.dev.yaml build

# Interactive CLI
docker compose -f compose.yaml -f compose.dev.yaml run --rm agent

# One-shot agent command
docker compose -f compose.yaml -f compose.dev.yaml run --rm agent \
  -p "explain pyproject.toml"

# Default benchmark evaluation (BrowseComp, one task)
docker compose -f compose.yaml -f compose.dev.yaml run --rm eval
```

Compose writes session records and deliverables to `.apodex/runs/<session-id>/`.
Its named state volume is retained for legacy sessions. Attached inputs are
copied into a separate volume that tools can only read. See
[run artifacts and timestamps](../run-artifacts.md) for the on-disk layout.
The agent receives `.env` through its process environment; Compose masks the
on-disk file inside `/project` so model commands cannot read it. The SGLang and
Transformers overrides also mask their respective env files. Keep any custom
credential file outside the mounted project, since project files are available
to the agent by design.

The convenience helper wraps the same thing:

```bash
./docker/run.sh -p "analyze repository structure"
./docker/run.sh eval --limit 5
```

`run.sh` uses `compose.yaml` on its own, so it needs registry access to the
private package. Keep the `compose.dev.yaml` override instead when building
locally.

## Pin a release or another image

Set `FRONTIER_AGENT_IMAGE` before running Compose:

```bash
FRONTIER_AGENT_IMAGE=ghcr.io/apodexai/frontieragent:latest \
  docker compose run --rm agent -p "explain pyproject.toml"
```

Any image name works here, including one you built and tagged yourself, or one
mirrored to a registry you can reach.

A tag that exists only on this machine is the exception. `compose.yaml` sets
`pull_policy: always`, so Compose would still try to resolve it from a registry.
Pass `--pull never` so it uses the local image:

```bash
FRONTIER_AGENT_IMAGE=frontier-agent:local \
  docker compose run --pull never --rm agent
```

## Direct `docker run`

Compose is the supported path; this is the equivalent for environments that
cannot use it. The environment variables and mounts are not optional — they are
what tells the runtime it is inside a container and where the three sandbox
roots live.

The command below runs `frontier-agent:local`, which you build from this
checkout first, so it needs no registry access:

```bash
docker build -t frontier-agent:local .
```

To use the private published image instead, replace that tag with
`ghcr.io/apodexai/frontieragent:latest` and `docker login ghcr.io` first.

```bash
docker run --rm -it \
  --env-file .env \
  -e APODEX_IN_CONTAINER=1 \
  -e SANDBOX_BACKEND=container \
  -e FRONTIER_AGENT_REQUIRE_TOOL_USER=1 \
  -e FRONTIER_AGENT_WORKSPACE_DIR=/workspace \
  -e APODEX_RUNS_ROOT=/apodex-runs \
  -e APODEX_RUNS_ROOT_PINNED=1 \
  -e APODEX_HOST_RUNS_ROOT="$(pwd)/.apodex/runs" \
  -e APODEX_OUTPUTS_LINK=/outputs \
  -e APODEX_INPUT_STAGING_ROOT=/apodex-inputs \
  -e FRONTIER_AGENT_INPUTS_ROOT=/inputs \
  -v "$(pwd):/workspace" \
  -v /dev/null:/workspace/.env:ro \
  -v "$(pwd)/.apodex/runs:/apodex-runs" \
  -v frontier-agent-inputs:/apodex-inputs \
  -v frontier-agent-inputs:/inputs:ro \
  -v frontier-agent-state:/root/.apodex \
  -v frontier-agent-config:/root/.config/apodex \
  -w /workspace \
  frontier-agent:local \
  -p "explain main workflow"
```

## Cloud server (AWS EC2 / Aliyun ECS)

For a terminal deployment accessed over SSH:

1. Provision an EC2 or ECS Linux instance with Docker and the Compose plugin.
2. Clone this repository and create `.env` from `.env.example`.
3. Launch the container:

```bash
git clone https://github.com/ApodexAI/FrontierAgent.git
cd FrontierAgent
cp .env.example .env
# Edit .env, then — one of:

# With registry access, use the published image:
docker login ghcr.io
docker compose pull agent
docker compose run --rm agent

# Or build this checkout on the instance, no registry access needed:
docker compose -f compose.yaml -f compose.dev.yaml build
docker compose -f compose.yaml -f compose.dev.yaml run --rm agent
```

The container itself is disposable; Compose persists sessions, configuration,
attachments, and deliverables in volumes or the checked-out workspace. Rebuild
(or `docker compose pull agent`) to upgrade. This is an interactive SSH/TUI
deployment, not a long-running HTTP service.

## Build from the current checkout

The development override builds this checkout instead of using the published
image, which is the only path that needs no registry access:

```bash
cp .env.example .env
docker compose -f compose.yaml -f compose.dev.yaml build
docker compose -f compose.yaml -f compose.dev.yaml run --rm agent --version
docker compose -f compose.yaml -f compose.dev.yaml run --rm eval \
  --benchmark browsecomp --limit 1 --out /app/results/smoke
```

Once the image is built, keep the `compose.dev.yaml` override on the run
command too. `compose.yaml` sets `pull_policy: always`, so it reaches for the
private registry on every start and fails for anyone without access — the
override's `pull_policy: build` keeps the local image in use.

`docker build -t apodex:local .` builds the same image under the name that the
macOS `--docker` path expects. See [Contributing](../../CONTRIBUTING.md) for the
rest of the development loop.

Return to the [installation chooser](README.md).
