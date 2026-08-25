#!/usr/bin/env bash
# Build the redistributable FrontierChallenge task image.
#
#   ./shared_images/build.sh open    # 81 of the 97 tasks   (~16.8 GB)
#
# This repository does not build or distribute an ORCA image. The 16 ORCA
# tasks require a separately licensed, user-created local runtime; see
# docs/providers/orca.md.
set -euo pipefail

root=$(cd "$(dirname "$0")" && pwd)
platform=${PLATFORM:-linux/amd64}
open_image=${OPEN_IMAGE:-frontierchallenge/cpu-open:2026.08}

target=${1:-open}
case "$target" in
  open) ;;
  *)
    echo "usage: $0 [open]" >&2
    echo "ORCA runtimes are user-supplied and are never built here; see docs/providers/orca.md" >&2
    exit 2
    ;;
esac

# Validate the toolchain before starting a build that takes tens of minutes.
# `docker info` is not a daemon test -- it exits 0 with only the client
# installed -- so probe with something that needs the daemon to answer.
if ! docker ps >/dev/null 2>&1; then
  echo "error: no Docker daemon is answering here." >&2
  echo "       Start Docker, or build on a host that has it." >&2
  exit 1
fi

# These Dockerfiles carry a `# syntax=` directive and are built with
# --progress, both of which need BuildKit, which needs the buildx plugin.
# Without it the failure is a bare "unknown flag: --progress" tens of lines
# into docker's usage text, which says nothing about the actual cause.
if ! docker buildx version >/dev/null 2>&1; then
  cat >&2 <<'EOF'
error: the docker buildx plugin is missing, and these images need BuildKit.

  Recent Docker Desktop and docker-ce ship buildx already. To install it by
  hand on Linux (no root needed):

    mkdir -p ~/.docker/cli-plugins
    curl -sSL -o ~/.docker/cli-plugins/docker-buildx \
      https://github.com/docker/buildx/releases/download/v0.20.1/buildx-v0.20.1.linux-amd64
    chmod +x ~/.docker/cli-plugins/docker-buildx
    docker buildx version
EOF
  exit 1
fi

build() {
  local dockerfile="$1" image="$2"
  shift 2
  echo "== building $image from $(basename "$dockerfile") =="
  DOCKER_BUILDKIT=1 docker build \
    --platform "$platform" \
    --progress plain \
    -f "$dockerfile" \
    -t "$image" \
    "$@" \
    "$root"
}

build "$root/Dockerfile.open" "$open_image"

echo
echo "Built:"
docker images --format '  {{.Repository}}:{{.Tag}}  {{.Size}}' | grep frontierchallenge || true
echo
echo "Verify with: ./shared_images/verify.sh"
