#!/usr/bin/env bash
# Download and verify the split HF datasets, then prepare the task image.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec python3 "$ROOT/scripts/setup_release.py" "$@"
