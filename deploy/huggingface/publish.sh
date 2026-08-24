#!/usr/bin/env bash
# Assemble a Hugging Face Docker Space working tree from this repository.
#
# Hugging Face builds from a Dockerfile at the *root* of the Space repo, while
# this repo's root Dockerfile is the CLI image. So a Space tree is not just a
# copy of the checkout: the Space Dockerfile has to become the root one, and
# README.space.md has to become the Space's README (its YAML front matter is
# what tells HF the SDK and port).
#
# This script only *prepares* the tree and prints the push commands. It never
# pushes and never creates a Space, so it cannot publish anything — or leak a
# credential — on its own.
#
#   ./deploy/huggingface/publish.sh /tmp/space-tree
#
set -euo pipefail

TARGET="${1:-}"
if [[ -z "$TARGET" ]]; then
  echo "usage: $0 <target-directory>" >&2
  echo "example: $0 /tmp/frontier-agent-space" >&2
  exit 2
fi

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"

if [[ -e "$TARGET" && -n "$(ls -A "$TARGET" 2>/dev/null)" ]]; then
  echo "error: $TARGET exists and is not empty; refusing to overwrite" >&2
  exit 2
fi
mkdir -p "$TARGET"

echo "assembling a Space tree in $TARGET"

# Everything the runtime imports, plus the app itself. Local benchmark datasets,
# generated tasks, results, tests, and the git history are excluded below: a
# Space build does not need them, and some corpora are license-gated.
for path in \
  pyproject.toml uv.lock LICENSE \
  frontier_agent plugins workflows benchmarks apodex config deploy
do
  if [[ ! -e "$ROOT/$path" ]]; then
    echo "error: missing $path in $ROOT" >&2
    exit 1
  fi
  cp -R "$ROOT/$path" "$TARGET/"
done

# Never ship credentials, caches, or a virtualenv.
find "$TARGET" -name '__pycache__' -type d -prune -exec rm -rf {} + 2>/dev/null || true
find "$TARGET" -name '*.pyc' -delete 2>/dev/null || true
find "$TARGET" -name '.env' -o -name '.env.*' -not -name '*.example' \
  | while read -r leaked; do rm -f "$leaked"; done
rm -rf \
  "$TARGET/benchmarks/public/datasets" \
  "$TARGET/benchmarks/public/results" \
  "$TARGET/benchmarks/public/tasks-generated" \
  "$TARGET/benchmarks/datasets" \
  "$TARGET/benchmarks/results" \
  "$TARGET/benchmarks/tasks-generated" \
  "$TARGET/.venv"

# The Space entry points: Dockerfile and README must be at the root.
cp "$HERE/Dockerfile" "$TARGET/Dockerfile"
cp "$HERE/README.space.md" "$TARGET/README.md"

# The Dockerfile copies from the build context root, which is now $TARGET, so
# its COPY paths already line up. Keep the ignore list too.
cp "$ROOT/.dockerignore" "$TARGET/.dockerignore" 2>/dev/null || true

# Fail loudly rather than shipping a tree that cannot build.
missing=0
for required in Dockerfile README.md pyproject.toml uv.lock \
                deploy/huggingface/app.py frontier_agent workflows plugins
do
  [[ -e "$TARGET/$required" ]] || { echo "error: $required missing from tree" >&2; missing=1; }
done
# A credential would reach the tree as a dotenv file, not as prose — matching
# "OPENAI_API_KEY=" anywhere would just flag the docs that explain the variable.
leaked_env="$(find "$TARGET" \( -name '.env' -o -name '.env.*' \) \
                ! -name '*.example' -print 2>/dev/null || true)"
if [[ -n "$leaked_env" ]]; then
  echo "error: the assembled tree still contains dotenv file(s):" >&2
  echo "$leaked_env" >&2
  missing=1
fi
[[ "$missing" -eq 0 ]] || exit 1

cat <<EOF

Space tree ready: $TARGET

Next, publish it (this script intentionally does not):

  cd "$TARGET"
  git init -q && git add -A && git commit -qm "FrontierAgent react demo"
  git remote add space https://huggingface.co/spaces/<org>/<space-name>
  git push --force space HEAD:main

Then set, in the Space's Settings → Variables and secrets:

  Variables : OPENAI_BASE_URL, OPENAI_MODEL   (see deploy/huggingface/README.md §3)
  Secrets   : OPENAI_API_KEY, SERPER_API_KEY  (see deploy/huggingface/README.md §4)

Wait for Building → Running, then open the Space URL.
EOF
