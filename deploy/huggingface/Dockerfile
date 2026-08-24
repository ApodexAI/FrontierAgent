# FrontierAgent — Hugging Face Docker Space
#
# Build context is the REPOSITORY ROOT, not this directory:
#     docker build -f deploy/huggingface/Dockerfile -t frontier-agent-hf-space .
#
# Deliberately NOT in this image:
#   * model weights — the agent talks to an OpenAI-compatible endpoint over
#     HTTP (OPENAI_BASE_URL); nothing is served locally.
#   * secrets — OPENAI_API_KEY and friends arrive at runtime as Space Secrets.
#   * bubblewrap / GPU / inference server — the demo executes no model-authored
#     commands at all (see DEMO_ALLOWED_TOOLS below).
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_LINK_MODE=copy \
    PATH=/opt/venv/bin:$PATH

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir uv

# ── dependencies (cached independently of the source) ────────────────────
WORKDIR /app
COPY pyproject.toml uv.lock README.md LICENSE ./
COPY frontier_agent ./frontier_agent
COPY plugins ./plugins
COPY workflows ./workflows
COPY benchmarks ./benchmarks
COPY apodex ./apodex
COPY config ./config
COPY deploy ./deploy
RUN uv sync --frozen --extra hf-space --extra document-readers

# ── the app user ─────────────────────────────────────────────────────────
# Hugging Face Spaces expect uid 1000. /app stays root-owned and read-only to
# this user, so the running agent cannot rewrite the service's own source; all
# mutable state lives under $HOME.
RUN useradd --create-home --uid 1000 --shell /bin/bash user \
    && mkdir -p /home/user/run /home/user/.cache \
    && chown -R user:user /home/user

USER user
ENV HOME=/home/user \
    XDG_CACHE_HOME=/home/user/.cache \
    HF_HOME=/home/user/.cache/huggingface \
    MPLCONFIGDIR=/home/user/.cache/matplotlib \
    GRADIO_ALLOW_FLAGGING=never \
    GRADIO_ANALYTICS_ENABLED=False
# DEMO_RUNTIME_ROOT is intentionally unset: the app picks /data when Hugging
# Face persistent storage is mounted and writable, else $HOME (see
# config._default_runtime_root). Set it as a Space Variable to override.

# ``native`` makes the react prompt name this session's real directories, so
# each visitor gets their own workspace/outputs. The tool policy — not a
# filesystem namespace — is what contains the agent here: every
# command-executing tool is denied (see deploy/huggingface/security.py).
ENV SANDBOX_BACKEND=native \
    BASH_ALLOWLIST_MODE=enforce \
    FRONTIER_AGENT_TOOL_USER=off

# Conservative demo defaults. Every one is overridable as a Space *Variable*.
ENV DEMO_WORKFLOW=react \
    DEMO_PUBLIC_MODE=true \
    DEMO_MAX_TURNS=24 \
    DEMO_MAX_OUTPUT_TOKENS=4096 \
    DEMO_TASK_TIMEOUT_SECONDS=600 \
    DEMO_MAX_CONCURRENCY=1 \
    DEMO_QUEUE_SIZE=4 \
    DEMO_SESSION_TTL_SECONDS=3600 \
    HF_MODEL_ID=apodex/Apodex-1.1-mini

# The runtime's own bootstrap creates ./data relative to the working
# directory, so the app runs from a writable dir rather than the read-only
# source tree. The FrontierAgent packages are importable regardless: the
# workflow loader resolves the project root from its own __file__.
WORKDIR /home/user/run

EXPOSE 7860

# Gradio binds 0.0.0.0:7860 inside the container; publish it to 127.0.0.1 on a
# development host. ``python -m`` keeps the app as PID 1 so SIGTERM reaches its
# handler directly (deploy/huggingface/app.py:_install_signal_handlers).
CMD ["python", "-m", "deploy.huggingface.app"]
