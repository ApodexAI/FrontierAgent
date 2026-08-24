# Recommended track: cu13 / SGLang 0.5.17 (NVIDIA driver 580+). The legacy
# cu12 build (driver 525-579) remains available via
# `--build-arg SGLANG_BASE_IMAGE=lmsysorg/sglang:v0.5.10.post1-runtime`.
ARG SGLANG_BASE_IMAGE=lmsysorg/sglang:v0.5.17-runtime
FROM ${SGLANG_BASE_IMAGE}

ARG FRONTIER_AGENT_VERSION=dev
ARG SGLANG_VERSION=unknown
ARG CUDA_TRACK=unknown

LABEL org.opencontainers.image.title="FrontierAgent GPU" \
      org.opencontainers.image.description="FrontierAgent and SGLang in one NVIDIA GPU runtime image" \
      org.opencontainers.image.source="https://github.com/ApodexAI/FrontierAgent" \
      org.opencontainers.image.version="${FRONTIER_AGENT_VERSION}" \
      io.apodex.sglang.version="${SGLANG_VERSION}" \
      io.apodex.cuda.track="${CUDA_TRACK}"

# Keep the agent environment separate from the SGLang system Python. SGLang's
# prebuilt CUDA/PyTorch stack is deliberately left untouched and is always
# launched with /usr/bin/python3 by gpu-entrypoint.sh.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    APODEX_IN_CONTAINER=1 \
    SANDBOX_BACKEND=container \
    FRONTIER_AGENT_WORKSPACE_DIR=/workspace \
    FRONTIER_AGENT_OUTPUTS_DIR=/outputs \
    FRONTIER_AGENT_INPUTS_ROOT=/inputs \
    SGLANG_PYTHON=/usr/bin/python3

# Some upstream runtime tags contain uv's Python wrapper without its binary.
# Force a pinned reinstall so /usr/local/bin/uv exists deterministically.
RUN /usr/bin/python3 -m pip install --no-cache-dir --force-reinstall "uv==0.11.5"

WORKDIR /app
COPY pyproject.toml uv.lock README.md LICENSE ./
COPY frontier_agent ./frontier_agent
COPY plugins ./plugins
COPY workflows ./workflows
COPY benchmarks ./benchmarks
COPY apodex ./apodex
COPY config ./config
COPY tools ./tools
COPY docker ./docker

RUN /usr/local/bin/uv sync --frozen --extra sandbox --extra document-readers \
    && useradd --system --create-home --home-dir /home/agent-tool --shell /bin/bash agent-tool \
    && /opt/venv/bin/python -m venv --system-site-packages /opt/tool-venv \
    && mkdir -p /workspace /outputs /inputs \
    && chown -R agent-tool:agent-tool /opt/tool-venv /home/agent-tool \
    && chmod +x /app/docker/entrypoint.sh /app/docker/gpu-entrypoint.sh

EXPOSE 30000
WORKDIR /workspace

# Preserve NVIDIA's base-image initialization, then select server/tui/shell via
# the FrontierAgent entrypoint. Cloud platforms may override CMD as usual.
ENTRYPOINT ["/opt/nvidia/nvidia_entrypoint.sh", "/app/docker/gpu-entrypoint.sh"]
CMD ["server"]
