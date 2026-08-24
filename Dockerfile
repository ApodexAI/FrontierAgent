FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    SANDBOX_BACKEND=bwrap

RUN apt-get update \
    && apt-get install -y --no-install-recommends bubblewrap ca-certificates libreoffice-core libreoffice-impress \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir uv

WORKDIR /app
COPY pyproject.toml uv.lock README.md LICENSE ./
COPY frontier_agent ./frontier_agent
COPY plugins ./plugins
COPY workflows ./workflows
COPY benchmarks ./benchmarks
COPY apodex ./apodex
# Runtime config the harness reads (provider registry).
COPY config ./config
# tools/ carries the import/symbol gates, so they can be run inside the image.
COPY tools ./tools
COPY docker ./docker
RUN chmod +x /app/docker/entrypoint.sh

RUN uv sync --frozen --extra eval --extra sandbox --extra document-readers

# An unprivileged account for model-run commands, and a writable overlay venv
# for it. Needed by SANDBOX_BACKEND=container, which is the path used by the
# supported Docker CLI/Compose launchers when the runtime does not grant
# bubblewrap its namespaces. In container mode the container is the boundary
# and this uid is what keeps the model out of the harness's own credentials,
# which are readable through /proc/<pid>/environ by the same uid and nobody
# else.
#
# The overlay is a separate venv rather than a writable /opt/venv: a writable
# harness venv would let a model-installed package be imported by the harness
# itself on a later turn. A venv's ``--system-site-packages`` only exposes the
# base interpreter's packages, not packages from the venv that created it, so
# explicitly bridge the baked /opt/venv site-packages with a .pth file. The
# harness stack remains read-only while the tool user can install additions in
# its own overlay.
#
# The .pth holds an ``import site; site.addsitedir(...)`` line, not the bare
# path: ``site.addpackage`` treats a plain path as ``sys.path.append`` and does
# NOT process the .pth files inside the directory it names. Bare-path bridging
# therefore breaks any distribution whose import hook lives in a .pth — a
# namespace package's ``-nspkg.pth``, ``distutils-precedence.pth``, an editable
# install — which would import in the harness and fail in the sandbox, the
# "environment looks broken" class of error the overlay exists to avoid.
# ``addsitedir`` recurses into them, so the sandbox sees the same import graph
# the harness does (including this project itself, read-only, where /app is
# present).
RUN useradd --system --create-home --home-dir /home/agent-tool --shell /bin/bash agent-tool \
    && python -m venv --system-site-packages /opt/tool-venv \
    && tool_site="$(/opt/tool-venv/bin/python -c 'import site; print(site.getsitepackages()[0])')" \
    && baked_site="$(python -c 'import site; print(site.getsitepackages()[0])')" \
    && printf 'import site; site.addsitedir("%s")\n' "$baked_site" > "$tool_site/frontier_agent_baked.pth" \
    && chown -R agent-tool:agent-tool /opt/tool-venv /home/agent-tool

ENTRYPOINT ["/app/docker/entrypoint.sh"]
