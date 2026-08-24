# Install FrontierAgent on macOS

For a detailed Chinese guide and a clone-to-run helper, see
[macOS 中文安装与运行指南](macos.zh-CN.md).

Use macOS for the FrontierAgent terminal application and connect it to a hosted
or remote OpenAI-compatible model endpoint. macOS cannot run the NVIDIA SGLang
images in this repository; an Apple GPU is not an NVIDIA CUDA device.

## Install

```bash
brew install uv

git clone https://github.com/ApodexAI/FrontierAgent.git
cd FrontierAgent
uv sync --python 3.12 --extra dev
cp .env.example .env
```

Configure the endpoint in `.env`:

```dotenv
OPENAI_API_KEY=your-key
OPENAI_BASE_URL=https://your-openai-compatible-endpoint/v1
OPENAI_MODEL=your-model-name
```

Then start either workflow:

```bash
uv run frontier-agent --native --mode react --cwd /path/to/project
uv run frontier-agent --native --mode agent_team --cwd /path/to/project
```

`--native` is explicit here on purpose. **On macOS, an invocation with neither
`--native` nor `--docker` selects the container path whenever a Docker daemon is
running** — which builds `apodex:local` from this repository on first use and
takes several minutes. Without a daemon it falls back to the native runtime and
says so on stderr. Both are supported; which one you get should be your choice
rather than a property of whether Docker happens to be running.

## Native or Docker on the Mac

Docker Desktop is optional. Where the model runs does not change between these
two: both call the hosted endpoint configured above. What changes is where the
agent's own commands run.

`--native` runs them as the current macOS user. Mutable runtime state, dependency
caches, temporary files, and inputs are redirected under
`<project>/.apodex/runtime/native`, and approvals remain in effect, but this is a
convenience boundary and not an OS sandbox. Note that the redirected `HOME` means
the agent does not see `~/.gitconfig`, `~/.ssh` or `~/.config/gh`; see [the note
on native mode and your
credentials](linux.md#native-mode-and-your-git-and-ssh-credentials).

Scientific, plotting, spreadsheet, and document-reader packages are not installed
by default. When a task needs one, the workflow may install that specific package
with `python -m pip install <package>`; native mode redirects it into the
workspace-local Python overlay rather than modifying the CLI venv.

`--docker` re-executes the whole CLI inside the Linux agent image, and the
container is the boundary — the supported way to get a real one on macOS, since
macOS has no bubblewrap:

```bash
brew install --cask docker
docker info
uv run frontier-agent --docker --mode react --cwd /path/to/project
```

The first `--docker` run builds `apodex:local` from this repository and may take
several minutes; later runs reuse the image. The project is mounted at
`/project`, per-session scratch at `/workspace`, and session deliverables at
`/outputs`, which appears on the host under
`<project>/.apodex/runs/<session-id>/outputs/`. Clones, drafts, and other
intermediate files stay under that run's `workspace/` instead of spilling into
the project root.

`APODEX_IMAGE=<tag>` selects a different prebuilt image instead. A tag that is
not present locally is pulled, never built under that name. For Compose and
`docker run` deployments, see [Run FrontierAgent in Docker](docker.md).

`--bwrap` is therefore not available on macOS; it reports that and names these
two paths instead.

Docker Desktop does not turn an Apple GPU into a CUDA GPU. To use a local
Qwen/SGLang server, run it on a remote Linux NVIDIA machine and set
`OPENAI_BASE_URL` to a securely reachable endpoint.

Return to the [installation chooser](README.md).
