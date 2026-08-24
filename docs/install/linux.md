# Install FrontierAgent on Linux

Use this guide when FrontierAgent calls a hosted or already-running
OpenAI-compatible endpoint. If this machine must also serve a model on an
NVIDIA GPU, choose the [Docker](linux-nvidia.md) or
[native](linux-nvidia-native.md) GPU guide instead.

## Native quick start

After cloning the repository, the bootstrap helper installs `uv` and managed
Python 3.12 when needed, configures the endpoint, and starts the native runtime:

```bash
./scripts/run-linux.sh

# Alternatives with explicit isolation requirements:
./scripts/run-linux.sh --bwrap
./scripts/run-linux.sh --docker
```

It does not install or run a local LLM. Use `--setup-only` for provisioning
without starting the TUI, and put additional CLI arguments after `--`, for
example `./scripts/run-linux.sh -- --theme mono`.

For manual installation, install Python 3.12, Git, and
[uv](https://docs.astral.sh/uv/), then:

```bash
git clone https://github.com/ApodexAI/FrontierAgent.git
cd FrontierAgent
uv sync --python 3.12 --extra dev
cp .env.example .env
```

Configure `.env`:

```dotenv
OPENAI_API_KEY=your-key
OPENAI_BASE_URL=https://your-openai-compatible-endpoint/v1
OPENAI_MODEL=your-model-name
```

Run:

```bash
uv run frontier-agent --mode react --cwd /path/to/project
uv run frontier-agent --mode agent_team --cwd /path/to/project
```

The default native runtime stores run records below `<project>/.apodex/runs`
and caches, temporary files, dependencies, and staged inputs below
`<project>/.apodex/runtime/native`. Commands still run with the current user's
permissions, so review approvals carefully on untrusted repositories.

## Native mode and your git and ssh credentials

Native mode is the default on Linux, and it redirects `HOME` and the XDG
directories into `<project>/.apodex/runtime/native` so a task cannot fill your
real home with caches. One consequence is worth knowing before the agent's first
commit: the commands it runs **do not see `~/.gitconfig`, `~/.ssh`, or
`~/.config/gh`**. `git commit` fails with `Author identity unknown`, and a push
over ssh finds no key.

This matches how the bubblewrap and container boundaries behave — none of them
expose your home directory — so it is deliberate rather than an oversight. When
you do want the agent committing on your behalf, give the repository its own
identity, which lives in the workspace and is therefore visible:

```bash
git -C /path/to/project config user.name "Your Name"
git -C /path/to/project config user.email "you@example.com"
```

Prefer that over widening the boundary. Pushing is best left to you, outside the
agent's session.

## Optional isolation

Use `--docker` when a Docker daemon is available. For a lighter Linux-only
boundary, install bubblewrap and select it explicitly:

```bash
sudo apt-get update
sudo apt-get install -y bubblewrap
uv sync --python 3.12 --extra sandbox --extra dev
uv run frontier-agent --bwrap --mode react --cwd /path/to/project
```

Docker access is root-equivalent on a conventional daemon. Bubblewrap depends
on usable Linux user namespaces and may be disabled by a host or provider.

## Windows and WSL2

Native Windows Python is not currently in the release matrix. Install WSL2,
clone the repository inside its Linux filesystem, and follow this guide from
the WSL shell. Docker Desktop is optional; do not assume PowerShell paths or
commands are interchangeable with the Linux examples.

Return to the [installation chooser](README.md).
