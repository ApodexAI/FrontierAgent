# FrontierAgent TUI quickstart for macOS and Linux

[中文](tui-endpoint-quickstart.zh-CN.md) · [Documentation index](../README.md)

This tutorial is for developers who want to run the FrontierAgent TUI on
macOS or Linux and connect it to an existing OpenAI-compatible LLM endpoint
configured in `.env`. It does not deploy a model, require an NVIDIA GPU, or
require Docker.

## What you will run

```text
Your macOS or Linux terminal
├── FrontierAgent full-screen TUI
├── target code repository (selected by --cwd)
└── HTTPS → an existing OpenAI-compatible LLM endpoint
```

FrontierAgent and the model service are independent. Only the agent and its
tools run on your machine; the endpoint configured in `.env` performs model
inference.

## 1. Prerequisites

You need only:

- a macOS or Linux terminal;
- Git;
- network access to GitHub, Python package sources, and the LLM endpoint;
- the endpoint's API key, base URL, and model ID.

The project uses `uv` to manage Python 3.12 and its dependencies. The launcher
installs `uv` from Astral's official site if it is missing. If your organization
disallows automatic downloads, ask an administrator to install
[`uv`](https://docs.astral.sh/uv/getting-started/installation/) first.

## 2. Clone the project and configure `.env`

```bash
git clone https://github.com/ApodexAI/FrontierAgent.git
cd FrontierAgent
cp .env.example .env
chmod 600 .env
```

Open `.env` in an editor and set at least these three values:

```dotenv
OPENAI_API_KEY=your-api-key
OPENAI_BASE_URL=https://your-endpoint.example/v1
OPENAI_MODEL=your-model-id
```

Configuration rules:

- the endpoint must support the OpenAI Chat Completions API;
- `OPENAI_BASE_URL` normally ends in `/v1`; do not include
  `/v1/chat/completions`;
- `OPENAI_MODEL` must exactly match a model ID accepted by the endpoint;
- `.env` is ignored by Git. Never commit, screenshot, or share it;
- environment variables take precedence over `.env`, so existing shell values
  with the same names override the file.

Search and document settings such as `SERPER_API_KEY` and `JINA_API_KEY` are
optional and are not needed for the first TUI run.

## 3. Start the TUI

Run these commands from the cloned `FrontierAgent` directory. Replace
`/absolute/path/to/your-project` with the absolute path of the repository the
agent should work on.

### macOS

```bash
./scripts/run-macos.sh \
  --mode react \
  --cwd /absolute/path/to/your-project
```

### Linux

```bash
./scripts/run-linux.sh \
  --mode react \
  --cwd /absolute/path/to/your-project
```

The launcher installs Python 3.12 and the required dependencies, validates
`.env`, and opens the full-screen TUI. It does not install or start a local
LLM. The first run downloads dependencies; later runs reuse the cache.

Default runtime selection is platform-specific:

- Linux uses the workspace-local native runtime by default;
- macOS selects automatically: it uses Docker when the daemon is available and
  falls back to native mode otherwise;
- pass `--native` to explicitly require native mode;
- on Linux, pass `--bwrap` when bubblewrap is installed and supported for a
  lightweight filesystem boundary;
- pass `--docker` when a reachable Docker daemon is available for container
  isolation.

Linux runtime selection can be summarized as follows:

| Invocation | Behavior |
|---|---|
| No runtime option | Defaults to native; does not probe or switch to Docker/bubblewrap |
| `--native` | Explicitly requires native mode |
| `--bwrap` | Explicitly requires bubblewrap; startup fails if unavailable |
| `--docker` | Explicitly requires Docker; startup fails if unavailable |

Explicit isolation choices are fail-closed. If bubblewrap or Docker is
unavailable, FrontierAgent exits with an error instead of silently falling back
to native mode. This prevents a developer from mistakenly believing the task
is still isolated.

To install bubblewrap on Debian/Ubuntu and start FrontierAgent:

```bash
sudo apt-get update
sudo apt-get install -y bubblewrap
./scripts/run-linux.sh --bwrap --mode react --cwd /absolute/path/to/your-project
```

Installing the binary alone may not be sufficient: the host or cloud platform
must also permit Linux user namespaces. The launcher checks actual usability.

For example, to explicitly use Docker on either platform:

```bash
# Docker must be installed and running
./scripts/run-macos.sh --docker --mode react --cwd /absolute/path/to/your-project
./scripts/run-linux.sh --docker --mode react --cwd /absolute/path/to/your-project
```

Docker is only an isolation boundary for agent commands. FrontierAgent still
calls the remote LLM endpoint from `.env`; it does not deploy a model locally.

Add `--setup-only` to either command if you want to install dependencies and
validate configuration without opening the TUI.

## 4. Run your first task

When the TUI opens, enter this read-only task in the prompt and press Enter:

```text
Read this repository's README and project configuration. Summarize its purpose, entry points, and local test commands. Analyze only; do not modify files.
```

Start with `react`: one stateful agent performs code reading, command execution,
and file work in sequence. For work that genuinely splits into independent
subtasks, exit and start Agent Team instead:

```bash
# macOS
./scripts/run-macos.sh --mode agent_team --cwd /absolute/path/to/your-project

# Linux
./scripts/run-linux.sh --mode agent_team --cwd /absolute/path/to/your-project
```

`agent_team` uses a coordinator and sub-agents and will normally consume more
endpoint tokens.

## 5. Essential TUI controls

| Control | Purpose |
|---|---|
| `/help` or `F1` | Show complete help |
| `/config` | Show redacted provider, model, and endpoint diagnostics |
| `/mode react` | Switch to the single-agent workflow |
| `/mode agent_team` | Switch to Agent Team |
| `/plan` | Investigate and propose a plan; edits stay locked until approval |
| `/attach <path>` | Add a read-only file or directory input |
| `/log` | Show the current run's trace path |
| `/revert` | Undo changes recorded by file-editing tools in this session |
| `/resume` | Pick and resume a saved session inside the TUI |
| `Ctrl-C` | Interrupt the current task |
| `/exit` | Exit the TUI |

Writes show a diff and request approval by default. Do not add `--yes` on your
first run. Native mode is not an operating-system sandbox: approved commands
run with the permissions of your current macOS or Linux user. Select `--docker`
explicitly when you want a container isolation boundary.

## 6. Files and sessions

For a native run started with `--cwd /absolute/path/to/your-project`, state is
stored under:

```text
/absolute/path/to/your-project/.apodex/
├── runs/<session-id>/       # trace, engine log, checkpoint, and outputs
└── runtime/native/          # workspace-local caches, temporary files, and dependencies
```

List resumable sessions:

```bash
uv run frontier-agent --cwd /absolute/path/to/your-project --resume
```

Resume a specific session ID:

```bash
uv run frontier-agent \
  --cwd /absolute/path/to/your-project \
  --resume SESSION_ID
```

## 7. Troubleshooting

### The full-screen TUI does not open

The TUI requires both stdin and stdout to be attached to a normal TTY. Do not
pipe or redirect the launcher, use `TERM=dumb`, or add `--print`, `--no-tui`,
`--no-color`, or `--theme mono`.

### Required `.env` values are missing

Make sure `OPENAI_API_KEY`, `OPENAI_BASE_URL`, and `OPENAI_MODEL` are non-empty
in `FrontierAgent/.env`, then rerun the launcher from the FrontierAgent root.

### 401 or 403

The API key is invalid, expired, or lacks access to the selected model. Correct
`.env` and restart FrontierAgent.

### 404 or model not found

Check that the base URL normally includes `/v1` but not `/chat/completions`, and
that the model ID exactly matches one exposed by the endpoint.

### Timeout or connection failure

Check endpoint reachability from this machine along with VPN, proxy, DNS, TLS
certificate, and firewall policies. FrontierAgent's startup preflight validates
local configuration only; it does not make a network request to test the key.

### `uv` is still not found after installation

Open a new terminal, or follow the official `uv` instructions to add its binary
directory to `PATH`. The launchers also check `$HOME/.local/bin/uv` and
`$HOME/.cargo/bin/uv` directly.

## 8. Shortest command for daily use

After the first setup, continue to launch from the FrontierAgent repository:

```bash
# macOS
./scripts/run-macos.sh --cwd /absolute/path/to/your-project

# Linux
./scripts/run-linux.sh --cwd /absolute/path/to/your-project
```

Next, read the [TUI user guide](../tui-user-guide.md) for the three sidebar tabs,
Space previews, approvals, and Agent Team asynchronous intervention.

For the lower-level CLI/TUI, approval, attachment, theme, and session reference,
see [`apodex/README.md`](../../apodex/README.md). For platform details, see the
[macOS guide](macos.md) and [Linux guide](linux.md).
