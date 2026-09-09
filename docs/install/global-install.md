# Install once and launch from any project

[Documentation index](../README.md) · [Installation chooser](README.md)

This page is for people who want to run `frontier-agent` the way they run any
other command-line tool. Install it once with `uv`. Open a terminal in a
project. Run `frontier-agent`. The TUI opens against that directory.

The installation and its Python dependencies live in uv's tool directory. They
do not depend on a repository checkout and they do not touch the projects you
run the agent in. Everything below was verified with a wheel built from this
repository and installed with the commands shown. Install directly from this
repository, or from a local clone of it.

The clone-and-`uv sync` workflow in the [quickstart](tui-endpoint-quickstart.md)
keeps working unchanged. Use it when you develop FrontierAgent itself.

## 1. Install the tool

You need Git and [uv](https://docs.astral.sh/uv/getting-started/installation/).
uv downloads a Python 3.12 for the tool if the machine has none.

```bash
uv tool install --python 3.12 git+https://github.com/ApodexAI/FrontierAgent.git
```

From a local clone instead:

```bash
uv tool install --python 3.12 /path/to/FrontierAgent
```

Either command installs two executables, `frontier-agent` and its
compatibility alias `apodex`, into uv's tool bin directory.

### If the command is not found

uv prints a warning when its bin directory is not on `PATH`. Fix it once:

```bash
uv tool update-shell
```

Then open a new terminal. To see the directory it is talking about, run
`uv tool dir --bin` and add it to `PATH` yourself if you prefer.

## 2. Configure the endpoint once

FrontierAgent is a bring-your-own-key tool. There is no login command and the
TUI never asks for or displays a key. Credentials come from environment
variables. A globally installed tool has no `.env` next to it, so it also reads
one optional user file:

```text
$XDG_CONFIG_HOME/apodex/env        # $HOME/.config/apodex/env when XDG_CONFIG_HOME is unset
```

Create it with the same three values the quickstart puts into `.env`. The
commands below respect `XDG_CONFIG_HOME` and fall back to `~/.config`:

```bash
config_dir="${XDG_CONFIG_HOME:-$HOME/.config}/apodex"
mkdir -p "$config_dir"
cat > "$config_dir/env" <<'EOF'
OPENAI_API_KEY=your-key
OPENAI_BASE_URL=https://your-openai-compatible-endpoint/v1
OPENAI_MODEL=your-model-name
EOF
chmod 600 "$config_dir/env"
```

Optional web tools take `SERPER_API_KEY` and `JINA_API_KEY` in the same file.
`APODEX_ENV_FILE=/path/to/file` points the CLI at a different file.

Rules for the file:

- plain `KEY=value` lines, comments with `#`;
- values are read literally. `${VAR}` is not expanded, so a value can never
  resolve against something that happens to be exported;
- blank values are ignored;
- the CLI warns when the file is readable by other users and tells you the
  `chmod` to run;
- the file is optional. Exported variables alone are enough.

### Precedence

When the same variable is set in several places, the first match wins:

| Priority | Source |
|---|---|
| 1 | explicit CLI options such as `--model` and `--max-tokens` |
| 2 | variables exported in the shell |
| 3 | `.env` in the directory you launch from, or the nearest parent that has one |
| 4 | the user file above |

Row 3 is the behaviour the checkout workflow has always had, and it is why a
project can pin its own `OPENAI_MODEL` in a local `.env` while the key stays in
your user file. `.env` files are only looked up from the launch directory, not
from a directory given with `--cwd`.

### A key stays with its endpoint

If the user file defines `OPENAI_API_KEY` and `OPENAI_BASE_URL` together, the
CLI treats them as a pair. When a project `.env` or an exported variable sets
`OPENAI_BASE_URL` to a different endpoint but provides no key, the key from the
user file is withheld. Startup then reports that the key is missing and names
the file and the variable, without printing either value. Add the key next to
that endpoint, or remove the override. The same rule applies to every
`<PREFIX>_API_KEY` / `<PREFIX>_BASE_URL` pair. A key defined on its own, with
no endpoint next to it, is applied wherever the CLI points.

## 3. Launch from a project

```bash
cd /path/to/project

frontier-agent                          # Stateful ReAct, full-screen TUI
frontier-agent --mode agent_team        # coordinator plus parallel sub-agents
frontier-agent -p "explain src/main.py" # one-shot, prints, exits
frontier-agent --cwd /other/project     # another project without leaving this shell
frontier-agent --resume                 # list this project's saved sessions
```

Run records, traces, and deliverables stay under `<project>/.apodex/`, exactly
as they do for a checkout launch. See
[run artifacts and timestamps](../run-artifacts.md).

On Linux the agent's commands run in the workspace-local native runtime by
default. Native mode is not an operating-system sandbox. Approved commands run
with your user's permissions, and the `python3` the tools see is the tool's own
Python environment.

## 4. macOS and Docker

On macOS the CLI prefers a container whenever a Docker daemon is reachable, and
falls back to native mode when it is not. That preference is unchanged for a
global install. What changes is that the installed tool carries no
`Dockerfile`, so it cannot build the `apodex:local` image by itself. Do one of
these once:

```bash
# a) build the image from a clone; later launches reuse it
git clone https://github.com/ApodexAI/FrontierAgent.git
docker build -t apodex:local FrontierAgent

# b) or let frontier-agent build from a clone when the image is missing
export APODEX_BUILD_CONTEXT=/path/to/FrontierAgent

# c) or name an image you are already able to pull
export APODEX_IMAGE=registry.example/your-org/frontieragent:tag
```

Without one of these, a launch that would have entered the container stops and
prints the same three options plus `--native`. It does not fall back to native
mode on its own, because you were promised a container. Pass `--native` when
you want the workspace-local runtime instead:

```bash
frontier-agent --native
```

The image this repository publishes to `ghcr.io/apodexai/frontieragent` is
private. Pulling it needs a GitHub account or token that is already authorized
for that package. Signing in with `docker login ghcr.io` does not grant that
access by itself. Build from source, options a or b, unless you have it.

Inside the container the CLI sees the values you configured. Variables from the
user file, the launch directory `.env`, and the exported environment are
forwarded by name, so the value itself never appears on a command line.

## 5. Update and uninstall

```bash
uv tool install --reinstall --python 3.12 git+https://github.com/ApodexAI/FrontierAgent.git
uv tool uninstall frontier-agent
```

`--reinstall` implies uv's `--refresh`, so the Git source is fetched again
rather than served from the cache. For a local clone, pull it and run the same
command with the clone path.

Uninstalling removes the tool environment only. Your user file under
`apodex/` in your config directory, the session history under `~/.apodex/`,
and each project's `.apodex/` directory stay where they are until you delete
them.

## 6. Developing FrontierAgent

Contributors keep the checkout workflow from [CONTRIBUTING](../../CONTRIBUTING.md):

```bash
git clone https://github.com/ApodexAI/FrontierAgent.git
cd FrontierAgent
uv sync --python 3.12 --extra dev
cp .env.example .env
uv run frontier-agent --cwd /path/to/project
```

Return to the [installation chooser](README.md).
