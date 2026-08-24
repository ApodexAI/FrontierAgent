# FrontierAgent — a terminal-native agent

Command: `frontier-agent` (compatibility alias: `apodex`; module: `python -m apodex`).

Point it at a local repository and give it work in plain language: Stateful
ReAct reads, searches, edits files, runs commands, and iterates on failures.
Agent Team adds a coordinator, task board, parallel sub-agents, report collection,
and synthesis.

Thinking and tool calls stream as they happen. Anything that writes shows a
unified diff and waits for approval. Every action lands in a local JSONL trace,
every file write is snapshotted so `/revert` can undo the whole session, and
`--resume` picks a session back up after Ctrl-C.

## What it reuses

| Reused | From |
|---|---|
| The ReAct engine (loop, streaming, compaction, guardrails) | `frontier_agent.core.runtime.loop.agent_loop.run_agent_loop` |
| Observer contract (streaming, approval, path rewriting) | `frontier_agent.core.loop_types` |
| Robustness observers | `frontier_agent.components.observers` (`TextRepetitionGuard`, `LeakedToolCallRetryObserver`) |
| LLM binding, streaming, retries, and response normalization | `frontier_agent.core.runtime.loop.llm_client` |
| Workflow prompts and profiles | `workflows.stateful_react_agent` and `workflows.agent_team` |
| Editing and web tools | `plugins.tools` (`write_file`, `file_editor_*`, `web_search`, `web_fetch`) |

`bash`, `read_file`, `grep_search`, `glob_search` and `delete_file` are
reimplemented in `local_tools.py`: they work against your actual working
directory and skip whatever your `.gitignore` declares, which the shared
sandbox-backed variants are not built for.

## Terminal UI stack

The UI stack is intentionally frozen to two layers:

- **Textual** owns the interactive full-screen application: layout, input,
  workers, modal approval and responsive terminal behaviour.
- **Rich** owns reusable renderables such as Markdown, diffs and panels, and
  the line-mode fallback used by `--no-tui`, one-shot and piped runs.

Urwid, PyTermTk, PyTermGUI and web dashboard frameworks are alternatives to
this stack, not add-ons. They are not dependencies: adding a second widget or
event-loop system would duplicate input, rendering and tests without adding a
required capability. Revisit this decision only if a confirmed product
requirement cannot be implemented with Textual and Rich.

## Where commands run

This is the part worth reading before pointing it at a real repository. The
strategy is resolved once at startup, and announced if it is not isolating you.

| Platform | Strategy | What it means |
|---|---|---|
| Linux | **native** (default) | Runs as the current user. Run records stay under `<project>/.apodex/runs`; runtime caches and temporary state stay under `<project>/.apodex/runtime/native`. This is not an OS sandbox. |
| Linux with `--bwrap` | **bubblewrap jail** | Explicit isolation option: working directory bound read-write **at its own absolute path**; the system read-only; the rest of `$HOME` hidden. Requires bubblewrap and usable user namespaces. |
| macOS with Docker | **container** | The whole CLI re-executes inside the repo's Docker image. Your project is mounted at `/project`, run-private scratch at `/workspace`, final workflow artifacts at `/outputs`, and `~/.apodex` is bind-mounted. |
| macOS without Docker | **native** | Starts directly without requiring Docker. Run records stay under `<project>/.apodex/runs`; runtime caches and temporary state stay under `<project>/.apodex/runtime/native`. Commands still have the permissions of your macOS user and remain approval-gated; this is not an OS sandbox. |

Native mode is announced on every run so it is never mistaken for an OS
security boundary. Approval, path authorization, journaling, and `/revert`
remain active, but native shell commands have the current user's permissions.

`APODEX_SANDBOX` accepts `native`, `bwrap`, `host` or `container` to override
the resolution. Note that a Linux host can ship `bwrap` and still refuse to
mount a fresh `/proc` — common inside an unprivileged container — so explicit
bubblewrap mode probes with the real arguments instead of assuming the binary
is enough.

## Install and run

Run from the repository root, so `frontier_agent`, `plugins` and `workflows`
import:

```bash
uv sync
cp .env.example .env

# Add your own model credentials to .env:
#   OPENAI_API_KEY=...
#   OPENAI_BASE_URL=https://api.openai.com/v1
#   OPENAI_MODEL=gpt-4o
# research mode additionally wants SERPER_API_KEY / JINA_API_KEY

# Interactive TUI, Stateful ReAct, against another repository
frontier-agent --mode react --cwd /path/to/your/repo

# Coordinator with parallel sub-agents
frontier-agent --mode agent_team --cwd /path/to/your/repo

# One-shot: run, print, exit
frontier-agent --cwd /repo -p "explain src/foo.py"

# Auto-approve every tool call (batch / trusted use)
frontier-agent --cwd /repo --yes "add a --verbose flag to the CLI"

# Resume (the session id is printed at startup and by /log)
frontier-agent --resume 20260804-153000-coding-ab12

# List sessions that can be resumed
frontier-agent --resume

# Run the whole CLI inside Docker
frontier-agent --docker --cwd /path/to/your/repo

# Force the workspace-local native runtime (already the Linux default; macOS
# uses it automatically when Docker is unavailable)
frontier-agent --native --cwd /path/to/your/repo

# Explicit Linux bubblewrap isolation
frontier-agent --bwrap --cwd /path/to/your/repo

# Start with read-only PDF/image inputs (repeat --input as needed)
frontier-agent --cwd /path/to/your/repo \
  --input ~/Downloads/claim.pdf --input ~/Desktop/photo.jpg
```

In the full-screen TUI on macOS, `Ctrl+V` or `/paste` reads Finder file
selections and image data from the system clipboard. Absolute-path text is
attached through the same session input manager; ordinary text is inserted into
the prompt. `Cmd+V` remains the terminal's normal text paste shortcut.

This is a local, open-source BYOK tool: there is no account or `login` command.
Keys stay in your environment or local `.env`; the TUI never asks for or displays
them. Startup validates the local configuration before opening the TUI, and
`/config` shows only safe diagnostics such as provider, model, endpoint host and
whether the required key is configured.

The first `--docker` run builds the image, which takes a few minutes
(LibreOffice and the document readers are large); later runs reuse it.
`APODEX_IMAGE` overrides the tag.

On Linux, native mode is the default. On macOS, Docker remains preferred when
its daemon is reachable, with automatic fallback to native mode. Native mode
redirects common Python, Node, Rust, Go, and Ruby caches/state into
`.apodex/runtime/native`; the CLI's already-installed Python environment is reused
read-only from its install location. Heavy scientific, plotting, spreadsheet,
and document packages are optional: the agent installs only a package required
by the current task, and native Python installs land in
`.apodex/runtime/native/home/.local/site-packages`.

## Two workflow modes

| Mode | Tools | Prompt |
|---|---|---|
| **react** (default) | stateful web, shell, and file tools | focused single-agent research and file work |
| **agent_team** | coordinator tools plus bounded sub-agent tools | decomposition, parallel investigation, collection, and synthesis |

`/workflow react` and `/workflow agent_team` switch workflows in the TUI.

## Options

| Option | Meaning |
|---|---|
| `task` (positional) | the task; omit it for the interactive REPL |
| `--mode react\|agent_team` | workflow mode (default `react`) |
| `--resume [id]` | list saved sessions without an id; restore history, changes, mode and working directory with one |
| `--model` | model id (defaults to `$OPENAI_MODEL` / `$APODEX_MODEL`) |
| `--cwd` | working directory the agent operates in |
| `--input PATH` | attach a file or directory as a read-only session input; repeatable |
| `--max-turns` / `--max-tokens` | turns per task (50) / output tokens per call (8192) |
| `-y, --yes` | auto-approve every tool call |
| `-p, --print` | one-shot: run, print, exit |
| `--plan` | plan mode: investigate and propose first; edits locked until approved |
| `--docker` | run the whole CLI in a container (implied on macOS) |
| `--native` | use the workspace-local native runtime (default on Linux) |
| `--bwrap` | require the optional bubblewrap filesystem jail on Linux |
| `--no-sandbox` | run commands directly on this machine |
| `--theme <name>` / `--no-color` | application palette (defaults to Catppuccin); supports dark, light, Tokyo Night, Dracula, Nord, Gruvbox, One Dark, and Solarized variants |
| `--no-tui` | plain line-mode UI instead of the full-screen TUI |

The full-screen UI is selected only when stdin and stdout are both terminals.
Pipes, `TERM=dumb`, `NO_COLOR` (including an empty value), `--theme mono`,
`--no-color`, `--no-tui`, and one-shot runs use line mode automatically. SSH
and tmux use the full-screen UI when they expose a normal TTY and terminal type.
This makes line mode the dependable fallback rather than an error path.
In line mode a palette changes FrontierAgent's ANSI output only; the terminal
emulator still owns its own background and full colour scheme. Use the
full-screen TUI to apply the palette to the complete application surface.

### How the palettes are built

One palette per theme (`apodex/tui/themes.py`) drives both surfaces, so the
line UI and the TUI can't disagree about what a theme looks like, and selecting
a theme recolours everything rather than only the widget chrome.

**Each upstream palette is the source of truth.** Gruvbox's orange is `#fe8019`
and Solarized's yellow is `#b58900` because that is what makes them those
themes. Semantic colours are used verbatim unless they fall below 3:1, and are
then corrected by the smallest lightness step that clears it — hue and chroma
are never touched. 68 of 84 are untouched.

The three text tiers come from each designer's own ramp (gruvbox `fg`/`fg3`/`fg4`,
Solarized `base1`/`base0`/`base00`, Catppuccin `text`/`subtext0`/`overlay2`),
which is why quiet text keeps the palette's cast — gruvbox warm, Solarized teal
— instead of going grey. Floors match what each role actually is: `foreground`
≥ 6:1 and `muted` ≥ 4.5:1 are body text and owe WCAG AA, while `subtle` and the
semantic colours are short bold labels, glyphs, borders and diff markers, so
they owe 3:1 (WCAG 1.4.3 / 1.4.11). `apodex/tests/test_themes.py` checks the
floors *and* asserts the semantic colours still match upstream byte-for-byte —
holding accents to the 4.5:1 body-text floor is what once turned Catppuccin
Latte's amber into a dark brown, and contrast alone cannot catch that.

Two rules follow from that and are worth knowing before editing colours:

- **No Rich `dim`.** `dim` is a terminal-side blend by an unspecified amount, so
  it destroys a measured ratio — it was why thinking text collided with the
  background on darker palettes. Quiet text uses `muted` / `subtle` instead.
- **No colour emoji in the UI.** A colour emoji paints itself and ignores the
  surrounding foreground, so it can't follow a theme. Every glyph is a
  monochrome character from the shared `GLYPHS` vocabulary and takes the active
  theme's colour like any other text.

`dark` and `light` are our own palettes, not Textual's built-ins — those had no
Rich half, so the transcript fell back to Catppuccin and painted dark-theme
colours onto a light background.

### The sidebar

The right-hand workspace has three tabs by default, with **Plan** selected. A
fourth **Diff** tab appears as soon as the session has file changes:

- **Plan** — the todo list, or Agent Team's current task board.
- **Activity** — a selectable tool timeline. Arrow keys move between calls;
  Space or a mouse click opens the complete call details.
- **Files** — final reports and workflow deliverables. Arrow keys select a file;
  Space or a mouse click previews source code, Markdown, PDF documents, Office files (`.docx`, `.xlsx`, `.pptx`), Jupyter Notebooks (`.ipynb`), 3D/macromolecular structures (`.pdb`, `.stl`, `.obj`, `.gltf`), images (ANSI pixel art), and archives (`.zip`, `.tar`).
- **Diff** — every change accumulated since the session first touched each
  file, rendered as a colourized unified diff and scrollable with the arrow /
  page keys. File tools are journaled by path; a non-read-only `bash` call is
  journaled by scanning the working directory and the session's outputs
  directory around the tool phase, so what a shell script wrote shows up too —
  though `/revert` leaves those alone, since a scan cannot separate the shell's
  writes from a concurrent editor's. The tab count and footer show changed
  files plus total added and removed lines.

When a task completes, the workspace opens **Diff** when changes exist and
**Files** otherwise. Removing or reverting every change hides **Diff** again.

`Ctrl+Tab` / `Ctrl+Shift+Tab` cycle the tabs, while `Ctrl+O` jumps directly to
Files (and returns to Plan when pressed again). Each tab receives the full pane
instead of competing vertically for height. Tab labels carry compact counts
(`Plan 8/14`, `Activity 31 ◐1 ✗2`, `Diff 3`) when content extends off-screen. `Ctrl+B`
hides the sidebar entirely; it also hides itself below 100 columns.

### If the colours look grey or wrong

A palette needs colours to exist in. Rich and Textual pick their colour depth
from `COLORTERM`, then `TERM`, and never query the terminal. At 8 colours the
themes are not approximated but destroyed: every value snaps to one of eight
ANSI slots your *terminal* defines, so gruvbox's `#ebdbb2` cream becomes pure
white, its muted tan becomes `#aaaaaa` grey, and its orange and red both become
the same red. That is the "everything is black or grey" failure, and no palette
change can fix it.

apodex prints a warning when it detects fewer than 256 colours. To fix it:

```bash
export COLORTERM=truecolor
```

The containerised path (the default on macOS) forwards your host's `TERM` and
`COLORTERM` inward and falls back to a 256-colour floor, because Docker
otherwise sets `TERM=xterm` inside the container and forwards no `COLORTERM` at
all — the container's `TERM` is a Docker default, not a measurement of the
terminal actually painting the pixels. `TERM=dumb` is still honoured as-is.

## Slash commands

`/help` · `/mode <name>` · `/model <name>` · `/config` (safe local settings) ·
`/cwd [<path>]` · `/new` (save and start fresh) · `/fork` (branch the current
context) · `/sessions` · `/rename <name>` · `/clear` (context and plan) ·
`/context` (window and cumulative token usage) · `/revert` (undo every file
change this session) · `/log` (trace
path) · `/auto` (toggle auto-approve) · `/theme <name>` · `/exit`

Available names: `dark`, `light`, `catppucin`, `catppucin-latte`,
`tokyo-night`, `tokyo-night-day`, `dracula`, `nord`, `gruvbox`,
`gruvbox-light`, `one-dark`, `one-light`, `solarized`, `solarized-light`, and
`mono`. The correctly spelled `catppuccin` and `catppuccin-latte` aliases also
work.

`/compact` creates a fresh context checkpoint: it first condenses every tool
result (keeping useful leading/trailing output and source URLs), then asks the
summary model to produce one concise session summary.

In the full-screen TUI, entering `/theme` opens a mouse- and keyboard-selectable
theme list. Entering `/workflow` similarly opens a picker for `react` and
`agent_team`; explicit forms such as `/workflow react` work in both TUI and
line mode.

In the full-screen UI, press `F1` for interaction help, `Ctrl-P` for the command
palette, `Ctrl-B` to toggle the plan/activity sidebar, and `↑`/`↓` to revisit
submitted input without losing the current draft. `Tab` completes an unambiguous
slash command. Approval starts on **No**; long command and diff previews scroll
with `Ctrl-U` / `Ctrl-D`.

## Safety

- **Approval gate.** Read-only calls (read/grep/glob/view/web_search/web_fetch,
  and read-only bash such as `ls` / `git status`) run straight through. Writing,
  deleting, installing and dangerous shell need confirmation, which `-y` skips.
  A hard-denied command (`rm -rf /`, writing outside the working directory,
  `xargs` feeding `rm`) is refused **regardless of `-y`**.
- **Revertible.** Every write, edit and delete snapshots the previous content
  first, so `/revert` restores the whole session. `delete_file` exists as a
  first-class tool for exactly this reason — prefer it over `bash rm`.
- **Local trace.** Every LLM and tool action, including refused ones, is written
  to `<cwd>/.apodex/runs/<id>/trace.jsonl`.
- **Working-directory boundary.** Reads, writes and deletes stay inside the
  working directory, and `find` prunes the artifact directories your
  `.gitignore` declares.
- **Interrupt and resume.** Ctrl-C stops the run; state is persisted per turn,
  so `--resume <id>` continues from the last completed turn.

Pointing `--cwd` at this repository itself is not the intended use — the file
tools' checkout protection may refuse writes. Use it on another repository.

## Layout

```
apodex/
├── cli.py          # argparse entry, .env load, sandbox resolution, docker dispatch
├── native.py       # workspace-local runtime home, caches, dependencies, inputs
├── sandbox.py      # native / bwrap / container / host strategy + execution
├── docker.py       # the macOS path: re-exec the CLI inside the repo image
├── config.py       # ModelConfig from env (OPENAI_* / APODEX_*)
├── llm.py          # terminal LLM construction and provider configuration
├── prompts.py      # legacy generic-profile prompts
├── prompts_base.py # shared generic-profile prompt builders
├── profiles/       # terminal workflow selection and compatibility profiles
├── agent_tools.py  # tool lists, risk levels, path rewriting, read-only bash test
├── local_tools.py  # local bash/read_file/glob/grep/delete_file
├── todo.py         # todo_write tool + plan panel state
├── diff_preview.py # unified diff shown before anything is written
├── changes.py      # WorkspaceJournal: snapshot / diffstat / revert
├── trace.py        # TraceObserver: every action to JSONL
├── observers.py    # streaming render, approval, path rewriting, journal
├── render.py       # Rich / plain-text rendering
├── session.py      # REPL, slash commands, and terminal-session coordination
├── task_runner.py  # generic-loop and native-workflow task execution
├── session_state.py # session IDs, checkpoints, and resume listings
├── middleware.py   # terminal skill-injection wiring
├── tui/            # full-screen Textual UI
└── tests/          # mock-LLM end-to-end plus journal / trace / persistence units
```

## Known gaps

- No second confirmation for large diffs or deletions beyond the single
  approval plus diff preview.
- The trace records `is_error` and duration but does not classify failures
  (timeout vs test failure vs syntax error).
- No skills are bundled. The loader is wired, so a profile's `skills:` list
  picks up any `plugins/skills/<id>/SKILL.md` you add.
- The bubblewrap path runs in CI on Linux; the macOS container path has not been
  exercised on macOS hardware.
