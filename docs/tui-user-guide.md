# FrontierAgent TUI user guide

[中文](tui-user-guide.zh-CN.md) · [Endpoint setup](install/tui-endpoint-quickstart.md) · [Documentation index](README.md)

This guide begins after the TUI has opened. It explains the layout, the four
right-hand tabs, file previews, approvals, and asynchronous intervention in
`agent_team`. If FrontierAgent is not running yet, complete the
[macOS/Linux endpoint quickstart](install/tui-endpoint-quickstart.md) first.

## 1. Interface map

```text
┌──────────────────────────────────────────────────────────────────────┐
│ FrontierAgent · workflow · session · workspace                 F2   │
├──────────────────────────────────────────────┬───────────────────────┤
│                                              │ Plan Activity Files Diff│
│ Transcript                                   │                       │
│                                              │ Active tab            │
│ Tasks, thinking, tools, approvals, report    │                       │
├──────────────────────────────────────────────┴───────────────────────┤
│ phase · elapsed · workflow · model · context · tools · queued        │
│ attachments, when present                                           │
│ task input or a mid-run steering instruction                        │
└──────────────────────────────────────────────────────────────────────┘
```

The top bar identifies the workflow, session, and working directory. The left
side is the complete transcript; the right side shows working state. The footer
reports the current phase, elapsed time, model, remaining context, tool count,
queued interventions, and session file-change totals. Below 100 terminal columns the sidebar is hidden;
widen the terminal or press `Ctrl-B` to restore it.

> Screenshot 1: the complete initial TUI, annotated with the top bar,
> transcript, sidebar tabs, status bar, and prompt.

## 2. The sidebar tabs

Press `Ctrl-Tab` to cycle through `Plan → Activity → Files`, or
`Ctrl-Shift-Tab` to cycle backwards. When the session has file changes, a
clickable `Diff` tab appears and joins the cycle after `Files`.

### Plan: plan and task board

`Plan` answers “what is planned, and how far has it progressed?”

- In `react`, it shows steps maintained by the agent's todo tool.
- In `agent_team`, it shows tasks created and updated by the coordinator.
- Glyphs distinguish pending, in-progress, completed, and cancelled work.
- The tab heading retains a `completed/total` summary when items scroll away.
- `no plan yet` simply means the current task has not created a plan.

Plan is a read-only status view and does not need keyboard focus. A new task
automatically returns the sidebar to Plan so decomposition is immediately
visible.

> Screenshot 2: Plan during an Agent Team run, preferably with several states
> and the completion count visible in the tab label.

### Activity: tools and sub-agent activity

`Activity` answers “who is doing what, did the tool succeed, and how long did
it take?” Each row carries a state glyph, name, duration, and short summary.

With Activity focused:

1. Select a row with `↑` / `↓`.
2. Press `Space` on a normal tool call to inspect its state, duration, call ID,
   and details.
3. Press `Space` or `Esc` to close the detail view.
4. Press `Esc` in the list to return focus to the prompt.

In `agent_team`, Activity is divided into:

- `SUB-AGENTS`: queued, running, ready, or failed background workers.
- `COORDINATOR`: the coordinator's own tool calls.

Select a sub-agent row and press `Space` to expand or collapse its thinking,
message, tool-call, tool-result, and error events in place. Select an expanded
event and press `Space` to open its details. Group headings retain total, live,
and failed counts.

Every Agent Team worker shares one role internally, so each row is identified by
two things the run state cannot supply: a colour taken from the active theme's
palette, and a specialty marker inferred from the name the coordinator chose.

| Marker | Specialty | Example name |
|---|---|---|
| `⌕` | research / discovery | `market_research` |
| `⊕` | web / browsing | `news_web` |
| `▦` | data / metrics | `revenue_data` |
| `∑` | analysis / synthesis | `policy_analysis` |
| `⊙` | verification / review | `fact_check` |
| `✎` | writing / reporting | `final_writeup` |
| `⌗` | code / build | `chart_code` |
| `☑` | planning | `delivery_plan` |
| `◆` | anything else | `misc_helper` |

While `collect_reports` is blocked, the transcript also shows one live card that
rewrites itself in place — one line per worker, with the same markers and
colours — and removes itself once the fan-in returns.

> Screenshot 3: Agent Team Activity with both `SUB-AGENTS` and `COORDINATOR`
> visible and one worker expanded.

### Files: session deliverables

`Files` answers “what did the agent produce?” Press `Ctrl-O` to toggle directly
between Files and Plan. When a completed task produces outputs, the TUI also
reveals Files. The top of the pane shows the host deliverables path; Docker runs
additionally show the path visible to the agent. A separate `Work:` line points
to this run's intermediate workspace, which is intentionally not mixed into
the deliverables list.

In Files:

1. Select a file with `↑` / `↓`.
2. Press `Space` for a read-only preview.
3. Page with `Ctrl-U` / `Ctrl-D`.
4. Press `Space` or `Esc` to close the preview.
5. Press `Esc` in the file list to return to Plan.

The TUI previews common source and text files, Markdown, CSV, PDF, Word, Excel,
PowerPoint, Jupyter notebooks, images, archives, and selected 3D or molecular
formats. Some Office, PDF, and image formats require optional reader packages.
Unsupported files remain listed but cannot be opened in the TUI. Previews are
bounded; the file on disk is authoritative.

The final answer is automatically saved as `final-report.md`. In native mode,
the default output directory is:

```text
<your --cwd>/.apodex/runs/<session-id>/outputs/
```

> Screenshot 4: the Files list and a preview opened with `Space`. Source code,
> Markdown, or an image demonstrates the preview most clearly.

### Diff: session file changes

`Diff` shows a colourized unified diff of what this session changed on disk.
It stays hidden while there are no changes. A completed task opens `Diff` when
changes exist and opens `Files` otherwise.
It accumulates repeated edits across turns from the version that existed when
the current session first touched each file. Existing unrelated working-tree
changes are therefore not attributed to the agent. Created and deleted files
use the familiar `/dev/null` headers.

Two kinds of change reach it. File tools (`write_file`, the `file_editor`
family, `delete_file`) name their target, so it is snapshotted directly. A
`bash` command that is not read-only names nothing, so the working directory
and the session's outputs directory are scanned before and after the tool
phase, and only the files that actually changed are kept — a shell script, a
`sed -i`, a generated file and a `rm` all show up. Binary and very large files
are listed as changed but carry no baseline, so they are left out rather than
misreported.

**`/revert` only undoes the first kind.** A scan sees that the tree changed,
not who changed it: anything else writing during the same window — your own
editor, a watcher, a dev server — is indistinguishable from the shell command,
and restoring those files would destroy work the session never did. So
scan-discovered files are shown here but left on disk, and `/revert` lists them
separately for you to handle yourself.

The tab label shows the number of changed files. The pane and status bar show
total `+added` and `-removed` lines in green and red. The view refreshes in the
background; arrow keys, `PgUp`/`PgDn` and `Home`/`End` scroll it, and `Esc`
returns to Plan.

## 3. A normal task workflow

Enter a task at the bottom and press Enter:

```text
Inspect this repository, identify the test entry points and major modules, and produce a Markdown architecture note. Explain the plan before editing.
```

A useful observation sequence is:

1. Watch decomposition and progress in Plan.
2. Inspect tool calls in Activity; expand workers in Agent Team.
3. Review the command or diff when an approval dialog appears.
4. Type a plain-text instruction in the prompt if the direction needs changing.
5. On completion, press `Ctrl-G` for the final report or `Ctrl-O` for files.
6. Ask a follow-up in the same prompt; the session context and modified
   workspace are retained.

## 4. Asynchronous steering and intervention

The prompt remains usable while the status bar says the agent is working. For
example:

```text
Do not edit code yet. Focus on test-coverage gaps and cite the evidence in the report.
```

After Enter, the transcript shows a queued notice and the status bar's `queued`
or `q` count increases. The message is injected as a new user instruction at
the next safe agent-turn boundary.

Its precise semantics matter:

- Steering is queued; it does not terminate an in-flight LLM request or tool.
- In `react`, the current agent receives it.
- In `agent_team`, the coordinator receives it. Already-dispatched sub-agents
  are not interrupted directly; the coordinator can change later delegation,
  verification, or synthesis.
- If the instruction arrives after the final useful boundary, it is preserved
  and run immediately as a follow-up task.
- Multiple interventions queue in order, so keep them short and state clearly
  when a new instruction supersedes an earlier one.
- Slash commands are not queued during a run. The TUI asks you to interrupt
  first with `Ctrl-C`.

Use `Ctrl-C` when you need to stop the task immediately instead of sending the
word “stop.” State is saved through the most recently completed turn. Pressing
`Ctrl-C` while idle exits the application.

> Screenshot 5: Agent Team running after a steer was entered. Capture the
> queued transcript notice, `queued 1` in the footer, and workers still active.

## 5. Approval dialogs

For a write or protected command, the TUI shows the target, reason, and command
or diff preview. `No` is initially selected so an accidental Enter cannot run
the action.

| Key / option | Meaning |
|---|---|
| `y` | Approve this action once |
| `n` or `Esc` | Reject |
| `m` | Auto-approve bash for this session; use only in Docker or a trusted environment |
| `a` | Allow all ordinary approvals for this session |
| `A` | Persistently allow this command class |
| `e` | Reject and give the agent a replacement instruction |
| `Ctrl-U` / `Ctrl-D` | Page through a long command or diff |

Dangerous actions do not accept the single-key `y`; type the full word `yes`.
Approve actions individually on a first run. The native runtime is not an OS
sandbox, and approved commands have the current user's permissions.

## 6. Reviewing a long transcript

| Action | Purpose |
|---|---|
| `Alt-J` / `Alt-K` | Move through visible transcript blocks |
| `Alt-Enter` | Expand or collapse the selected thinking/process block |
| `Ctrl-G` or `/report` | Jump to the latest final report |
| `Ctrl-Y` or `/copy` | Copy the latest final report |
| `/filter thinking` | Show only thinking |
| `/filter tools` | Show only tool calls and results |
| `/filter errors` | Show only errors |
| `/filter report` | Show only the final report |
| `/filter all` | Restore everything |
| `/find <text>` | Search the transcript |

Very long sessions hide older rendered blocks to keep the TUI responsive, but
the complete history remains in the session. Use `/compact` when the context is
approaching its limit.

## 7. Attachments, sessions, and workflows

Common commands:

```text
/attach <path>       copy a file or directory into read-only session inputs
/attachments         list attachments
/detach <name>       delete the attachment copy, not its source
/workflow react      select one sequential stateful agent
/workflow agent_team select a coordinator with parallel sub-agents
/new                 save this session and start an empty one
/fork                branch the current context into a new session
/rename <name>       give the session a readable name
/resume              select a previous session
/context             inspect context and token usage
/config              inspect redacted endpoint and model configuration
/log                 show the JSONL trace path
/revert              undo edits recorded by session file-editing tools
```

Type `@` followed by part of a filename to search explicit attachments and files
under the current `--cwd`, then press `Tab` to complete a reference. Workspace
files are already mounted and are not copied into the attachment area. On macOS,
`Ctrl-V` or `/paste` can read Finder files and clipboard images; on Linux, use
`/attach <path>`. Relative `--input` and `/attach` paths start at the current
`--cwd` and may reference files or directories at any depth below it; absolute
paths remain supported. Switching workflows resets conversation context, so
switch before beginning the next task.

Multiline or very large terminal text paste is displayed as a compact
`[Pasted text …]` marker. Pressing Enter expands it and sends the complete text,
including its original line breaks, as one prompt.

## 8. Choosing a workflow

- Start with `react` for code navigation, focused repository edits, and
  sequential follow-ups.
- Use `agent_team` when work splits into independent investigations or needs
  cross-checking and synthesis from several reports.
- Agent Team makes concurrent endpoint calls. It can gather diverse evidence
  faster, but normally consumes more tokens and provider concurrency.
- Native, bubblewrap, and Docker are runtime choices independent of the
  `react` versus `agent_team` workflow choice.

Press `F1` at any time for built-in shortcuts. Press `F2` for theme, workflow,
behavior, permissions, and session settings.

For ready-to-run demonstrations of ReAct, Agent Team, attachments, Files
previews, and asynchronous steering, see the [TUI demo queries](tui-demo-queries.md).
