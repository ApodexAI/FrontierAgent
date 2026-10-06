# Changelog

All notable changes to FrontierAgent will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] — Unreleased

Initial open-source release of FrontierAgent.

### Changed

- **Runtime engine moved to [`apodex-agent-core`](https://pypi.org/project/apodex-agent-core/)
  (pinned `==0.12.2`).** The agent loop, loop contracts, tool execution,
  compaction, observers, AgentBus, DAG and providers now come from `agent_core`;
  `frontier_agent.*` keeps its import paths as `sys.modules` aliases or thin
  adapters, so workflows, apodex and benchmarks are unchanged. Product policy is
  injected through `AgentLoopHooks` / `ToolExecutionHooks` and the `configure_*`
  resolvers (`core/runtime/loop/{agent_loop,tool_exec}.py`,
  `components/agent_bus/bus.py`, `infra/openai_client.py`). Behaviour now
  follows AgentCore where the fork had diverged, notably: compaction pins the
  first user message verbatim and replaces legacy prose spill indexes, and
  `Any`-typed tool parameters generate `{"type": "string"}` (`create_file`
  now annotates its `rows` / `data` shorthand shapes explicitly).

### Added

- **ReAct workflow**: single stateful agent with tool use, sandboxed execution,
  and iterative reasoning.
- **Agent Team workflow**: coordinator + parallel sub-agents with task board,
  fan-in report collection, and synthesis.
- Built-in tools: web search, web fetch (with academic routing), file I/O,
  bash execution, document readers/writers (PDF, DOCX, PPTX, XLSX).
- Sandbox backends: `bwrap` (bubblewrap), container, and E2B cloud.
- Terminal UI (TUI) with Rich rendering and Textual interface.
- Benchmark evaluation suite with support for BrowseComp, HLE, DeepSearchQA,
  WideSearch, SuperChem, Frontier Science, and XBench.
- Bundled FrontierSearchBench with 41 verifiable research queries, official
  batch scorers, collection/export adapters, and external-scoring result mode.
- Model support: OpenAI, Anthropic (direct and Bedrock), and
  OpenAI-compatible endpoints.
- Profile-based configuration system.
- Local NVIDIA/SGLang doctor and lifecycle commands with API/tool-parser smoke
  checks, optional VPN-safe Compose subnet selection, and host UID/GID mapping.
- Separate 0.8B infrastructure-smoke and candidate 35B RTX 4090, RTX 5090, and
  multi-GPU configuration templates, with SGLang input/output budgets kept
  inside the configured context window.
- Clean-machine Linux + NVIDIA installation and release-certification guide,
  distinguishing deployment health from production agent correctness.

- Standalone installation with `uv tool install`: the wheel now ships the
  provider registry, and `frontier-agent` runs from any directory without a
  checkout. An optional user env file (`$XDG_CONFIG_HOME/apodex/env`, default
  `~/.config/apodex/env`, override with `APODEX_ENV_FILE`) holds the endpoint
  below exported variables and the launch directory's `.env`; a key defined
  next to a base URL is only applied together with that base URL.
- `APODEX_BUILD_CONTEXT` names a checkout to build `apodex:local` from when the
  installed CLI is not one. Without an image, a checkout, or an explicit
  `APODEX_IMAGE`, the Docker path stops with the options instead of silently
  running natively.

### Fixed

- **A write the path gate refused no longer reaches the host.** `write_file` and
  `file_editor_create` fall back to the sandbox when local authorization fails,
  and for the in-process `CurrentSandbox` that fallback is an `open()` in the
  harness (as root, in container mode), so a refused path was written anyway.
  That branch now authorizes the path itself; remote backends are unaffected.
- **Paths are quoted before reaching a sandbox shell.** `file_editor`'s view /
  create / str_replace commands, `write_file`'s `mkdir -p`, and the Docker
  `mkdir -p` interpolated the path unquoted, so a path containing `;` ran
  commands that never passed the bash policy.
- **A system directory named as the workspace root grants nothing.** The root
  arrives through `ExecutionScope` metadata (workload input), and
  `{"workspace_root": "/etc"}` made `/etc` readable and writable. System roots
  and direct children of `/usr` and `/etc` are refused; a run directory deep
  under `/var` or `/opt` (a container volume, macOS `$TMPDIR`) still works.
- Bash policy: privilege escalation (`sudo`/`su`/…), remote/exfil clients
  (`ssh`/`nc`/`rsync`/…) and signal senders (`kill`/`pkill`/`killall`) are now
  refused in every allowlist mode, including the default `off`. The local CLI
  sends them to the human as a typed confirmation that auto-approve, `auto_for_me`
  and saved rules cannot answer.
- Bash policy: command substitutions are located by one scanner for both
  masking and extraction, so quoted parens, apostrophes in double quotes and
  unterminated `$(` no longer hide a nested command; `$((…))` arithmetic is no
  longer assessed as a command.
- Bash policy: the host-shutdown/`mkfs`/fork-bomb word screens only see text the
  shell executes, so quoted arguments and heredoc data mentioning `halt` or
  `reboot` are no longer refused, while `bash -c`, shell heredocs, pipes into a
  shell, evaluators (`watch`/`tmux`/…) and `systemctl` shutdown units still are.
  `DROP TABLE` keeps screening the whole text.
- Spill recovery: oversized and compacted tool results are stored through
  AgentCore's `SpillStore` (one store, one registry). A single scope key
  (`task:llm_session`) now decides the store directory, the bwrap `/spill` mount
  and read authorization: a jail sees only its own store and its sub-agents',
  siblings and other conversations see nothing, and `_path_auth` no longer
  authorizes every store the process created. A path is advertised only when
  the backend actually running commands can open it (`auto` going to E2B, or
  bwrap unusable, gets none), and the compaction spill callback is withheld
  then. Compacting a truncated preview points at the original full body
  instead of storing the preview as "[Full text]". The `recover_result` footer
  is only shown when a trajectory JSONL exists. Known limitation: `container`
  mode without the inner bwrap jail shares one tool uid, so model commands can
  still read other scopes there.
- Surface finalize-gate bypasses on the final turn: an answer delivered despite
  open task-board items now carries an unfinished-work note and a
  `finalize_gate_bypassed` marker instead of reading as a clean success.
- Native mode puts the CLI's own Python environment ahead of the inherited
  `PATH`, so `read_file`, `download_file`, and `python3` inside `bash` use the
  interpreter the CLI was installed with rather than a system Python.
- The Docker launcher forwards the resolved runtime variables (exported
  environment, launch directory `.env`, user env file) into the container by
  name with `docker run -e NAME`, so an exported value now takes precedence over
  the checkout's `.env` inside the container as it already did natively.
- Apply benchmark question limits after seeded shuffling so repeated runs can
  sample different questions while `--no-shuffle` keeps canonical ordering.
