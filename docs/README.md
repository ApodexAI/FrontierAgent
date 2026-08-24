# FrontierAgent documentation

Use this page as the map for the repository documentation. Pick the section
that matches what you are trying to do; files listed as **reference** are most
useful when changing that subsystem rather than when getting started.

## Start here

| Goal | Read this |
|---|---|
| Understand the project and run a first task | [Project README](../README.md#quick-start) |
| Run the TUI with an existing LLM endpoint | [English quickstart](install/tui-endpoint-quickstart.md) / [中文教程](install/tui-endpoint-quickstart.zh-CN.md) |
| Learn the TUI panes, previews, approvals, and live steering | [English user guide](tui-user-guide.md) / [中文使用教程](tui-user-guide.zh-CN.md) |
| Install on any environment | [Installation and deployment](#installation-and-deployment) |
| Run FrontierAgent from a container | [Docker and Compose](install/docker.md) |
| Run or configure a local NVIDIA/SGLang model | [GPU and SGLang](#gpu-and-sglang) |
| Understand ReAct versus Agent Team | [Workflow implementations](#workflow-implementations) |
| Run benchmark evaluation | [Evaluation guide](eval.md) |
| Extend the framework | [Developer guides](#developer-guides) |

## Installation and deployment

Start with the [installation chooser](install/README.md). It routes by operating
model—hosted endpoint, Docker GPU host, custom cloud image, or an existing GPU
container—instead of asking you to guess from the operating system alone.

| Environment | Canonical guide |
|---|---|
| macOS or Linux, fastest hosted-endpoint TUI setup | [English quickstart](install/tui-endpoint-quickstart.md) / [中文教程](install/tui-endpoint-quickstart.zh-CN.md) |
| macOS, hosted or remote endpoint | [macOS](install/macos.md) / [中文详细版](install/macos.zh-CN.md) |
| Linux, hosted or remote endpoint | [Linux](install/linux.md) |
| Windows | [WSL2 section in the Linux guide](install/linux.md#windows-and-wsl2) |
| Any host with Docker, no local Python | [Docker and Compose](install/docker.md) |
| Linux NVIDIA host with Docker | [Docker SGLang](install/linux-nvidia.md) |
| Existing Linux GPU environment without nested Docker | [Native SGLang](install/linux-nvidia-native.md) |
| GPU cloud or custom OCI image | [GPU platforms](install/gpu-platforms.md) |

## GPU and SGLang

Read these in order when bringing up a local model:

1. [GPU compatibility](install/gpu-compatibility.md) explains supported runtime
   tracks and what must be recorded when certifying hardware.
2. Choose the matching deployment guide from the table above.
3. [SGLang configuration reference](../config/sglang/README.md) documents every
   `.env.sglang` setting, safe tuning order, and upstream references.

The files under `config/sglang/*.env.example` are runnable templates; their
comments and the SGLang reference are the source of truth for those variables.

## Using FrontierAgent

| Topic | Document |
|---|---|
| TUI panes, keyboard workflow, previews, approvals, and Agent Team intervention | [English user guide](tui-user-guide.md) / [中文使用教程](tui-user-guide.zh-CN.md) |
| CLI, TUI, themes, slash commands, approvals, and terminal internals | [`apodex/README.md`](../apodex/README.md) **(reference)** |
| High-level architecture and sandbox boundaries | [Framework architecture](framework.md) |
| Security policy and vulnerability reporting | [Security policy](../SECURITY.md) |

The root README's job is the product overview, a short quick start, and results.
Detailed operational behavior belongs in the focused documents above, so new
instructions go there rather than into a second copy.

## Workflow implementations

| Workflow | Use it for | Configuration reference |
|---|---|---|
| Stateful ReAct | One stateful agent doing sequential research and file work | [Workflow README](../workflows/stateful_react_agent/README.md) |
| Agent Team | Coordinator plus parallel sub-agents and optional reporter | [Workflow README](../workflows/agent_team/README.md) |

Those READMEs own profile selection and workflow-specific configuration. To
create a new workflow plugin, use [Writing a workflow](workflows.md) instead.

## Evaluation

- [Evaluation guide](eval.md): installation, judge configuration, dataset
  downloads, execution, filesystem contract, results, and progress.
- [FrontierSearchBench evaluation](eval-frontier-search.md): the one benchmark
  with an external cross-query scorer — isolation requirement, collect / export /
  score workflow, and scorer options.
- [Benchmark registry](../benchmarks/README.md): supported dataset keys,
  default pipelines, scoring implementations, repository layout, and how to
  add a benchmark.

`docs/eval.md` is the operator guide for everything the benchmarks share;
`docs/eval-frontier-search.md` carries only what is specific to
FrontierSearchBench. `benchmarks/README.md` is the registry and extension
reference; keeping those roles separate avoids duplicating setup instructions.

## Developer guides

| Task | Document |
|---|---|
| Understand runtime flow, observers, teams, and package boundaries | [Framework architecture](framework.md) |
| Author and register a workflow plugin | [Writing a workflow](workflows.md) |
| Develop and submit changes | [Contributing](../CONTRIBUTING.md) |
| Understand the frozen Textual + Rich decision | [TUI framework decision](python_terminal_tui_ai_agent_guide.md) **(design record, Chinese)** |
| Compare context-offloading changes with an A/B run | [Tool-result truncation A/B](tool-result-truncation-ab.md) |
| Pick up the deferred context-offloading work | [Context offloading follow-ups](context-offloading-followups.md) |
| Inspect release-facing changes | [Changelog](../CHANGELOG.md) |

Subsystem READMEs live beside their code when they are primarily useful to
maintainers: [`apodex/`](../apodex/README.md),
[`benchmarks/`](../benchmarks/README.md),
[`deploy/huggingface/`](../deploy/huggingface/README.md), and
[`tools/golden/`](../tools/golden/README.md). Publishing the public demo Space
has a condensed Chinese walkthrough in
[`README.zh-CN.md`](../deploy/huggingface/README.zh-CN.md),
with that directory's English `README.md` as the authoritative version;
`README.space.md` beside them is the Space's own landing page, not repository
documentation.

## Documentation ownership

When updating documentation, change the canonical page instead of copying a
second set of commands:

- root `README.md`: product story, capabilities, short quick start, and results;
- `docs/install/`: environment-specific installation and deployment;
- `docs/install/docker.md`: Compose, image pinning, `docker run`, and cloud
  deployment of the CPU agent container;
- `.env.example`: the runtime agent, web-tool, and document-reader variables,
  documented by its own comments;
- `config/sglang/README.md`: SGLang variables and tuning;
- workflow READMEs: workflow behavior and profiles;
- `docs/eval.md`: running and interpreting evaluations, including datasets and
  the eval-only credentials that are not in `.env.example`;
- `docs/eval-frontier-search.md`: FrontierSearchBench's external scorer,
  isolation requirement, and three-step workflow;
- `benchmarks/README.md`: benchmark registry and extension points;
- `apodex/README.md`: CLI/TUI subsystem reference;
- `docs/framework.md` and `docs/workflows.md`: framework development.

Prefer linking to a canonical section over repeating it. A small command that
gets a reader started is fine; a second complete setup or troubleshooting guide
usually is not.
