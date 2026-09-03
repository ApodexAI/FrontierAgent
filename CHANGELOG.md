# Changelog

All notable changes to FrontierAgent will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] — Unreleased

Initial open-source release of FrontierAgent.

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

### Fixed

- Apply benchmark question limits after seeded shuffling so repeated runs can
  sample different questions while `--no-shuffle` keeps canonical ordering.
