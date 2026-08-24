# Contributing to FrontierAgent

Thank you for your interest in contributing to FrontierAgent! This document
provides guidelines and instructions for contributing.

## Getting Started

1. **Fork** the repository on GitHub.
2. **Clone** your fork locally:
   ```bash
   git clone https://github.com/<your-username>/FrontierAgent.git
   cd FrontierAgent
   ```
3. **Install** dependencies:
   ```bash
   pip install uv
   uv sync --all-extras
   ```
4. **Create** a feature branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## Development Setup

### Prerequisites

- Python ≥ 3.12
- [uv](https://docs.astral.sh/uv/) (recommended) or pip

### Fully loaded environment

The quick start in the [README](README.md#quick-start) installs the lightweight
terminal runtime. Development wants every optional extra present up front:

```bash
uv sync --frozen \
  --extra sandbox \
  --extra document-readers \
  --extra eval \
  --extra dev

uv run frontier-agent --no-tui -p "say hello"
```

Linux uses the workspace-local native runtime for that command. Pass `--bwrap` or
`--docker` when development requires an OS isolation boundary; `--no-sandbox`
deliberately uses the real home directory and host caches as well.

To build and run the container image, see
[Run FrontierAgent in Docker](docs/install/docker.md#build-from-the-current-checkout).

### Running Tests

```bash
# Fast checks for TUI/CLI changes
uv run pytest apodex/tests -q

# Workflow and framework tests
uv run pytest tests -q

# Everything, plus lint
uv run pytest -q
uv run ruff check .
```

### Debugging a session

Line mode removes full-screen rendering from the equation:

```bash
uv run frontier-agent --no-tui --mode agent_team --cwd /path/to/project
```

Inside the session, `/log` prints its trace location. After a failure, inspect:

```bash
tail -f /path/to/project/.apodex/runs/<session-id>/engine.log
python -m json.tool /path/to/project/.apodex/runs/<session-id>/session.json
```

On macOS, rebuild after Docker or runtime changes:

```bash
docker build -t apodex:local .
uv run frontier-agent --docker --no-tui --mode agent_team --cwd /path/to/project
```

For benchmark debugging, keep both question and team concurrency small and write
each run to a dedicated result directory.

### Code Quality

We use [Ruff](https://docs.astral.sh/ruff/) for linting and formatting:

```bash
uv run ruff check .
uv run ruff format .
```

### Pre-flight Checks

Before submitting, run the pre-flight validation:

```bash
uv run python tools/preflight.py
uv run python tools/import_smoke.py
```

## Submitting Changes

1. Ensure all tests pass and pre-flight checks are clean.
2. Write clear, concise commit messages.
3. Push to your fork and open a Pull Request against `main`.
4. Describe what changed and why in the PR description.

## Reporting Issues

- Use [GitHub Issues](https://github.com/ApodexAI/FrontierAgent/issues) to
  report bugs or request features.
- Include steps to reproduce, expected behavior, and actual behavior.
- Attach relevant logs or error messages.

## Code Style

- Follow existing conventions in the codebase.
- Add docstrings to public functions and classes.
- Use type annotations throughout.
- Keep lines ≤ 100 characters (configured in `pyproject.toml`).

## License

By contributing, you agree that your contributions will be licensed under the
[Apache License 2.0](LICENSE).
