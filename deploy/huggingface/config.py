"""Environment-driven configuration for the Hugging Face Space demo.

Every knob is an environment variable. Switching model, endpoint or secret is
an ops action (Space *Variables* / *Secrets*), never a code edit — that is the
contract this module enforces, and ``preflight`` is what turns a
misconfiguration into an actionable message instead of a 500 on first prompt.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from frontier_agent.components.task_board_types import BOARD_TOOLS

#: The only workflow this demo exposes (P0). ``agent_team`` is out of scope.
SUPPORTED_WORKFLOWS: tuple[str, ...] = ("react",)

#: Hosts that serve Hugging Face *web pages*, never an OpenAI-compatible API.
#: Pointing ``OPENAI_BASE_URL`` at a model repo page is the single most likely
#: deployment mistake, so it is rejected by name.
_HF_WEBSITE_HOSTS: frozenset[str] = frozenset(
    {"huggingface.co", "www.huggingface.co", "hf.co", "www.hf.co"},
)

#: Demo-safe toolset: real research plus text deliverables, no arbitrary shell.
#: ``bash`` / ``run_python_code`` / ``download_file`` are deliberately absent —
#: see ``security.py`` for the full rationale and the deny list.
#:
#: ``create_file`` (the office-document deliverable tool) is deliberately NOT
#: here, but no longer because it is broken: the ``E2BIG`` defect that made every
#: call fail was fixed upstream (the writer bundle goes over stdin now) and its
#: Office writers are hard dependencies of the package. It stays out because it
#: is the only tool left that spawns a subprocess — running a model-authored JSON
#: program under this app's own interpreter — and because ``containment`` checks
#: top-level path arguments, not the secondary destinations nested inside its
#: ``ops``. See §"What the agent cannot do" in README.md. ``write_file`` covers
#: text deliverables and needs no shell at all.
DEFAULT_ALLOWED_TOOLS: tuple[str, ...] = (
    "web_search",
    "web_fetch",
    "read_file",
    "write_file",
    "glob_search",
    "grep_search",
    "add_task",
    "update_task",
)

_TRUE = frozenset({"1", "true", "yes", "on"})
_FALSE = frozenset({"0", "false", "no", "off"})


class ConfigError(ValueError):
    """A demo configuration value is unusable."""


@dataclass(frozen=True)
class Issue:
    """One preflight finding. ``field`` names the env var to fix."""

    field: str
    message: str

    def __str__(self) -> str:  # pragma: no cover - trivial
        return f"{self.field}: {self.message}"


@dataclass(frozen=True)
class Preflight:
    """Result of validating a :class:`DemoConfig` without any network call."""

    errors: tuple[Issue, ...] = ()
    warnings: tuple[Issue, ...] = ()

    @property
    def ok(self) -> bool:
        return not self.errors

    def format(self) -> str:
        lines: list[str] = []
        if self.errors:
            lines.append("Configuration errors (the demo cannot run):")
            lines += [f"  ✗ {issue}" for issue in self.errors]
        if self.warnings:
            lines.append("Configuration warnings:")
            lines += [f"  ! {issue}" for issue in self.warnings]
        return "\n".join(lines)


def _get(env: Mapping[str, str], name: str, default: str = "") -> str:
    return str(env.get(name, default) or "").strip()


def _get_bool(env: Mapping[str, str], name: str, default: bool) -> bool:
    raw = _get(env, name).lower()
    if raw in _TRUE:
        return True
    if raw in _FALSE:
        return False
    return default


def _get_int(env: Mapping[str, str], name: str, default: int, *, minimum: int = 1) -> int:
    raw = _get(env, name)
    if not raw:
        return default
    try:
        value = int(float(raw))
    except ValueError as exc:
        raise ConfigError(f"{name} must be an integer, got {raw!r}") from exc
    return max(value, minimum)


def _get_float(
    env: Mapping[str, str], name: str, default: float, *, minimum: float = 1.0,
) -> float:
    raw = _get(env, name)
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be a number, got {raw!r}") from exc
    return max(value, minimum)


def _get_list(env: Mapping[str, str], name: str, default: tuple[str, ...]) -> tuple[str, ...]:
    raw = _get(env, name)
    if not raw:
        return default
    items = [part.strip() for part in raw.replace("\n", ",").split(",")]
    return tuple(dict.fromkeys(item for item in items if item))


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(value, high))


def _default_runtime_root(env: Mapping[str, str]) -> Path:
    """Pick a writable root for session data.

    ``/data`` is Hugging Face's persistent-storage mount when enabled; the
    ``$HOME`` fallback keeps an ephemeral Space (and a plain ``docker run``)
    working without extra setup. Never inside the service checkout — the
    runtime's own path guard refuses to write there (see
    ``plugins/tools/_path_auth._is_isolated_workspace_root``).
    """
    explicit = _get(env, "DEMO_RUNTIME_ROOT")
    if explicit:
        return Path(explicit).expanduser()
    data = Path("/data")
    if data.is_dir() and os.access(data, os.W_OK):
        return data / "frontier-agent-demo"
    home = _get(env, "HOME") or "/tmp"
    return Path(home).expanduser() / "frontier-agent-demo"


@dataclass(frozen=True)
class DemoConfig:
    """The resolved, immutable demo configuration."""

    # What the UI advertises. ``model_id`` / ``model_url`` are *labels*: the
    # request always goes to ``openai_base_url``.
    workflow: str
    model_id: str
    model_url: str

    # The OpenAI-compatible endpoint the FrontierAgent runtime calls.
    openai_model: str
    openai_base_url: str
    openai_api_key: str = field(repr=False, default="")

    # Runtime limits.
    max_turns: int = 24
    max_output_tokens: int = 4096
    task_timeout_s: float = 600.0
    max_concurrency: int = 1
    queue_size: int = 4
    session_ttl_s: float = 3600.0
    max_prompt_chars: int = 4000

    # Behaviour / safety.
    public_mode: bool = True
    reporter: bool = False
    allowed_tools: tuple[str, ...] = DEFAULT_ALLOWED_TOOLS
    runtime_root: Path = Path("/tmp/frontier-agent-demo")
    #: Selects the runtime's filesystem convention (``plugins/tools/_sandbox``).
    #: ``native`` is what the demo wants: it makes the react prompt name this
    #: session's *real* directories (``_runtime.render_system_prompt_notes``),
    #: so each visitor's workspace and outputs are genuinely separate paths.
    #: ``container`` instead hard-codes ``/workspace`` and ``/outputs`` in the
    #: prompt, which cannot be per-session. Neither mode weakens the demo,
    #: because every command-executing tool (``bash``, ``run_python_code``) is
    #: denied outright — the only filesystem access left is the in-process,
    #: path-authorised file tools.
    sandbox_backend: str = "native"

    # Carried so ``security.py`` can redact them and preflight can check that
    # the web tools have what they need. Never rendered.
    hf_token: str = field(repr=False, default="")
    search_api_key: str = field(repr=False, default="")
    fetch_api_key: str = field(repr=False, default="")
    #: Endpoint used by ``web_fetch`` to extract page content. A *full*
    #: chat-completions URL, not an API base — that is what the tool expects.
    summary_llm_base_url: str = ""
    summary_llm_api_key: str = field(repr=False, default="")

    @property
    def workflow_profile(self) -> str:
        """The workflow profile YAML the ``react`` mode runs on.

        Read from ``apodex/profiles/react.yaml`` so the Space and the CLI can
        never drift apart on which profile ``react`` means.
        """
        from apodex.profiles import get_profile

        profile = get_profile(self.workflow)
        if not profile.workflow_profile:
            raise ConfigError(
                f"profile {self.workflow!r} declares no workflow_profile",
            )
        return profile.workflow_profile

    @property
    def pipeline_id(self) -> str:
        """The registered pipeline id backing this workflow."""
        from apodex.profiles import get_profile

        profile = get_profile(self.workflow)
        if not profile.workflow:
            raise ConfigError(f"profile {self.workflow!r} declares no workflow")
        return profile.workflow

    @property
    def research_wall_time_s(self) -> float:
        """Research budget: the loop must stop in time to still answer."""
        return max(30.0, self.task_timeout_s * 0.6)

    @property
    def wall_deadline_reserve_s(self) -> float:
        """Time held back from research for finalisation/reporting."""
        return max(20.0, self.task_timeout_s * 0.25)

    @property
    def tool_timeout_s(self) -> float:
        """Per-tool ceiling.

        Kept well under half the task wall: a tool that starts just before the
        research deadline must still finish inside the wall, or the run is
        killed with no answer (see ``check_wall_feasibility``).
        """
        return _clamp(self.task_timeout_s * 0.25, 30.0, 120.0)

    @property
    def llm_timeout_s(self) -> float:
        """Per-LLM-call ceiling, and the finalisation rescue budget.

        The upper bound is generous because the intended models are large
        reasoning models: they can spend minutes on ``reasoning_content``
        before the first visible token, and a tighter cap would abort healthy
        calls. The task wall, not this value, is what bounds a run.
        """
        return _clamp(self.task_timeout_s * 0.4, 60.0, 600.0)

    @property
    def logical_call_timeout_s(self) -> float:
        """Ceiling for one *logical* LLM call — admission wait, every attempt,
        and the backoff between them.

        The profile ships 900s, which is longer than this demo's whole task
        wall: a single wedged call would then be cut off by the adapter's
        ``asyncio.wait_for`` backstop instead of the loop landing cleanly with
        whatever answer it has. Half the wall leaves room for one full retry of
        a ``llm_timeout_s`` attempt while keeping the cooperative path in
        charge; the floor keeps it from ever undercutting a single attempt.
        """
        return _clamp(
            self.task_timeout_s * 0.5, self.llm_timeout_s, self.task_timeout_s,
        )

    @property
    def reasoning_only_timeout_s(self) -> float:
        """How long a reply may stay reasoning-only before it is resampled.

        Wall clock, so its token equivalent shrinks as endpoint concurrency
        rises — see the note in the workflow profile. Proportional to the task
        wall rather than fixed, because a demo configured for a short wall
        cannot afford the profile's 120s on one runaway attempt.
        """
        return _clamp(self.task_timeout_s * 0.2, 30.0, 120.0)

    @property
    def reasoning_only_max_tokens(self) -> int:
        """Reasoning-only token cap, tied to the demo's own output budget.

        The profile's 16384 is inert here: reasoning tokens are drawn from
        ``max_tokens``, so a 4096-token budget can never reach it and the
        load-invariant half of the guard would never fire — leaving only the
        wall-clock half, which is the half the profile says not to trust.

        The configured output budget may be as low as 256 tokens. Keep the
        guard strictly below it even when the preferred 512-token floor cannot
        fit, otherwise the token guard becomes unreachable again.
        """
        preferred = max(512, int(self.max_output_tokens * 0.6))
        return min(preferred, self.max_output_tokens - 1)

    @property
    def hard_wall_time_s(self) -> int:
        """Scheduler-level hard ceiling for the whole graph execution."""
        return int(self.task_timeout_s)

    @property
    def effective_concurrency(self) -> int:
        """Runs actually executed at once, whatever ``max_concurrency`` asks for.

        The single source of truth for the clamp, so nothing can report the
        requested number as though it were the enforced one. See the warning in
        :func:`preflight` for why it is 1.
        """
        return 1

    @property
    def secrets(self) -> tuple[str, ...]:
        """Every value that must never reach logs, the UI, or a download."""
        return tuple(value for value in (
            self.openai_api_key, self.hf_token,
            self.search_api_key, self.fetch_api_key, self.summary_llm_api_key,
        ) if value)

    def profile_overrides(self) -> dict[str, Any]:
        """Demo limits expressed as ``load_react_profile`` overrides.

        Using the runtime's own ``metadata['profile_overrides']`` seam keeps
        every Space-specific bound out of the profile YAML and out of the
        core runtime.
        """
        return {
            "llm": {
                "model": self.openai_model,
                "base_url": self.openai_base_url,
                "api_key": self.openai_api_key,
                "max_tokens": self.max_output_tokens,
            },
            "agent": {
                "main_max_turns": self.max_turns,
                "max_turns": self.max_turns,
                "agent_tools": list(self.allowed_tools),
                "reporter": self.reporter,
                "research_wall_time_s": self.research_wall_time_s,
                "wall_deadline_reserve_s": self.wall_deadline_reserve_s,
                # Keep tool and LLM calls inside the task wall.
                "tool_timeout_s": self.tool_timeout_s,
                "llm_timeout_s": self.llm_timeout_s,
                # Runaway guardrails. The profile arms these for a TUI session
                # with a 9000s research wall; a public demo's wall is two
                # orders of magnitude smaller, so every one of them has to be
                # rescaled or it is either unreachable or larger than the run.
                "logical_call_timeout_s": self.logical_call_timeout_s,
                "reasoning_only_timeout_s": self.reasoning_only_timeout_s,
                "reasoning_only_max_tokens": self.reasoning_only_max_tokens,
                "reporter_timeout_s": _clamp(self.task_timeout_s * 0.3, 30.0, 180.0),
                "reporter_phase_timeout_s": _clamp(
                    self.task_timeout_s * 0.4, 45.0, 300.0,
                ),
                # The board tools are only meaningful when they are allowed.
                "task_board": BOARD_TOOLS.issubset(self.allowed_tools),
            },
        }

    def public_summary(self) -> dict[str, str]:
        """Redaction-safe facts for the UI header. No secrets, ever."""
        return {
            "workflow": self.workflow,
            "model": self.model_id,
            "model_url": self.model_url,
            "endpoint": redact_url(self.openai_base_url),
            "served_model": self.openai_model,
            "max_turns": str(self.max_turns),
            "timeout": f"{int(self.task_timeout_s)}s",
            # The enforced value, never the requested one: reporting
            # "8" while running one at a time would be a false claim.
            "concurrency": str(self.effective_concurrency),
        }


def redact_url(url: str) -> str:
    """Return ``url`` without userinfo or query string (both can carry keys)."""
    if not url:
        return ""
    try:
        parts = urlparse(url)
    except ValueError:
        return "(unparseable URL)"
    if not parts.scheme or not parts.netloc:
        return url
    host = parts.hostname or ""
    if parts.port:
        host = f"{host}:{parts.port}"
    return f"{parts.scheme}://{host}{parts.path}" if host else url


def load_config(env: Mapping[str, str] | None = None) -> DemoConfig:
    """Build a :class:`DemoConfig` from ``env`` (defaults to ``os.environ``)."""
    env = os.environ if env is None else env

    model_id = _get(env, "HF_MODEL_ID", "apodex/Apodex-1.1-mini")
    model_url = _get(env, "HF_MODEL_URL") or (
        f"https://huggingface.co/{model_id}" if "/" in model_id else ""
    )

    max_concurrency = _get_int(env, "DEMO_MAX_CONCURRENCY", 1)

    return DemoConfig(
        workflow=_get(env, "DEMO_WORKFLOW", "react").lower(),
        model_id=model_id,
        model_url=model_url,
        openai_model=_get(env, "OPENAI_MODEL"),
        openai_base_url=_get(env, "OPENAI_BASE_URL"),
        openai_api_key=_get(env, "OPENAI_API_KEY"),
        max_turns=_get_int(env, "DEMO_MAX_TURNS", 24),
        max_output_tokens=_get_int(env, "DEMO_MAX_OUTPUT_TOKENS", 4096, minimum=256),
        task_timeout_s=_get_float(env, "DEMO_TASK_TIMEOUT_SECONDS", 600.0, minimum=30.0),
        max_concurrency=max_concurrency,
        queue_size=_get_int(env, "DEMO_QUEUE_SIZE", 4),
        session_ttl_s=_get_float(env, "DEMO_SESSION_TTL_SECONDS", 3600.0, minimum=60.0),
        max_prompt_chars=_get_int(env, "DEMO_MAX_PROMPT_CHARS", 4000, minimum=16),
        public_mode=_get_bool(env, "DEMO_PUBLIC_MODE", True),
        reporter=_get_bool(env, "DEMO_REPORTER", False),
        allowed_tools=_get_list(env, "DEMO_ALLOWED_TOOLS", DEFAULT_ALLOWED_TOOLS),
        runtime_root=_default_runtime_root(env),
        sandbox_backend=_get(env, "SANDBOX_BACKEND", "native").lower(),
        hf_token=_get(env, "HF_TOKEN"),
        search_api_key=_get(env, "SERPER_API_KEY"),
        fetch_api_key=_get(env, "JINA_API_KEY"),
        summary_llm_base_url=_get(env, "SUMMARY_LLM_BASE_URL"),
        summary_llm_api_key=_get(env, "SUMMARY_LLM_API_KEY"),
    )


def preflight(config: DemoConfig) -> Preflight:
    """Validate ``config`` locally — no network, no credentials sent anywhere."""
    errors: list[Issue] = []
    warnings: list[Issue] = []

    if config.workflow not in SUPPORTED_WORKFLOWS:
        errors.append(Issue(
            "DEMO_WORKFLOW",
            f"{config.workflow!r} is not available in this demo; "
            f"supported: {', '.join(SUPPORTED_WORKFLOWS)}",
        ))

    if not config.openai_model:
        errors.append(Issue(
            "OPENAI_MODEL",
            "not set — this is the model *name the endpoint serves* "
            "(what goes in the request body), e.g. 'Apodex-1.1-mini'",
        ))

    errors.extend(_check_base_url(config.openai_base_url))

    if not config.openai_api_key:
        errors.append(Issue(
            "OPENAI_API_KEY",
            "not set — configure it as a Space *Secret*, never a Variable",
        ))

    if not config.allowed_tools:
        errors.append(Issue(
            "DEMO_ALLOWED_TOOLS",
            "resolved to an empty toolset; the agent would have no tools",
        ))

    if config.sandbox_backend not in ("container", "native"):
        errors.append(Issue(
            "SANDBOX_BACKEND",
            f"must be 'native' (recommended: per-session workspace paths) or "
            f"'container', got {config.sandbox_backend or '(unset)'!r}. The "
            "react workflow refuses to bind filesystem tools without one of "
            "these — it fails closed rather than falling back to unisolated "
            "host execution.",
        ))
    elif config.sandbox_backend == "container":
        warnings.append(Issue(
            "SANDBOX_BACKEND",
            "'container' makes the agent's prompt name the fixed /workspace and "
            "/outputs paths, which cannot be per-session; use 'native' so each "
            "visitor gets their own directories.",
        ))

    if config.max_concurrency > config.effective_concurrency:
        warnings.append(Issue(
            "DEMO_MAX_CONCURRENCY",
            f"{config.max_concurrency} requested but the run is serialised to 1: "
            "the FrontierAgent runtime keeps per-process global state "
            "(service registry, sandbox/bash policy, mount-dir env), so "
            "concurrent runs in one process are not isolated. Scale with "
            "replicas, not with this value.",
        ))

    if config.public_mode:
        from deploy.huggingface.security import HARD_DENIED_TOOLS

        risky = sorted(set(config.allowed_tools) & HARD_DENIED_TOOLS)
        if risky:
            errors.append(Issue(
                "DEMO_ALLOWED_TOOLS",
                f"{', '.join(risky)} cannot be enabled while "
                "DEMO_PUBLIC_MODE is on (arbitrary code/download execution)",
            ))

    # web_search returns *no results* (not an error) without a Serper key, which
    # looks to a visitor like an agent that cannot research anything.
    if "web_search" in config.allowed_tools and not config.search_api_key:
        warnings.append(Issue(
            "SERPER_API_KEY",
            "not set, so web_search will return zero results and the agent will "
            "appear unable to research. Add it as a Space Secret, or drop "
            "web_search from DEMO_ALLOWED_TOOLS.",
        ))

    # The react profile uses the "aligned" web_fetch, whose page extraction is
    # itself an LLM call. Without an extraction endpoint every fetch comes back
    # as "Extraction failed", leaving the agent with search snippets only — it
    # still answers, just markedly worse, and nothing else says why.
    if "web_fetch" in config.allowed_tools and not config.summary_llm_base_url:
        warnings.append(Issue(
            "SUMMARY_LLM_BASE_URL",
            "not set, so web_fetch cannot extract page content and will return "
            "'Extraction failed' for every URL. Set it to a FULL "
            "chat-completions URL (…/v1/chat/completions — not an API base), "
            "with SUMMARY_LLM_MODEL_NAME and SUMMARY_LLM_API_KEY. Pointing it "
            "at the same endpoint as OPENAI_BASE_URL works.",
        ))
    elif config.summary_llm_base_url and not config.summary_llm_base_url.rstrip(
        "/",
    ).endswith("/chat/completions"):
        # Documented as a trap, so it has to be checked: unlike every other
        # endpoint variable here this one wants the full route, and copying the
        # OPENAI_BASE_URL style is the natural mistake.
        warnings.append(Issue(
            "SUMMARY_LLM_BASE_URL",
            f"{redact_url(config.summary_llm_base_url)} does not end in "
            "'/chat/completions'. Unlike OPENAI_BASE_URL this must be the FULL "
            "route, not an API base, or web_fetch's page extraction will fail "
            "for every URL.",
        ))

    if config.task_timeout_s > 1800:
        warnings.append(Issue(
            "DEMO_TASK_TIMEOUT_SECONDS",
            f"{int(config.task_timeout_s)}s is long for a shared public demo; "
            "one run blocks the queue for that whole time",
        ))

    # The per-call ceilings are derived from the wall but have floors of their
    # own (a large reasoning model needs a minute before its first visible
    # token). Below those floors the arithmetic stops meaning anything: one
    # model call can outlast the run that contains it, so the run is killed by
    # the adapter's backstop instead of landing with a partial answer.
    if config.llm_timeout_s > config.task_timeout_s:
        warnings.append(Issue(
            "DEMO_TASK_TIMEOUT_SECONDS",
            f"{int(config.task_timeout_s)}s is shorter than a single model call "
            f"is allowed to take ({int(config.llm_timeout_s)}s), so a run can be "
            "cut off mid-call with no answer. Raise it to at least "
            f"{int(config.llm_timeout_s * 2)}s.",
        ))

    if config.model_url and _is_hf_website(config.model_url) is False:
        warnings.append(Issue(
            "HF_MODEL_URL",
            "does not look like a Hugging Face model page; it is used only as "
            "a display link",
        ))

    return Preflight(tuple(errors), tuple(warnings))


def effective_sandbox_mode() -> str:
    """The filesystem mode the runtime will *actually* use, or ``""``.

    Not the same thing as reading ``SANDBOX_BACKEND``:
    ``plugins/tools/_sandbox._get_sandbox_backend`` resolves it through
    ``frontier_agent.infra.config.get_config()``, a singleton built on first
    access and cached thereafter. A value exported *after* something has already
    read that config is therefore ignored. Asking the runtime itself is the only
    way to know what a run will really get.
    """
    try:
        from plugins.tools._sandbox import resolve_sandbox_mode

        return str(resolve_sandbox_mode() or "")
    except Exception:
        return ""


def runtime_preflight(config: DemoConfig) -> Preflight:
    """Validate what this *process* can really do, beyond the env mapping.

    :func:`preflight` answers "is the configuration coherent?" from values
    alone, so it stays pure and cheap. This answers "will a run actually work
    here?", which means interrogating the loaded runtime.
    """
    errors: list[Issue] = []

    # Session storage is created eagerly at startup, so an unwritable root would
    # otherwise surface as a traceback from SessionStore rather than as the
    # ops-fixable configuration problem it is.
    root = config.runtime_root
    probe = root if root.exists() else root.parent
    try:
        root.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        errors.append(Issue(
            "DEMO_RUNTIME_ROOT",
            f"{root} cannot be created ({exc}). Point it at a writable "
            "directory, or give the app user write access — on a Space that "
            "usually means enabling persistent storage or leaving it unset so "
            "it falls back to $HOME.",
        ))
    else:
        if not os.access(root, os.W_OK):
            errors.append(Issue(
                "DEMO_RUNTIME_ROOT",
                f"{root} is not writable by uid {os.getuid()} "
                f"(checked via {probe}). Sessions could not be stored.",
            ))

    mode = effective_sandbox_mode()
    if mode and mode not in ("container", "native"):
        errors.append(Issue(
            "SANDBOX_BACKEND",
            f"is configured as {config.sandbox_backend!r} but the runtime "
            f"resolved {mode!r}, so the react workflow will refuse to bind its "
            "filesystem tools. This value is cached by "
            "frontier_agent.infra.config on first read, so it must be present "
            "in the environment before the process starts — set it in the "
            "image (as the Space Dockerfile does), not at runtime.",
        ))
    return Preflight(tuple(errors), ())


def _is_hf_website(url: str) -> bool:
    try:
        host = (urlparse(url).hostname or "").lower()
    except ValueError:
        return False
    return host in _HF_WEBSITE_HOSTS


def _check_base_url(base_url: str) -> list[Issue]:
    """Reject the endpoint mistakes that would otherwise fail at first prompt."""
    if not base_url:
        return [Issue(
            "OPENAI_BASE_URL",
            "not set — must be an OpenAI-compatible endpoint ending in '/v1', "
            "e.g. https://my-endpoint.example.com/v1",
        )]

    try:
        parts = urlparse(base_url)
    except ValueError:
        return [Issue("OPENAI_BASE_URL", f"is not a valid URL: {base_url!r}")]

    if parts.scheme not in ("http", "https"):
        return [Issue(
            "OPENAI_BASE_URL",
            f"must start with http:// or https://, got {base_url!r}",
        )]
    if not parts.hostname:
        return [Issue("OPENAI_BASE_URL", f"has no host: {base_url!r}")]

    if _is_hf_website(base_url):
        return [Issue(
            "OPENAI_BASE_URL",
            f"{redact_url(base_url)} is a Hugging Face *model repository web "
            "page*, not an inference API. It cannot serve "
            "POST /chat/completions. Point this at the OpenAI-compatible "
            "endpoint that serves the model (its URL ends in '/v1'); keep the "
            "model page in HF_MODEL_URL instead.",
        )]

    path = parts.path.rstrip("/")
    if path.endswith("/chat/completions"):
        return [Issue(
            "OPENAI_BASE_URL",
            "must be the API *base* URL (…/v1), not the full "
            "…/chat/completions path — the client appends the route itself.",
        )]

    return []


__all__ = [
    "DEFAULT_ALLOWED_TOOLS",
    "SUPPORTED_WORKFLOWS",
    "ConfigError",
    "DemoConfig",
    "Issue",
    "Preflight",
    "effective_sandbox_mode",
    "load_config",
    "preflight",
    "redact_url",
    "runtime_preflight",
]
