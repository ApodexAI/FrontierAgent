"""LLM connection + sampling settings for the terminal agent.

``ModelConfig`` is the resolved LLM config a session runs with. It is built
by the YAML profile loader (:mod:`apodex.profiles`), which pulls
model/base_url/limits from a profile and secrets (``${OPENAI_API_KEY}`` …)
from ``.env`` — either via ``config/providers.yaml`` (``llm.provider:``) or
explicit ``${VAR}`` refs. Secrets live only in ``.env``; everything else
lives in the profile YAML.
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING
from urllib.parse import urlsplit

if TYPE_CHECKING:
    from apodex.profiles import AgentProfile


@dataclass
class ModelConfig:
    """LLM connection + sampling settings for the coding agent."""

    model: str
    api_key: str
    base_url: str | None
    temperature: float = 0.0
    # Nucleus / top-k sampling. ``None`` leaves the server's own default, which
    # is what every profile got before these existed. Neither is a
    # Chat-Completions parameter on ``OpenAIClient``, so ``build_llm`` sends both
    # through ``extra_body`` (vLLM / SGLang read them there).
    #
    # They only mean anything once ``temperature`` is off 0: at 0 SGLang takes
    # the argmax path and does no probabilistic sampling, so both are inert.
    top_p: float | None = None
    top_k: int | None = None
    max_tokens: int = 8192
    # Model context window (tokens) — drives the context-fill indicator and the
    # compaction limit. Override with $OPENAI_CONTEXT_WINDOW for big/small models.
    context_window: int = 128_000

    @property
    def redacted_key(self) -> str:
        k = self.api_key or ""
        if len(k) <= 8:
            return "***" if k else "(none)"
        return f"{k[:4]}…{k[-4:]}"


@dataclass(frozen=True)
class RuntimeConfigIssue:
    """One secret-free runtime configuration finding."""

    code: str
    message: str
    env_var: str | None = None
    blocking: bool = True


@dataclass(frozen=True)
class RuntimeConfigStatus:
    """A safe, immutable view of resolved runtime configuration.

    It intentionally has no API-key or raw-URL field. This makes the object
    safe to pass to line-mode renderers, the TUI, logs, and screenshots.
    """

    mode: str
    profile_name: str
    profile_path: str
    provider: str
    model: str
    endpoint_host: str | None
    api_key_env: str | None
    api_key_configured: bool
    issues: tuple[RuntimeConfigIssue, ...] = ()

    @property
    def errors(self) -> tuple[RuntimeConfigIssue, ...]:
        return tuple(issue for issue in self.issues if issue.blocking)

    @property
    def warnings(self) -> tuple[RuntimeConfigIssue, ...]:
        return tuple(issue for issue in self.issues if not issue.blocking)

    @property
    def ok(self) -> bool:
        return not self.errors


_UNRESOLVED_ENV_RE = re.compile(r"\$(?:\{|[A-Z_])")


def _configured(value: str | None) -> bool:
    stripped = (value or "").strip()
    return bool(stripped) and not _UNRESOLVED_ENV_RE.search(stripped)


def inspect_runtime_config(
    cfg: ModelConfig,
    *,
    profile: AgentProfile,
    mode: str | None = None,
    environ: Mapping[str, str] | None = None,
) -> RuntimeConfigStatus:
    """Perform a local, structural runtime preflight with no network calls."""
    env = os.environ if environ is None else environ
    active_mode = mode or profile.name
    provider = profile.provider or "custom"
    api_key_env = profile.api_key_env
    key_configured = _configured(cfg.api_key)
    if provider != "local" and (cfg.api_key or "").strip() == "EMPTY":
        key_configured = False

    issues: list[RuntimeConfigIssue] = []
    if not key_configured:
        source = f" ({api_key_env})" if api_key_env else ""
        issues.append(RuntimeConfigIssue(
            code="missing_api_key",
            message=f"API key{source} is missing for provider {provider}.",
            env_var=api_key_env,
        ))

    model = (cfg.model or "").strip()
    if not model:
        issues.append(RuntimeConfigIssue(
            code="missing_model",
            message="The active model is empty.",
            env_var=profile.model_env,
        ))

    endpoint_host: str | None = None
    if cfg.base_url:
        try:
            parsed = urlsplit(cfg.base_url)
            if parsed.scheme.lower() not in {"http", "https"} or not parsed.hostname:
                raise ValueError
            endpoint_host = parsed.hostname
        except (TypeError, ValueError):
            issues.append(RuntimeConfigIssue(
                code="invalid_base_url",
                message="The provider base URL must be a valid HTTP(S) URL.",
                env_var=profile.base_url_env,
            ))

    # Gate the search credentials on the tools the profile actually binds, not
    # on the mode name. Keying this on ``research`` meant it never fired: the
    # terminal only exposes ``react`` and ``agent_team``, and both bind
    # web_search and web_fetch, so a missing key first surfaced as an error
    # string inside a tool result.
    tool_names = frozenset(getattr(profile, "tool_names", ()) or ())
    if "web_search" in tool_names and not _configured(env.get("SERPER_API_KEY")):
        issues.append(RuntimeConfigIssue(
            code="missing_serper_api_key",
            message=(
                "SERPER_API_KEY is not set; web_search will return an error "
                "instead of results for every query this session makes."
            ),
            env_var="SERPER_API_KEY",
            # A warning, not a blocker: web_search is one of seven tools these
            # profiles bind, so a local coding session has no use for the key
            # and must not be refused a startup over it.
            blocking=False,
        ))
    if "web_fetch" in tool_names and not _configured(env.get("JINA_API_KEY")):
        issues.append(RuntimeConfigIssue(
            code="missing_jina_api_key",
            message=(
                "JINA_API_KEY is missing; web_fetch will use its direct-fetch fallback."
            ),
            env_var="JINA_API_KEY",
            blocking=False,
        ))

    return RuntimeConfigStatus(
        mode=active_mode,
        profile_name=profile.name,
        profile_path=str(profile.path),
        provider=provider,
        model=model,
        endpoint_host=endpoint_host,
        api_key_env=api_key_env,
        api_key_configured=key_configured,
        issues=tuple(issues),
    )


def format_runtime_config_status(status: RuntimeConfigStatus) -> str:
    """Format safe runtime metadata for line mode or TUI display."""
    source = f" ({status.api_key_env})" if status.api_key_env else ""
    endpoint = status.endpoint_host or "provider default"
    profile_path = status.profile_path or "unknown"
    lines = [
        f"profile: {status.profile_name} ({profile_path})",
        f"mode: {status.mode}",
        f"provider: {status.provider}",
        f"model: {status.model or 'missing'}",
        f"endpoint: {endpoint}",
        f"API key{source}: {'configured' if status.api_key_configured else 'missing'}",
    ]
    lines.extend(f"error: {issue.message}" for issue in status.errors)
    lines.extend(f"warning: {issue.message}" for issue in status.warnings)
    return "\n".join(lines)


def format_preflight_errors(status: RuntimeConfigStatus) -> str:
    """Format actionable, copyable guidance without rendering secret data."""
    lines = ["error: runtime configuration preflight failed"]
    for issue in status.errors:
        lines.append(f"- {issue.message}")
    env_vars = list(dict.fromkeys(
        issue.env_var for issue in status.errors if issue.env_var
    ))
    if env_vars:
        lines.append("Set the missing or invalid values and retry, for example:")
        lines.extend(f"  export {name}=..." for name in env_vars)
    else:
        lines.append("Update the active profile's llm configuration and retry.")
    lines.append("  frontier-agent --cwd .")
    return "\n".join(lines)


_USER_SETTINGS_PATH = os.path.expanduser("~/.config/apodex/settings.json")


@dataclass
class UserSettings:
    """Persistent user preferences across CLI/TUI sessions (~/.config/apodex/settings.json)."""

    theme: str = "tokyo-night"
    workflow: str = "react"
    auto_approve: bool = False
    auto_for_me: bool = False
    verbose: bool = True
    plan_mode: bool = False
    path: str = _USER_SETTINGS_PATH

    @classmethod
    def load(cls, path: str = _USER_SETTINGS_PATH) -> UserSettings:
        import json
        try:
            with open(path, encoding="utf-8") as f:
                d = json.load(f)
            return cls(
                theme=str(d.get("theme") or "tokyo-night"),
                workflow=str(d.get("workflow") or "react"),
                auto_approve=bool(d.get("auto_approve", False)),
                auto_for_me=bool(d.get("auto_for_me", False)),
                verbose=bool(d.get("verbose", True)),
                plan_mode=bool(d.get("plan_mode", False)),
                path=path,
            )
        except Exception:
            return cls(path=path)

    def save(self) -> None:
        import json
        try:
            os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump({
                    "theme": self.theme,
                    "workflow": self.workflow,
                    "auto_approve": self.auto_approve,
                    "auto_for_me": self.auto_for_me,
                    "verbose": self.verbose,
                    "plan_mode": self.plan_mode,
                }, f, indent=2)
        except Exception:
            pass


__all__ = [
    "ModelConfig",
    "RuntimeConfigIssue",
    "RuntimeConfigStatus",
    "UserSettings",
    "format_preflight_errors",
    "format_runtime_config_status",
    "inspect_runtime_config",
]
