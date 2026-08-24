"""Runtime/infra configuration system — .env + optional YAML overlay."""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)

CONFIG_VERSION = 1

# Resolve config from the repository root, independent of the caller's cwd.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_YAML_PATH = _REPO_ROOT / "config.yaml"
_ENV_PATH = _REPO_ROOT / ".env"


def _load_env_file() -> None:
    """Load ``.env`` into ``os.environ`` for direct environment readers."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        logger.debug("python-dotenv not installed — .env not loaded into os.environ")
        return
    load_dotenv(_ENV_PATH, override=False)


_load_env_file()


def _resolve_env_vars(value: Any) -> Any:
    """Resolve ``$VAR``, defaults, and required variables recursively."""
    if isinstance(value, str):
        def _replace_braced(m: re.Match[str]) -> str:
            var_name, op, payload = m.group(1), m.group(2), m.group(3)
            if op == "?":
                resolved = os.environ.get(var_name)
                if not resolved:
                    raise ValueError(
                        payload or f"required environment variable {var_name} is not set"
                    )
                return resolved
            default = payload if payload is not None else ""
            return os.environ.get(var_name, default)

        value = re.sub(r"\$\{(\w+)(?::([-?])([^}]*))?\}", _replace_braced, value)

        def _replace_simple(m: re.Match[str]) -> str:
            # ``group(0)`` is the whole match and ``group(1)`` the mandatory
            # name subpattern, so both always participate; binding them states
            # that for the checker, which otherwise reads the environ lookup as
            # possibly returning None.
            whole = m.group(0)
            name = m.group(1) or ""
            return os.environ.get(name, whole)

        value = re.sub(r"\$([A-Z_][A-Z0-9_]*)", _replace_simple, value)
        return value

    if isinstance(value, dict):
        return {k: _resolve_env_vars(v) for k, v in value.items()}

    if isinstance(value, list):
        return [_resolve_env_vars(item) for item in value]

    return value


def _load_yaml_config() -> dict[str, Any]:
    """Load and resolve config.yaml, returning empty config on ordinary errors."""
    if not _YAML_PATH.is_file():
        return {}

    try:
        import yaml
    except ImportError:
        logger.debug("PyYAML not installed — skipping config.yaml")
        return {}

    try:
        raw = yaml.safe_load(_YAML_PATH.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            logger.warning("config.yaml root is not a dict — ignoring")
            return {}

        file_version = raw.pop("config_version", None)
        if file_version is not None and file_version != CONFIG_VERSION:
            logger.warning(
                "config.yaml version %s != expected %s. "
                "Run 'diff config.yaml config.example.yaml' to see changes.",
                file_version, CONFIG_VERSION,
            )

        resolved = _resolve_env_vars(raw)
        return resolved
    except ValueError:
        # Required variables must fail fast instead of discarding the YAML.
        raise
    except Exception as e:
        logger.warning("Failed to load config.yaml: %s", e)
        return {}


def _flatten_yaml(data: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """Flatten nested YAML keys with underscore-separated prefixes."""
    flat: dict[str, Any] = {}
    for key, value in data.items():
        full_key = f"{prefix}_{key}" if prefix else key
        if isinstance(value, dict):
            flat.update(_flatten_yaml(value, full_key))
        else:
            flat[full_key] = value
    return flat


class FrontierAgentConfig(BaseSettings):
    """Central configuration loaded from env + optional YAML overlay."""

    model_config = SettingsConfigDict(
        env_file=_ENV_PATH,
        env_file_encoding="utf-8",
        extra="ignore",
    )

    llm_provider: str = "openai"  # openai | anthropic | qwen | deepseek
    openai_api_key: str = ""
    openai_model: str = "gpt-4o"
    openai_base_url: str = "https://api.openai.com/v1"
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-sonnet-4-20250514"
    anthropic_base_url: str = ""
    anthropic_thinking: bool = False
    anthropic_thinking_type: str = "adaptive"
    anthropic_thinking_display: str = "summarized"
    anthropic_thinking_budget: int = 8192
    anthropic_effort: str = ""
    qwen_api_key: str = ""
    qwen_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    qwen_model: str = "qwen-max"
    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_model: str = "deepseek-chat"

    llm_max_tokens: int = 16384  # per-call max output tokens (thinking models need ≥8K)

    # Entries may override provider/model/credentials and select failure triggers.
    llm_fallback_chain: list[dict] = []
    llm_fallback_model: str = ""  # empty = disabled. e.g. "gpt-4o"
    llm_fallback_max_retries: int = 2  # retries on primary before fallback
    llm_fallback_cooldown: int = 60  # seconds to stay on fallback

    @field_validator("llm_fallback_chain", mode="before")
    @classmethod
    def _parse_fallback_chain(cls, value: Any) -> Any:
        """Accept either a list (from YAML) or a JSON-encoded string (from env)."""
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                return []
            try:
                parsed = json.loads(stripped)
            except json.JSONDecodeError:
                logger.warning(
                    "llm_fallback_chain: failed to JSON-parse %r — treating as empty",
                    stripped[:80],
                )
                return []
            return parsed if isinstance(parsed, list) else []
        return value or []

    # Empty vision credentials fall back to the primary OpenAI-compatible client.
    vision_api_key: str = ""
    vision_base_url: str = ""
    vision_model: str = ""
    vision_default_model: str = "google/gemini-2.5-flash"

    @property
    def effective_vision_api_key(self) -> str:
        return self.vision_api_key or self.openai_api_key

    @property
    def effective_vision_base_url(self) -> str:
        return self.vision_base_url or self.openai_base_url

    @property
    def effective_vision_model(self) -> str:
        if self.vision_model:
            return self.vision_model
        primary = (self.openai_model or "").lower()
        if (
            "flash" in primary or "gpt-4o" in primary or "gpt-5" in primary
            or "claude" in primary or "gemini" in primary
        ):
            return self.openai_model
        return self.vision_default_model

    serper_api_key: str = ""
    serper_base_url: str = "https://google.serper.dev"
    jina_api_key: str = ""
    jina_base_url: str = "https://r.jina.ai"
    # Suffix-matched domains blocked from both search and fetch.
    web_domain_blacklist_extra: str = ""
    # Domain-scoped, case-insensitive phrase filtering for search snippets.
    web_snippet_block_domains: str = ""
    web_snippet_block_phrases: str = ""

    unpaywall_email: str = ""     # required by Unpaywall — falls back to a no-reply
    ncbi_api_key: str = ""        # optional; raises PubMed/E-utils rate limit

    # Optional inexpensive model for extracting content from fetched pages.
    summary_llm_api_key: str = ""
    summary_llm_base_url: str = ""
    summary_llm_model: str = ""

    # auto tries E2B, then bwrap, and fails closed; local aliases bwrap.
    sandbox_backend: str = "auto"  # e2b | bwrap | local | auto
    # Memory caps are per process; concurrency determines aggregate use. Zero disables.
    sandbox_local_mem_mb: int = 640
    sandbox_local_max_concurrency: int = 2
    sandbox_e2b_mem_mb: int = 896
    sandbox_bwrap_mem_mb: int = 12 * 1024
    # The container backend uses RLIMIT_DATA to avoid rejecting large-VSZ runtimes.
    sandbox_container_mem_mb: int = 1024
    # Retain bounded output while continuing to drain the child process.
    sandbox_output_cap_kb: int = 8192
    run_python_timeout_s: int = 90
    # Clamp model-supplied timeouts separately from the default.
    run_python_max_timeout_s: int = 300
    tool_exec_result_max_chars: int = 8_000
    # Shape of the inline preview kept when a tool result overflows its cap.
    # ``middle`` keeps the head AND the tail, because the verdict of an exec
    # result (pytest summary, linker error, exit status) lives at the END and a
    # head-only cut hides exactly the line the model needs. ``head`` restores
    # the legacy shape so the two can be compared on one build.
    # ``auto`` decides per tool from ToolMeta.result_is_ranked: head for a
    # relevance-ranked search result (its tail is its worst hits), middle for
    # sequential output (its tail is the verdict). The default, because the live
    # A/B found no accuracy difference between the uniform shapes in either
    # direction and mildly favoured head on a search-heavy benchmark — which is
    # the case auto routes to head. ``middle`` / ``head`` force one shape on
    # every tool, which is what the A/B arms pin. See
    # docs/tool-result-truncation-ab.md.
    tool_result_truncation: str = "auto"  # auto | middle | head
    # Which shape of summary Tier 2 compaction asks for. ``research`` preserves
    # candidates / sources / queries; ``handoff`` preserves exact commands,
    # paths, returned values and unverified claims, for long-run coding.
    # ``auto`` dispatches on the conversation's tool mix, which leaves
    # web-dominated runs on exactly the research shape they already had.
    compaction_prompt_style: str = "auto"  # auto | research | handoff
    # Global inline cap applied by the loop to EVERY tool result, including the
    # tools that set ``max_result_chars=0`` (web_fetch / web_search / read_file).
    # Tunable so a stress run can force truncation on those tools the way
    # TOOL_EXEC_RESULT_MAX_CHARS already can for bash. 0 keeps the default.
    tool_result_max_chars: int = 0

    # Keep agent and E2B pool concurrency aligned to avoid predictable spillover.
    agent_bus_max_parallel: int = 8
    e2b_api_key: str = ""
    e2b_template: str = "base"
    e2b_timeout: int = 1800
    # Pool size is per worker process; account-wide capacity is not coordinated.
    e2b_pool_size: int = 8
    e2b_pool_lease_timeout_s: float = 5.0

    host: str = "0.0.0.0"
    port: int = 8000

_config: FrontierAgentConfig | None = None
_config_mtime: float = 0.0
_yaml_mtime: float = 0.0


def _get_mtimes() -> tuple[float, float]:
    """Get modification times of .env and config.yaml."""
    env_mt = _ENV_PATH.stat().st_mtime if _ENV_PATH.exists() else 0.0
    yaml_mt = _YAML_PATH.stat().st_mtime if _YAML_PATH.exists() else 0.0
    return env_mt, yaml_mt


def get_config(force_reload: bool = False) -> FrontierAgentConfig:
    """Singleton config accessor with mtime-based auto-reload.

    Checks both .env and config.yaml modification times — if either file
    has been updated since last load, automatically reconstructs the config.
    """
    global _config, _config_mtime, _yaml_mtime

    try:
        env_mt, yaml_mt = _get_mtimes()
    except OSError:
        env_mt, yaml_mt = 0.0, 0.0

    if _config is None or force_reload or env_mt > _config_mtime or yaml_mt > _yaml_mtime:
        yaml_overrides = _load_yaml_config()

        if yaml_overrides:
            flat_overrides = _flatten_yaml(yaml_overrides)

            # Environment variables override the YAML overlay.
            for key, value in flat_overrides.items():
                env_key = key.upper()
                if env_key in os.environ:
                    continue
                if isinstance(value, str):
                    os.environ[env_key] = value
                elif isinstance(value, (list, dict)):
                    os.environ[env_key] = json.dumps(value)
                else:
                    os.environ[env_key] = str(value)

        _config = FrontierAgentConfig()
        _config_mtime = env_mt
        _yaml_mtime = yaml_mt

        if yaml_overrides:
            logger.debug("Config loaded from .env + config.yaml (%d YAML overrides)", len(yaml_overrides))

    return _config
