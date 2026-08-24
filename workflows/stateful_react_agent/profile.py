"""Native profile loader and LLM builders for the stateful ReAct agent."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from frontier_agent.core.runtime.loop.model_profile import ThinkingFormat

from frontier_agent.core.llm import LLMClient

logger = logging.getLogger(__name__)

_PROFILES_DIR = Path(__file__).resolve().parent / "profiles"
_PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Retired profile names kept resolvable so saved command lines, CI jobs and
# pinned ``workflow_profile:`` values don't hard-fail with FileNotFoundError
# after a rename. Mirrors ``workflows/agent_team/profile.py``.
_PROFILE_ALIASES = {
    "default": "simple",
    "keep5": "benchmark",
    "Apodex1.1-solve": "tui",
}

# An alias must not silently change what a pinned name DOES. The retired
# ``default`` profile retained every tool result (``keep_last_k: -1``) while
# ``simple`` blanks after 5, and eight ``scripts/evaluate-*.sh`` runs plus the
# ``officeqa``/``gdpval``/``apex`` sandbox defaults are still pinned to
# ``--profile default``; remapping the name alone would quietly make those runs
# incomparable with every earlier result. Applied before caller ``overrides``,
# so an explicit override still wins.
_ALIAS_OVERRIDES: dict[str, dict[str, Any]] = {
    "default": {"agent": {"keep_last_k": -1}},
}


def _deep_merge(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge profile overrides without importing another workflow."""
    merged = dict(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_react_profile(
    name: str,
    *,
    overrides: dict[str, Any] | None = None,
    inline: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Load a ReAct profile YAML with env-var resolution."""
    import yaml
    from dotenv import load_dotenv

    from frontier_agent.infra.config import _resolve_env_vars

    load_dotenv(_PROJECT_ROOT / ".env", override=False)

    if inline is not None:
        raw: Any = inline
    else:
        path = _PROFILES_DIR / f"{_PROFILE_ALIASES.get(name, name)}.yaml"
        if not path.exists():
            raise FileNotFoundError(f"ReAct profile not found: {path}")
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))

    resolved = _resolve_env_vars(raw)
    if inline is None and (alias_overrides := _ALIAS_OVERRIDES.get(name)):
        resolved = _deep_merge(resolved, alias_overrides)
    if overrides:
        resolved = _deep_merge(resolved, overrides)
    return resolved


def _resolve_thinking_format(profile: dict[str, Any]) -> ThinkingFormat:
    from frontier_agent.core.runtime.loop.model_profile import (
        infer_thinking_format,
        is_thinking_format,
    )
    from frontier_agent.infra.protocol_client import (
        protocol_of,
        thinking_format_for_protocol,
    )

    explicit = (profile.get("agent") or {}).get("thinking_format")
    if explicit:
        if is_thinking_format(explicit):
            return explicit
        logger.warning(
            "profile agent.thinking_format=%r is not a known format — ignoring",
            explicit,
        )
    # Native Anthropic / OpenAI Responses return content as a typed block list
    # → content_block so the parser keeps the verbatim blocks.
    by_protocol = thinking_format_for_protocol(protocol_of(profile.get("llm") or {}))
    if by_protocol:
        return by_protocol
    model_id = (profile.get("llm") or {}).get("model")
    return infer_thinking_format(model_id, default="tag")


def create_react_llm(profile: dict[str, Any]) -> LLMClient:
    """Create the ReAct LLM from a profile dict, keyed on ``llm.protocol``.

    - ``anthropic`` → native Anthropic Messages API (thinking + signature).
    - ``responses`` → OpenAI Responses API (reasoning + encrypted_content).
    - ``chat_completions`` (default) → OpenAI-compatible Chat Completions.
    """
    from frontier_agent.infra.llm import with_provider_stamp
    from frontier_agent.infra.openai_client import OpenAIClient
    from frontier_agent.infra.protocol_client import build_protocol_client

    cfg = profile["llm"]
    provider = cfg.get("_provider_label") or cfg.get("provider") or "openai"
    protocol_client = build_protocol_client(cfg, title="FrontierAgent-StatefulReAct")
    if protocol_client is not None:
        return with_provider_stamp(protocol_client, str(provider))

    extra_body: dict[str, Any] = dict(cfg.get("extra_body") or {})

    if (top_p := cfg.get("top_p")) is not None:
        extra_body.setdefault("top_p", top_p)
    repetition_penalty = cfg.get("repetition_penalty")
    if (
        repetition_penalty is not None
        and repetition_penalty != 1.0
        and "repetition_penalty" not in extra_body
    ):
        extra_body["repetition_penalty"] = repetition_penalty

    client = OpenAIClient(
        model=cfg["model"],
        api_key=cfg.get("api_key", "dummy"),
        base_url=cfg.get("base_url"),
        temperature=cfg.get("temperature", 0.0),
        # int(): ``max_tokens`` may come from a ``${OPENAI_MAX_TOKENS:-…}``
        # placeholder, and env-var substitution always yields a string. It is
        # forwarded verbatim into the request body, where a quoted number is a
        # 400 from the endpoint.
        max_completion_tokens=int(cfg.get("max_tokens") or 65536),
        default_headers={
            "HTTP-Referer": "frontier_agent",
            "X-Title": "FrontierAgent-StatefulReAct",
        },
        extra_body=extra_body or None,
    )
    return with_provider_stamp(client, str(provider))


def build_react_model_profile(profile: dict[str, Any]) -> Any:
    """Build a ModelProfile from a ReAct profile dict."""
    from frontier_agent.core.runtime.loop.model_profile import ModelProfile
    from frontier_agent.infra.protocol_client import protocol_of

    return ModelProfile(
        model_id=profile["llm"]["model"],
        provider=str(profile["llm"].get("provider") or "openai"),
        thinking_format=_resolve_thinking_format(profile),
        protocol=protocol_of(profile.get("llm") or {}),
    )
