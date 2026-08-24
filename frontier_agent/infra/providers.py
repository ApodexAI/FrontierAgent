"""LLM provider registry loader."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_PATH = _PROJECT_ROOT / "config" / "providers.yaml"


# Process-wide cache keyed by absolute path so an in-process test that
# points ``FRONTIER_AGENT_PROVIDERS_PATH`` somewhere else gets a distinct
# cache slot. Value is the fully env-expanded raw yaml so callers can
# pull any top-level section (``providers``, ``model_chains``) without
# re-reading the file.
_cache: dict[str, dict[str, Any]] = {}


@dataclass(frozen=True)
class ProviderMetadata:
    """Non-secret provider registry metadata safe for diagnostics."""

    name: str
    provider_type: str
    api_key_env: str | None
    base_url_env: str | None


_ENV_REF_RE = re.compile(
    r"\$(?:\{([A-Z_][A-Z0-9_]*)(?::[-?][^}]*)?\}|([A-Z_][A-Z0-9_]*))"
)


def environment_variable_source(value: Any) -> str | None:
    """Return the first referenced environment variable, never its value."""
    if not isinstance(value, str):
        return None
    match = _ENV_REF_RE.search(value)
    return (match.group(1) or match.group(2)) if match else None


class ProviderNotFound(KeyError):
    """Raised when a profile references a provider that's not in the
    registry. Carries the requested name + the available names to make
    the failure diagnosable without re-reading the YAML."""

    def __init__(self, name: str, available: list[str]) -> None:
        self.requested = name
        self.available = available
        super().__init__(
            f"Provider {name!r} not registered in providers.yaml; "
            f"available: {sorted(available)}",
        )


def _providers_path() -> Path:
    """Resolve the providers.yaml path, honoring the env override."""
    override = os.environ.get("FRONTIER_AGENT_PROVIDERS_PATH")
    if override:
        return Path(override)
    return _DEFAULT_PATH


def _load_raw_yaml(*, refresh: bool = False) -> dict[str, Any]:
    """Load and cache the full provider yaml (env-expanded).

    Returns the fully ``${VAR}``-expanded raw dict. Callers extract
    the section they need (``providers``, ``model_chains``).
    """
    import yaml
    from dotenv import load_dotenv

    from frontier_agent.infra.config import _resolve_env_vars

    path = _providers_path()
    key = str(path.resolve())
    if not refresh and key in _cache:
        return _cache[key]

    # Load .env so env-expansion sees the same values a profile loader
    # would (matches load_swarm_profile's behavior).
    load_dotenv(_PROJECT_ROOT / ".env", override=False)

    if not path.exists():
        raise FileNotFoundError(
            f"Provider registry not found: {path}. Set "
            f"FRONTIER_AGENT_PROVIDERS_PATH or create the file from the "
            f"checked-in template."
        )

    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    resolved = _resolve_env_vars(raw)
    if not isinstance(resolved, dict):
        raise ValueError(
            f"providers.yaml must be a top-level mapping; "
            f"got {type(resolved).__name__}",
        )
    _cache[key] = resolved
    return resolved


def load_providers(*, refresh: bool = False) -> dict[str, dict[str, Any]]:
    """Load and cache the provider registry.

    Returns a dict keyed by provider name. Each value is a fully
    env-expanded dict (at minimum: ``type``, ``api_key``, ``base_url``).

    ``refresh=True`` bypasses the cache — useful for tests that mutate
    ``os.environ`` between calls.
    """
    raw = _load_raw_yaml(refresh=refresh)
    providers = raw.get("providers") or {}
    if not isinstance(providers, dict):
        raise ValueError(
            f"providers.yaml must have a top-level ``providers:`` mapping; "
            f"got {type(providers).__name__}",
        )
    return providers


def load_model_chains(*, refresh: bool = False) -> dict[str, list[dict[str, Any]]]:
    """Load and cache the canonical-model → provider-leg chain mapping.

    Returns ``{canonical_model_name: [{provider, model?}, ...]}``. Each
    leg references a provider by name from the ``providers:`` registry;
    optional ``model:`` overrides the model_id sent for that leg.

    Returns an empty dict when the ``model_chains:`` section is absent —
    legacy resolution via ``MODEL_*`` env vars remains the only source
    in that case.
    """
    raw = _load_raw_yaml(refresh=refresh)
    chains_raw = raw.get("model_chains") or {}
    if not isinstance(chains_raw, dict):
        raise ValueError(
            f"providers.yaml ``model_chains:`` must be a mapping; "
            f"got {type(chains_raw).__name__}",
        )
    # Normalise: every chain value must be a list of dicts with ``provider``.
    normalised: dict[str, list[dict[str, Any]]] = {}
    for name, legs in chains_raw.items():
        if not isinstance(legs, list):
            raise ValueError(
                f"model_chains[{name!r}] must be a list of legs; "
                f"got {type(legs).__name__}",
            )
        out: list[dict[str, Any]] = []
        for i, leg in enumerate(legs):
            if not isinstance(leg, dict) or "provider" not in leg:
                raise ValueError(
                    f"model_chains[{name!r}][{i}] must be a mapping with "
                    f"``provider:`` key; got {leg!r}",
                )
            out.append(dict(leg))
        normalised[str(name)] = out
    return normalised


def resolve_provider(name: str) -> dict[str, Any]:
    """Look up a provider by name. Raises :class:`ProviderNotFound`
    when the name is not registered."""
    providers = load_providers()
    if name not in providers:
        raise ProviderNotFound(name, list(providers))
    return dict(providers[name])  # defensive copy — caller may mutate


def provider_metadata(name: str) -> ProviderMetadata:
    """Return safe metadata from the unexpanded provider registry.

    Unlike :func:`resolve_provider`, this deliberately reads the source YAML so
    diagnostics retain names such as ``OPENAI_API_KEY`` without retaining or
    returning the corresponding secret value.
    """
    import yaml

    path = _providers_path()
    if not path.exists():
        raise FileNotFoundError(
            f"Provider registry not found: {path}. Set "
            "FRONTIER_AGENT_PROVIDERS_PATH or create the checked-in template."
        )
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    providers = raw.get("providers") or {}
    if not isinstance(providers, dict):
        raise ValueError("providers.yaml must have a top-level ``providers:`` mapping")
    if name not in providers:
        raise ProviderNotFound(name, list(providers))
    entry = providers[name]
    if not isinstance(entry, dict):
        raise ValueError(f"provider {name!r} must be a mapping")
    return ProviderMetadata(
        name=name,
        provider_type=str(entry.get("type") or ""),
        api_key_env=environment_variable_source(entry.get("api_key")),
        base_url_env=environment_variable_source(entry.get("base_url")),
    )


def _reset_cache() -> None:
    """Clear the in-process cache. For tests only."""
    _cache.clear()


__all__ = [
    "ProviderMetadata",
    "ProviderNotFound",
    "_reset_cache",
    "environment_variable_source",
    "load_model_chains",
    "load_providers",
    "provider_metadata",
    "resolve_provider",
]
