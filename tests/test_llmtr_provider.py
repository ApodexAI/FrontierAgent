"""LLMTR registry entry: preset URL, key name, no secret in metadata."""

from __future__ import annotations

import apodex.onboarding as onboarding
from frontier_agent.infra.providers import load_providers, provider_metadata


def test_llmtr_is_registered_openai_compat_with_preset_url(monkeypatch):
    monkeypatch.delenv("LLMTR_API_KEY", raising=False)
    monkeypatch.delenv("LLMTR_BASE_URL", raising=False)
    providers = load_providers(refresh=True)
    assert "llmtr" in providers
    entry = providers["llmtr"]
    assert entry["type"] == "openai_compat"
    assert entry["base_url"] == "https://llmtr.com/v1"
    # The key expands to empty when unset; the value itself is never asserted.
    assert entry.get("api_key") in ("", None)


def test_llmtr_metadata_names_the_env_var_not_a_secret(monkeypatch):
    monkeypatch.setenv("LLMTR_API_KEY", "fixture-llmtr-key-should-not-appear")
    meta = provider_metadata("llmtr")
    assert meta.name == "llmtr"
    assert meta.provider_type == "openai_compat"
    assert meta.api_key_env == "LLMTR_API_KEY"
    assert meta.base_url_env == "LLMTR_BASE_URL"
    assert "fixture-llmtr-key-should-not-appear" not in repr(meta)


def test_onboarding_lists_llmtr_key_var():
    assert "LLMTR_API_KEY" in onboarding._KEY_VARS
