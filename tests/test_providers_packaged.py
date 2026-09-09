"""The provider registry resolves inside an installed wheel, not only a checkout."""

from __future__ import annotations

import tomllib
from pathlib import Path

from frontier_agent.infra import providers

_REPO_ROOT = Path(__file__).resolve().parents[1]


def test_checkout_registry_is_preferred_when_present(monkeypatch) -> None:
    monkeypatch.delenv("FRONTIER_AGENT_PROVIDERS_PATH", raising=False)
    assert providers._DEFAULT_PATH.is_file()  # this is a checkout

    assert providers._providers_path() == providers._DEFAULT_PATH


def test_packaged_copy_is_used_when_the_checkout_file_is_absent(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.delenv("FRONTIER_AGENT_PROVIDERS_PATH", raising=False)
    monkeypatch.setattr(providers, "_DEFAULT_PATH", tmp_path / "missing" / "providers.yaml")
    packaged = tmp_path / "site-packages" / "frontier_agent" / "infra" / "providers.yaml"
    packaged.parent.mkdir(parents=True)
    packaged.write_text("providers:\n  openai:\n    type: openai\n", encoding="utf-8")
    monkeypatch.setattr(providers, "_packaged_default_path", lambda: packaged)

    assert providers._providers_path() == packaged
    assert "openai" in providers.load_providers(refresh=True)


def test_explicit_override_still_wins_over_both(monkeypatch, tmp_path) -> None:
    override = tmp_path / "custom.yaml"
    override.write_text("providers: {}\n", encoding="utf-8")
    monkeypatch.setenv("FRONTIER_AGENT_PROVIDERS_PATH", str(override))
    monkeypatch.setattr(providers, "_packaged_default_path", lambda: tmp_path / "unused")

    assert providers._providers_path() == override


def test_missing_everywhere_names_the_checkout_path(monkeypatch, tmp_path) -> None:
    # A source tree that lost the file gets the same message it always did.
    monkeypatch.delenv("FRONTIER_AGENT_PROVIDERS_PATH", raising=False)
    absent = tmp_path / "config" / "providers.yaml"
    monkeypatch.setattr(providers, "_DEFAULT_PATH", absent)
    monkeypatch.setattr(providers, "_packaged_default_path", lambda: None)

    assert providers._providers_path() == absent


def test_wheel_ships_the_registry_next_to_its_loader() -> None:
    """pyproject force-includes config/providers.yaml where the loader looks."""
    pyproject = tomllib.loads((_REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    force_include = pyproject["tool"]["hatch"]["build"]["targets"]["wheel"]["force-include"]

    assert force_include["config/providers.yaml"] == "frontier_agent/infra/providers.yaml"
    # The loader resolves the packaged copy relative to its own package.
    assert providers.__package__ == "frontier_agent.infra"
