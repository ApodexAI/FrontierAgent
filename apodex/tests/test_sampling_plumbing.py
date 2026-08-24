"""apodex profiles can express the model's sampling triple, and deliberately don't.

`ModelConfig` carried only `temperature`, so a profile pointed at a self-hosted
vLLM / SGLang endpoint could not set `top_p` or `top_k` at all — the keys parsed
and were dropped. `build_llm` now forwards both through `extra_body`, the same
route the workflow profile loaders use.

No shipped apodex profile sets them, and that is a decision rather than an
omission: every one defaults to `provider: openai`, and OpenAI-family endpoints
reject these fields outright (`tools/preflight.py` detects exactly that error).
The deadlock that forces temperature off 0 under `workflows/*/profiles/` is
specific to SGLang's argmax path on a self-hosted Apodex checkpoint.
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml

from apodex.config import ModelConfig
from apodex.llm import build_llm

REPO = Path(__file__).resolve().parents[2]
APODEX_PROFILES = sorted((REPO / "apodex" / "profiles").glob("*.yaml"))


def _cfg(**kw: object) -> ModelConfig:
    base: dict[str, object] = {
        "model": "test-model", "api_key": "k", "base_url": "https://x.test/v1",
    }
    base.update(kw)
    return ModelConfig(**base)  # type: ignore[arg-type]


# ── the plumbing ─────────────────────────────────────────────────────────────


def test_an_unset_pair_sends_nothing() -> None:
    """The default must stay byte-identical for every existing profile: sending an
    explicit top_p/top_k nobody asked for would change sampling everywhere."""
    client = build_llm(_cfg(temperature=0.0))

    assert getattr(client, "extra_body", {}) == {}


def test_top_p_and_top_k_reach_extra_body() -> None:
    """Neither is an `OpenAIClient` parameter nor a Chat-Completions field, so
    `extra_body` is the only route to a vLLM / SGLang server."""
    client = build_llm(_cfg(temperature=1.0, top_p=0.95, top_k=20))

    assert client.extra_body == {"top_p": 0.95, "top_k": 20}


def test_either_one_alone_still_travels() -> None:
    assert build_llm(_cfg(top_p=0.9)).extra_body == {"top_p": 0.9}
    assert build_llm(_cfg(top_k=40)).extra_body == {"top_k": 40}


def test_a_zero_is_carried_rather_than_treated_as_absent() -> None:
    """`0` is a real top_k value (disable top-k) and must not be swallowed by a
    falsy check — the guard is `is not None` for that reason."""
    assert build_llm(_cfg(top_k=0)).extra_body == {"top_k": 0}


# ── the profile → ModelConfig read ───────────────────────────────────────────


def test_a_profile_can_actually_set_them() -> None:
    """End to end through the profile loader, so the read cannot silently drop
    the keys the way it did before."""
    from apodex.profiles import _resolve_llm

    llm = {
        "models": ["m"],
        "api_key": "k",
        "base_url": "https://x.test/v1",
        "temperature": 1.0,
        "top_p": 0.95,
        "extra_body": {"top_k": 20},
    }
    cfg, *_ = _resolve_llm(dict(llm), dict(llm))

    assert (cfg.temperature, cfg.top_p, cfg.top_k) == (1.0, 0.95, 20)


def test_a_profile_that_says_nothing_leaves_both_unset() -> None:
    from apodex.profiles import _resolve_llm

    llm = {"models": ["m"], "api_key": "k", "temperature": 0.0}
    cfg, *_ = _resolve_llm(dict(llm), dict(llm))

    assert cfg.top_p is None
    assert cfg.top_k is None


# ── and why no shipped profile uses it ───────────────────────────────────────


def _load(path: Path) -> dict:
    return yaml.safe_load(re.sub(r"\$\{[^}]*\}", "PLACEHOLDER", path.read_text())) or {}


def test_the_profiles_are_being_scanned() -> None:
    assert len(APODEX_PROFILES) >= 4, [str(p) for p in APODEX_PROFILES]


def test_no_shipped_apodex_profile_sends_top_k_to_an_openai_provider() -> None:
    """The hazard this asserts against is real, not hypothetical:
    `tools/preflight.py` already carries a dedicated hint for an endpoint
    rejecting these fields. Every apodex profile defaults to `provider: openai`,
    so setting them here would turn a working default into a 400.

    A profile that switches to a self-hosted provider may set them — this only
    forbids the combination.
    """
    for path in APODEX_PROFILES:
        llm = _load(path).get("llm") or {}
        provider = str(llm.get("provider") or "")
        extra_body = llm.get("extra_body") or {}
        if provider in ("", "openai") or provider == "PLACEHOLDER":
            assert extra_body.get("top_k") is None, (
                f"{path.name} sends top_k to an OpenAI-family provider, which "
                f"rejects it — see tools/preflight.py"
            )
            assert llm.get("top_p") is None, (
                f"{path.name} sends top_p to an OpenAI-family provider, which "
                f"rejects it — see tools/preflight.py"
            )


def test_the_generic_modes_document_why_they_stay_deterministic() -> None:
    """`coding` and `research` run apodex's own agent loop at temperature 0, which
    looks like the defect fixed in `workflows/*/profiles/`. It is not — they
    default to gpt-4o — and the files have to say so or someone will "fix" them.
    """
    for name in ("coding", "research"):
        text = (REPO / "apodex" / "profiles" / f"{name}.yaml").read_text()
        assert "argmax" in text, f"{name}.yaml does not explain its temperature 0"
        assert "gpt-4o" in text
