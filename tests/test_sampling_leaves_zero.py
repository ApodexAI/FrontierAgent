"""An agent loop must not sample at temperature 0 on this model.

Zero temperature looks like the obvious choice for an evaluation, and that is the
trap: on this model it trades occasional reasoning divergence for a
DETERMINISTIC deadlock. A context already in reasoning-runaway, resampled four
times at temperature 0 on a self-hosted EAS, returned byte-identical output every
time — there is no probability of leaving the absorbing state. One question
accumulated 117 runaways and burned 176 minutes before exiting on `no_tool`.

Against the official endpoint, which ignores temperature and therefore samples,
the same 19 questions went from 516 runaways to 37, from 9/9 `no_tool` to 0/9,
and from 0/9 correct to 5/9.

These tests pin the rule rather than the three numbers, because the numbers are
easy to "helpfully" revert in the name of reducing evaluation variance.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
PROFILES = sorted(REPO.glob("workflows/*/profiles/*.yaml"))

# `report_llm` is the ONE section allowed at temperature 0: it is a single
# tool-free synthesis call with no loop to get stuck in, so an absorbing
# reasoning state has nothing to absorb. Determinism is worth more there.
_NON_LOOP_SECTIONS = frozenset({"report_llm"})


def _load(path: Path) -> dict:
    """Parse a profile with `${VAR}` substitutions stubbed out."""
    return yaml.safe_load(re.sub(r"\$\{[^}]*\}", "PLACEHOLDER", path.read_text())) or {}


def _loop_sections(path: Path) -> list[tuple[str, dict]]:
    cfg = _load(path)
    return [
        (name, section)
        for name, section in cfg.items()
        if name.endswith("llm")
        and name not in _NON_LOOP_SECTIONS
        and isinstance(section, dict)
        and "temperature" in section
    ]


def test_the_profiles_are_actually_being_scanned() -> None:
    """A glob that silently matches nothing would make every test below vacuous."""
    assert len(PROFILES) >= 6, [str(p) for p in PROFILES]
    assert any(_loop_sections(p) for p in PROFILES)


@pytest.mark.parametrize("path", PROFILES, ids=lambda p: f"{p.parent.parent.name}/{p.stem}")
def test_no_agent_loop_samples_at_temperature_zero(path: Path) -> None:
    for name, section in _loop_sections(path):
        assert float(section["temperature"]) != 0.0, (
            f"{path.name}:{name} is at temperature 0. On this model that is a "
            f"deterministic deadlock, not merely low variance — sglang takes the "
            f"argmax path and a reasoning runaway can never resolve itself. Use "
            f"--runs N to control evaluation variance instead."
        )


@pytest.mark.parametrize("path", PROFILES, ids=lambda p: f"{p.parent.parent.name}/{p.stem}")
def test_a_sampling_loop_carries_the_whole_triple(path: Path) -> None:
    """temperature alone leaves the distribution unspecified.

    Moving off 0 is what fixes the deadlock, but it also switches sglang from
    argmax to sampling — at which point `top_p` and `top_k` start mattering and
    their absence means whatever the server defaults to, not the values the model
    ships in `generation_config.json`.
    """
    for name, section in _loop_sections(path):
        extra_body = section.get("extra_body") or {}
        assert section.get("top_p") is not None, (
            f"{path.name}:{name} samples (temperature "
            f"{section['temperature']}) but sets no top_p"
        )
        assert extra_body.get("top_k") is not None, (
            f"{path.name}:{name} samples but sets no extra_body.top_k — top_k is "
            f"not a Chat-Completions field, so it only reaches SGLang this way"
        )


@pytest.mark.parametrize("path", PROFILES, ids=lambda p: f"{p.parent.parent.name}/{p.stem}")
def test_the_shipped_values_are_the_models_own_defaults(path: Path) -> None:
    """Pinned so a drift is a deliberate edit with a reason, not a stray tweak."""
    for _name, section in _loop_sections(path):
        assert float(section["temperature"]) == 1.0
        assert float(section["top_p"]) == 0.95
        assert int((section.get("extra_body") or {})["top_k"]) == 20


def test_enabling_thinking_and_setting_top_k_do_not_evict_each_other() -> None:
    """Both live under `extra_body`, so adding one by rewriting the mapping is an
    easy way to silently drop the other."""
    for path in PROFILES:
        for _name, section in _loop_sections(path):
            extra_body = section.get("extra_body") or {}
            if "chat_template_kwargs" not in extra_body:
                continue
            assert extra_body.get("top_k") == 20, f"{path.name} lost top_k"
            assert extra_body["chat_template_kwargs"].get("enable_thinking") is True, (
                f"{path.name} lost enable_thinking"
            )


# ── the plumbing the values depend on ────────────────────────────────────────


def test_top_p_reaches_the_request_for_both_workflows() -> None:
    """`top_p` is not an `OpenAIClient` parameter — each workflow's profile loader
    folds it into `extra_body`. Without that fold the key parses fine and is
    silently dropped, which is the failure mode this asserts against.
    """
    for module in (
        "workflows/agent_team/profile.py",
        "workflows/stateful_react_agent/profile.py",
    ):
        source = (REPO / module).read_text()
        assert 'extra_body.setdefault("top_p", top_p)' in source, (
            f"{module} no longer folds top_p into extra_body — every profile's "
            f"top_p would be accepted and ignored"
        )
