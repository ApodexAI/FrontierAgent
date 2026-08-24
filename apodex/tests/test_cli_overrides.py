from __future__ import annotations

from apodex.cli import (
    apply_model_overrides,
    build_parser,
    main,
    publish_model_overrides,
)
from apodex.config import ModelConfig


def _cfg() -> ModelConfig:
    """A stand-in for what a profile resolves to (react ships max_tokens 32768)."""
    return ModelConfig(
        model="profile-model", api_key="k", base_url="https://example.test/v1",
        max_tokens=32768,
    )


def test_max_tokens_defaults_to_the_profile_rather_than_a_parser_constant() -> None:
    # A concrete parser default would silently outrank the profile for every run
    # that does not pass the flag.
    assert build_parser().parse_args([]).max_tokens is None
    assert apply_model_overrides(_cfg()).max_tokens == 32768


def test_max_tokens_flag_reaches_the_model_config() -> None:
    args = build_parser().parse_args(["--max-tokens", "4096"])
    assert apply_model_overrides(_cfg(), max_tokens=args.max_tokens).max_tokens == 4096


def test_model_flag_outranks_a_resumed_session() -> None:
    resumed = apply_model_overrides(_cfg(), resumed_model="saved-model")
    assert resumed.model == "saved-model"
    redirected = apply_model_overrides(
        _cfg(), model="flag-model", resumed_model="saved-model",
    )
    assert redirected.model == "flag-model"


def test_blank_resumed_model_keeps_the_profile_model() -> None:
    assert apply_model_overrides(_cfg(), resumed_model="   ").model == "profile-model"


def test_overrides_are_republished_where_the_workflow_will_read_them() -> None:
    """Reaching ModelConfig is not enough to reach the agent's own LLM calls.

    The native workflows render their LLM config from their profile YAML, which
    interpolates ${OPENAI_MODEL} and ${OPENAI_MAX_TOKENS} from the environment
    and never sees this object. Confirmed against a recording endpoint: without
    this step, --max-tokens 24 still sent max_completion_tokens 32768.
    """
    environ: dict[str, str] = {}
    cfg = apply_model_overrides(_cfg(), model="flag-model", max_tokens=24)
    publish_model_overrides(cfg, environ=environ)
    assert environ == {"OPENAI_MODEL": "flag-model", "OPENAI_MAX_TOKENS": "24"}


def test_republishing_without_flags_restates_the_profile_values() -> None:
    # Both profile fields resolve *from* these variables, so an unflagged run
    # must write back what was already there rather than a parser constant.
    environ = {"OPENAI_MODEL": "profile-model", "OPENAI_MAX_TOKENS": "32768"}
    publish_model_overrides(apply_model_overrides(_cfg()), environ=environ)
    assert environ == {"OPENAI_MODEL": "profile-model", "OPENAI_MAX_TOKENS": "32768"}


def test_non_positive_max_tokens_is_rejected_before_a_session_starts(
    monkeypatch, tmp_path, capsys,
) -> None:
    monkeypatch.chdir(tmp_path)  # keep .env discovery away from the checkout
    assert main(["--max-tokens", "0"]) == 2
    assert "--max-tokens must be a positive integer" in capsys.readouterr().err
