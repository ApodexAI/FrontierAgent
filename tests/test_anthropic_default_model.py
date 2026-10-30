from frontier_agent.infra.config import FrontierAgentConfig
from frontier_agent.infra.llm_adapter import _anthropic_temperature

# claude-sonnet-4-20250514 was retired by Anthropic on 2026-06-15.
RETIRED_MODEL = "claude-sonnet-4-20250514"


def test_default_anthropic_model_is_not_retired():
    assert FrontierAgentConfig.model_fields["anthropic_model"].default != RETIRED_MODEL


def test_default_anthropic_model_gets_no_temperature():
    default = FrontierAgentConfig.model_fields["anthropic_model"].default

    assert _anthropic_temperature(default, None) is None


def test_older_models_keep_the_temperature():
    assert _anthropic_temperature("claude-sonnet-4-6", None) == 0.3


def test_thinking_drops_the_temperature():
    assert _anthropic_temperature("claude-sonnet-4-6", {"type": "adaptive"}) is None


def test_prefixed_claude_5_ids_get_no_temperature():
    assert _anthropic_temperature("anthropic/claude-sonnet-5.5", None) is None
    assert _anthropic_temperature("global.anthropic.claude-sonnet-5-5", None) is None


def test_opus_4_7_and_4_8_get_no_temperature():
    assert _anthropic_temperature("claude-opus-4-7", None) is None
    assert _anthropic_temperature("claude-opus-4-8", None) is None
    assert _anthropic_temperature("claude-opus-4-6", None) == 0.3


def test_role_temperature_is_kept_for_older_models():
    assert _anthropic_temperature("claude-sonnet-4-6", None, 0.7) == 0.7
    assert _anthropic_temperature("claude-sonnet-5-5", None, 0.7) is None


def test_overridden_provider_omits_temperature_for_claude_5(monkeypatch):
    import frontier_agent.infra.anthropic_client as anthropic_client
    from frontier_agent.infra.llm_adapter import _create_overridden_provider

    captured = {}

    class FakeClient:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(anthropic_client, "AnthropicClient", FakeClient)
    config = FrontierAgentConfig(llm_provider="anthropic", anthropic_api_key="test-key")

    _create_overridden_provider(config, temperature=0.7)
    assert captured["model"] == config.anthropic_model
    assert captured["temperature"] is None

    _create_overridden_provider(config, "claude-sonnet-4-6", temperature=0.7)
    assert captured["temperature"] == 0.7
