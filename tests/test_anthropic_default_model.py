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
