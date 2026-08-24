"""The compaction trigger must look at the request about to be sent.

`should_compact` used to read only `InputTokenGauge`, which describes the request
already SENT. It runs at turn end, after this turn's assistant message and tool
results were appended, so the next request is strictly larger. That left a blind
spot a single turn could cross: measured over 51 sub-agents killed by
`llm_error`, 51/51 had a last successful prompt just UNDER the 209,715 trigger
(median 207,803) and then issued a request of 264,743 tokens median, which the
endpoint rejected with HTTP 400 — losing the sub-agent and every report it had
gathered.

One assistant turn can add 40k+ tokens once reasoning is replayed into history as
`<think>` (measured max 130,965 characters), against a trigger-to-ceiling margin
of only 52,429. No reactive read covers that.
"""

from __future__ import annotations

import asyncio

import pytest

from frontier_agent.core.runtime.loop.compact import INPUT_ESTIMATE_KEY
from frontier_agent.core.runtime.loop.tiered_compact import (
    InputTokenGauge,
    InputTokenThresholdPolicy,
    TieredCompactor,
    compaction_trigger_tokens,
)

MAX_LEN = 262_144
TRIGGER = 209_715  # int(262144 * 0.8)


def _gauge(*, real: int, estimate: int) -> InputTokenGauge:
    """A gauge holding one calibration pair, as the loop would leave it."""
    gauge = InputTokenGauge()
    gauge.tokens = real
    gauge.estimate = estimate
    return gauge


def _policy(gauge: InputTokenGauge, limit: int = TRIGGER) -> InputTokenThresholdPolicy:
    return InputTokenThresholdPolicy(gauge, limit)


# ── the failure this fixes ───────────────────────────────────────────────────


def test_the_measured_death_scenario_now_triggers() -> None:
    """The exact numbers from the 51 killed sub-agents.

    Last successful prompt 207,803 — under the 209,715 trigger, so the old
    gauge-only read returned False. The request about to go out was 264,743, well
    past the 262,144 ceiling. Reading the gauge alone could not see it coming.
    """
    # tiktoken under-states this checkpoint by ~14% (measured median scale 1.14).
    gauge = _gauge(real=207_803, estimate=182_283)
    policy = _policy(gauge)

    # Turn end: the history now estimates 232,231, which scales to ~264,743 real.
    assert policy.should_compact(turn=40, messages=[], estimated_tokens=232_231) is True


def test_the_gauge_alone_would_have_said_no() -> None:
    """Pin the counterfactual, so the regression is legible if the projection is
    ever removed: the real token count on its own is under the limit."""
    gauge = _gauge(real=207_803, estimate=182_283)

    assert gauge.tokens <= TRIGGER


def test_a_quiet_turn_still_does_not_trigger() -> None:
    """The fix must not compact every turn — that was measured to cost history and
    4% of searches per trial for no accuracy gain."""
    gauge = _gauge(real=120_000, estimate=105_263)
    policy = _policy(gauge)

    assert policy.should_compact(turn=10, messages=[], estimated_tokens=110_000) is False


def test_the_gauge_still_triggers_on_its_own() -> None:
    """The original signal is kept: what the endpoint really charged is
    authoritative when it is already over the line."""
    gauge = _gauge(real=220_000, estimate=220_000)
    policy = _policy(gauge)

    assert policy.should_compact(turn=5, messages=[], estimated_tokens=1) is True


# ── the calibration ─────────────────────────────────────────────────────────


def test_the_scale_comes_from_one_snapshot_not_from_the_turn_end_history() -> None:
    """The subtle half of this fix.

    Deriving the ratio from `messages` (the turn-END history) instead of the
    gauge's own pair would inflate the denominator with the very tool results that
    make this turn dangerous — understating the scale exactly when it is needed
    most, and making the projection under-fire. The gauge carries `estimate` from
    the same request as `tokens` precisely so that cannot happen.
    """
    gauge = _gauge(real=182_000, estimate=140_000)  # scale 1.30 on the SENT request

    # A huge turn-end history must not change the scale that gets applied.
    policy = _policy(gauge)
    small = policy.should_compact(turn=1, messages=[], estimated_tokens=161_320)
    assert small is True, "161,320 * 1.30 = 209,716 — one token over the trigger"

    assert gauge.real_to_estimate_scale() == pytest.approx(1.3)


def test_an_over_stating_estimator_is_not_allowed_to_shrink_the_projection() -> None:
    """Floor of 1.0. Only under-estimation can cost a request here, so a scale
    below 1.0 is discarded for this purpose rather than trusted."""
    gauge = _gauge(real=100_000, estimate=200_000)  # raw scale 0.5
    policy = _policy(gauge)

    # With the raw 0.5 scale, 220,000 would project to 110,000 and not fire.
    assert policy.should_compact(turn=9, messages=[], estimated_tokens=220_000) is True


def test_an_absurd_ratio_is_capped() -> None:
    """Ceiling of 3.0 — a sanity bound, so one strange usage report cannot make
    every subsequent turn compact."""
    gauge = _gauge(real=300_000, estimate=1_000)  # raw scale 300

    # 60,000 * 3.0 = 180,000 < trigger. Uncapped it would be 18,000,000.
    # The gauge's own 300,000 is over the trigger, so raise the limit above it to
    # isolate what the projection contributes.
    isolated = InputTokenThresholdPolicy(gauge, 400_000)
    assert isolated.should_compact(turn=3, messages=[], estimated_tokens=60_000) is False


def test_the_first_turn_projects_the_raw_estimate() -> None:
    """Before any reply there is no pair, so the scale is neutral and the estimate
    is used as-is rather than being discarded."""
    gauge = InputTokenGauge()
    policy = _policy(gauge)

    assert gauge.real_to_estimate_scale() == 1.0
    assert policy.should_compact(turn=1, messages=[], estimated_tokens=250_000) is True
    assert policy.should_compact(turn=1, messages=[], estimated_tokens=1_000) is False


def test_a_gauge_with_no_published_estimate_stays_neutral() -> None:
    """A missing estimate must not be read as an infinite ratio."""
    gauge = _gauge(real=200_000, estimate=0)

    assert gauge.real_to_estimate_scale() == 1.0


# ── the two consumers clamp differently, on purpose ─────────────────────────


def test_the_relief_scale_is_deliberately_left_unclamped() -> None:
    """`TieredCompactor` uses the same ratio for a different decision.

    For the trigger, a scale below 1.0 is unsafe to trust. For the relief target
    it is a real measurement — the estimator over-stating — and flooring it would
    escalate to a Tier 2 summary for volume that is not there. The two clampings
    must not be unified into one.
    """
    gauge = _gauge(real=100_000, estimate=200_000)
    compactor = TieredCompactor(
        keep_tool_result=1, summary_llm=None, relief_target=1, gauge=gauge,
    )

    assert compactor._real_token_scale() == pytest.approx(0.5)


# ── the trigger ratio has one definition ────────────────────────────────────


def test_the_default_ratio_matches_what_the_profiles_document() -> None:
    assert compaction_trigger_tokens(MAX_LEN) == TRIGGER


def test_the_ratio_is_overridable(monkeypatch: pytest.MonkeyPatch) -> None:
    """Widening was measured and rejected, but the knob stays so the question can
    be revisited without editing a profile."""
    monkeypatch.setenv("AGENT_COMPACTION_TRIGGER_RATIO", "0.65")

    assert compaction_trigger_tokens(MAX_LEN) == int(MAX_LEN * 0.65)


@pytest.mark.parametrize("bad", ["", "abc", "0", "1", "1.5", "-0.5"])
def test_an_unusable_ratio_falls_back_rather_than_disabling_compaction(
    bad: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A ratio of 0 or 1 either compacts every turn or never — a typo must not
    silently produce either."""
    monkeypatch.setenv("AGENT_COMPACTION_TRIGGER_RATIO", bad)

    assert compaction_trigger_tokens(MAX_LEN) == TRIGGER


def test_no_workflow_still_computes_the_trigger_itself() -> None:
    """Three sites each carried their own `int(max_len * 0.8)`. A fourth copy
    would drift from the value the check and the docs both quote."""
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    offenders = [
        str(path.relative_to(root))
        for path in (root / "workflows").rglob("*.py")
        if "max_len * 0.8" in path.read_text()
    ]

    assert offenders == []


# ── end to end through the gauge's observer hook ────────────────────────────


def test_the_pair_the_loop_publishes_drives_the_projection() -> None:
    """The gauge is fed by `on_llm_response`, not set by hand. Read it the way the
    loop does, so this cannot pass on a hand-built gauge alone.
    """
    from frontier_agent.core.loop_types import TurnContext

    gauge = InputTokenGauge()
    asyncio.run(gauge.on_llm_response(TurnContext(
        turn=1,
        max_turns=10,
        task_id="task",
        role_id="researcher",
        ai_text="",
        thinking="",
        tool_calls=[],
        messages=[],
        usage={"prompt_tokens": 182_000},
        metadata={INPUT_ESTIMATE_KEY: 140_000},
    )))

    assert gauge.tokens == 182_000
    assert gauge.estimate == 140_000
    assert _policy(gauge).should_compact(
        turn=1, messages=[], estimated_tokens=161_320,
    ) is True
