"""The three token budgets must be able to hold at once.

`config/sglang/README.md` describes the context/output relationship and
`docker/sglang-doctor.sh` checks it — but only for `SGLANG_*` on the compose
path, by a script an operator has to remember to run. The values that actually
build the loop are the profile's `max_len` / `max_input_tokens` and the LLM's
`max_tokens`, and those can be set straight through `OPENAI_*` against any
endpoint with nothing checking them.

A violation does not look like a misconfiguration. It surfaces as a provider
rejection mid-run, or as a reasoning watchdog that cannot reliably pre-empt the
provider's output cap — both of which read as runtime faults.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from frontier_agent.core.runtime.loop.budget_consistency import check_context_budget

REPO = Path(__file__).resolve().parents[1]
SHIPPED_PROFILES = sorted(
    [
        *(REPO / "workflows/agent_team/profiles").glob("*.yaml"),
        *(REPO / "workflows/stateful_react_agent/profiles").glob("*.yaml"),
    ]
)


def _check(**kw: object) -> list[str]:
    kw.setdefault("label", "test")
    return check_context_budget(**kw)  # type: ignore[arg-type]


# ── the invariants ───────────────────────────────────────────────────────────


def test_the_shipped_combination_is_consistent() -> None:
    """max_len 262144 / max_input 229376 / max_tokens 32768 / watchdog 16384 —
    the defaults, which fit a `--context-length 262144` server exactly."""
    assert _check(
        max_len=262_144,
        max_input_tokens=229_376,
        max_tokens=32_768,
        reasoning_only_max_tokens=16_384,
    ) == []


def test_a_full_reply_that_cannot_fit_the_served_context_is_flagged() -> None:
    """229,376 + 65,536 > 262,144. Nothing catches this before the provider,
    and by then the turn is lost."""
    problems = _check(
        max_len=262_144, max_input_tokens=229_376, max_tokens=65_536,
    )
    assert len(problems) == 1
    assert "exceeds max_len" in problems[0]


def test_an_input_guard_below_the_compaction_trigger_is_not_flagged() -> None:
    """Tiered workflows force one compaction pass when the input guard trips,
    so the guard may safely sit below the normal threshold.
    """
    assert _check(
        max_len=262_144, max_input_tokens=196_608, max_tokens=32_768,
    ) == []


def test_lowering_the_input_guard_can_make_a_65536_reply_fit() -> None:
    """The forced compaction path keeps a lower guard compatible with tiered
    compaction, so an exactly-full prompt/reply pairing is consistent."""
    assert len(_check(
        max_len=262_144, max_input_tokens=229_376, max_tokens=65_536,
    )) == 1
    assert _check(
        max_len=262_144, max_input_tokens=196_608, max_tokens=65_536,
    ) == []


def test_a_watchdog_at_or_above_max_tokens_cannot_reliably_preempt_it() -> None:
    """The provider cap competes with the watchdog, so the runaway path may
    handle the reply instead — a different mechanism with a different recovery."""
    problems = _check(
        max_len=262_144,
        max_input_tokens=229_376,
        max_tokens=32_768,
        reasoning_only_max_tokens=32_768,
    )
    assert len(problems) == 1
    assert "reasoning watchdog" in problems[0]


# ── what must NOT be flagged ─────────────────────────────────────────────────


def test_an_unset_bound_is_a_choice_not_an_inconsistency() -> None:
    assert _check(max_len=262_144, max_input_tokens=None, max_tokens=None) == []
    assert _check(max_len=262_144, max_input_tokens=0, max_tokens=0) == []


def test_an_unknown_context_window_disables_every_check() -> None:
    """Without max_len there is no ratio to compare against, and guessing one
    would report a violation the operator cannot act on."""
    assert _check(max_len=0, max_input_tokens=229_376, max_tokens=65_536) == []


def test_the_watchdog_is_checked_without_a_context_window() -> None:
    """The watchdog/output-cap relationship is independent of max_len and must
    still be validated for simple profiles that do not enable compaction."""
    problems = _check(
        max_len=0,
        max_input_tokens=229_376,
        max_tokens=16_384,
        reasoning_only_max_tokens=16_384,
    )
    assert len(problems) == 1
    assert "reasoning watchdog" in problems[0]


def test_the_trigger_margin_is_reported_but_never_judged() -> None:
    """Deliberate, and worth pinning so it is not "helpfully" promoted later.

    Whether one turn actually crosses the trigger-to-wall margin is dynamic —
    the reply, its tool result, and whatever the reasoning watchdog really
    allowed. PR #66 measured single-turn growth of 40-70K against a 52,429
    margin, and also measured that the SAME max_tokens was harmless while the
    watchdog held real output near 15K. No static rule separates those two, so
    turning this ratio into a warning would invent a threshold the evidence does
    not support. The standing fix is to project the next request in
    `should_compact`, not to bound this ratio here.
    """
    # 32,768 is 62% of the 52,429 margin, and this is the shipped default.
    assert _check(
        max_len=262_144, max_input_tokens=229_376, max_tokens=32_768,
    ) == []
    # Even a reply that alone eats 94% of the margin is not judged here.
    assert _check(
        max_len=262_144, max_input_tokens=212_992, max_tokens=49_152,
    ) == []


# ── the shipped profiles must satisfy their own rules ────────────────────────


def test_every_shipped_profile_satisfies_the_invariants() -> None:
    """The check is worth little if what we ship fails it. Env-substituted
    `${VAR:-default}` values are read at their defaults, which is what an
    operator who sets nothing gets.
    """
    assert SHIPPED_PROFILES, "no profiles found — the glob is wrong"
    failures: dict[str, list[str]] = {}
    for path in SHIPPED_PROFILES:
        raw = path.read_text()
        # Resolve only the `${VAR:-default}` default, the way a bare run does.
        for pattern, replacement in (
            ("${OPENAI_CONTEXT_WINDOW:-262144}", "262144"),
            ("${OPENAI_MAX_INPUT_TOKENS:-229376}", "229376"),
            ("${OPENAI_MAX_TOKENS:-32768}", "32768"),
        ):
            raw = raw.replace(pattern, replacement)
        if "${" in raw:
            # Any other substitution is not part of this invariant; blank it so
            # the YAML still parses.
            raw = "\n".join(
                line for line in raw.splitlines() if "${" not in line
            )
        cfg = yaml.safe_load(raw) or {}
        agent = cfg.get("agent") or {}
        llm = cfg.get("llm") or {}
        problems = check_context_budget(
            max_len=int(agent.get("max_len") or 0),
            max_input_tokens=int(agent.get("max_input_tokens") or 0) or None,
            max_tokens=int(llm.get("max_tokens") or 0) or None,
            reasoning_only_max_tokens=(
                int(agent.get("reasoning_only_max_tokens") or 0) or None
            ),
            label=path.name,
        )
        if problems:
            failures[str(path.relative_to(REPO))] = problems
    assert not failures, failures


def test_the_reasoning_watchdog_stays_under_the_output_cap_in_every_profile() -> None:
    """The pairing that makes the watchdog reachable at all. Pinned separately
    because raising OPENAI_MAX_TOKENS is easy and this relationship is not
    obvious from either key's name.
    """
    for path in SHIPPED_PROFILES:
        raw = path.read_text().replace("${OPENAI_MAX_TOKENS:-32768}", "32768")
        raw = "\n".join(line for line in raw.splitlines() if "${" not in line)
        cfg = yaml.safe_load(raw) or {}
        watchdog = (cfg.get("agent") or {}).get("reasoning_only_max_tokens")
        out_cap = (cfg.get("llm") or {}).get("max_tokens")
        if watchdog and out_cap:
            assert int(watchdog) < int(out_cap), (
                f"{path.name}: reasoning_only_max_tokens {watchdog} is not below "
                f"llm.max_tokens {out_cap}, so the watchdog cannot reliably "
                "pre-empt the provider cap"
            )
