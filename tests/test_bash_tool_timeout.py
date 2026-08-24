"""bash must run for the configured ``tool_timeout_s``, not a module constant.

Regression guard for #53: the deadline used to be
``int(os.environ.get("BASH_TIMEOUT", "300"))`` frozen at import, so a profile
asking for 1800s still killed every command at 5 minutes and heavy-compute
tasks could not be run at all.
"""
from __future__ import annotations

import asyncio
import importlib

import pytest

from frontier_agent.core.execution_context import (
    reset_current_tool_budget,
    set_current_tool_budget,
)
from frontier_agent.core.runtime.loop import tool_exec as TE
from frontier_agent.core.tool import tool

# ``plugins.tools.__init__`` re-exports the ``bash`` Tool under that name, which
# shadows the submodule on the package — reach the module itself explicitly.
B = importlib.import_module("plugins.tools.bash")


# ── The resolver ────────────────────────────────────────────────────────


def test_resolves_to_the_loop_budget(monkeypatch) -> None:
    monkeypatch.delenv("BASH_TIMEOUT", raising=False)
    token = set_current_tool_budget(1800.0)
    try:
        assert B._resolve_timeout() == 1800
    finally:
        reset_current_tool_budget(token)


def test_falls_back_to_the_shipped_default_outside_the_loop(monkeypatch) -> None:
    monkeypatch.delenv("BASH_TIMEOUT", raising=False)
    assert B._resolve_timeout() == B._DEFAULT_BASH_TIMEOUT == 300


def test_env_override_wins_over_the_default(monkeypatch) -> None:
    monkeypatch.setenv("BASH_TIMEOUT", "900")
    assert B._resolve_timeout() == 900


def test_env_override_is_read_per_call_not_at_import(monkeypatch) -> None:
    """The old module constant made exporting BASH_TIMEOUT from a launcher a
    no-op unless it happened before import."""
    monkeypatch.setenv("BASH_TIMEOUT", "600")
    assert B._resolve_timeout() == 600
    monkeypatch.setenv("BASH_TIMEOUT", "700")
    assert B._resolve_timeout() == 700


def test_env_override_is_clamped_to_the_loop_budget(monkeypatch) -> None:
    """Overshooting the budget would only swap bash's own diagnosis for the
    loop's bare cancel — the command dies either way."""
    monkeypatch.setenv("BASH_TIMEOUT", "3600")
    token = set_current_tool_budget(900.0)
    try:
        assert B._resolve_timeout() == 900
    finally:
        reset_current_tool_budget(token)


def test_env_override_below_the_budget_is_honoured(monkeypatch) -> None:
    monkeypatch.setenv("BASH_TIMEOUT", "120")
    token = set_current_tool_budget(900.0)
    try:
        assert B._resolve_timeout() == 120
    finally:
        reset_current_tool_budget(token)


@pytest.mark.parametrize("bad", ["", "   ", "abc", "0", "-5"])
def test_unusable_env_override_is_ignored(monkeypatch, bad: str) -> None:
    monkeypatch.setenv("BASH_TIMEOUT", bad)
    token = set_current_tool_budget(750.0)
    try:
        assert B._resolve_timeout() == 750
    finally:
        reset_current_tool_budget(token)


# ── The tool passes the resolved value to the sandbox ───────────────────


class _FakeResult:
    stdout = "ok"
    stderr = ""
    exit_code = 0


async def test_bash_hands_the_budget_to_the_sandbox(monkeypatch) -> None:
    seen: dict[str, int] = {}

    async def fake_run(sandbox, command, *, timeout, **kwargs):
        seen["timeout"] = timeout
        return _FakeResult()

    monkeypatch.setattr(B, "aget_sandbox", _fake_sandbox)
    monkeypatch.setattr(B, "arun_sandbox_cmd", fake_run)
    monkeypatch.setattr(B, "ensure_guard_file", _noop_async)
    monkeypatch.delenv("BASH_TIMEOUT", raising=False)

    token = set_current_tool_budget(1800.0)
    try:
        await B.bash.ainvoke({"command": "echo hi"})
    finally:
        reset_current_tool_budget(token)
    assert seen["timeout"] == 1800


async def test_timeout_message_reports_the_configured_limit(monkeypatch) -> None:
    async def fake_run(sandbox, command, *, timeout, **kwargs):
        raise TimeoutError

    monkeypatch.setattr(B, "aget_sandbox", _fake_sandbox)
    monkeypatch.setattr(B, "arun_sandbox_cmd", fake_run)
    monkeypatch.setattr(B, "ensure_guard_file", _noop_async)
    monkeypatch.delenv("BASH_TIMEOUT", raising=False)

    token = set_current_tool_budget(1800.0)
    try:
        out = await B.bash.ainvoke({"command": "sleep 99999"})
    finally:
        reset_current_tool_budget(token)
    assert "timed out after 1800 seconds" in out
    # The recovery hint must offer backgrounding, not only "do something
    # smaller" — chunk-and-restart plumbing is what #53 measured as turn waste.
    assert "nohup" in out


async def _fake_sandbox():
    return object()


async def _noop_async(*args, **kwargs):
    return None


# ── The loop side ───────────────────────────────────────────────────────


def test_outer_wait_leaves_bash_room_to_report_its_own_timeout() -> None:
    outer = TE._effective_tool_timeout("bash", {}, 900)
    budget = TE._tool_budget("bash", outer)
    assert budget == 900
    assert outer > budget, "equal deadlines make the winner a coin flip"


def test_only_budget_aware_tools_get_a_budget() -> None:
    assert TE._tool_budget("web_search", 900) is None
    assert TE._tool_budget("download_file", 660) is None


def test_unrelated_tools_keep_their_outer_wait() -> None:
    assert TE._effective_tool_timeout("web_search", {}, 120) == 120
    # download_file's own fixed deadline is still floored, not graced.
    assert TE._effective_tool_timeout("download_file", {}, 360) == 660
    # Aggregation tools still read their per-call argument.
    assert TE._effective_tool_timeout("collect_reports", {"timeout": 1800}, 360) == 1805


async def test_execute_tools_publishes_the_budget_to_the_running_tool() -> None:
    """End-to-end: cfg.tool_timeout reaches a tool named ``bash``."""
    from frontier_agent.core.execution_context import get_current_tool_budget

    @tool(name="bash")
    async def fake_bash(command: str) -> str:
        return f"budget={get_current_tool_budget()}"

    results = await TE.execute_tools(
        [{"name": "bash", "args": {"command": "x"}, "id": "c1"}],
        {"bash": fake_bash},
        timeout=1800,
        turn=1,
        count_offset=0,
    )
    assert results[0].result == "budget=1800.0"
    assert not results[0].is_error


async def test_the_budget_does_not_leak_between_parallel_calls() -> None:
    from frontier_agent.core.execution_context import get_current_tool_budget

    @tool(name="bash")
    async def fake_bash(command: str) -> str:
        await asyncio.sleep(0)
        return f"bash={get_current_tool_budget()}"

    @tool(name="web_search")
    async def fake_search(query: str) -> str:
        await asyncio.sleep(0)
        return f"search={get_current_tool_budget()}"

    results = await TE.execute_tools(
        [
            {"name": "bash", "args": {"command": "x"}, "id": "c1"},
            {"name": "web_search", "args": {"query": "y"}, "id": "c2"},
        ],
        {"bash": fake_bash, "web_search": fake_search},
        timeout=600,
        turn=1,
        count_offset=0,
    )
    assert results[0].result == "bash=600.0"
    assert results[1].result == "search=None"


# ── The remote sandbox must outlive a long exec ─────────────────────────


class _RemoteSandbox:
    """Stand-in for an E2B sandbox: the TTL is settable and observable."""

    def __init__(self) -> None:
        self.ttl: int | None = None

    def set_timeout(self, seconds: int) -> None:
        self.ttl = seconds


async def test_long_exec_extends_the_remote_ttl(monkeypatch) -> None:
    """A command allowed 1800s must not be killed by an 1800s sandbox TTL that
    was refreshed at the command's own start."""
    import plugins.tools._sandbox as S

    monkeypatch.setattr(S, "_get_e2b_config", lambda: ("key", "base", 1800))
    sandbox = _RemoteSandbox()
    await S._ensure_ttl_outlives_exec(sandbox, 1800)
    assert sandbox.ttl == 1860


async def test_short_exec_leaves_the_ttl_alone(monkeypatch) -> None:
    """No extra API round-trip on ordinary commands."""
    import plugins.tools._sandbox as S

    monkeypatch.setattr(S, "_get_e2b_config", lambda: ("key", "base", 1800))
    sandbox = _RemoteSandbox()
    await S._ensure_ttl_outlives_exec(sandbox, 300)
    assert sandbox.ttl is None


async def test_local_backends_have_no_ttl_to_extend() -> None:
    import plugins.tools._sandbox as S

    await S._ensure_ttl_outlives_exec(object(), 1800)  # no set_timeout, no error


# ── The wall-clock reserve must cover the grace, not just the timeout ────


def test_worst_case_tool_time_includes_the_grace() -> None:
    """The reserve is sized from this, so it has to be the OUTER wait."""
    assert TE.max_tool_wall_time_s(900) == TE._effective_tool_timeout("bash", {}, 900)
    assert TE.max_tool_wall_time_s(900) == 930
    assert TE.max_tool_wall_time_s(0) == TE._BUDGET_GRACE_S
    assert TE.max_tool_wall_time_s(-5) == TE._BUDGET_GRACE_S


@pytest.mark.parametrize(
    "workflow_module",
    [
        "workflows.stateful_react_agent.nodes.main_agent",
        "workflows.agent_team.nodes.main_agent",
    ],
)
def test_both_main_loops_reserve_the_full_tool_overrun(workflow_module: str) -> None:
    """A bash call started on the LAST research turn runs to ``tool_timeout +
    grace``. If the reserve only covers ``tool_timeout`` it eats into the
    finalization budget, or crosses the hard wall on a tight config."""
    import importlib
    import inspect

    src = inspect.getsource(importlib.import_module(workflow_module))
    # The reserve and the feasibility check must both budget the outer wait.
    assert "max_tool_wall_time_s(tool_timeout)" in src, (
        f"{workflow_module} must size its reserve from the outer wait"
    )
    assert "worst_case_tool_s + landing_budget_s" in src
    assert "tool_timeout_s=worst_case_tool_s" in src
    assert "max(tool_timeout, 0.0) + landing_budget_s" not in src, (
        f"{workflow_module} still reserves only the configured timeout"
    )


def test_a_last_turn_tool_call_fits_inside_the_reserve() -> None:
    """The arithmetic the reserve exists to guarantee: research stops, one tool
    runs to its full outer wait, and finalization still gets its whole budget."""
    tool_timeout, landing_budget_s = 900.0, 600.0
    reserve = TE.max_tool_wall_time_s(tool_timeout) + landing_budget_s
    hard_total_s = 5000.0
    research_deadline_s = hard_total_s - reserve

    # Worst case: a bash call starts one instant before the deadline check.
    outer_wait = TE._effective_tool_timeout("bash", {}, int(tool_timeout))
    finishes_at = research_deadline_s + outer_wait
    assert finishes_at + landing_budget_s <= hard_total_s

    from frontier_agent.components.finalization.budget import check_wall_feasibility
    assert check_wall_feasibility(
        hard_total_s=hard_total_s,
        research_deadline_s=research_deadline_s,
        tool_timeout_s=TE.max_tool_wall_time_s(tool_timeout),
        landing_budget_s=landing_budget_s,
        label_prefix="test",
    )


# ── A routine TTL refresh must never shorten an active long exec ─────────


async def test_a_concurrent_short_call_cannot_shorten_a_long_exec_ttl(monkeypatch) -> None:
    """The shared singleton refreshes with no minimum on every access. Since
    ``set_timeout`` is relative to NOW, that would reset a long command's
    extension back to the configured window and kill the sandbox mid-exec."""
    import plugins.tools._sandbox as S

    # tool_timeout_s above e2b_timeout — the case the review flagged.
    monkeypatch.setattr(S, "_get_e2b_config", lambda: ("key", "base", 1800))
    sandbox = _RemoteSandbox()

    await S._ensure_ttl_outlives_exec(sandbox, 3600)   # long bash starts
    raised = sandbox.ttl
    assert raised >= 3660

    S._extend_sandbox_ttl(sandbox)                     # ordinary call, no minimum
    assert sandbox.ttl is not None
    # Must still outlast the 3600s command, not fall back to 1800.
    assert sandbox.ttl > 3500, f"routine refresh shortened the TTL to {sandbox.ttl}"


async def test_a_routine_refresh_still_extends_a_short_lived_sandbox(monkeypatch) -> None:
    """Never-lower must not become never-extend: the singleton stays alive."""
    import plugins.tools._sandbox as S

    monkeypatch.setattr(S, "_get_e2b_config", lambda: ("key", "base", 1800))
    sandbox = _RemoteSandbox()
    S._extend_sandbox_ttl(sandbox)
    assert sandbox.ttl >= 1800, f"first refresh gave {sandbox.ttl}, want >= 1800"
    S._extend_sandbox_ttl(sandbox)
    # Refreshed from the new "now" rather than decaying toward the old deadline.
    assert sandbox.ttl >= 1800, f"second refresh gave {sandbox.ttl}, want >= 1800"


@pytest.mark.parametrize("clock", [0.0, 8_628_776.372424237, 1.5e9, 9.007e15])
async def test_the_ttl_window_is_never_shorter_than_configured(
    monkeypatch, clock: float,
) -> None:
    """``set_timeout`` is a promise of "at least this long", at any clock value.

    CI once produced 1799 from a configured 1800 on the first call to a fresh
    sandbox. That was never reproduced locally and the cause is not understood,
    so this pins the contract across a spread of ``time.monotonic()`` magnitudes
    rather than asserting one exact number.
    """
    import plugins.tools._sandbox as S

    monkeypatch.setattr(S, "_get_e2b_config", lambda: ("key", "base", 1800))
    monkeypatch.setattr(S.time, "monotonic", lambda: clock)
    sandbox = _RemoteSandbox()
    S._extend_sandbox_ttl(sandbox)
    assert sandbox.ttl >= 1800, (
        f"TTL {sandbox.ttl} is shorter than the configured 1800 at clock={clock}"
    )


async def test_ttl_tracking_tolerates_a_non_weakrefable_sandbox(monkeypatch) -> None:
    """Falls back to the old behaviour rather than losing the refresh."""
    import plugins.tools._sandbox as S

    monkeypatch.setattr(S, "_get_e2b_config", lambda: ("key", "base", 1800))

    class _Slotted:
        __slots__ = ("ttl",)

        def __init__(self) -> None:
            self.ttl = None

        def set_timeout(self, seconds: int) -> None:
            self.ttl = seconds

    sandbox = _Slotted()
    S._extend_sandbox_ttl(sandbox, min_seconds=3660)
    assert sandbox.ttl >= 3660
