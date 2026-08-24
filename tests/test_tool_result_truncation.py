"""Shape of the inline preview kept when a tool result overflows its cap.

Three invariants live here, one per way the old shape misled the model:

* what survives inline is the head AND the tail, because an exec result states
  its verdict last;
* the recovery pointer is charged to the cap instead of being appended past it;
* the same body spills to one file however many times it is spilled.

Plus the per-turn aggregate budget, which had no call site at all.
"""

from __future__ import annotations

import json

import pytest

from frontier_agent.core.execution_context import (
    ExecutionScope,
    reset_current_execution_scope,
    set_current_execution_scope,
)
from frontier_agent.core.tool import tool
from plugins.tools import _overflow


@pytest.fixture(autouse=True)
def _isolate_spill_registry():
    """``_created_stores`` is module state; a test must not see another's stores."""
    saved = set(_overflow._created_stores)
    _overflow._created_stores.clear()
    try:
        yield
    finally:
        _overflow._created_stores.clear()
        _overflow._created_stores.update(saved)


# A pytest run in the shape that matters: the verdict is in the last four lines
# and nothing before them names the failing test.
_PYTEST_TAIL = (
    "=========================== short test summary info ===========================\n"
    "FAILED tests/test_billing.py::test_proration - AssertionError: 1199 != 1200\n"
    "1 failed, 812 passed in 41.03s\n"
)


def _pytest_output(chars: int) -> str:
    filler = "".join(
        f"tests/test_module_{i}.py::test_case_{i} PASSED                          [ 42%]\n"
        for i in range(chars // 80 + 1)
    )
    return "collected 813 items\n\n" + filler[:chars] + "\n" + _PYTEST_TAIL


@pytest.fixture
def spill_store(tmp_path, monkeypatch):
    """A private, agent-visible spill store bound to one execution scope."""
    store = tmp_path / "store"
    monkeypatch.setenv("SANDBOX_BACKEND", "container")
    monkeypatch.setenv("APODEX_SPILL_DIR", str(store))
    token = set_current_execution_scope(
        ExecutionScope(task_id="trunc-test", metadata={"llm_session_id": "s1"}),
    )
    try:
        yield store
    finally:
        reset_current_execution_scope(token)


def _files(spill_root):
    return sorted(spill_root.rglob("*.md"))


# ── 1. head AND tail ────────────────────────────────────────────────────


def test_preview_keeps_the_verdict_that_lives_in_the_tail() -> None:
    body = _pytest_output(60_000)

    out = _overflow.truncate_preview(body, 8_000)

    assert len(out) <= 8_000
    assert out.startswith("collected 813 items")
    # The whole reason for the change: the failing test is still readable.
    assert "FAILED tests/test_billing.py::test_proration" in out
    assert "1 failed, 812 passed" in out
    assert "chars elided" in out


def test_preview_splices_only_on_line_boundaries() -> None:
    body = "".join(f"line {i:04d} " + "x" * 60 + "\n" for i in range(2_000))

    out = _overflow.truncate_preview(body, 4_000)

    head, _, rest = out.partition("\n… ")
    _, _, tail = rest.partition(" …\n")
    assert head.endswith("x" * 60)
    assert tail.startswith("line ")


def test_preview_returns_short_text_untouched() -> None:
    assert _overflow.truncate_preview("short", 8_000) == "short"


def test_preview_falls_back_to_head_when_the_budget_cannot_be_split() -> None:
    """Two 200-char slivers plus a marker say less than one 300-char head."""
    body = "a" * 5_000

    out = _overflow.truncate_preview(body, 300)

    assert out == "a" * 300
    assert "elided" not in out


def test_head_mode_restores_the_legacy_shape(monkeypatch) -> None:
    """The A/B switch has to change the shape, or the arms are not comparable."""
    from frontier_agent.infra.config import get_config

    monkeypatch.setattr(get_config(), "tool_result_truncation", "head")
    body = _pytest_output(60_000)

    out = _overflow.truncate_preview(body, 8_000)

    assert len(out) <= 8_000
    assert "FAILED tests/test_billing.py::test_proration" not in out
    assert "elided" not in out


def test_an_unknown_mode_keeps_the_safe_default(monkeypatch) -> None:
    from frontier_agent.infra.config import get_config

    monkeypatch.setattr(get_config(), "tool_result_truncation", "sideways")
    assert _overflow._truncation_mode() == "middle"


# ── 2. the pointer is inside the cap ───────────────────────────────────


def test_overflow_fits_the_cap_with_the_pointer_inside_it(spill_store) -> None:
    from plugins.tools.meta import get_tool_meta

    cap = get_tool_meta("bash").max_result_chars
    assert cap > 0
    body = _pytest_output(cap * 6)

    out = _overflow.maybe_overflow("bash", body)

    assert len(out) <= cap
    assert f"saved read-only at {spill_store}/" in out
    assert "1 failed, 812 passed" in out


def test_a_result_within_its_cap_is_returned_verbatim(spill_store) -> None:
    body = "small enough\n"
    assert _overflow.maybe_overflow("bash", body) == body
    assert _files(spill_store) == []


def test_an_uncapped_tool_is_left_alone(spill_store) -> None:
    body = _pytest_output(50_000)
    # web_fetch sets max_result_chars=0; the loop's global cap is its only bound.
    assert _overflow.maybe_overflow("web_fetch", body) == body


def test_overflow_degrades_to_a_pointerless_footer_when_the_write_fails(
    spill_store, monkeypatch,
) -> None:
    from plugins.tools.meta import get_tool_meta

    monkeypatch.setattr(
        _overflow, "_spill_document",
        lambda *a, **k: (_ for _ in ()).throw(OSError("read-only fs")),
    )
    cap = get_tool_meta("bash").max_result_chars
    body = _pytest_output(cap * 6)

    out = _overflow.maybe_overflow("bash", body)

    assert len(out) <= cap
    assert "saved read-only at" not in out
    assert "not readable from this backend" in out
    # No half-written file, and no stray temp file, under the name the digest owns.
    assert _files(spill_store) == []
    assert list(spill_store.rglob("*.tmp")) == []


# ── 3. one file per body ────────────────────────────────────────────────


def test_the_same_body_spills_to_one_file(spill_store) -> None:
    body = _pytest_output(20_000)

    first = _overflow.spill_compacted_body("collect_reports", body)
    second = _overflow.spill_compacted_body("collect_reports", body)

    assert first == second
    assert len(_files(spill_store)) == 1


def test_two_tools_with_one_body_stay_distinguishable(spill_store) -> None:
    """The document header names the tool, so the digest has to include it."""
    body = _pytest_output(20_000)

    bash_ref = _overflow.spill_compacted_body("bash", body)
    report_ref = _overflow.spill_compacted_body("collect_reports", body)

    assert bash_ref != report_ref
    assert len(_files(spill_store)) == 2


def test_a_body_too_small_to_be_worth_a_file_is_not_spilled(spill_store) -> None:
    assert _overflow.spill_compacted_body("bash", "x" * 100) is None
    assert _files(spill_store) == []


def test_the_spilled_document_holds_the_body_verbatim(spill_store) -> None:
    body = _pytest_output(20_000)

    ref = _overflow.spill_compacted_body("bash", body)

    assert ref
    written = _files(spill_store)[0].read_text(encoding="utf-8")
    assert written.endswith(body)
    assert _overflow.get_overflow_content(str(_files(spill_store)[0])) == body


# ── 4. the per-turn aggregate budget ────────────────────────────────────


def test_aggregate_budget_bounds_one_turn_of_parallel_results(spill_store) -> None:
    from frontier_agent.core.loop_types import ToolResult
    from frontier_agent.core.runtime.loop.tool_exec import (
        TOOL_RESULT_MAX_CHARS,
        _apply_aggregate_budget,
    )

    def _result(idx: int) -> ToolResult:
        return ToolResult(
            name="web_fetch",
            args={},
            result=f"PAPER-{idx}-HEAD\n" + _pytest_output(TOOL_RESULT_MAX_CHARS),
            duration_ms=1,
            tool_call_id=f"c{idx}",
            is_error=False,
        )

    results = _apply_aggregate_budget([_result(0), _result(1)])

    assert sum(len(r.result) for r in results) <= _overflow.MAX_AGGREGATE_RESULT_CHARS
    # Both are still recognisable, and what was cut is on disk.
    assert results[0].result.startswith("PAPER-0-HEAD")
    assert results[1].result.startswith("PAPER-1-HEAD")
    assert any("per-turn tool-result budget" in r.result for r in results)
    assert _files(spill_store)


def test_aggregate_budget_leaves_a_turn_that_fits_untouched() -> None:
    from frontier_agent.core.loop_types import ToolResult
    from frontier_agent.core.runtime.loop.tool_exec import _apply_aggregate_budget

    original = [
        ToolResult(
            name="bash", args={}, result="ok\n", duration_ms=1,
            tool_call_id="c1", is_error=False,
        ),
    ]

    assert _apply_aggregate_budget(original) is original


def test_aggregate_budget_never_cuts_a_result_to_nothing(spill_store) -> None:
    bodies = [_pytest_output(120_000) for _ in range(6)]

    adjusted = _overflow.check_aggregate_budget(bodies, ["bash"] * len(bodies))

    assert sum(len(b) for b in adjusted) <= _overflow.MAX_AGGREGATE_RESULT_CHARS
    # ``_MIN_AGGREGATE_KEEP`` is the budget, not the exact length: snapping the
    # splice to line boundaries can land a little under it.
    assert all(len(b) > _overflow._MIN_AGGREGATE_KEEP // 2 for b in adjusted)
    # Nothing is cut without a way back to it.
    for body, original in zip(adjusted, bodies, strict=True):
        assert body == original or "saved read-only at" in body


async def test_execute_tools_applies_the_turn_budget(spill_store) -> None:
    from frontier_agent.core.runtime.loop import tool_exec

    @tool(name="web_fetch")
    async def fake_fetch(url: str) -> str:
        return _pytest_output(tool_exec.TOOL_RESULT_MAX_CHARS)

    results = await tool_exec.execute_tools(
        [
            {"name": "web_fetch", "args": {"url": "a"}, "id": "c1"},
            {"name": "web_fetch", "args": {"url": "b"}, "id": "c2"},
        ],
        {"web_fetch": fake_fetch},
        timeout=30,
        turn=1,
        count_offset=0,
    )

    assert len(results) == 2
    assert sum(len(r.result) for r in results) <= _overflow.MAX_AGGREGATE_RESULT_CHARS
    assert all(r.tool_call_id in {"c1", "c2"} for r in results)


# ── the A/B gate ────────────────────────────────────────────────────────


def test_the_offline_ab_gate_runs_and_passes(tmp_path) -> None:
    """``scripts/truncation_ab.py`` is also the regression gate.

    Run it in a subprocess: it sets process-wide env and config on purpose, and
    the point is to check the shipped entry point, not an import of its helpers.
    """
    import subprocess
    import sys

    proc = subprocess.run(
        [
            sys.executable, "scripts/truncation_ab.py",
            "--caps", "2000,8000",
            "--json", str(tmp_path / "ab.json"),
        ],
        capture_output=True, text=True, timeout=300, check=False,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    report = json.loads((tmp_path / "ab.json").read_text(encoding="utf-8"))
    for cap in ("2000", "8000"):
        assert report["head"][cap]["recall"]["tail"] == 0.0
        assert report["middle"][cap]["recall"]["tail"] == 1.0
        # Both arms must respect the cap they advertise.
        assert report["head"][cap]["results_over_cap"] == 0
        assert report["middle"][cap]["results_over_cap"] == 0


# ── the tunable global cap ──────────────────────────────────────────────


def test_the_global_cap_is_tunable_for_stress_runs(monkeypatch, spill_store) -> None:
    """web_fetch/web_search/read_file opt out of a per-tool cap, so the global one
    is the only lever a stress run has over them."""
    from frontier_agent.core.runtime.loop import tool_exec
    from frontier_agent.infra.config import get_config

    monkeypatch.setattr(get_config(), "tool_result_max_chars", 5_000)
    assert tool_exec._result_cap() == 5_000

    out = tool_exec._truncate_with_recovery("web_fetch", _pytest_output(60_000))

    assert len(out) <= 5_000
    assert "1 failed, 812 passed" in out


def test_an_unset_or_bad_global_cap_keeps_the_default(monkeypatch) -> None:
    from frontier_agent.core.runtime.loop import tool_exec
    from frontier_agent.infra.config import get_config

    monkeypatch.setattr(get_config(), "tool_result_max_chars", 0)
    assert tool_exec._result_cap() == tool_exec.TOOL_RESULT_MAX_CHARS
    # A non-numeric value must not uncap the context.
    monkeypatch.setattr(get_config(), "tool_result_max_chars", "sideways")
    assert tool_exec._result_cap() == tool_exec.TOOL_RESULT_MAX_CHARS


def test_the_footer_names_a_route_that_exists_without_a_reader(spill_store) -> None:
    """Profiles differ in what they bind: the stateful_react benchmark profile has
    bash/grep_search/glob_search and NO read_file, so a footer that only names
    read_file sends the agent at a tool it does not have."""
    from plugins.tools.meta import get_tool_meta

    body = _pytest_output(get_tool_meta("bash").max_result_chars * 6)

    out = _overflow.maybe_overflow("bash", body)

    assert "grep_search(pattern=" in out
    assert "bash" in out.rsplit("[...", 1)[-1]
    assert "read-only; do not write there" in out


# ── auto: shape per output kind ─────────────────────────────────────────


def test_auto_keeps_the_tail_for_sequential_output(monkeypatch) -> None:
    from frontier_agent.infra.config import get_config

    monkeypatch.setattr(get_config(), "tool_result_truncation", "auto")
    body = _pytest_output(60_000)

    out = _overflow.truncate_preview(body, 8_000, tool_name="bash")

    assert "1 failed, 812 passed" in out
    assert "chars elided" in out


def test_auto_keeps_a_contiguous_head_for_ranked_output(monkeypatch) -> None:
    """A relevance-ordered result inverts the rule: its tail is its worst hits,
    so half the budget spent there buys less than more good hits."""
    from frontier_agent.infra.config import get_config

    monkeypatch.setattr(get_config(), "tool_result_truncation", "auto")
    ranked = "".join(f"{i}. hit {i} " + "x" * 60 + "\n" for i in range(1, 600))

    out = _overflow.truncate_preview(ranked, 4_000, tool_name="web_search")

    assert "elided" not in out
    assert out.startswith("1. hit 1")
    # Deeper-but-still-useful ranks survive that a split budget would drop.
    assert "\n40. hit 40 " in out


def test_auto_without_a_tool_name_keeps_the_safe_default(monkeypatch) -> None:
    from frontier_agent.infra.config import get_config

    monkeypatch.setattr(get_config(), "tool_result_truncation", "auto")
    assert _overflow._truncation_mode() == "middle"


def test_auto_is_the_default_and_dispatches_per_tool() -> None:
    """The default has to route per tool, not force one shape on everything.

    A uniform default is what the live A/B measured, and it found no accuracy
    difference either way; the arms pin `middle`/`head` explicitly, so they are
    unaffected by what the default is.
    """
    from frontier_agent.infra.config import FrontierAgentConfig, get_config

    assert FrontierAgentConfig.model_fields["tool_result_truncation"].default == "auto"
    # And with nothing overridden, the live config really does dispatch.
    if get_config().tool_result_truncation == "auto":
        assert _overflow._truncation_mode("web_search") == "head"
        assert _overflow._truncation_mode("bash") == "middle"


def test_only_relevance_ranked_tools_are_marked_ranked() -> None:
    from plugins.tools.meta import get_tool_meta

    assert get_tool_meta("web_search").result_is_ranked
    assert get_tool_meta("scholar_search").result_is_ranked
    # A fetched document, a grep over files in traversal order, and an exec log
    # are all sequential — their ends carry information.
    for name in ("web_fetch", "grep_search", "glob_search", "bash", "read_file"):
        assert not get_tool_meta(name).result_is_ranked, name


# ── Site 3 must not cut a tool's own continuation pointer ──────────────────
#
# Neither post-processor had any test at all, which is how the two drifted: the
# React one whitelists read_file and budgets grep/glob at 10K, both explained in
# comments; the agent-team one carried the same explanation for recover_result
# while read_file and grep/glob fell to the 6K default. Measured on a live
# agent-team run, that cut the tail off 5 of 154 results — including the
# ``read_file again with offset=N`` line the caller needs to continue, replaced
# by generic "re-fetch" advice. Parametrised over BOTH processors so a future
# divergence fails here instead of in a benchmark trajectory.

_POINTER = "\n\n[Full results saved to /spill/abc123.md — use read_file to see the rest]"


def _pointer_bearing_body() -> str:
    """A gate-①-sized result whose recovery pointer sits at the very end."""
    filler = "match line here\n" * 500
    return filler[: 8_000 - len(_POINTER)] + _POINTER


def _processors():
    from workflows.agent_team.subagent_runtime import SwarmToolResultPostProcessor
    from workflows.stateful_react_agent._runtime import ReactToolResultPostProcessor

    return [
        pytest.param(ReactToolResultPostProcessor(), id="stateful_react"),
        pytest.param(SwarmToolResultPostProcessor(), id="agent_team"),
    ]


@pytest.mark.parametrize("processor", _processors())
@pytest.mark.parametrize("tool", ["read_file", "grep_search", "glob_search"])
def test_site3_keeps_the_self_pagination_pointer(processor, tool: str) -> None:
    from frontier_agent.core.loop_types import ToolResult

    body = _pointer_bearing_body()
    out = processor.process(ToolResult(
        name=tool, args={}, result=body, duration_ms=0,
        tool_call_id="call-1", is_error=False,
    ))

    # The pointer is the whole point: without it the agent is told to redo the
    # work instead of being handed the path or offset that continues it.
    assert "/spill/abc123.md" in out, f"{tool}: recovery pointer was cut"
    assert out == body, f"{tool}: body was shortened despite fitting gate ①"
