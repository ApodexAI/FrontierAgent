from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import Any

import pytest

from frontier_agent.core.loop_types import TurnContext, notify_observers
from plugins.tools import task_board as tb
from workflows._shared.citation_contract import (
    finalize_report_with_canonical_references,
)
from workflows.agent_team.nodes import fast_reporter_v1
from workflows.agent_team.nodes.reporter import agent_team_reporter
from workflows.agent_team.observers import bare_text_finalize as btf
from workflows.agent_team.observers.bare_text_finalize import (
    BareTextFinalizeObserver,
)
from workflows.agent_team.spec import SWARM_SPEC
from workflows.agent_team.spec_report import AGENT_TEAM_REPORT_SPEC

_TASK = "task"
_ANSWER = "Deployment complete; 100% of attacks blocked [1]."
_REFERENCES = [{"url": "https://example.com/a", "title": "Example"}]
_BYPASS_ERR = (
    "Cannot finish: task board has unresolved item(s) ['t1']. "
    "For each, call update_task(...)"
)


def _seed_board(resolutions: dict[str, str]) -> None:
    tb._BOARDS[_TASK] = {
        "seq": len(resolutions),
        "tasks": {
            tid: {
                "description": f"work {tid}",
                "resolution": resolution,
                "owners": [],
            }
            for tid, resolution in resolutions.items()
        },
    }


def _context(turn: int, max_turns: int = 20) -> TurnContext:
    return TurnContext(
        turn=turn,
        max_turns=max_turns,
        task_id=_TASK,
        role_id="coordinator",
        ai_text=_ANSWER,
        thinking="",
        tool_calls=[],
        messages=[],
        usage=None,
        metadata={},
    )


def _dispatch(ctx: TurnContext) -> list:
    try:
        return asyncio.run(
            notify_observers(
                [BareTextFinalizeObserver()],
                "on_llm_response",
                ctx,
            ),
        )
    finally:
        tb._BOARDS.pop(_TASK, None)


def test_mid_run_unfinished_board_still_blocks() -> None:
    """Away from the last turn the gate wins: rejected, never latched."""
    _seed_board({"t1": "open"})
    ctx = _context(turn=5)

    interventions = _dispatch(ctx)

    (intervention,) = interventions
    assert intervention.continue_to_next_turn is True
    assert intervention.stop_reason is None
    assert "final_answer" not in ctx.metadata


def test_final_turn_bypass_stores_marker_and_warning() -> None:
    """The bypass must leave machine- and human-readable traces in metadata.

    ``finalize_gate`` blocks while board items are open, but the final turn
    accepts the answer anyway (never lose it to max_turns). The observer
    stores the gate message (``finalize_gate_bypassed``) and a ready-to-append
    warning built while the board is still live
    (``finalize_gate_warning``); delivery nodes append the warning after any
    finalization that would otherwise strip it (issue #19, review on #48).
    The latched answer itself stays clean — appending here would be defeated
    by the reporter's References cleanup.
    """
    _seed_board({"t1": "open", "t2": "open"})
    ctx = _context(turn=19)

    interventions = _dispatch(ctx)

    (intervention,) = interventions
    assert intervention.stop_reason == "final_answer"
    assert ctx.metadata.get("finalize_gate_bypassed")
    warning = str(ctx.metadata.get("finalize_gate_warning") or "")
    assert "unfinished" in warning.casefold()
    assert "t1" in warning and "t2" in warning
    # The answer is latched verbatim; the warning travels separately.
    assert ctx.metadata["final_answer"] == _ANSWER


def test_final_turn_clean_board_latches_untouched() -> None:
    """Gate passes → the answer is latched verbatim, no marker, no warning."""
    _seed_board({"t1": "resolved", "t2": "cancelled"})
    ctx = _context(turn=19)

    interventions = _dispatch(ctx)

    (intervention,) = interventions
    assert intervention.stop_reason == "final_answer"
    assert ctx.metadata["final_answer"] == _ANSWER
    assert "finalize_gate_bypassed" not in ctx.metadata
    assert "finalize_gate_warning" not in ctx.metadata


def test_references_finalizer_strips_a_trailing_warning() -> None:
    """Documents WHY the warning must be re-attached after finalization.

    ``finalize_report_with_canonical_references`` drops everything from the
    ``References`` heading to the end of the body — a warning appended after
    that section does not survive the reporter's citation cleanup.

    ``strip_trailing_references`` refuses to cut when the heading starts
    before 30% of the body (mid-body sections are legitimate), so the body
    must be realistically long — as in a real report, where the reviewer's
    reproduction showed the warning being stripped.
    """
    body = (
        "Attackers probed hidden paths, enumerated backup archives, and "
        "fuzzed administrative endpoints across several weeks of access "
        "logs before the intrusion was detected. " * 3
        + "[1]\n\n"
        "## References\n\n"
        "[1] https://example.com/a\n"
    )
    warning = "\n\n---\n\n> ⚠ Unfinished work at submission: task t1."
    finalized = finalize_report_with_canonical_references(
        body + warning,
        references=_REFERENCES,
        language="en",
    )
    assert "Unfinished" not in finalized
    assert "https://example.com/a" in finalized  # canonical block re-appended


def test_append_bypass_warning_is_idempotent() -> None:
    """Delivery boundaries append exactly once, even on repeated calls."""
    warning = "\n\n---\n\n> ⚠ Unfinished work at submission: task t1."
    source: Mapping[str, Any] = {"finalize_gate_warning": warning}

    once = btf.append_bypass_warning("Report body.", source)
    twice = btf.append_bypass_warning(once, source)
    assert once.endswith(warning)
    assert twice == once
    # No stored warning → text untouched.
    assert btf.append_bypass_warning("Report body.", {}) == "Report body."


@pytest.mark.asyncio
async def test_reporter_reappends_warning_after_finalization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The reporter's output must carry the warning exactly once (review #48).

    ``_run_fast_reporter`` is the post-finalization report: References
    cleanup already ran inside the chain, so the node itself is the seam
    that re-attaches the stored bypass warning.
    """
    finalized_report = (
        "Rewritten report with citation [1].\n\n"
        "## References\n\n- [1] https://example.com/a\n"
    )

    async def _stub_reporter(
        _state: dict[str, Any], _ctx: Any,
    ) -> str:
        return finalized_report

    monkeypatch.setattr(fast_reporter_v1, "_run_fast_reporter", _stub_reporter)
    warning = "\n\n---\n\n> ⚠ Unfinished work at submission: task t1."
    state: dict[str, Any] = {
        "reporter_backend": "fast",
        "metadata": {},
        "finalize_gate_bypassed": _BYPASS_ERR,
        "finalize_gate_warning": warning,
    }

    out = await agent_team_reporter(state, None)  # type: ignore[arg-type]

    final_answer = str(out.get("final_answer") or "")
    assert final_answer.endswith(warning.rstrip("\n")) or warning in final_answer
    assert final_answer.count("Unfinished work at submission") == 1
    # The marker must pass through so downstream consumers keep it.
    assert out.get("finalize_gate_bypassed") == _BYPASS_ERR


def _node(spec: Any, node_id: str) -> Any:
    return next(node for node in spec.nodes if node.node_id == node_id)


@pytest.mark.parametrize("spec", [SWARM_SPEC, AGENT_TEAM_REPORT_SPEC])
def test_specs_publish_and_forward_the_bypass_marker(spec: Any) -> None:
    """main_agent must publish the marker; the reporter must receive it.

    Without these list entries the marker dies in loop-local metadata and
    consumers only ever see ``answer_status="complete"`` (review on #48).
    """
    marker_keys = {"finalize_gate_bypassed", "finalize_gate_warning"}
    main_fields = set(_node(spec, "main_agent").output_fields)
    reporter = _node(spec, "agent_team_reporter")
    assert marker_keys <= main_fields
    assert marker_keys <= set(reporter.context_policy.include_fields)
    assert marker_keys <= set(reporter.output_fields)
