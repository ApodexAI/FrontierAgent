from __future__ import annotations

import asyncio
import itertools
from pathlib import Path
from types import SimpleNamespace

import pytest

from frontier_agent.components.observers.context_size_guard import ContextSizeGuard
from frontier_agent.core.execution_context import (
    ExecutionScope,
    reset_current_execution_scope,
    set_current_execution_scope,
)
from frontier_agent.core.llm import LLMResponse
from frontier_agent.core.loop_types import LoopConfig, TurnContext
from frontier_agent.core.messages import assistant_msg, text_of
from frontier_agent.core.runtime.loop.agent_loop import run_agent_loop
from frontier_agent.core.runtime.loop.compact import (
    COMPACTION_SEQ_KEY,
    FORCE_COMPACTION_KEY,
    INPUT_ESTIMATE_KEY,
    KeepLastNToolResultsCompactor,
    compact_messages,
    compress_tool_results,
    estimate_tokens,
)
from frontier_agent.core.runtime.loop.compact_llm import LLMSummaryCompactor
from frontier_agent.core.runtime.loop.tiered_compact import (
    _SPILL_MANIFEST_HEADER,
    InputTokenGauge,
    TieredCompactor,
)
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



def _turn(*, messages: list[dict], tokens: int, metadata: dict | None = None) -> TurnContext:
    return TurnContext(
        turn=3,
        max_turns=1000,
        task_id="task",
        role_id="researcher",
        ai_text="",
        thinking="",
        tool_calls=[],
        messages=messages,
        usage={"prompt_tokens": tokens},
        metadata=metadata if metadata is not None else {},
    )


def _tool_history(body: str) -> list[dict]:
    return [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "research"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "call-1", "name": "web_fetch", "args": {}}],
        },
        {"role": "tool", "tool_call_id": "call-1", "content": body},
        {"role": "user", "content": "continue"},
    ]


def test_compress_tool_results_is_bounded_and_never_grows() -> None:
    body = "x" * 2_000 + " https://example.com/" + "y" * 2_000
    result = compress_tool_results(_tool_history(body), max_chars=300)[3]["content"]
    assert len(result) <= 300
    assert len(result) < len(body)


def _fanin_history(body: str) -> list[dict]:
    history = _tool_history(body)
    history[2]["tool_calls"] = [
        {"id": "call-1", "name": "collect_reports", "args": {}},
    ]
    return history


def test_compress_tool_results_never_truncates_protected_results() -> None:
    """Fan-in results are pinned out of Tier 1's spill path, so a candidate that
    goes straight back to the provider must not shrink them either."""
    body = "report " * 1_000
    history = _fanin_history(body)

    unprotected = compress_tool_results(history, max_chars=300)[3]["content"]
    protected = compress_tool_results(
        history, max_chars=300, protect_tool_names=frozenset({"collect_reports"}),
    )[3]["content"]

    assert len(unprotected) <= 300
    assert protected == body


def test_compress_tool_results_bounds_protected_results_for_a_summarizer() -> None:
    """Tier 2's pre-pass keeps reports readable without pricing out the request."""
    body = "report " * 1_000
    protected = compress_tool_results(
        _fanin_history(body),
        max_chars=300,
        protect_tool_names=frozenset({"collect_reports"}),
        protect_max_chars=2_000,
    )[3]["content"]

    assert 300 < len(protected) <= 2_000


def test_gauge_reads_the_estimate_of_the_request_that_was_sent() -> None:
    """Both sides of the scale must describe ONE request. ``ctx.messages`` at
    turn end cannot: it already carries this turn's completion, and the per-call
    system addendum was never in it."""
    gauge = InputTokenGauge()
    ctx = _turn(
        messages=_tool_history("small"), tokens=777,
        metadata={INPUT_ESTIMATE_KEY: 500},
    )
    asyncio.run(gauge.on_llm_response(ctx))
    assert gauge.tokens == 777
    assert gauge.estimate == 500


def test_gauge_without_a_published_estimate_keeps_the_scale_neutral() -> None:
    """A missing estimate must not silently understate the ratio."""
    gauge = InputTokenGauge()
    asyncio.run(gauge.on_llm_response(_turn(messages=_tool_history("x" * 4_000), tokens=777)))
    assert gauge.estimate == 0
    compactor = TieredCompactor(
        keep_tool_result=1, summary_llm=None, relief_target=1, gauge=gauge,
    )
    assert compactor._real_token_scale() == 1.0


@pytest.mark.asyncio
async def test_loop_publishes_the_estimate_of_the_request_it_sent() -> None:
    """End to end: the gauge's denominator is byte-for-byte the list the provider
    received — addendum included, this turn's completion excluded."""
    gauge = InputTokenGauge()

    class _OneShotLLM:
        model = "stub"

        def __init__(self) -> None:
            self.seen: list[int] = []

        async def chat(self, messages: list, **kwargs: object) -> LLMResponse:
            del kwargs
            self.seen.append(estimate_tokens(messages))
            return LLMResponse(
                content="conclusion " * 500, tool_calls=[], finish_reason="stop",
            )

    llm = _OneShotLLM()
    await run_agent_loop(
        system_prompt="system",
        user_message="go",
        llm=llm,
        tools=[],
        config=LoopConfig(
            max_turns=1,
            system_addendum_per_call="REMEMBER: " + "policy " * 400,
            system_addendum_min_turn=0,
        ),
        observers=[gauge],
    )

    assert len(llm.seen) == 1
    assert gauge.estimate == llm.seen[0]


def test_keep_last_spills_before_discarding_body() -> None:
    history = _tool_history("evidence " * 500)
    compacted = KeepLastNToolResultsCompactor(
        keep_tool_result=0,
        spill=lambda name, body: f"/workspace/.spill/{name}.md",
    ).compact(history, keep_recent=1)
    assert "[Full text] /workspace/.spill/web_fetch.md" in compacted[3]["content"]


def test_tier2_summarizes_real_history_not_tier1_placeholder() -> None:
    class CaptureLLM:
        def __init__(self) -> None:
            self.prompts: list[str] = []

        async def chat(self, messages: list[dict]) -> SimpleNamespace:
            self.prompts.append(messages[0]["content"])
            return SimpleNamespace(content="summary")

    llm = CaptureLLM()
    history = _tool_history("source https://example.com/fact " + "x" * 5_000)
    compactor = TieredCompactor(
        keep_tool_result=0,
        summary_llm=llm,
        relief_target=1,
    )
    asyncio.run(compactor.compact(history, keep_recent=1))
    assert llm.prompts
    assert "https://example.com/fact" in llm.prompts[0]


def test_tier2_no_relief_is_not_retried_every_turn() -> None:
    class GrowingLLM:
        calls = 0

        async def chat(self, messages: list[dict]) -> SimpleNamespace:
            self.calls += 1
            return SimpleNamespace(content="summary " * 10_000)

    llm = GrowingLLM()
    history = _tool_history("x" * 5_000)
    compactor = TieredCompactor(
        keep_tool_result=0,
        summary_llm=llm,
        relief_target=1,
    )
    asyncio.run(compactor.compact(history, keep_recent=1))
    asyncio.run(compactor.compact(history, keep_recent=1))
    assert llm.calls == 1


def test_tier2_keeps_bounded_spill_manifest(caplog) -> None:
    class SummaryLLM:
        async def chat(self, messages: list[dict]) -> SimpleNamespace:
            return SimpleNamespace(content="short summary")

    paths: list[str] = []

    def spill(name: str, body: str) -> str:
        path = f"/workspace/.spill/session/{len(paths):02d}-{name}.md"
        paths.append(path)
        return path

    history = _tool_history("evidence " * 1_000)
    history.insert(2, {"role": "assistant", "content": "analysis " * 1_000})
    compactor = TieredCompactor(
        keep_tool_result=0,
        summary_llm=SummaryLLM(),
        relief_target=1,
        spill=spill,
    )

    with caplog.at_level("INFO"):
        result = asyncio.run(compactor.compact(history, keep_recent=1))

    content = "\n".join(str(message.get("content", "")) for message in result)
    assert _SPILL_MANIFEST_HEADER in content
    assert paths[0] in content
    assert "selected=tier2" in caplog.text


def test_tool_compression_keeps_spill_manifest_without_tier2(caplog) -> None:
    class UnexpectedLLM:
        calls = 0

        async def chat(self, messages: list[dict]) -> SimpleNamespace:
            self.calls += 1
            raise AssertionError("Tier 2 should not run")

    llm = UnexpectedLLM()
    history = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "research"},
        {
            "role": "assistant", "content": "",
            "tool_calls": [{"id": "old", "name": "web_fetch", "args": {}}],
        },
        {"role": "tool", "tool_call_id": "old", "content": "old " * 1_000},
        {
            "role": "assistant", "content": "",
            "tool_calls": [{"id": "new", "name": "bash", "args": {}}],
        },
        {"role": "tool", "tool_call_id": "new", "content": "new " * 3_000},
        {"role": "user", "content": "continue"},
    ]
    path = "/workspace/.spill/session/old.md"
    compactor = TieredCompactor(
        keep_tool_result=1,
        summary_llm=llm,
        relief_target=500,
        spill=lambda name, body: path,
    )

    with caplog.at_level("INFO"):
        result = asyncio.run(compactor.compact(history, keep_recent=1))

    content = "\n".join(str(message.get("content", "")) for message in result)
    assert path in content
    assert llm.calls == 0
    assert "selected=tool_compression_" in caplog.text


def test_tool_compression_spills_a_fresh_result_before_shortening_it(caplog) -> None:
    """The latest tool result has not reached the model when turn-end
    compaction runs, so a selected compression candidate must make its discarded
    body recoverable just like Tier 1 does for older results."""

    class UnexpectedLLM:
        async def chat(self, messages: list[dict]) -> SimpleNamespace:
            raise AssertionError("the cheap candidate should satisfy relief")

    fresh_body = "FRESH_CANARY\n" + "new evidence " * 3_000 + "\nFRESH_END"
    history = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "research"},
        {
            "role": "assistant", "content": "",
            "tool_calls": [{"id": "old", "name": "web_fetch", "args": {}}],
        },
        {"role": "tool", "tool_call_id": "old", "content": "old " * 1_000},
        {
            "role": "assistant", "content": "",
            "tool_calls": [{"id": "fresh", "name": "web_fetch", "args": {}}],
        },
        {"role": "tool", "tool_call_id": "fresh", "content": fresh_body},
    ]
    spilled: list[tuple[str, str, str]] = []

    def _spill(name: str, body: str) -> str:
        path = f"/spill/session/{len(spilled)}.md"
        spilled.append((name, body, path))
        return path

    compactor = TieredCompactor(
        keep_tool_result=1,
        summary_llm=UnexpectedLLM(),
        relief_target=500,
        spill=_spill,
    )

    with caplog.at_level("INFO"):
        result = asyncio.run(compactor.compact(history, keep_recent=1))

    fresh = next(m for m in result if m.get("tool_call_id") == "fresh")
    fresh_spills = [path for _name, body, path in spilled if body == fresh_body]
    manifest_refs = TieredCompactor._spill_refs(result)
    assert "selected=tool_compression_" in caplog.text
    assert len(text_of(fresh["content"])) < len(fresh_body)
    assert fresh_spills and fresh_spills[0] in manifest_refs


def test_tool_compression_keeps_fresh_results_when_spill_is_disabled(caplog) -> None:
    """Without a spill store there is nowhere to put a discarded body, so the
    latest tool-call turn — which compaction reaches before the model has read it
    even once — has to survive a winning candidate verbatim.

    Driven through ``compact`` rather than by handing ``compress_tool_results``
    an id set, because the behaviour under test IS the compactor's decision to
    populate that set only when ``spill is None``. Sized so a compression
    candidate genuinely wins: ``keep_recent`` covers all three results, so Tier 1
    frees nothing and the candidate has to be the one that does.
    """

    class UnexpectedLLM:
        async def chat(self, messages: list[dict]) -> SimpleNamespace:
            raise AssertionError("the cheap candidate should satisfy relief")

    stale_a = "aaa evidence " * 4_000
    stale_b = "bbb evidence " * 4_000
    fresh_body = "FRESH_CANARY\n" + "new evidence " * 2_000 + "\nFRESH_END"
    history = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "research"},
        {
            "role": "assistant", "content": "",
            "tool_calls": [{"id": "a", "name": "web_fetch", "args": {}}],
        },
        {"role": "tool", "tool_call_id": "a", "content": stale_a},
        {
            "role": "assistant", "content": "",
            "tool_calls": [{"id": "b", "name": "web_fetch", "args": {}}],
        },
        {"role": "tool", "tool_call_id": "b", "content": stale_b},
        {
            "role": "assistant", "content": "",
            "tool_calls": [{"id": "fresh", "name": "web_fetch", "args": {}}],
        },
        {"role": "tool", "tool_call_id": "fresh", "content": fresh_body},
    ]

    compactor = TieredCompactor(
        keep_tool_result=3,
        summary_llm=UnexpectedLLM(),
        # Above every candidate that preserves the fresh body, below Tier 1 —
        # which keeps all three results whole here and so frees nothing.
        relief_target=8_000,
        spill=None,
    )

    with caplog.at_level("INFO"):
        result = asyncio.run(compactor.compact(history, keep_recent=3))

    by_id = {m.get("tool_call_id"): text_of(m.get("content")) for m in result}
    assert "selected=tool_compression_" in caplog.text
    # Non-vacuous: the candidate really did shorten the results it was allowed to.
    assert len(by_id["a"]) < len(stale_a)
    assert len(by_id["b"]) < len(stale_b)
    assert by_id["fresh"] == fresh_body


def test_tool_compression_candidate_keeps_protected_fanin_intact(caplog) -> None:
    """The winning candidate is the final history, and a blanked fan-in report
    has no spill file behind it — so ``protect_tool_names`` has to reach it."""
    history = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "research"},
        {
            "role": "assistant", "content": "",
            "tool_calls": [{"id": "fanin", "name": "collect_reports", "args": {}}],
        },
        {"role": "tool", "tool_call_id": "fanin", "content": "findings " * 500},
        {
            "role": "assistant", "content": "",
            "tool_calls": [{"id": "new", "name": "bash", "args": {}}],
        },
        {"role": "tool", "tool_call_id": "new", "content": "new " * 3_000},
        {"role": "user", "content": "continue"},
    ]
    compactor = TieredCompactor(
        keep_tool_result=1,
        summary_llm=None,
        relief_target=2_000,
        protect_tool_names=frozenset({"collect_reports"}),
        spill=lambda name, body: f"/spill/session/{name}.md",
    )

    with caplog.at_level("INFO"):
        result = asyncio.run(compactor.compact(history, keep_recent=1))

    assert "selected=tool_compression_" in caplog.text
    fanin = next(m for m in result if m.get("tool_call_id") == "fanin")
    assert fanin["content"] == "findings " * 500


def test_spill_refs_recovers_run_dir_manifest_paths() -> None:
    """The run-dir branch emits ``spill/`` (no dot). Keying the harvest on the
    workspace ``.spill/`` spelling loses every ref on a second Tier 2 pass."""
    refs = ["/runs/task-1/spill/2f9c/00-web_fetch.md"]
    manifest = TieredCompactor._with_spill_manifest(
        [{"role": "system", "content": "system"}], refs,
    )
    assert TieredCompactor._spill_refs(manifest) == refs


def test_spill_refs_keeps_native_paths_with_spaces() -> None:
    refs = ["/tmp/My Workspace/spill/2f9c/00-web_fetch.md"]
    manifest = TieredCompactor._with_spill_manifest(
        [{"role": "system", "content": "system"}], refs,
    )
    assert TieredCompactor._spill_refs(manifest) == refs


def test_prose_is_never_harvested_as_a_ref() -> None:
    """The index carries its paths in ``spill_refs``; text is only for the model.

    This replaced a shape guess (absolute path, no whitespace after the store
    marker, header followed only by bullets) that existed to tell a real index
    from prose that resembled one — a distinction the field makes for free. A
    path mentioned in a sentence, or list-shaped prose spliced under the header
    by a deterministic rollback, is now simply not a ref.
    """
    prose = {
        "role": "user",
        "content": (
            f"{_SPILL_MANIFEST_HEADER}\n"
            "- the agent searched for prior art and found none\n"
            "- /workspace/.spill/2f9c/00-web_fetch.md\n"
            "Key evidence: see /workspace/.spill/2f9c/other.md\n"
        ),
    }

    assert TieredCompactor._spill_refs([prose]) == []

    carrier = {
        "role": "user",
        "content": "rendered for the model",
        "spill_refs": ["/workspace/.spill/2f9c/00-web_fetch.md"],
    }
    assert TieredCompactor._spill_refs([carrier]) == [
        "/workspace/.spill/2f9c/00-web_fetch.md",
    ]


def test_a_legacy_prose_index_is_left_in_place_rather_than_lost() -> None:
    """A history checkpointed before the field existed keeps its index message,
    so the paths stay readable by the model even though they are not harvested.
    The fresh index is added alongside; it is the one later passes replace."""
    legacy = {
        "role": "user",
        "content": f"{_SPILL_MANIFEST_HEADER}\n- /workspace/.spill/old/a.md\n",
    }

    out = TieredCompactor._with_spill_manifest(
        [{"role": "system", "content": "s"}, legacy],
        ["/workspace/.spill/new/b.md"],
    )

    bodies = [text_of(m.get("content")) for m in out]
    assert any("/workspace/.spill/old/a.md" in b for b in bodies)
    assert any("/workspace/.spill/new/b.md" in b for b in bodies)
    assert sum(1 for m in out if m.get("spill_refs")) == 1


def test_deterministic_summary_drops_a_manifest_instead_of_truncating_it() -> None:
    """``compact_messages`` folds a user message in as ``content[:400]``. On a
    manifest that cut lands mid-path, and the fragment is still path-shaped
    enough to be re-harvested as a dead ref into every later manifest. Reachable
    whenever ``LLMSummaryCompactor(failure_fallback="deterministic")`` falls back.
    """
    long_path = "/workspace/.spill/2f9c/" + "d" * 380 + "/00-web_fetch.md"
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": f"{_SPILL_MANIFEST_HEADER}\n- {long_path}"},
        {"role": "user", "content": "an older question"},
        {"role": "user", "content": "the recent one"},
    ]

    compacted = compact_messages(messages, keep_recent=1)
    content = "\n".join(str(m.get("content", "")) for m in compacted)

    assert _SPILL_MANIFEST_HEADER not in content
    assert "/workspace/.spill/2f9c/" not in content
    assert TieredCompactor._spill_refs(compacted) == []


def test_manifest_cost_does_not_push_a_passing_candidate_into_tier2(caplog) -> None:
    """The recovery index is a bounded ~3 KB the compaction pays for itself.
    Charging it against the relief target discarded a candidate that had ALREADY
    freed enough, falling through to a Tier 2 round-trip that is slower and under
    no obligation to come back smaller."""

    class UnexpectedLLM:
        calls = 0

        async def chat(self, messages: list, **kwargs: object) -> SimpleNamespace:
            del messages, kwargs
            type(self).calls += 1
            return SimpleNamespace(content="summary")

    history: list[dict] = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "research"},
    ]
    for idx in range(20):
        history.append({
            "role": "assistant", "content": "",
            "tool_calls": [{"id": f"c{idx}", "name": "web_fetch", "args": {}}],
        })
        history.append({
            "role": "tool", "tool_call_id": f"c{idx}", "content": "evidence " * 400,
        })
    history.append({"role": "user", "content": "continue"})

    # Target set so the cheap candidate passes EXACTLY: any manifest at all puts
    # it over, which is the condition that used to trigger the wasted Tier 2 call.
    target = estimate_tokens(compress_tool_results(history, max_chars=300))
    tier1_only = KeepLastNToolResultsCompactor(keep_tool_result=3).compact(history, 1)
    assert estimate_tokens(tier1_only) > target, "tier1 must not win outright here"

    paths = iter(f"/workspace/.spill/2f9c/{i:02d}-{'d' * 80}.md" for i in range(20))
    compactor = TieredCompactor(
        keep_tool_result=3,
        summary_llm=UnexpectedLLM(),
        relief_target=target,
        spill=lambda name, body: next(paths),
    )

    with caplog.at_level("INFO"):
        result = asyncio.run(compactor.compact(history, keep_recent=1))

    content = "\n".join(str(m.get("content", "")) for m in result)
    assert UnexpectedLLM.calls == 0
    assert "selected=tool_compression_" in caplog.text
    assert _SPILL_MANIFEST_HEADER in content


def test_tier2_sees_and_spills_full_protected_fanin() -> None:
    """Tier 2 must not erase middle reports before its LLM sees them, and a
    winning summary needs a recovery pointer because it replaces the originals.
    """
    canary = "MIDDLE_REPORT_CANARY_7d2a"
    body = (
        '<report agent="first">' + "A" * 9_000 + "</report>\n"
        f'<report agent="middle">{canary}</report>\n'
        '<report agent="last">' + "B" * 9_000 + "</report>"
    )

    class CapturingLLM:
        seen = ""

        async def chat(self, messages: list, **kwargs: object) -> SimpleNamespace:
            del kwargs
            type(self).seen = "\n".join(
                str(message.get("content", "")) for message in messages
            )
            return SimpleNamespace(content="summary")

    spilled: list[tuple[str, str]] = []

    def _spill(name: str, content: str) -> str:
        spilled.append((name, content))
        return "/tmp/My Workspace/spill/session/collect-reports.md"

    history = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "delegate"},
        {
            "role": "assistant", "content": "",
            "tool_calls": [{"id": "fanin", "name": "collect_reports", "args": {}}],
        },
        {"role": "tool", "tool_call_id": "fanin", "content": body},
        {"role": "user", "content": "synthesize now"},
    ]
    compactor = TieredCompactor(
        keep_tool_result=0,
        summary_llm=CapturingLLM(),
        relief_target=1,
        protect_tool_names=frozenset({"collect_reports"}),
        spill=_spill,
    )

    result = asyncio.run(compactor.compact(history, keep_recent=1))
    rendered = "\n".join(str(message.get("content", "")) for message in result)

    assert canary in CapturingLLM.seen
    assert spilled == [("collect_reports", body)]
    assert "/tmp/My Workspace/spill/session/collect-reports.md" in rendered


def test_spill_manifest_keeps_only_latest_twenty_paths() -> None:
    assert "Read-only recovery index" in _SPILL_MANIFEST_HEADER
    assert "Never write here" in _SPILL_MANIFEST_HEADER
    refs = [f"/workspace/.spill/session/{idx:02d}.md" for idx in range(25)]
    result = TieredCompactor._with_spill_manifest(
        [{"role": "system", "content": "system"}], refs,
    )
    content = "\n".join(str(message.get("content", "")) for message in result)
    assert refs[0] not in content
    assert refs[4] not in content
    assert all(path in content for path in refs[5:])


def test_spill_manifest_update_is_idempotent() -> None:
    first = "/workspace/.spill/session/first.md"
    second = "/workspace/.spill/session/second.md"
    result = TieredCompactor._with_spill_manifest(
        [{"role": "system", "content": "system"}], [first],
    )
    refs = TieredCompactor._spill_refs(result)
    result = TieredCompactor._with_spill_manifest(result, [*refs, second])
    content = "\n".join(str(message.get("content", "")) for message in result)
    assert content.count(_SPILL_MANIFEST_HEADER) == 1
    assert first in content
    assert second in content


def test_context_guard_forces_one_completed_compaction_before_stop() -> None:
    guard = ContextSizeGuard(100, force_compaction_first=True)
    asyncio.run(guard.on_loop_start(LoopConfig()))
    metadata: dict[str, object] = {}

    first = asyncio.run(guard.on_llm_response(
        _turn(messages=[], tokens=101, metadata=metadata),
    ))
    assert first is None
    assert metadata[FORCE_COMPACTION_KEY] is True

    metadata.pop(FORCE_COMPACTION_KEY)
    metadata[COMPACTION_SEQ_KEY] = 1
    second = asyncio.run(guard.on_llm_response(
        _turn(messages=[], tokens=101, metadata=metadata),
    ))
    assert second is not None
    assert second.stop_reason == "budget_exhausted"


def test_context_guard_stops_when_the_forced_compaction_never_runs() -> None:
    """A ``no_tool`` nudge and ``continue_to_next_turn`` both ``continue`` before
    turn end, and only turn end advances COMPACTION_SEQ_KEY. Re-arming on an
    unadvanced seq without a bound kept the loop issuing over-limit requests
    until the attempt budget ran out."""
    guard = ContextSizeGuard(100, force_compaction_first=True)
    asyncio.run(guard.on_loop_start(LoopConfig()))
    metadata: dict[str, object] = {}

    def _over_limit_response():
        # The loop consumed the request but never reached the compaction step,
        # so the sequence stays put.
        metadata.pop(FORCE_COMPACTION_KEY, None)
        return asyncio.run(guard.on_llm_response(
            _turn(messages=[], tokens=101, metadata=metadata),
        ))

    assert _over_limit_response() is None  # arms the forced pass
    assert _over_limit_response() is None  # one retry for a skipped turn
    stopped = _over_limit_response()
    assert stopped is not None
    assert stopped.stop_reason == "budget_exhausted"


def test_context_guard_rearms_after_dropping_back_under_the_limit() -> None:
    """Recovering under the limit restores the full one-forced-pass budget."""
    guard = ContextSizeGuard(100, force_compaction_first=True)
    asyncio.run(guard.on_loop_start(LoopConfig()))
    metadata: dict[str, object] = {}

    assert asyncio.run(guard.on_llm_response(
        _turn(messages=[], tokens=101, metadata=metadata),
    )) is None
    metadata.pop(FORCE_COMPACTION_KEY)
    metadata[COMPACTION_SEQ_KEY] = 1
    assert asyncio.run(guard.on_llm_response(
        _turn(messages=[], tokens=50, metadata=metadata),
    )) is None

    # Growing back over the limit earns a fresh forced pass, not an instant stop.
    assert asyncio.run(guard.on_llm_response(
        _turn(messages=[], tokens=101, metadata=metadata),
    )) is None
    assert metadata[FORCE_COMPACTION_KEY] is True


def test_spill_lands_outside_the_workspace(tmp_path, monkeypatch) -> None:
    """The store must not be inside the tree the agent writes.

    It used to live at ``<workspace>/.spill``, which under ``native`` is
    frequently the user's own repository, and which put it inside the one tree
    every write guard has to make an exception for.
    """
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.setattr(
        "plugins.tools._sandbox.current_local_workspace", lambda: str(workspace),
    )
    monkeypatch.setenv("SANDBOX_BACKEND", "container")
    monkeypatch.setenv("APODEX_SPILL_DIR", str(tmp_path / "store"))

    token = set_current_execution_scope(ExecutionScope(task_id="task/unsafe"))
    try:
        visible = _overflow.spill_compacted_body("web_fetch", "detail " * 400)
    finally:
        reset_current_execution_scope(token)

    assert visible is not None
    assert visible.startswith(str(tmp_path / "store"))
    # An unsafe task id never reaches the filesystem, and nothing lands in the
    # workspace.
    assert "task/unsafe" not in visible
    assert not (workspace / ".spill").exists()
    assert list(workspace.iterdir()) == []

    from plugins.tools._sandbox import resolve_runtime_path

    physical = Path(resolve_runtime_path(visible))
    assert physical.is_relative_to(tmp_path / "store")
    assert _overflow.get_overflow_content(str(physical)) == "detail " * 400
    assert _overflow.cleanup_overflow("task/unsafe") == 1
    assert not physical.exists()


def test_a_spilled_body_is_readable_but_not_writable_by_the_file_tools(
    tmp_path, monkeypatch,
) -> None:
    """Read authorization is what recovery needs; the absence of write
    authorization is what replaces a per-writer special case."""
    from plugins.tools._path_auth import _is_path_allowed

    monkeypatch.setenv("APODEX_SPILL_DIR", str(tmp_path / "store"))
    token = set_current_execution_scope(ExecutionScope(task_id="t", metadata={}))
    try:
        visible = _overflow.spill_compacted_body("bash", "body " * 400)
    finally:
        reset_current_execution_scope(token)
    assert visible

    from plugins.tools._sandbox import resolve_runtime_path

    physical = resolve_runtime_path(visible)
    readable, _ = _is_path_allowed(physical)
    writable, reason = _is_path_allowed(physical, write_access=True)

    assert readable, "recovery cannot work if the store is unreadable"
    assert not writable, reason


def test_a_run_directory_is_preferred_over_the_temp_root(tmp_path, monkeypatch) -> None:
    """A harness with a run directory keeps its recovery files beside the run."""
    monkeypatch.setenv("SANDBOX_BACKEND", "native")
    monkeypatch.delenv("APODEX_SPILL_DIR", raising=False)
    monkeypatch.setenv("APODEX_RUN_DIR", str(tmp_path / "run"))

    token = set_current_execution_scope(ExecutionScope(task_id="native-task"))
    try:
        visible = _overflow.spill_compacted_body("web_fetch", "native " * 400)
    finally:
        reset_current_execution_scope(token)

    assert visible is not None
    assert visible.startswith(str(tmp_path / "run" / "spill"))
    assert _overflow.get_overflow_content(visible) == "native " * 400


def test_native_names_the_physical_path_not_the_mount(tmp_path, monkeypatch) -> None:
    """Under ``native`` there is no mount, so the canonical ``/spill`` would name
    whatever that path happens to be on the host — or nothing."""
    workspace = tmp_path / "native-workspace"
    workspace.mkdir()
    monkeypatch.setenv("SANDBOX_BACKEND", "native")
    monkeypatch.setenv("APODEX_SPILL_DIR", str(tmp_path / "store"))
    monkeypatch.setattr(
        "plugins.tools._sandbox.current_local_workspace", lambda: str(workspace),
    )
    token = set_current_execution_scope(ExecutionScope(task_id="native-task"))
    try:
        visible = _overflow.spill_compacted_body("web_fetch", "native " * 400)
    finally:
        reset_current_execution_scope(token)

    assert visible is not None
    assert visible.startswith(str(tmp_path / "store"))
    assert not visible.startswith("/spill")
    # Even with a private workspace available, nothing is written into it.
    assert list(workspace.iterdir()) == []
    assert _overflow.get_overflow_content(visible) == "native " * 400


def test_compaction_spills_are_isolated_by_session(tmp_path, monkeypatch) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.setattr(
        "plugins.tools._sandbox.current_local_workspace",
        lambda: str(workspace),
    )

    paths: list[str] = []
    for session_id in ("session-a", "session-b"):
        token = set_current_execution_scope(ExecutionScope(
            task_id="shared-task",
            metadata={"llm_session_id": session_id},
        ))
        try:
            path = _overflow.spill_compacted_body("web_fetch", "detail " * 400)
        finally:
            reset_current_execution_scope(token)
        assert path is not None
        paths.append(path)

    assert paths[0].rsplit("/", 1)[0] != paths[1].rsplit("/", 1)[0]


def test_cleanup_overflow_handles_empty_task_id_safely(tmp_path, monkeypatch) -> None:
    workspace = tmp_path / "workspace"
    spill_root = workspace / ".spill"
    spill_root.mkdir(parents=True)
    sentinel = spill_root / "must-survive.md"
    sentinel.write_text("unscoped evidence", encoding="utf-8")
    monkeypatch.setattr(
        "plugins.tools._sandbox.current_local_workspace",
        lambda: str(workspace),
    )

    assert _overflow.cleanup_overflow("") == 0
    assert sentinel.read_text(encoding="utf-8") == "unscoped evidence"


def test_cleanup_overflow_removes_only_target_session(tmp_path, monkeypatch) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.setattr(
        "plugins.tools._sandbox.current_local_workspace",
        lambda: str(workspace),
    )
    session_a, _ = _overflow._overflow_dir("task:session-a")
    session_b, _ = _overflow._overflow_dir("task:session-b")
    file_a = session_a / "a.md"
    file_b = session_b / "b.md"
    file_a.write_text("session a", encoding="utf-8")
    file_b.write_text("session b", encoding="utf-8")

    assert _overflow.cleanup_overflow("task:session-a") == 1
    assert not file_a.exists()
    assert file_b.read_text(encoding="utf-8") == "session b"


def test_cleanup_overflow_defaults_to_the_callers_own_scope(tmp_path, monkeypatch) -> None:
    """The writers key on ``f"{task_id}:{llm_session_id}"``. A caller passing a
    bare ``task_id`` hashes to a different directory and silently removes
    nothing, so the no-argument form has to resolve the same scope they used."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.setattr(
        "plugins.tools._sandbox.current_local_workspace", lambda: str(workspace),
    )
    scope = ExecutionScope(task_id="t1", metadata={"llm_session_id": "s1"})
    token = set_current_execution_scope(scope)
    try:
        _overflow.spill_compacted_body("web_fetch", "evidence " * 200)
        assert _overflow.cleanup_overflow("") == 0
        assert _overflow.cleanup_overflow("t1") == 0  # bare task_id matches nothing
        assert _overflow.cleanup_overflow() == 1
    finally:
        reset_current_execution_scope(token)


def test_cleanup_overflow_does_not_create_the_store_it_resolves(
    tmp_path, monkeypatch,
) -> None:
    """Resolving a directory in order to delete it must not first mkdir it."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.setattr(
        "plugins.tools._sandbox.current_local_workspace", lambda: str(workspace),
    )

    assert _overflow.cleanup_overflow("never-spilled:session") == 0
    assert not (workspace / ".spill").exists()


def test_process_cleanup_removes_only_stores_this_process_created(
    tmp_path, monkeypatch,
) -> None:
    """A discarded conversation drops its own recovery files and nothing else.

    The guarantee used to come from confining the walk to one session-private
    workspace, with symlink and filesystem-root defences because that tree was
    the agent's own. It now comes from only ever deleting directories this
    process created — a neighbour's store in the same root is not ours to remove,
    so none of those defences are needed.
    """
    store = tmp_path / "store"
    monkeypatch.setenv("APODEX_SPILL_DIR", str(store))

    # Another session's store, in the same root, which we did not create.
    stranger = store / "someone-else"
    stranger.mkdir(parents=True)
    (stranger / "theirs.md").write_text("theirs", encoding="utf-8")

    token = set_current_execution_scope(
        ExecutionScope(task_id="mine", metadata={"llm_session_id": "s"}),
    )
    try:
        ref = _overflow.spill_compacted_body("bash", "mine " * 400)
    finally:
        reset_current_execution_scope(token)
    assert ref

    assert _overflow.cleanup_overflow_process() == 1
    assert (stranger / "theirs.md").read_text(encoding="utf-8") == "theirs"


def test_overflow_dir_sees_a_backend_configured_only_in_config(
    tmp_path, monkeypatch,
) -> None:
    """``native`` runs commands on this filesystem, so the canonical ``/spill``
    is not an alias there and may name an unrelated directory. Reading the raw
    env var misses a backend that only ``get_config()`` knows about."""
    store = tmp_path / "store"
    monkeypatch.setenv("APODEX_SPILL_DIR", str(store))
    monkeypatch.delenv("SANDBOX_BACKEND", raising=False)
    monkeypatch.setattr(
        "plugins.tools._overflow._resolved_backend", lambda: "native",
    )

    visible = _overflow._overflow_dir("t:s")[1]
    assert visible.startswith(str(store))
    assert not visible.startswith("/spill")


def test_resolved_backend_survives_a_misconfigured_backend(monkeypatch) -> None:
    """Spill is a diagnostic aid; a bad backend value must not raise through it."""
    monkeypatch.setenv("SANDBOX_BACKEND", "not-a-backend")
    assert _overflow._resolved_backend() == ""


def test_llm_summary_empty_rollback_uses_source_messages() -> None:
    class EmptyLLM:
        async def chat(self, messages: list[dict]) -> SimpleNamespace:
            return SimpleNamespace(content="")

    history = _tool_history("x" * 5_000)
    compactor = LLMSummaryCompactor(
        summary_llm=EmptyLLM(),
        failure_fallback="deterministic",
    )
    result = asyncio.run(compactor.compact(
        history,
        keep_recent=1,
        compress_all_tool_results=True,
    ))
    assert len(result) == 3
    # This marker only exists in ``source_messages`` after tool-result compression;
    # rolling back over the original ``messages`` would contain raw x characters.
    assert "[Compressed tool result:" in result[1]["content"]


def test_a_summary_quoting_the_header_is_left_alone() -> None:
    """``format_conversation_for_summary`` renders the previous index verbatim, so
    the summarizer can quote its header back.

    That used to matter twice over: the index was appended INTO the summary
    message, so updating it meant locating where the index began inside prose the
    summarizer wrote, and cutting at the first occurrence of the header discarded
    every finding after the quote. The index is now its own message, so a summary
    is never edited at all.
    """
    summary = {
        "role": "user",
        "content": (
            "[Compacted summary of earlier turns]\n\n"
            "FINDINGS A: the study reports a 3.2x speedup\n"
            f"The recovery index said: {_SPILL_MANIFEST_HEADER}\n"
            "- the agent then compared three vendors\n"
            "FINDINGS B: the ablation contradicts the abstract"
        ),
    }
    messages = [{"role": "system", "content": "system"}, summary]

    out = TieredCompactor._with_spill_manifest(
        messages, ["/workspace/.spill/session/real.md"],
    )

    kept = next(m for m in out if "FINDINGS A" in text_of(m.get("content")))
    assert kept["content"] == summary["content"]
    assert "FINDINGS B: the ablation contradicts the abstract" in kept["content"]
    index = next(m for m in out if m.get("spill_refs"))
    assert index["spill_refs"] == ["/workspace/.spill/session/real.md"]
    assert "/workspace/.spill/session/real.md" in text_of(index["content"])


def test_bullets_in_a_summary_body_are_not_refs() -> None:
    """Appending the index into the summary made every ``- `` bullet in the
    summary a harvest candidate, filling the bounded index with dead paths and
    evicting real ones each pass. Nothing reads bullets any more."""
    summary = {
        "role": "user",
        "content": (
            "[Compacted summary]\n\n"
            "- Key evidence: see /workspace/.spill/session/cited.md\n"
            "- /workspace/.spill/session/quoted.md is the important one.md\n"
        ),
    }

    assert TieredCompactor._spill_refs([summary]) == []


@pytest.mark.asyncio
async def test_tier2_summarizer_never_sees_the_manifest_header() -> None:
    """Belt to the brace above: the summarizer cannot echo a header it was
    never shown. ``compact_messages`` already drops manifests on the
    deterministic path; the LLM path did not."""
    prompts: list[str] = []

    class _CapturingLLM:
        async def chat(self, messages):
            prompts.append(str(messages[0].get("content", "")))
            return SimpleNamespace(content="[Compacted summary] tidy summary")

    compactor = LLMSummaryCompactor(summary_llm=_CapturingLLM())
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "research the topic"},
        {
            "role": "user",
            "content": f"{_SPILL_MANIFEST_HEADER}\n- /workspace/.spill/s/a.md",
        },
        *[
            item
            for idx in range(4)
            for item in (
                {"role": "assistant", "content": f"step {idx}"},
                {"role": "user", "content": f"continue {idx}"},
            )
        ],
    ]

    await compactor.compact(messages, keep_recent=2)

    assert prompts, "summarizer was not called"
    assert all(_SPILL_MANIFEST_HEADER not in prompt for prompt in prompts)


@pytest.mark.asyncio
async def test_manifest_cost_does_not_flip_the_no_relief_verdict() -> None:
    """A compaction win SMALLER than the recovery index must still count as relief.

    ``_last_no_relief_estimate`` suppresses the Tier 2 attempt on every later
    pass until the estimate grows 10%. Measuring it AFTER the manifest was
    appended charged the compactor up to ~3 KB of its own index, so a pass that
    genuinely freed history — by less than the index costs — was recorded as a
    failure and stopped being retried, penalising a compactor that worked. The
    relief-target check a few lines earlier already excludes the manifest for
    exactly this reason; the two disagreed.

    The orderings only differ while ``pre_manifest < tier1 <= post_manifest``, so
    the summary is sized into that band before the assertion means anything.
    """
    def _history() -> list[dict]:
        messages: list[dict] = [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "research the topic"},
        ]
        # >= 22 tool results so Tier 1 spills the 20 the manifest can hold.
        for idx in range(22):
            call_id = f"call-{idx}"
            messages.append({
                "role": "assistant",
                "content": f"fetching {idx}",
                "tool_calls": [
                    {"id": call_id, "function": {"name": "web_fetch", "arguments": "{}"}},
                ],
            })
            messages.append({
                "role": "tool", "content": "b" * 200, "tool_call_id": call_id,
            })
        messages.append({"role": "user", "content": "continue"})
        return messages

    def _ref(index: int) -> str:
        return f"/workspace/nested/project/checkout/.spill/s-{index:02d}/{'a' * 40}.md"

    async def _compact(filler: str) -> tuple[int, int]:
        counter = itertools.count()

        class _LLM:
            async def chat(self, messages):
                return SimpleNamespace(content=filler)

        compactor = TieredCompactor(
            keep_tool_result=2,
            summary_llm=_LLM(),
            relief_target=1,  # unreachable, so the whole tier ladder runs
            spill=lambda name, body: _ref(next(counter)),
        )
        out = await compactor.compact(_history(), keep_recent=2)
        return estimate_tokens(out), compactor._last_no_relief_estimate

    # Tier 1 must be measured WITH the spill placeholders it emits: each carries
    # a ``[Full text]`` line, so a spill-less Tier 1 is not the real threshold.
    tier1_counter = itertools.count()
    tier1_tokens = estimate_tokens(
        KeepLastNToolResultsCompactor(
            keep_tool_result=2,
            spill=lambda name, body: _ref(next(tier1_counter)),
        ).compact(_history(), 2),
    )
    manifest_tokens = estimate_tokens([{
        "role": "user",
        "content": _SPILL_MANIFEST_HEADER + "\n"
        + "\n".join(f"- {_ref(idx)}" for idx in range(20)),
    }])

    # Bisect the summary length so the POST-manifest size lands just above
    # Tier 1 while the pre-manifest size is still below it. Bounded tightly:
    # a wide upper bound would build multi-hundred-KB summaries per probe.
    low, high = 1, 20_000
    for _ in range(20):
        mid = (low + high) // 2
        if (await _compact("d" * mid))[0] < tier1_tokens + manifest_tokens // 3:
            low = mid
        else:
            high = mid

    post_manifest, no_relief = await _compact("d" * low)
    assert post_manifest - manifest_tokens < tier1_tokens <= post_manifest, (
        f"not in the disputed band: tier1={tier1_tokens} "
        f"post={post_manifest} manifest={manifest_tokens}"
    )
    assert no_relief == 0, (
        "a compaction pass that freed real history was recorded as no-relief "
        "because its own recovery index was charged against it"
    )


@pytest.mark.parametrize(
    ("backend", "expected"),
    [("native", "physical"), ("e2b", ""), ("container", "physical"), ("bwrap", "canonical")],
)
def test_mount_fallback_names_spill_per_backend(
    tmp_path, monkeypatch, backend: str, expected: str,
) -> None:
    """What a model command can name differs per backend, and must.

    ``native`` and production ``container`` commands run on THIS filesystem, so
    the physical path IS the path they can name. bwrap reaches the store through
    the read-only mount and gets the canonical path. A remote backend cannot see
    this filesystem at all and must be told nothing rather than something
    unresolvable.
    """
    store = tmp_path / "store"
    monkeypatch.setenv("SANDBOX_BACKEND", backend)
    monkeypatch.setenv("APODEX_SPILL_DIR", str(store))
    # ``container`` here means production container mode. The inner-bwrap opt-in
    # is read from the environment on every call, so an exported flag would flip
    # this case to the canonical mount and fail the assertion for a reason that
    # has nothing to do with the code under test.
    monkeypatch.delenv("FRONTIER_AGENT_CONTAINER_INNER_BWRAP", raising=False)

    target, visible = _overflow._overflow_dir("task:session")

    # Outside every write root, whatever the backend.
    assert target.is_relative_to(store), target
    if expected == "physical":
        assert visible == str(target)
    elif expected == "canonical":
        assert visible == f"/spill/{target.name}"
    else:
        assert visible == ""


def test_container_inner_bwrap_names_the_canonical_spill_mount(
    tmp_path, monkeypatch,
) -> None:
    from plugins.tools import _sandbox

    monkeypatch.setenv("SANDBOX_BACKEND", "container")
    monkeypatch.setenv("APODEX_SPILL_DIR", str(tmp_path / "store"))
    monkeypatch.setenv("FRONTIER_AGENT_CONTAINER_INNER_BWRAP", "1")
    monkeypatch.setattr(_sandbox, "bwrap_available", lambda: True)

    target, visible = _overflow._overflow_dir("task:session")

    assert target.is_relative_to(tmp_path / "store")
    assert visible == f"/spill/{target.name}"


def test_global_result_cap_spills_before_discarding_the_middle(tmp_path, monkeypatch) -> None:
    """The 150K cap is the ONLY one that applies to the uncapped tools.

    web_fetch / web_search / read_file set ``max_result_chars=0`` because content
    density is judged to reward full bodies, which makes ``maybe_overflow`` a
    documented no-op for them. This cap then truncated with nothing on disk, so a
    300K paper lost its second half irrecoverably — the opposite of what every
    other truncation here does.

    What is cut is now the MIDDLE, and the pointer is charged to the cap rather
    than added on top of it.
    """
    from frontier_agent.core.runtime.loop.tool_exec import (
        TOOL_RESULT_MAX_CHARS,
        _truncate_with_recovery,
    )

    store = tmp_path / "store"
    monkeypatch.setenv("SANDBOX_BACKEND", "container")
    monkeypatch.setenv("APODEX_SPILL_DIR", str(store))

    body = "HEAD\n" + ("x" * (TOOL_RESULT_MAX_CHARS * 2)) + "\nTAIL-MARKER"
    out = _truncate_with_recovery("web_fetch", body)

    # The pointer is inside the cap, not appended past it.
    assert len(out) <= TOOL_RESULT_MAX_CHARS
    # Head AND tail survive; the gap between them is named.
    assert out.startswith("HEAD\n")
    assert "TAIL-MARKER" in out
    assert "chars elided" in out
    assert f"saved read-only at {store}/" in out
    spilled = list(store.rglob("*.md"))
    assert len(spilled) == 1, spilled
    # The elided middle is recoverable from the file the marker names.
    assert spilled[0].read_text(encoding="utf-8").endswith(body)
    assert spilled[0].name in out


def test_global_result_cap_degrades_when_spill_is_unavailable(monkeypatch) -> None:
    """Spill is a diagnostic aid; a backend with no readable filesystem must get
    the plain byte count rather than a failed tool call."""
    from frontier_agent.core.runtime.loop import tool_exec

    monkeypatch.setattr(
        _overflow, "spill_compacted_body",
        lambda name, body: (_ for _ in ()).throw(OSError("read-only fs")),
    )
    body = "y" * (tool_exec.TOOL_RESULT_MAX_CHARS + 10)

    out = tool_exec._truncate_with_recovery("web_fetch", body)

    assert out.endswith(
        f"[... only part of this {len(body):,}-char result is shown; "
        "the remainder is not readable from this backend.]"
    )
    assert ".spill" not in out
    assert len(out) <= tool_exec.TOOL_RESULT_MAX_CHARS


# ── the typed recovery index ────────────────────────────────────────────


def test_tier1_placeholder_carries_its_spill_path_as_data() -> None:
    """The text is for the model, the field is for the compactor."""
    compactor = KeepLastNToolResultsCompactor(
        keep_tool_result=0, spill=lambda name, body: f"/ws/.spill/s/{name}.md",
    )
    messages = [
        {"role": "user", "content": "go"},
        assistant_msg("", tool_calls=[{
            "id": "c1", "type": "function",
            "function": {"name": "bash", "arguments": "{}"},
        }]),
        {"role": "tool", "tool_call_id": "c1", "content": "x" * 4_000},
    ]

    out = compactor.compact(messages, keep_recent=1)

    placeholder = next(m for m in out if m.get("role") == "tool")
    assert placeholder["spill_refs"] == ["/ws/.spill/s/bash.md"]
    # Still legible to the model, which is what actually recovers the body.
    assert "/ws/.spill/s/bash.md" in text_of(placeholder["content"])
    assert TieredCompactor._spill_refs(out) == ["/ws/.spill/s/bash.md"]


def test_a_placeholder_without_a_spill_path_carries_no_field() -> None:
    compactor = KeepLastNToolResultsCompactor(keep_tool_result=0, spill=None)
    messages = [
        {"role": "user", "content": "go"},
        assistant_msg("", tool_calls=[{
            "id": "c1", "type": "function",
            "function": {"name": "bash", "arguments": "{}"},
        }]),
        {"role": "tool", "tool_call_id": "c1", "content": "x" * 4_000},
    ]

    out = compactor.compact(messages, keep_recent=1)

    placeholder = next(m for m in out if m.get("role") == "tool")
    assert "spill_refs" not in placeholder


def test_a_legacy_full_text_placeholder_is_still_harvested() -> None:
    """A history checkpointed before the field existed. A fixed prefix on its own
    line is a format, not a shape guess, so reading it stays exact."""
    legacy = {
        "role": "tool",
        "tool_call_id": "c1",
        "content": "[tool result omitted]\n[Full text] /ws/.spill/old/a.md",
    }

    assert TieredCompactor._spill_refs([legacy]) == ["/ws/.spill/old/a.md"]


def test_the_index_is_replaced_not_duplicated_across_passes() -> None:
    messages = [{"role": "system", "content": "s"}, {"role": "user", "content": "q"}]

    first = TieredCompactor._with_spill_manifest(messages, ["/ws/.spill/s/a.md"])
    second = TieredCompactor._with_spill_manifest(
        first, ["/ws/.spill/s/a.md", "/ws/.spill/s/b.md"],
    )

    indexes = [m for m in second if m.get("spill_refs")]
    assert len(indexes) == 1
    assert indexes[0]["spill_refs"] == ["/ws/.spill/s/a.md", "/ws/.spill/s/b.md"]
    assert len(second) == len(first)


def test_the_index_never_reaches_a_provider() -> None:
    """``spill_refs`` is in-process bookkeeping; ``for_wire`` is what enforces it."""
    from frontier_agent.core.messages import WIRE_MESSAGE_KEYS, for_wire

    out = TieredCompactor._with_spill_manifest(
        [{"role": "user", "content": "q"}], ["/ws/.spill/s/a.md"],
    )
    index = next(m for m in out if m.get("spill_refs"))

    assert "spill_refs" not in WIRE_MESSAGE_KEYS
    wire = for_wire(out)
    assert all("spill_refs" not in m for m in wire)
    # The paths still reach the model as text, which is the point of rendering.
    assert "/ws/.spill/s/a.md" in text_of(
        next(m for m in wire if text_of(m.get("content")) == text_of(index["content"]))["content"],
    )
