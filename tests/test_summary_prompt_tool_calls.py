"""The compaction summarizer must be able to see what the prompt asks it to keep.

`COMPACTION_PROMPT` asks, under PRESERVE EXACTLY, for the exact search queries
already issued. A query lives in the assistant message's
`tool_calls[i]["function"]["arguments"]`, which the renderer used to drop — so
the instruction was asking for something absent from the input, and a model
asked to preserve an absent thing invents it.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

from frontier_agent.core.messages import assistant_msg, tool_msg, user_msg
from frontier_agent.infra.llm.summary_prompt import (
    _TOOL_ARGS_MAX_CHARS,
    COMPACTION_PROMPT,
    format_conversation_for_summary,
)


def _search_call(query: str, call_id: str = "call_1") -> dict:
    return {
        "id": call_id,
        "type": "function",
        "function": {
            "name": "web_search",
            "arguments": json.dumps({"q": query}),
        },
    }


def test_the_query_reaches_the_summarizer() -> None:
    rendered = format_conversation_for_summary([
        user_msg("who is the doctor youtuber born june 24"),
        assistant_msg("", tool_calls=[_search_call("doctor youtuber june 24")]),
        # The successful single-query result does NOT echo the query, which is
        # why the tool result cannot stand in for the tool call.
        tool_msg("## Search Results\n[1] **A page** — https://example.test", "call_1"),
    ])

    assert "-> web_search(" in rendered
    assert "doctor youtuber june 24" in rendered


def test_the_prompt_and_the_renderer_agree() -> None:
    """Guard against the instruction and the input drifting apart again."""
    assert "exact search queries already issued" in COMPACTION_PROMPT
    assert "## Queries already run" in COMPACTION_PROMPT

    rendered = format_conversation_for_summary([
        assistant_msg("", tool_calls=[_search_call("alpha")]),
    ])
    assert "alpha" in rendered


def test_visible_text_and_tool_calls_both_survive() -> None:
    rendered = format_conversation_for_summary([
        assistant_msg("I will check two angles.", tool_calls=[_search_call("beta")]),
    ])

    assert "I will check two angles." in rendered
    assert "beta" in rendered


def test_long_arguments_are_capped() -> None:
    payload = "x" * 5000
    rendered = format_conversation_for_summary([
        assistant_msg("", tool_calls=[{
            "id": "call_1",
            "type": "function",
            "function": {"name": "bash", "arguments": payload},
        }]),
    ])

    assert "…" in rendered
    assert len(rendered) < _TOOL_ARGS_MAX_CHARS + 200


def test_messages_without_tool_calls_are_unchanged() -> None:
    rendered = format_conversation_for_summary([user_msg("plain question")])

    assert rendered == "[user]\nplain question"


def test_a_nameless_tool_call_is_skipped() -> None:
    rendered = format_conversation_for_summary([
        assistant_msg("", tool_calls=[{
            "id": "call_1",
            "type": "function",
            "function": {"name": "", "arguments": "{}"},
        }]),
    ])

    assert "->" not in rendered


# ── prompt style selection ──────────────────────────────────────────────


def test_the_handoff_prompt_asks_for_what_a_coding_resume_needs() -> None:
    """The research shape preserves candidates and queries; a resumed coding run
    needs the commands, the paths, what they returned, and what is unverified."""
    from frontier_agent.infra.llm.summary_prompt import HANDOFF_COMPACTION_PROMPT as H

    assert "{conversation}" in H
    for demand in (
        "exact commands",
        "exact file paths",
        "error text",
        "unverified",
        "spill",           # recovery paths must survive the summary as prose too
        "forward plan",
        "do NOT know",
    ):
        assert demand in H, demand
    # First person, and explicitly not a third-party report.
    assert "note to YOURSELF" in H
    assert "not a third-party report" in H


def test_the_style_is_selectable_and_defaults_to_auto() -> None:
    from frontier_agent.infra.config import FrontierAgentConfig

    assert (
        FrontierAgentConfig.model_fields["compaction_prompt_style"].default == "auto"
    )


def test_an_explicit_style_forces_one_shape_on_everything(monkeypatch) -> None:
    """The A/B arms pin a style; pinning must beat the tool-mix dispatch."""
    from frontier_agent.infra.config import get_config
    from frontier_agent.infra.llm.summary_prompt import (
        HANDOFF_COMPACTION_PROMPT,
        RESEARCH_COMPACTION_PROMPT,
        compaction_prompt,
    )

    coding = [assistant_msg("", tool_calls=[_call("bash", {"command": "pytest"})])]
    monkeypatch.setattr(get_config(), "compaction_prompt_style", "research")
    assert compaction_prompt(coding) is RESEARCH_COMPACTION_PROMPT
    monkeypatch.setattr(get_config(), "compaction_prompt_style", "handoff")
    assert compaction_prompt([]) is HANDOFF_COMPACTION_PROMPT


def test_an_unknown_style_keeps_the_shape_already_in_use(monkeypatch) -> None:
    """A typo in an env var must not silently change what compaction preserves."""
    from frontier_agent.infra.config import get_config
    from frontier_agent.infra.llm.summary_prompt import (
        RESEARCH_COMPACTION_PROMPT,
        compaction_prompt,
    )

    monkeypatch.setattr(get_config(), "compaction_prompt_style", "handoffs")
    assert compaction_prompt() is RESEARCH_COMPACTION_PROMPT


async def test_the_summarizer_uses_the_configured_style(monkeypatch) -> None:
    """End to end: the style reaches the prompt the summary LLM is sent."""
    from frontier_agent.core.runtime.loop.compact_llm import LLMSummaryCompactor
    from frontier_agent.infra.config import get_config

    seen: list[str] = []

    class _LLM:
        async def chat(self, messages, **kwargs):
            seen.append(messages[0]["content"])
            return SimpleNamespace(content="## Summary\nwork so far")

    monkeypatch.setattr(get_config(), "compaction_prompt_style", "handoff")
    compactor = LLMSummaryCompactor(summary_llm=_LLM())
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "fix the failing test"},
        {"role": "assistant", "content": "ran pytest"},
        {"role": "user", "content": "and now?"},
    ]

    await compactor.compact(messages, keep_recent=1)

    assert seen, "the summarizer was never called"
    assert "note to YOURSELF" in seen[0]


# ── auto: prompt shape per conversation ─────────────────────────────────


def _call(name: str, args: dict) -> dict:
    return {
        "id": f"c_{name}",
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(args)},
    }


def test_auto_picks_handoff_for_machine_work(monkeypatch) -> None:
    from frontier_agent.infra.config import get_config
    from frontier_agent.infra.llm.summary_prompt import (
        HANDOFF_COMPACTION_PROMPT,
        compaction_prompt,
    )

    monkeypatch.setattr(get_config(), "compaction_prompt_style", "auto")
    messages = [
        assistant_msg("", tool_calls=[_call("bash", {"command": "pytest -x"})]),
        assistant_msg("", tool_calls=[_call("grep_search", {"pattern": "def main"})]),
        assistant_msg("", tool_calls=[_call("write_file", {"path": "a.py"})]),
        # A coding task legitimately reads documentation; one fetch is not enough
        # to make it research.
        assistant_msg("", tool_calls=[_call("web_fetch", {"url": "https://docs"})]),
    ]

    assert compaction_prompt(messages) is HANDOFF_COMPACTION_PROMPT


def test_auto_leaves_web_research_on_the_shape_it_already_had(monkeypatch) -> None:
    from frontier_agent.infra.config import get_config
    from frontier_agent.infra.llm.summary_prompt import (
        RESEARCH_COMPACTION_PROMPT,
        compaction_prompt,
    )

    monkeypatch.setattr(get_config(), "compaction_prompt_style", "auto")
    messages = [
        assistant_msg("", tool_calls=[_call("web_search", {"query": "who wrote x"})]),
        assistant_msg("", tool_calls=[_call("web_fetch", {"url": "https://a"})]),
        assistant_msg("", tool_calls=[_call("scholar_search", {"query": "y"})]),
        # A research task legitimately shells out to tabulate what it found.
        assistant_msg("", tool_calls=[_call("bash", {"command": "sort hits.txt"})]),
    ]

    assert compaction_prompt(messages) is RESEARCH_COMPACTION_PROMPT


def test_auto_breaks_ties_and_empties_toward_the_incumbent(monkeypatch) -> None:
    """Misrouting research to handoff loses candidate/query preservation;
    misrouting coding to research only keeps what it always had."""
    from frontier_agent.infra.config import get_config
    from frontier_agent.infra.llm.summary_prompt import (
        RESEARCH_COMPACTION_PROMPT,
        compaction_prompt,
    )

    monkeypatch.setattr(get_config(), "compaction_prompt_style", "auto")
    tie = [
        assistant_msg("", tool_calls=[_call("bash", {"command": "ls"})]),
        assistant_msg("", tool_calls=[_call("web_search", {"query": "z"})]),
    ]
    assert compaction_prompt(tie) is RESEARCH_COMPACTION_PROMPT
    assert compaction_prompt([]) is RESEARCH_COMPACTION_PROMPT
    assert compaction_prompt(None) is RESEARCH_COMPACTION_PROMPT
    # Orchestration and tool discovery say nothing about the kind of work.
    neutral = [
        assistant_msg("", tool_calls=[_call("delegate_subtask", {"task": "t"})]),
        assistant_msg("", tool_calls=[_call("tool_search", {"q": "t"})]),
    ]
    assert compaction_prompt(neutral) is RESEARCH_COMPACTION_PROMPT


def test_auto_reads_the_flattened_call_shape_too() -> None:
    """Observers and tests carry {"name": ...} rather than the wire form."""
    from frontier_agent.infra.llm.summary_prompt import _tool_call_names

    flat = [{"role": "assistant", "tool_calls": [{"name": "bash", "args": {}}]}]
    assert _tool_call_names(flat) == ["bash"]
