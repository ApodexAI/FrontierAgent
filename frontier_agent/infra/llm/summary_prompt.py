"""Shared structured-summary prompt for message-history compaction."""

from __future__ import annotations

from frontier_agent.core.messages import Message, text_of

__all__ = [
    "COMPACTION_PROMPT",
    "HANDOFF_COMPACTION_PROMPT",
    "RESEARCH_COMPACTION_PROMPT",
    "compaction_prompt",
    "format_conversation_for_summary",
]


# Shaped for research / QA: the units it preserves are candidates, sources and
# queries. See ``HANDOFF_COMPACTION_PROMPT`` for the long-run coding shape, and
# ``compaction_prompt`` for how one is chosen.
RESEARCH_COMPACTION_PROMPT = """\
Summarize the conversation that follows so the assistant agent can \
continue from a smaller context. The ORIGINAL question is in the \
system message and is NOT being summarized — it remains accessible.

PRESERVE EXACTLY (do NOT compress these):
- All entity names ever mentioned as candidates (people, places, \
organizations, works, songs, etc).
- Candidates that were RULED OUT and the specific reason \
(so future turns don't re-explore them).
- Source URLs and titles already consulted (so future turns don't \
re-fetch the same pages).
- The exact search queries already issued, verbatim (so future turns \
don't re-run them). List them even when they returned nothing useful.
- Verified facts that confirm or contradict each clue against each \
candidate.
- Any partial answer hypotheses that are still in play.
- Numeric values, dates, or quoted phrases that came up in evidence.

OK TO COMPRESS:
- Tool call mechanics (drop the JSON args, EXCEPT a search query — that \
one is preserved above; keep the substantive result).
- The RESULTS of a search that returned nothing useful (keep the query \
itself under PRESERVE above; the finding compresses to "no useful hits").
- Reasoning deliberation prose (keep conclusions, drop intermediate \
musings).
- Repeated framing or restatements of the question.

OUTPUT FORMAT — write a concise structured summary that the agent can \
read at the start of its next turn:

## Investigation so far
<one paragraph: what high-level direction has been tried>

## Candidates
- Confirmed / strong: <name + which clues it satisfies>
- Ruled out: <name + reason>
- Still open: <name + status>

## Sources consulted
- <URL or title> — <one-line takeaway>

## Queries already run
- <exact query text> — <what it yielded, or "no useful hits">

## Verified facts
- <fact + which candidate / clue it relates to>

## Next steps suggested by current state
<one or two lines>

Conversation to summarize:
{conversation}"""


# Shaped for long-run coding and file work, where what the next turn needs is
# not a list of candidates but the exact state of the work: which commands ran,
# which paths they touched, what they actually returned, and what is still
# unverified. Written first-person on the same reasoning that kimi-code's
# compaction instruction uses — a note the agent writes to itself, not a report
# about someone else's work — because a third-party summary drops the operational
# detail (the exact command, the exact error line) that a resuming agent has to
# re-derive by re-running things.
HANDOFF_COMPACTION_PROMPT = """\
You are about to run out of context. Write a note to YOURSELF so you can pick \
this task up after the conversation below is discarded.

Write it first person, present tense, as your own continuing train of thought — \
not a third-party report about someone else's work. Write it in the language the \
conversation has been using. The next turn will see your most recent user \
messages and this note, and nothing else: every tool call and tool result below \
will be gone.

PRESERVE EXACTLY — these cannot be re-derived cheaply, and a resuming agent that \
guesses them will redo work or corrupt state:
- The exact commands that were run, verbatim, and whether each succeeded.
- The exact file paths that were read or written, and what changed in each.
- What the results actually SAID: the concrete values returned, the key output \
lines, the exact error text, the signature or schema a lookup revealed. Not \
"the tests failed" but which test and what the assertion said.
- Any recovery path already mentioned in the conversation (a spilled tool result \
under a .spill or spill directory). Reproduce the path character for character; \
it is how the discarded detail is retrieved.
- Decisions already settled, kept SEPARATE from questions still open, so the \
next turn neither reopens a closed choice nor treats an undecided one as decided.
- Anything an earlier step CLAIMED but never verified — "tests pass", "the fix \
works", "the file was created". Say plainly that it is unverified.

OK TO COMPRESS OR DROP:
- Intermediate attempts that were superseded. Keep only the final working \
version of any code or command.
- Errors already diagnosed and fixed.
- Deliberation prose: keep the conclusion, drop the reasoning that led to it.
- Repeated restatements of the task.

Then set out the forward plan, and invest in it: right now you hold more context \
on this task than you ever will again. Give the exact next command or tool call, \
the remaining sequence after it, the decisions you have already made for those \
later steps, and the obstacles you can already foresee with how you mean to \
handle each. Anything you settle here is one less thing to rediscover.

Name what you still do NOT know that the next step depends on — files referenced \
but never read, an API assumed but never inspected, a question the user has not \
answered — so the next turn goes and checks instead of assuming.

Be concise and proportional: a long multi-step task earns detail, a nearly \
finished one earns a few sentences. Do not pad.

Conversation to summarize:
{conversation}"""


# Backwards-compatible name for the research shape, which was the only one.
COMPACTION_PROMPT = RESEARCH_COMPACTION_PROMPT

_PROMPT_STYLES = {
    "research": RESEARCH_COMPACTION_PROMPT,
    "handoff": HANDOFF_COMPACTION_PROMPT,
}

# ``ToolMeta.category`` values that mean the agent was doing research: reading
# the web. Everything else that a tool call can be — compute, file, search
# (glob/grep over the local filesystem), finance sandboxes — is work on a
# machine, whose state is what a resumed run needs to know about.
_RESEARCH_CATEGORIES = frozenset({"web"})
# Categories that say nothing either way and must not cast a vote.
_NEUTRAL_CATEGORIES = frozenset({"meta", "orchestration", ""})


def _tool_call_names(messages: list[Message]) -> list[str]:
    """Every tool name called across ``messages``, in order, with repeats.

    Accepts both shapes a call can arrive in: the wire form
    ``{"function": {"name": ...}}`` that ``Message.tool_calls`` carries, and the
    flattened ``{"name": ...}`` that some observers and tests use.
    """
    names: list[str] = []
    for message in messages:
        for call in message.get("tool_calls") or []:
            function = call.get("function") if isinstance(call, dict) else None
            name = ""
            if isinstance(function, dict):
                name = str(function.get("name") or "")
            if not name and isinstance(call, dict):
                name = str(call.get("name") or "")
            if name:
                names.append(name)
    return names


def _is_machine_work(messages: list[Message]) -> bool:
    """Whether this conversation is work on a machine rather than web research.

    Majority vote over the categorised tool calls, ties and empties going to
    research — the incumbent shape. Deliberately conservative in that direction:
    a research run misclassified as coding would lose the candidate and query
    preservation that is the whole point of the research prompt, while a coding
    run misclassified as research merely keeps the summary it has had all along.

    The signal is the tool MIX, not any single call: a coding task legitimately
    fetches documentation, and a research task legitimately runs ``bash`` to
    tabulate what it found.
    """
    from plugins.tools.meta import get_tool_meta

    machine = research = 0
    for name in _tool_call_names(messages):
        category = get_tool_meta(name).category
        if category in _NEUTRAL_CATEGORIES:
            continue
        if category in _RESEARCH_CATEGORIES:
            research += 1
        else:
            machine += 1
    return machine > research


def compaction_prompt(messages: list[Message] | None = None) -> str:
    """The compaction prompt for the configured style.

    Read per call so an A/B run can switch arms through
    ``COMPACTION_PROMPT_STYLE`` without a rebuild. An unknown or unreadable
    value falls back to ``research``, the shape that has been in use — a
    misconfigured value must not silently change what compaction preserves.

    ``auto`` (the default) dispatches on the conversation's own tool mix, for the
    reason the apex A/B surfaced: at identical task success the handoff shape
    produced smaller prompts on a coding benchmark, because the research shape's
    candidate / source / query sections are empty or padded there. The converse
    risk is real too, which is why this routes rather than replaces — on a
    web-dominated conversation ``auto`` returns exactly the prompt that was in
    use before it existed. Without ``messages`` there is nothing to dispatch on,
    so it also falls back to research.
    """
    from frontier_agent.infra.config import get_config

    try:
        style = str(get_config().compaction_prompt_style).strip().lower()
    except Exception:
        return RESEARCH_COMPACTION_PROMPT
    if style == "auto":
        return (
            HANDOFF_COMPACTION_PROMPT
            if messages and _is_machine_work(messages)
            else RESEARCH_COMPACTION_PROMPT
        )
    return _PROMPT_STYLES.get(style, RESEARCH_COMPACTION_PROMPT)


# Per-call cap on rendered tool arguments. Generous enough for a search
# batch or a file path, small enough that a hundred turns of ``bash``
# payloads cannot dominate the summarizer's window.
_TOOL_ARGS_MAX_CHARS = 300


def _render_tool_calls(message: Message) -> str:
    """Render an assistant message's tool calls as ``-> name(args)`` lines.

    Without this the summarizer never sees a single tool argument, because
    only ``role`` and ``content`` were rendered — and a search query lives in
    ``tool_calls[i]["function"]["arguments"]``. The prompt above asks for the
    exact queries already issued under PRESERVE EXACTLY, and a model asked to
    preserve something absent from its input will invent it. A fabricated
    "already run" list is worse than no list: it steers later turns away from
    searches the agent never actually tried.

    The tool result text cannot substitute. ``web_search``'s single-query path
    formats results without echoing ``q``, and ``web_search_aligned`` never
    echoes it at all — while the *empty*-result strings DO carry the query, so
    relying on results would preserve exactly the failed searches and lose the
    useful ones.
    """
    calls = message.get("tool_calls") or []
    lines: list[str] = []
    for call in calls:
        function = call.get("function") or {}
        name = str(function.get("name") or "")
        if not name:
            continue
        args = str(function.get("arguments") or "")
        if len(args) > _TOOL_ARGS_MAX_CHARS:
            args = args[:_TOOL_ARGS_MAX_CHARS] + "…"
        lines.append(f"-> {name}({args})")
    return "\n".join(lines)


def format_conversation_for_summary(
    messages: list[Message],
    *,
    preserve_tool_result_ids: frozenset[str] = frozenset(),
) -> str:
    """Render messages as a plain-text dialogue for the summarizer LLM.

    Long tool results (>4000 chars) are truncated to 3500 chars + ellipsis
    so a single noisy turn doesn't dominate the summarizer's input window.
    ``preserve_tool_result_ids`` exempts structured fan-in whose middle must
    remain visible; callers use it only for explicitly protected tool names.

    Assistant tool calls are rendered with their (truncated) arguments — see
    :func:`_render_tool_calls` for why the prompt's query-preservation rule
    depends on it.
    """
    parts: list[str] = []
    for m in messages:
        role = m.get("role", "") or ""
        content = text_of(m.get("content"))
        preserve_full = (
            role == "tool"
            and str(m.get("tool_call_id") or "") in preserve_tool_result_ids
        )
        if len(content) > 4000 and not preserve_full:
            content = content[:3500] + "\n...[truncated]..."
        rendered_calls = _render_tool_calls(m)
        if rendered_calls:
            content = f"{content}\n{rendered_calls}" if content else rendered_calls
        parts.append(f"[{role}]\n{content}")
    return "\n\n".join(parts)
